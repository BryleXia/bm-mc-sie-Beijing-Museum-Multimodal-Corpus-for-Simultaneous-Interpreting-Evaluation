#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保加利亚语对齐终极修复版 v2

问题根因：
  - whisper_transcript_bg.json 有740个fragment，每句话被切成了~3.5个片段
  - 之前的方法都是逐segment匹配，或者词级匹配，都会失败
  - alignment_bg_whisper.json 的贪心窗口太小(50词)且允许重叠

解决方案：
  - 合并连续segment → 形成完整句子 → 再与读稿匹配
  - 严格单调推进（current_seg只增不减）
  - 动态调整合并数量（1~max_merge个segment）
  - 允许在当前位置前方一定范围内搜索（处理停顿/重录产生的空白片段）

用法：
  python bg_align_final.py
  python bg_align_final.py --skip-split   # 只对齐不切音频
  python bg_align_final.py --dry-run      # 只打印结果不保存
"""

import json
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ─────────────────────────── 路径配置 ───────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
RECORDING_DIR  = PROJECT_ROOT / "交付" / "录音"
PROCESSED_DIR  = RECORDING_DIR / "处理后"
ALIGNMENT_DIR  = RECORDING_DIR / "对齐"
SCRIPT_DIR     = PROJECT_ROOT / "交付" / "朗读稿件"

LANG           = "bg"
AUDIO_FILE     = PROCESSED_DIR / LANG / f"{LANG}_full.wav"
TRANSCRIPT_FILE= ALIGNMENT_DIR / f"whisper_transcript_{LANG}.json"
READING_SCRIPT = SCRIPT_DIR / f"reading_script_{LANG}.txt"
OUTPUT_ALIGN   = ALIGNMENT_DIR / f"alignment_{LANG}_final.json"
OUTPUT_SENTENCES= PROCESSED_DIR / LANG / "sentences_final"


# ─────────────────────────── 文本工具 ───────────────────────────
def normalize(text: str) -> str:
    """标准化文本用于比较（保留西里尔字母）"""
    text = text.lower()
    # 保留字母、数字、空格（包含西里尔Unicode范围）
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    """计算两个归一化文本的相似度"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ─────────────────────────── 数据加载 ───────────────────────────
def load_transcript() -> List[Dict]:
    """
    加载Whisper转录并过滤幻觉段。

    背景：whisper_transcript_bg.json中，segment 7~49（39s~1417s）全是
    Whisper幻觉重复 "Макар и само…"（句子6的内容）。这是因为录音在63s后
    有一段长时间的静音/噪音，Whisper遇到静音时会把上下文重复输出。
    过滤后保留：句子1-6对应的真实段(0-61.5s) + 句子7+对应的真实段(1417.8s+)。
    """
    with open(TRANSCRIPT_FILE, encoding='utf-8') as f:
        segs = json.load(f)
    print(f"[加载] 原始 Whisper segments: {len(segs)}个，覆盖 {segs[-1]['end']:.1f}s")

    # 幻觉检测：找出重复文本的段（与其前5段中任一段相似度>0.7）
    HALL_SIM = 0.70
    LOOK_BACK = 5

    clean = []
    for i, seg in enumerate(segs):
        seg_norm = normalize(seg['text'])
        is_hall = False
        for j in range(max(0, i - LOOK_BACK), i):
            prev_norm = normalize(segs[j]['text'])
            if prev_norm and SequenceMatcher(None, seg_norm, prev_norm).ratio() > HALL_SIM:
                is_hall = True
                break
        if not is_hall:
            clean.append(seg)

    n_removed = len(segs) - len(clean)
    print(f"[过滤] 移除幻觉段: {n_removed}个 → 保留真实段: {len(clean)}个")
    if clean:
        print(f"[过滤] 真实段时间范围: {clean[0]['start']:.1f}s ~ {clean[-1]['end']:.1f}s")
    return clean


def load_reading_script() -> List[Dict]:
    """加载读稿句子列表"""
    with open(READING_SCRIPT, encoding='utf-8') as f:
        content = f.read()
    pattern = r'\[(\d+)\]\s*(.+?)(?=\n\[|\n==|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    sents = []
    for num, text in matches:
        text = text.strip()
        if text:
            sents.append({
                'index': int(num),
                'text':  text,
                'norm':  normalize(text),
            })
    print(f"[加载] 读稿句子: {len(sents)}句")
    return sents


# ─────────────────────────── 核心对齐算法 ───────────────────────────
def merged_text(segs: List[Dict], start: int, end: int) -> str:
    """合并 segs[start:end] 的文本"""
    return ' '.join(s['text'].strip() for s in segs[start:end])


def find_best_match(
    ref_norm: str,
    segs: List[Dict],
    search_start: int,
    max_offset: int,
    max_merge: int,
) -> Tuple[int, int, float, str]:
    """
    在 segs[search_start : search_start+max_offset+max_merge] 中
    找到与 ref_norm 最相似的连续segment窗口。

    策略：
      对每个起始偏移量 offset=0..max_offset：
        对每个合并数量 n=1..max_merge：
          计算 similarity(ref_norm, concat(segs[start+offset : start+offset+n]))
          记录最高分

    Returns:
        (abs_start, abs_end, score, whisper_text)
    """
    ref_len = len(ref_norm)
    best = (search_start, search_start + 1, 0.0, '')

    for offset in range(min(max_offset, len(segs) - search_start)):
        seg_start = search_start + offset
        if seg_start >= len(segs):
            break

        accumulated = ''
        for n in range(1, max_merge + 1):
            seg_end = seg_start + n
            if seg_end > len(segs):
                break

            # 增量拼接
            piece = segs[seg_end - 1]['text'].strip()
            accumulated = (accumulated + ' ' + piece).strip() if accumulated else piece
            acc_norm = normalize(accumulated)

            score = similarity(ref_norm, acc_norm)
            if score > best[2]:
                best = (seg_start, seg_end, score, accumulated)

            # 剪枝：合并文本已远超参考句子，继续合并无益
            if len(acc_norm) > ref_len * 2.5:
                break

        # 提前退出：已经找到足够好的匹配
        if best[2] >= 0.85:
            break

    return best


def align(
    segs: List[Dict],
    script: List[Dict],
    min_score: float   = 0.40,   # 低于此分数视为"未匹配"
    max_offset: int    = 40,     # 每句最多往前跳多少个segment找起点
    max_merge: int     = 12,     # 最多合并多少个segment
) -> List[Dict]:
    """
    贪心单调对齐：依次为每个读稿句子找最佳segment窗口。

    单调性保证：current_seg 只增不减，绝对不回头。
    """
    aligned = []
    current_seg = 0
    n_segs = len(segs)
    n_sents = len(script)

    for i, sent in enumerate(script):
        if current_seg >= n_segs:
            print(f"  ⚠ 句子 [{sent['index']}] — segment已耗尽，跳过剩余")
            break

        abs_start, abs_end, score, wtext = find_best_match(
            sent['norm'], segs, current_seg,
            max_offset=max_offset,
            max_merge=max_merge,
        )

        if score >= min_score:
            t_start = segs[abs_start]['start']
            t_end   = segs[abs_end - 1]['end']
            aligned.append({
                'index':       sent['index'],
                'text':        sent['text'],
                'start_time':  round(t_start, 3),
                'end_time':    round(t_end, 3),
                'duration':    round(t_end - t_start, 3),
                'match_score': round(score, 3),
                'whisper_text': wtext,
                'seg_range':   [abs_start, abs_end],
            })
            # 严格推进：下一句从这次匹配结束处开始
            current_seg = abs_end
        else:
            # 分数太低：可能是录音时跳过了这句，或者Whisper出错
            # 记录为"低可信度"，不推进current_seg（让下一句重新从这里找）
            prev_end_time = aligned[-1]['end_time'] if aligned else 0.0
            est_dur = max(2.0, len(sent['text']) / 12.0)
            aligned.append({
                'index':       sent['index'],
                'text':        sent['text'],
                'start_time':  round(prev_end_time + 0.3, 3),
                'end_time':    round(prev_end_time + 0.3 + est_dur, 3),
                'duration':    round(est_dur, 3),
                'match_score': round(score, 3),
                'whisper_text': wtext,
                'warning':     'low_confidence_estimated',
            })
            # 只小幅推进，不丢失后面可能匹配的segment
            current_seg = min(current_seg + 2, n_segs - 1)
            print(f"  ⚠ [{sent['index']:3d}] 最高分 {score:.3f} — {sent['text'][:45]}")

        if (i + 1) % 30 == 0:
            good_so_far = sum(1 for a in aligned if a.get('match_score', 0) >= 0.7)
            print(f"  进度: {i+1}/{n_sents}, seg={current_seg}/{n_segs}, "
                  f"好分={good_so_far}/{i+1}")

    return aligned


# ─────────────────────────── 音频切分 ───────────────────────────
def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def split_audio(aligned: List[Dict], audio_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg()
    ok = 0
    skip = 0

    for sent in aligned:
        if sent.get('warning'):
            skip += 1
            continue

        idx   = sent['index']
        start = sent['start_time']
        dur   = sent['duration']
        out_f = out_dir / f"{LANG}_{idx:04d}.wav"

        cmd = [
            ffmpeg, '-y', '-loglevel', 'error',
            '-i', str(audio_path),
            '-ss', str(start),
            '-t',  str(dur),
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            str(out_f)
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode == 0:
            ok += 1
        else:
            print(f"  切分失败 [{idx}]: {r.stderr.decode('utf-8','replace')[:80]}")

    print(f"[切分] 成功 {ok} 个，跳过(低可信) {skip} 个")
    return ok


# ─────────────────────────── 生成Metadata ───────────────────────────
def save_metadata(aligned: List[Dict], out_dir: Path):
    records = []
    for a in aligned:
        idx = a['index']
        fname = f"{LANG}_{idx:04d}.wav"
        fpath = out_dir / fname
        records.append({
            'audio_path':      fname if fpath.exists() else None,
            'text':            a['text'],
            'language':        LANG,
            'sentence_index':  idx,
            'start_time':      a['start_time'],
            'end_time':        a['end_time'],
            'duration_seconds': a['duration'],
            'match_score':     a.get('match_score', 0),
            'warning':         a.get('warning', None),
        })

    meta = {
        'dataset_info': {
            'name': f'museum_reading_corpus_{LANG}',
            'version': '3.0.0',
            'created_at': datetime.now().isoformat(),
            'language': LANG,
            'alignment_method': 'whisper_merge_match_v2',
            'total_samples': len(records),
            'audio_format': {'sample_rate': 16000, 'channels': 1, 'bit_depth': 16, 'format': 'wav'},
        },
        'samples': records,
    }

    meta_path = RECORDING_DIR / f"speech_corpus_metadata_{LANG}_final.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[元数据] 已保存: {meta_path}")


# ─────────────────────────── 统计报告 ───────────────────────────
def print_report(aligned: List[Dict]):
    import statistics as stats

    scores   = [a['match_score'] for a in aligned]
    durs     = [a['duration'] for a in aligned if not a.get('warning')]
    n_good   = sum(1 for s in scores if s >= 0.7)
    n_mid    = sum(1 for s in scores if 0.5 <= s < 0.7)
    n_low    = sum(1 for s in scores if s < 0.5)
    n_warn   = sum(1 for a in aligned if a.get('warning'))

    print("\n" + "=" * 55)
    print("  对齐质量报告")
    print("=" * 55)
    print(f"  总句子数:          {len(aligned)}")
    print(f"  优质(≥0.70):       {n_good:3d}  ({n_good/len(aligned)*100:.1f}%)")
    print(f"  中等(0.50~0.70):   {n_mid:3d}  ({n_mid/len(aligned)*100:.1f}%)")
    print(f"  低分(<0.50):       {n_low:3d}  ({n_low/len(aligned)*100:.1f}%)")
    print(f"  低可信度(估算):    {n_warn:3d}")
    if scores:
        print(f"  平均匹配分:        {stats.mean(scores):.3f}")
        print(f"  中位匹配分:        {stats.median(scores):.3f}")
    if durs:
        print(f"  平均句子时长:      {stats.mean(durs):.1f}s")
        total = sum(durs)
        print(f"  有效音频总时长:    {total:.0f}s ({total/60:.1f}min)")
    print("=" * 55)

    # 打印低分句子供人工核查
    low_sents = [(a['index'], a['match_score'], a['text'][:50])
                 for a in aligned if a['match_score'] < 0.5]
    if low_sents:
        print(f"\n  低分句子列表 (共{len(low_sents)}句，可人工核查):")
        for idx, sc, txt in low_sents[:20]:
            print(f"    [{idx:3d}] score={sc:.3f}  {txt}")
        if len(low_sents) > 20:
            print(f"    ... 还有 {len(low_sents)-20} 句")


# ─────────────────────────── 主函数 ───────────────────────────
def main():
    parser = argparse.ArgumentParser(description='保加利亚语对齐终极修复版')
    parser.add_argument('--skip-split', action='store_true', help='只对齐，不切割音频')
    parser.add_argument('--dry-run',    action='store_true', help='只打印结果，不保存文件')
    parser.add_argument('--min-score',  type=float, default=0.40, help='最低接受分数 (默认0.40)')
    parser.add_argument('--max-offset', type=int,   default=40,   help='每句最多跳过多少segment (默认40)')
    parser.add_argument('--max-merge',  type=int,   default=12,   help='每句最多合并多少segment (默认12)')
    args = parser.parse_args()

    print("=" * 55)
    print("  保加利亚语对齐终极修复版 v2")
    print(f"  参数: min_score={args.min_score}, max_offset={args.max_offset}, max_merge={args.max_merge}")
    print("=" * 55 + "\n")

    # 加载数据
    segs   = load_transcript()
    script = load_reading_script()

    # 对齐
    print("\n[对齐] 开始...")
    aligned = align(segs, script,
                    min_score=args.min_score,
                    max_offset=args.max_offset,
                    max_merge=args.max_merge)

    # 报告
    print_report(aligned)

    if args.dry_run:
        print("\n[DRY RUN] 不保存文件。")
        return

    # 保存对齐JSON
    result = {
        'language':         LANG,
        'language_name':    '保加利亚语',
        'audio_file':       AUDIO_FILE.name,
        'alignment_time':   datetime.now().isoformat(),
        'alignment_method': 'whisper_merge_match_v2',
        'params': {
            'min_score':  args.min_score,
            'max_offset': args.max_offset,
            'max_merge':  args.max_merge,
        },
        'total_sentences':  len(aligned),
        'sentences':        aligned,
    }
    with open(OUTPUT_ALIGN, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 对齐结果: {OUTPUT_ALIGN}")

    # 切分音频
    if not args.skip_split:
        if AUDIO_FILE.exists():
            print(f"\n[切分] 音频文件: {AUDIO_FILE}")
            split_audio(aligned, AUDIO_FILE, OUTPUT_SENTENCES)
            save_metadata(aligned, OUTPUT_SENTENCES)
        else:
            print(f"\n[警告] 音频文件不存在: {AUDIO_FILE}，跳过切分")
    else:
        print("\n[跳过] 音频切分（--skip-split）")

    print("\n✅ 全部完成！")
    print(f"   对齐结果: {OUTPUT_ALIGN}")
    if not args.skip_split:
        print(f"   音频文件: {OUTPUT_SENTENCES}/")


if __name__ == '__main__':
    main()

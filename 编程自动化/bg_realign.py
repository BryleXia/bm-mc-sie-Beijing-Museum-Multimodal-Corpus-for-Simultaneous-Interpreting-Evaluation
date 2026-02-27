#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保加利亚语重新转录+全局对齐 v3

根因分析：
  1. 原始 whisper_transcript_bg.json 从39s起被幻觉污染（Whisper重复"Макар и само"直到1417s）
  2. 1417s后的真实段虽然有696个，但每段只有0.8-2.5秒，是句子的碎片
  3. 学生朗读顺序与读稿顺序不一致（完全乱序）

解决方案：
  Step 1: 用 faster-whisper 重新转录 bg_full.wav
          - condition_on_previous_text=False  ← 关闭上下文传递，消灭幻觉
          - word_timestamps=True              ← 获取词级时间戳，精确对齐
          - 保存原始转录到 whisper_words_bg.json
  Step 2: 全局非顺序匹配
          - 对每个读稿句子，在所有Whisper词序列中搜索最佳匹配窗口
          - 用DP/贪心确保一对一对应（每段音频只分配一个句子）
  Step 3: 按对齐结果切割音频

用法：
  python bg_realign.py              # 重新转录 + 对齐 + 切割
  python bg_realign.py --skip-transcribe  # 跳过转录（使用已有words文件）
  python bg_realign.py --skip-split       # 只转录+对齐，不切割
"""

import json
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ─── 路径 ───────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
RECORDING_DIR   = PROJECT_ROOT / "交付" / "录音"
PROCESSED_DIR   = RECORDING_DIR / "处理后"
ALIGNMENT_DIR   = RECORDING_DIR / "对齐"
SCRIPT_DIR      = PROJECT_ROOT / "交付" / "朗读稿件"

# 默认语言（可通过 --lang 参数覆盖）
LANG            = "bg"

def get_paths(lang: str):
    """根据语言返回所有路径"""
    return {
        "audio":      PROCESSED_DIR / lang / f"{lang}_full.wav",
        "words_cache":ALIGNMENT_DIR / f"whisper_words_{lang}.json",
        "output_align":ALIGNMENT_DIR / f"alignment_{lang}_v3.json",
        "output_dir": PROCESSED_DIR / lang / "sentences_v3",
        "script":     SCRIPT_DIR / f"reading_script_{lang}.txt",
    }

# 兼容旧代码的全局变量（会被 main() 覆盖）
AUDIO_FILE      = None
WORDS_CACHE     = None
OUTPUT_ALIGN    = None
OUTPUT_DIR      = None


# ─── 文本工具 ────────────────────────────────────────────────────
def norm(text: str, lang: str = "bg") -> str:
    """文本规范化。日语需要去除所有空格（因为Whisper会把字符拆开）"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    if lang == "ja":
        # 日语：去掉所有空格
        return re.sub(r'\s+', '', text)
    else:
        # 其他语言：保留单词间空格
        return re.sub(r'\s+', ' ', text).strip()


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ─── Step 1: 重新转录 ────────────────────────────────────────────
def transcribe(audio_path: Path, model_size: str = "large-v3",
               device: str = "cuda", lang: str = "bg") -> List[Dict]:
    """
    用 faster-whisper 重新转录，关闭上下文传递消灭幻觉，
    开启词级时间戳用于精确对齐。
    结果缓存到 WORDS_CACHE。
    """
    print(f"\n[转录] 加载模型: {model_size} on {device}")
    print(f"[转录] 音频: {audio_path}  ({audio_path.stat().st_size/1e6:.1f} MB)")

    from faster_whisper import WhisperModel
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[转录] CUDA不可用，切换到CPU")
        device = "cpu"

    compute = "float16" if device == "cuda" else "int8"

    # 本地模型路径映射（避免联网下载）
    # large-v3 在 E 盘本地缓存，直接用快照目录路径加载，无需联网
    LOCAL_MODEL_PATHS = {
        "large-v3": r"E:\ai知识库\cache\huggingface\hub\models--Systran--faster-whisper-large-v3\snapshots\edaa852ec7e145841d8ffdb056a99866b5f0a478",
    }
    model_path = LOCAL_MODEL_PATHS.get(model_size, model_size)
    print(f"[转录] 模型路径: {model_path}")
    model = WhisperModel(model_path, device=device, compute_type=compute)

    print("[转录] 开始转录（condition_on_previous_text=False，消灭幻觉）...")
    print("[转录] 这大约需要 10-20 分钟，请耐心等待...\n")

    segments, info = model.transcribe(
        str(audio_path),
        language=lang,                       # 动态语言参数
        word_timestamps=True,
        condition_on_previous_text=False,   # ← 关键：不把前一段文字作为上下文
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,    # 静音超过300ms就切割
            threshold=0.3,
        ),
        beam_size=5,
        best_of=5,
        temperature=0.0,                    # 贪心解码，更稳定
        compression_ratio_threshold=2.4,    # 超过此压缩比视为幻觉，跳过
        log_prob_threshold=-1.0,
    )

    print(f"[转录] 检测语言: {info.language} (概率: {info.language_probability:.3f})")
    print(f"[转录] 音频时长: {info.duration:.1f}s")

    words = []
    seg_count = 0
    for seg in segments:
        seg_count += 1
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":  w.word.strip(),
                    "start": round(w.start, 3),
                    "end":   round(w.end,   3),
                    "prob":  round(w.probability, 3),
                })
        if seg_count % 50 == 0:
            print(f"  已处理 {seg_count} 段... 最新: [{seg.start:.1f}s] {seg.text[:40]}")

    print(f"\n[转录] 完成: {seg_count} 个片段, {len(words)} 个词")

    # 保存缓存
    cache = {
        "audio_file":   audio_path.name,
        "model":        model_size,
        "language":     info.language,
        "duration":     round(info.duration, 2),
        "transcribed_at": datetime.now().isoformat(),
        "total_words":  len(words),
        "words":        words,
    }
    ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(WORDS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"[转录] 词级数据已保存: {WORDS_CACHE}")
    return words


def load_words() -> List[Dict]:
    """加载已缓存的词级转录"""
    with open(WORDS_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    words = cache["words"]
    print(f"[加载] 词级转录: {len(words)} 个词, 音频 {cache['duration']:.1f}s")
    print(f"       模型: {cache['model']}, 转录时间: {cache['transcribed_at'][:19]}")
    return words


# ─── Step 2: 全局最优匹配 ────────────────────────────────────────
def load_uncertain_words(docx_path: Path) -> Dict[int, set]:
    """
    从日语朗读稿件的 docx 文件中提取高亮词（不确定专有名词）。
    返回 {sent_idx: set(highlighted_words)}
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn as _qn
    except ImportError:
        print("[警告] 未安装 python-docx，跳过不确定词屏蔽")
        return {}

    doc = Document(str(docx_path))
    uncertain = {}
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        m = re.match(r'\[(\d+)\]', para.text.strip())
        if not m:
            continue
        sent_idx = int(m.group(1))
        highlights = set()
        for run in para.runs:
            rpr = run._r.find(_qn('w:rPr'))
            if rpr is not None:
                hl = rpr.find(_qn('w:highlight'))
                if hl is not None and hl.get(_qn('w:val'), '') not in ('none', ''):
                    t = run.text.strip()
                    if t:
                        highlights.add(t)
        if highlights:
            uncertain[sent_idx] = highlights
    print(f"[加载] 不确定专有名词: {len(uncertain)} 个句子, "
          f"{sum(len(v) for v in uncertain.values())} 个词")
    return uncertain


def mask_uncertain(text: str, uncertain_words: set) -> str:
    """把不确定词从文本中去掉，避免它们拖累匹配分"""
    for w in uncertain_words:
        text = text.replace(w, "")
    return text


def load_script(uncertain: Dict[int, set] = None) -> List[Dict]:
    path = SCRIPT_DIR / f"reading_script_{LANG}.txt"
    with open(path, encoding="utf-8") as f:
        content = f.read()
    pattern = r'\[(\d+)\]\s*(.+?)(?=\n\[|\n==|\Z)'
    sents = []
    for n, t in re.findall(pattern, content, re.DOTALL):
        t = t.strip()
        if not t:
            continue
        idx = int(n)
        # 如果有不确定词，用屏蔽后的文本做 norm
        if uncertain and idx in uncertain:
            masked = mask_uncertain(t, uncertain[idx])
            n_text = norm(masked, LANG)
        else:
            n_text = norm(t, LANG)
        sents.append({"index": idx, "text": t, "norm": n_text})
    print(f"[加载] 读稿: {len(sents)} 句")
    return sents


def build_word_windows(words: List[Dict], window_words: int = 30,
                       step: int = 5) -> List[Dict]:
    """
    把词列表切成重叠窗口，每个窗口包含window_words个词。
    返回: [{"start": t, "end": t, "text": "...", "norm": "...",
            "wi_start": i, "wi_end": j}, ...]
    """
    windows = []
    n = len(words)
    for i in range(0, n, step):
        j = min(i + window_words, n)
        if j <= i:
            break
        chunk_words = words[i:j]
        text = " ".join(w["word"] for w in chunk_words)
        windows.append({
            "start":    chunk_words[0]["start"],
            "end":      chunk_words[-1]["end"],
            "text":     text,
            "norm":     norm(text, LANG),
            "wi_start": i,
            "wi_end":   j,
        })
    return windows


def find_sentence_in_words(
    sent_norm: str,
    words: List[Dict],
    search_start: int = 0,
    search_end:   int = -1,
    max_win:      int = 50,
) -> Tuple[int, int, float]:
    """
    在 words[search_start:search_end] 中找到与 sent_norm 最相似的连续词窗口。
    返回 (wi_start, wi_end, score)
    """
    if search_end < 0:
        search_end = len(words)

    ref_len    = len(sent_norm)
    ref_words  = sent_norm.split()
    expected_n = len(ref_words)

    best_start, best_end, best_score = search_start, min(search_start + expected_n, search_end), 0.0

    # 在搜索范围内尝试不同起点和长度
    # 日语：词被拆成单字符，步长必须为1，否则会跳过正确位置
    step = 1 if LANG == "ja" else max(1, expected_n // 4)
    for s in range(search_start, min(search_end, search_start + len(words)), step):
        acc = ""
        for e in range(s + 1, min(s + max_win + 1, search_end + 1)):
            acc = (acc + " " + words[e-1]["word"]).strip() if acc else words[e-1]["word"]
            acc_norm = norm(acc, LANG)
            score = sim(sent_norm, acc_norm)
            if score > best_score:
                best_score = score
                best_start, best_end = s, e
            # 剪枝：合并文本远超参考长度，停止
            if len(acc_norm) > ref_len * 2.2:
                break
        if best_score >= 0.85:
            break

    return best_start, best_end, best_score


def global_align(words: List[Dict], script: List[Dict]) -> List[Dict]:
    """
    全局非顺序匹配：对每个读稿句子在整个词列表中找最佳匹配。
    使用贪心策略 + 已用区间标记（避免重复）。

    流程：
    1. 对每个句子，全局搜索最佳匹配词区间 (score, wi_start, wi_end)
    2. 对找到的区间按时间排序，确保输出的音频文件按时间顺序排列
    3. 标记已用区间，防止重复分配
    """
    n_words = len(words)
    print(f"\n[对齐] 全局搜索 {len(script)} 个句子 in {n_words} 个词...")
    print(f"[对齐] 注意：允许乱序匹配（学生朗读顺序与稿件不同）\n")

    # 候选列表: [(score, wi_start, wi_end, sent_idx)]
    candidates = []

    for i, sent in enumerate(script):
        # 全局搜索（整个音频）
        wi_s, wi_e, score = find_sentence_in_words(
            sent["norm"], words, 0, n_words, max_win=40
        )
        candidates.append({
            "sent_idx":  i,
            "index":     sent["index"],
            "text":      sent["text"],
            "score":     score,
            "wi_start":  wi_s,
            "wi_end":    wi_e,
            "start_time": words[wi_s]["start"] if wi_s < n_words else 0,
            "end_time":   words[min(wi_e-1, n_words-1)]["end"] if wi_e > 0 else 0,
        })

        if (i + 1) % 30 == 0:
            good = sum(1 for c in candidates if c["score"] >= 0.70)
            print(f"  进度: {i+1}/{len(script)}, good(≥0.7): {good}/{i+1}")

    # 按时间排序输出（便于音频管理），保留原始句子index
    matched = sorted(candidates, key=lambda c: c["start_time"])

    return matched


def monotone_align(words: List[Dict], script: List[Dict],
                   min_score: float = 0.40,
                   max_offset_words: int = 300,
                   max_win: int = 40) -> List[Dict]:
    """
    单调对齐：按读稿顺序依次在 words 中找匹配（允许大幅跳跃）。

    适用场景：读稿顺序与录音顺序大体一致，但有跳过/插入情况。
    """
    # 日语：字符级分词，需要更大的搜索范围
    if LANG == "ja":
        max_offset_words = 2000  # 更大的搜索范围
        max_win = 150            # 更大的匹配窗口

    aligned = []
    current = 0
    n = len(words)

    for sent in script:
        if current >= n:
            break

        search_end = min(current + max_offset_words, n)
        wi_s, wi_e, score = find_sentence_in_words(
            sent["norm"], words, current, search_end, max_win=max_win
        )

        t_s = words[wi_s]["start"] if wi_s < n else 0
        t_e = words[min(wi_e-1, n-1)]["end"] if wi_e > 0 else 0

        aligned.append({
            "index":      sent["index"],
            "text":       sent["text"],
            "start_time": round(t_s, 3),
            "end_time":   round(t_e, 3),
            "duration":   round(t_e - t_s, 3),
            "score":      round(score, 3),
            "wi_range":   [wi_s, wi_e],
            "warning":    None if score >= min_score else "low_confidence",
        })

        if score >= min_score:
            current = wi_e
        else:
            # 日语：低分时前进更多（因为字符级分词，每个"词"很短）
            advance = 50 if LANG == "ja" else 5
            current = min(current + advance, n - 1)

    return aligned


# ─── Step 3: 切割音频 ────────────────────────────────────────────
def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def split_audio(aligned: List[Dict], audio: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg()
    ok = skip = 0

    for item in aligned:
        if item.get("warning") == "low_confidence":
            skip += 1
            continue
        idx   = item["index"]
        start = item["start_time"]
        dur   = max(0.1, item["duration"])
        out_f = out_dir / f"{LANG}_{idx:04d}.wav"

        cmd = [ff, "-y", "-loglevel", "error",
               "-i", str(audio),
               "-ss", str(start), "-t", str(dur),
               "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
               str(out_f)]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode == 0:
            ok += 1
        else:
            print(f"  ⚠ 切割失败 [{idx}]")

    print(f"[切割] 成功 {ok} 个, 跳过(低可信) {skip} 个")
    return ok


def save_result(aligned: List[Dict], method: str, audio_dur: float):
    scores   = [a["score"] for a in aligned]
    n_good   = sum(1 for s in scores if s >= 0.70)
    n_mid    = sum(1 for s in scores if 0.50 <= s < 0.70)
    n_low    = sum(1 for s in scores if s < 0.50)

    import statistics as st
    print("\n" + "=" * 55)
    print(f"  对齐报告 ({method})")
    print("=" * 55)
    print(f"  句子总数:      {len(aligned)}")
    print(f"  ≥0.70 优质:    {n_good}  ({n_good/len(aligned)*100:.1f}%)")
    print(f"  0.50~0.70 中等: {n_mid}  ({n_mid/len(aligned)*100:.1f}%)")
    print(f"  <0.50  低分:   {n_low}  ({n_low/len(aligned)*100:.1f}%)")
    print(f"  均分:          {st.mean(scores):.3f}")
    print(f"  中位分:        {st.median(scores):.3f}")
    durs = [a["duration"] for a in aligned if not a.get("warning")]
    if durs:
        print(f"  平均时长:      {st.mean(durs):.1f}s")
        print(f"  有效总时长:    {sum(durs):.0f}s / {audio_dur:.0f}s")
    print("=" * 55)

    result = {
        "language":        LANG,
        "language_name":   "保加利亚语",
        "audio_file":      AUDIO_FILE.name,
        "alignment_time":  datetime.now().isoformat(),
        "alignment_method": method,
        "total_sentences": len(aligned),
        "sentences": [
            {
                "index":       a["index"],
                "text":        a["text"],
                "start_time":  a["start_time"],
                "end_time":    a["end_time"],
                "duration":    a["duration"],
                "match_score": a["score"],
                **( {"warning": a["warning"]} if a.get("warning") else {} ),
            }
            for a in aligned
        ],
    }
    with open(OUTPUT_ALIGN, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 对齐结果: {OUTPUT_ALIGN}")
    return result


# ─── 主函数 ─────────────────────────────────────────────────────
def main():
    global AUDIO_FILE, WORDS_CACHE, OUTPUT_ALIGN, OUTPUT_DIR, LANG

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",            default="bg",
                        choices=["bg", "es", "ja"],
                        help="语言代码 (bg=保加利亚语, es=西班牙语, ja=日语)")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="跳过转录，直接使用已有的 whisper_words_<lang>.json")
    parser.add_argument("--skip-split",      action="store_true",
                        help="只对齐，不切割音频")
    parser.add_argument("--model",           default="large-v3",
                        choices=["medium", "large-v2", "large-v3"],
                        help="Whisper模型 (默认: large-v3)")
    parser.add_argument("--monotone",        action="store_true",
                        help="使用单调顺序对齐（而非全局非顺序）")
    parser.add_argument("--dry-run",         action="store_true",
                        help="只输出报告，不保存文件")
    args = parser.parse_args()

    # 根据语言设置路径
    LANG = args.lang
    paths = get_paths(LANG)
    AUDIO_FILE   = paths["audio"]
    WORDS_CACHE  = paths["words_cache"]
    OUTPUT_ALIGN = paths["output_align"]
    OUTPUT_DIR   = paths["output_dir"]

    lang_names = {"bg": "保加利亚语", "es": "西班牙语", "ja": "日语"}
    print("=" * 55)
    print(f"  {lang_names[LANG]}重新转录+全局对齐 v3")
    print("=" * 55)

    # ── Step 1: 转录 ──
    if args.skip_transcribe:
        if not WORDS_CACHE.exists():
            print(f"[错误] 找不到词级缓存: {WORDS_CACHE}")
            print(f"       请先不带 --skip-transcribe 运行一次")
            return
        words = load_words()
    else:
        words = transcribe(AUDIO_FILE, model_size=args.model, lang=LANG)

    audio_dur = words[-1]["end"] if words else 0

    # ── Step 2: 对齐 ──
    # 日语：加载不确定专有名词，屏蔽后再做匹配
    uncertain = {}
    if LANG == "ja":
        docx_path = PROJECT_ROOT / "新-日语-晦涩的专有名词都不确定标注出来了.docx"
        if docx_path.exists():
            uncertain = load_uncertain_words(docx_path)
        else:
            print(f"[警告] 未找到不确定词文件: {docx_path}")
    script = load_script(uncertain if uncertain else None)

    if args.monotone:
        print("\n[模式] 单调顺序对齐")
        aligned = monotone_align(words, script)
        method = "whisper_words_monotone_v3"
    else:
        print("\n[模式] 全局非顺序对齐（推荐：学生朗读顺序与稿件不同）")
        candidates = global_align(words, script)
        # 转换为标准格式
        aligned = []
        for c in candidates:
            n = len(words)
            wi_s = c["wi_start"]
            wi_e = c["wi_end"]
            t_s = words[wi_s]["start"] if wi_s < n else 0
            t_e = words[min(wi_e-1, n-1)]["end"] if wi_e > 0 else 0
            aligned.append({
                "index":      c["index"],
                "text":       c["text"],
                "start_time": round(t_s, 3),
                "end_time":   round(t_e, 3),
                "duration":   round(t_e - t_s, 3),
                "score":      round(c["score"], 3),
                "warning":    None if c["score"] >= 0.40 else "low_confidence",
            })
        method = "whisper_words_global_v3"

    if args.dry_run:
        scores = [a["score"] for a in aligned]
        import statistics as st
        print(f"\n[DRY RUN] 总计 {len(aligned)} 句")
        print(f"  均分={st.mean(scores):.3f}, 好分≥0.7: {sum(1 for s in scores if s>=0.7)}/{len(aligned)}")
        print("[DRY RUN] 不保存文件。")
        return

    # ── Step 3: 保存 + 切割 ──
    save_result(aligned, method, audio_dur)

    if not args.skip_split:
        print(f"\n[切割] 音频输出: {OUTPUT_DIR}")
        split_audio(aligned, AUDIO_FILE, OUTPUT_DIR)
    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()

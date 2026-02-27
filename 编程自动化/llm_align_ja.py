#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 语义级对齐 - 日语

原理：
  传统方法用 SequenceMatcher 做字符级相似度匹配，
  对日语专有名词效果差（读音不同 → 字符不同 → 匹配失败）。

  LLM 能理解语义等价：
  - "耀州窯" ≈ "ようしゅうよう"（读音相同）
  - "釉裏紅" ≈ "ゆうりこう"（语义相同）

  流程：
  1. 把 Whisper 转录分成 50 个词一组的块
  2. 对每个朗读稿句子，让 LLM 在转录块中找最佳匹配
  3. 返回词索引范围 → 转换为时间戳
"""

import json
import re
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# ─── 路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
WORDS_FILE   = PROJECT_ROOT / "交付/录音/对齐/whisper_words_ja.json"
SCRIPT_FILE  = PROJECT_ROOT / "交付/朗读稿件/reading_script_ja.txt"
OUTPUT_FILE  = PROJECT_ROOT / "交付/录音/对齐/alignment_ja_llm.json"
AUDIO_DIR    = PROJECT_ROOT / "交付/录音/处理后/ja/sentences_llm"
AUDIO_FILE   = PROJECT_ROOT / "交付/录音/处理后/ja/ja_full.wav"

# ─── 加载数据 ────────────────────────────────────────────────────
def load_words() -> List[Dict]:
    """加载 Whisper 词级转录"""
    data = json.loads(WORDS_FILE.read_text(encoding='utf-8'))
    words = data['words']
    print(f"[加载] Whisper 转录: {len(words)} 个词, 音频 {data['duration']:.1f}s")
    return words, data['duration']


def load_script() -> List[Dict]:
    """加载朗读稿件"""
    content = SCRIPT_FILE.read_text(encoding='utf-8')
    pattern = r'\[(\d+)\]\s*(.+?)(?=\n\[|\n==|\Z)'
    sents = []
    for n, t in re.findall(pattern, content, re.DOTALL):
        sents.append({"index": int(n), "text": t.strip()})
    print(f"[加载] 朗读稿件: {len(sents)} 句")
    return sents


def group_words(words: List[Dict], group_size: int = 50) -> List[Dict]:
    """把词列表分成组，方便 LLM 处理"""
    groups = []
    n = len(words)
    for i in range(0, n, group_size):
        j = min(i + group_size, n)
        chunk = words[i:j]
        text = ''.join(w['word'] for w in chunk)
        groups.append({
            "idx_range": [i, j],
            "text": text,
            "start": chunk[0]['start'],
            "end": chunk[-1]['end'],
        })
    print(f"[分组] {len(groups)} 个词组（每组 {group_size} 词）")
    return groups


# ─── LLM 调用 ────────────────────────────────────────────────────
# 阿里云通义千问 API 配置
ALIYUN_API_KEY = "sk-cdf8f27980b24974bf927c459b90adb8"
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_MODEL = "qwen-plus"  # qwen3.5-plus 的模型名


def call_llm(prompt: str, max_retries: int = 3) -> str:
    """调用阿里云通义千问 API"""
    client = OpenAI(api_key=ALIYUN_API_KEY, base_url=ALIYUN_BASE_URL)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=ALIYUN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [重试 {attempt+1}/{max_retries}] {e}")
                time.sleep(2)
            else:
                raise


def align_batch(sentences: List[Dict], word_groups: List[Dict],
                batch_size: int = 10) -> List[Dict]:
    """
    用 LLM 批量对齐句子到词组

    策略：把 10 个句子 + 所有词组发给 LLM，让它返回每个句子的匹配词组索引
    """
    # 构建词组索引文本
    group_text = "\n".join([
        f"[{g['idx_range'][0]}-{g['idx_range'][1]}] {g['text'][:80]}..."
        for g in word_groups
    ])

    # 构建句子文本
    sent_text = "\n".join([
        f"[{s['index']}] {s['text'][:100]}"
        for s in sentences
    ])

    prompt = f"""你是一个专业的日语语音对齐助手。

以下是 Whisper 从录音中转录的日语文本，按词索引分组：
{group_text}

请找出以下每个句子在转录中的位置。对于每个句子，返回最匹配的词索引范围。

句子列表：
{sent_text}

要求：
1. 即使专有名词的读音不同（如"耀州窯"读成"ようしゅうよう"），也要正确匹配
2. 返回 JSON 格式：{{"句子编号": [起始词索引, 结束词索引], ...}}
3. 只返回 JSON，不要其他解释
4. 如果某个句子完全找不到匹配，返回 null

示例输出：
{{"1": [0, 45], "2": [50, 95], "3": null}}
"""

    response = call_llm(prompt)

    # 解析 JSON
    try:
        # 提取 JSON 部分
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            print(f"  [警告] LLM 未返回有效 JSON: {response[:100]}")
            result = {}
    except json.JSONDecodeError as e:
        print(f"  [警告] JSON 解析失败: {e}")
        result = {}

    # 转换为对齐结果
    aligned = []
    for sent in sentences:
        idx_str = str(sent['index'])
        if idx_str in result and result[idx_str]:
            wi_start, wi_end = result[idx_str]
            aligned.append({
                "index": sent['index'],
                "text": sent['text'],
                "wi_start": wi_start,
                "wi_end": wi_end,
                "method": "llm_semantic",
            })
        else:
            aligned.append({
                "index": sent['index'],
                "text": sent['text'],
                "wi_start": None,
                "wi_end": None,
                "method": "llm_semantic",
            })

    return aligned


def align_all_sentences(sentences: List[Dict], word_groups: List[Dict],
                        words: List[Dict], batch_size: int = 20) -> List[Dict]:
    """分批对齐所有句子"""
    all_aligned = []
    n_batches = (len(sentences) + batch_size - 1) // batch_size

    print(f"\n[LLM对齐] {len(sentences)} 句，分 {n_batches} 批处理...")

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"  批次 {batch_num}/{n_batches}: 句子 {batch[0]['index']}-{batch[-1]['index']}")

        try:
            aligned = align_batch(batch, word_groups)
            all_aligned.extend(aligned)

            # 统计成功数
            success = sum(1 for a in aligned if a['wi_start'] is not None)
            print(f"    成功: {success}/{len(batch)}")

            # 避免 API 限流
            time.sleep(1)
        except Exception as e:
            print(f"    [错误] {e}")
            # 标记为失败
            for sent in batch:
                all_aligned.append({
                    "index": sent['index'],
                    "text": sent['text'],
                    "wi_start": None,
                    "wi_end": None,
                    "method": "llm_semantic",
                    "error": str(e),
                })

    return all_aligned


# ─── 后处理 ──────────────────────────────────────────────────────
def convert_to_timestamps(aligned: List[Dict], words: List[Dict]) -> List[Dict]:
    """把词索引转换为时间戳"""
    result = []
    n = len(words)

    for item in aligned:
        if item['wi_start'] is not None and item['wi_end'] is not None:
            wi_s = max(0, min(item['wi_start'], n-1))
            wi_e = max(wi_s+1, min(item['wi_end'], n))

            t_s = words[wi_s]['start']
            t_e = words[wi_e-1]['end']
            duration = t_e - t_s

            result.append({
                "index": item['index'],
                "text": item['text'],
                "start_time": round(t_s, 3),
                "end_time": round(t_e, 3),
                "duration": round(duration, 3),
                "wi_range": [wi_s, wi_e],
                "method": "llm_semantic",
            })
        else:
            result.append({
                "index": item['index'],
                "text": item['text'],
                "start_time": None,
                "end_time": None,
                "duration": None,
                "method": "llm_semantic",
                "warning": "no_match",
            })

    return result


def save_result(aligned: List[Dict], audio_dur: float):
    """保存对齐结果"""
    result = {
        "method": "llm_semantic_v1",
        "language": "ja",
        "audio_file": "ja_full.wav",
        "audio_duration": round(audio_dur, 2),
        "total_sentences": len(aligned),
        "aligned_sentences": sum(1 for a in aligned if a.get('start_time') is not None),
        "sentences": aligned,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] {OUTPUT_FILE}")

    # 统计报告
    success = sum(1 for a in aligned if a.get('start_time') is not None)
    total_dur = sum(a['duration'] for a in aligned if a.get('duration'))
    print(f"  成功: {success}/{len(aligned)} ({success/len(aligned)*100:.1f}%)")
    print(f"  总时长: {total_dur:.1f}s / {audio_dur:.1f}s ({total_dur/audio_dur*100:.1f}%)")


# ─── 主函数 ──────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  LLM 语义级对齐 - 日语")
    print("=" * 55)

    # 加载数据
    words, audio_dur = load_words()
    sentences = load_script()
    word_groups = group_words(words, group_size=60)  # 60词一组

    # LLM 对齐
    aligned = align_all_sentences(sentences, word_groups, words, batch_size=15)

    # 转换为时间戳
    aligned_ts = convert_to_timestamps(aligned, words)

    # 保存结果
    save_result(aligned_ts, audio_dur)

    print("\n完成！")


if __name__ == "__main__":
    main()

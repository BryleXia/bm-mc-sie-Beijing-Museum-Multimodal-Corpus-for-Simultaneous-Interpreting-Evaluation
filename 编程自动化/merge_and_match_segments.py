#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并Whisper短segment后重新匹配

问题：Whisper VAD将录音切成了740个fragment，其中364个<2秒
解决：合并相邻短segment，形成完整句子后再匹配
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
ALIGNMENT_DIR = PROJECT_ROOT / "交付" / "录音" / "对齐"
CORPUS_PATH = PROJECT_ROOT / "交付" / "multilingual_corpus.json"


def normalize_text(text: str) -> str:
    """标准化文本"""
    text = text.lower()
    text = re.sub(r'[^\w\s\u0400-\u04FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_corpus_sentences(lang: str) -> List[Dict]:
    """加载语料库句子"""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    sentences = []
    for board in corpus.get("boards", []):
        museum = board.get("source", {}).get("museum", "未知")
        board_id = board.get("board_id", "")
        for para in board.get("paragraphs", []):
            for sent in para.get("sentences", []):
                text = sent.get(lang, "").strip()
                if text:
                    sentences.append({
                        "museum": museum,
                        "board_id": board_id,
                        "text": text,
                        "normalized": normalize_text(text),
                    })
    return sentences


def merge_short_segments(segments: List[Dict], min_duration: float = 3.0) -> List[Dict]:
    """
    合并短segment，形成较长的完整句子

    策略：
    1. 遍历segments，累积文本直到达到min_duration
    2. 或者当累积文本与语料库某句子匹配度高时，提前结束当前合并
    """
    merged = []
    current = None

    for seg in segments:
        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "texts": [seg["text"]],
            }
        else:
            # 合并
            current["end"] = seg["end"]
            current["texts"].append(seg["text"])

        duration = current["end"] - current["start"]
        combined_text = " ".join(current["texts"])

        # 达到最小时长，保存并重置
        if duration >= min_duration:
            current["text"] = combined_text
            current["normalized"] = normalize_text(combined_text)
            merged.append(current)
            current = None

    # 处理最后一个未完成的segment
    if current is not None:
        current["text"] = " ".join(current["texts"])
        current["normalized"] = normalize_text(current["text"])
        merged.append(current)

    return merged


def match_to_corpus(merged_segments: List[Dict], corpus_sentences: List[Dict],
                   threshold: float = 0.65) -> List[Dict]:
    """将合并后的segment与语料库匹配"""
    matched = []
    used_corpus = set()

    for seg in merged_segments:
        seg_norm = seg["normalized"]
        if len(seg_norm) < 15:  # 跳过太短的
            continue

        best_match = None
        best_score = 0
        best_idx = -1

        for idx, sent in enumerate(corpus_sentences):
            if idx in used_corpus:
                continue

            score = SequenceMatcher(None, seg_norm, sent["normalized"]).ratio()

            if score > best_score:
                best_score = score
                best_match = sent
                best_idx = idx

        if best_score >= threshold and best_match:
            used_corpus.add(best_idx)
            matched.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["end"] - seg["start"],
                "whisper_text": seg["text"],
                "matched_text": best_match["text"],
                "matched_museum": best_match["museum"],
                "match_score": best_score,
            })

    return matched


def main():
    lang = "bg"

    # 加载原始segment
    transcript_path = ALIGNMENT_DIR / f"whisper_transcript_{lang}.json"
    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    print(f"加载 {len(raw_segments)} 个原始segment")

    # 合并短segment
    merged = merge_short_segments(raw_segments, min_duration=3.0)
    print(f"合并后: {len(merged)} 个segment")

    # 统计合并后时长分布
    durations = [s["end"] - s["start"] for s in merged]
    print(f"时长分布: <3s={sum(1 for d in durations if d < 3)}, 3-5s={sum(1 for d in durations if 3 <= d < 5)}, >=5s={sum(1 for d in durations if d >= 5)}")

    # 加载语料库
    corpus = load_corpus_sentences(lang)
    print(f"语料库: {len(corpus)} 个句子")

    # 匹配
    matched = match_to_corpus(merged, corpus, threshold=0.65)
    print(f"匹配成功: {len(matched)} 个")

    # 统计匹配质量
    scores = [m["match_score"] for m in matched]
    total_duration = sum(m["duration"] for m in matched)
    print(f"平均匹配分: {sum(scores)/len(scores):.3f}")
    print(f"总时长: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")

    # 保存结果
    output = {
        "method": "whisper_merged_matched",
        "total_segments": len(merged),
        "matched_count": len(matched),
        "total_duration": total_duration,
        "matches": matched,
    }

    output_path = ALIGNMENT_DIR / f"alignment_{lang}_merged.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path}")

    # 生成稿件
    script_lines = [f"保加利亚语朗读稿件（合并匹配版）", f"生成时间: 2026-02-23", f"句子数: {len(matched)}", ""]

    current_museum = None
    for i, m in enumerate(matched, 1):
        if m["matched_museum"] != current_museum:
            current_museum = m["matched_museum"]
            script_lines.append(f"\n{'='*50}\n  {current_museum}\n{'='*50}\n")
        script_lines.append(f"[{i}] {m['matched_text']}")
        script_lines.append(f"    # 匹配分: {m['match_score']:.2f}, 时长: {m['duration']:.1f}s\n")

    script_path = PROJECT_ROOT / "交付" / "朗读稿件" / f"reading_script_{lang}_merged.txt"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(script_lines))
    print(f"稿件已保存: {script_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反向匹配策略：用语料库句子在Whisper转录中搜索对应时间段

思路：
1. 对于每个语料库句子，在Whisper转录文本中寻找匹配
2. 找到匹配后，确定对应的segment范围
3. 计算该句子的时间戳
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

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
        board_title = board.get("board_title", {}).get("zh", "")
        for para in board.get("paragraphs", []):
            for sent in para.get("sentences", []):
                text = sent.get(lang, "").strip()
                if text:
                    sentences.append({
                        "museum": museum,
                        "board_id": board_id,
                        "board_title_zh": board_title,
                        "text": text,
                        "normalized": normalize_text(text),
                        "zh": sent.get("zh", ""),
                    })
    return sentences


def find_sentence_in_transcript(
    corpus_sentence: Dict,
    whisper_segments: List[Dict],
    start_idx: int = 0,
    search_window: int = 200
) -> Tuple[int, int, float]:
    """
    在Whisper转录中搜索语料库句子的位置

    Returns:
        (start_segment_idx, end_segment_idx, match_score)
    """
    target = corpus_sentence["normalized"]
    target_words = target.split()
    target_len = len(target_words)

    if target_len < 5:  # 太短的句子不好匹配
        return -1, -1, 0

    best_start = -1
    best_end = -1
    best_score = 0

    # 从start_idx开始搜索
    for i in range(start_idx, min(start_idx + search_window, len(whisper_segments))):
        # 尝试从segment i开始，组合不同数量的连续segment
        combined_text = ""
        combined_norm = ""

        for j in range(i, min(i + 30, len(whisper_segments))):  # 最多组合30个segment
            combined_text += " " + whisper_segments[j]["text"]
            combined_norm = normalize_text(combined_text)
            combined_words = combined_norm.split()

            # 如果组合文本比目标长太多，跳过
            if len(combined_words) > target_len * 1.5:
                break

            # 计算相似度
            score = SequenceMatcher(None, target, combined_norm).ratio()

            # 也检查目标是否在组合文本中
            if target in combined_norm:
                score = max(score, 0.9)

            if score > best_score:
                best_score = score
                best_start = i
                best_end = j + 1

                # 如果找到很好的匹配，提前退出
                if score > 0.85:
                    return best_start, best_end, best_score

    return best_start, best_end, best_score


def main():
    lang = "bg"

    # 加载Whisper转录
    transcript_path = ALIGNMENT_DIR / f"whisper_transcript_{lang}.json"
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # 为每个segment添加normalized
    for s in segments:
        s["normalized"] = normalize_text(s["text"])

    print(f"Whisper segments: {len(segments)}")
    total_duration = segments[-1]["end"]
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # 加载语料库句子
    corpus = load_corpus_sentences(lang)
    print(f"Corpus sentences: {len(corpus)}")

    # 按长度排序语料库句子（优先匹配长句子）
    corpus_sorted = sorted(corpus, key=lambda x: -len(x["normalized"].split()))

    # 搜索匹配
    matched = []
    current_segment = 0

    for sent in corpus_sorted:
        start, end, score = find_sentence_in_transcript(
            sent, segments, current_segment, search_window=100
        )

        if score >= 0.6 and start >= 0:
            # 计算时间戳
            seg_start = segments[start]["start"]
            seg_end = segments[end - 1]["end"]

            matched.append({
                "corpus_text": sent["text"],
                "museum": sent["museum"],
                "board_title_zh": sent["board_title_zh"],
                "whisper_text": " ".join(segments[i]["text"] for i in range(start, end)),
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": seg_end - seg_start,
                "match_score": score,
                "segment_range": [start, end],
            })

            # 更新搜索起点
            current_segment = end

    print(f"\n匹配结果: {len(matched)} 个句子")

    if matched:
        scores = [m["match_score"] for m in matched]
        total_dur = sum(m["duration"] for m in matched)
        print(f"平均匹配分: {sum(scores)/len(scores):.3f}")
        print(f"高质量匹配(>=0.7): {sum(1 for s in scores if s >= 0.7)}")
        print(f"总时长: {total_dur:.1f}s ({total_dur/60:.1f}min)")
        print(f"覆盖率: {total_dur/total_duration*100:.1f}%")

        # 按时间排序
        matched.sort(key=lambda x: x["start_time"])

        # 保存结果
        output = {
            "method": "corpus_first_matching",
            "total_matched": len(matched),
            "total_duration": total_dur,
            "sentences": matched,
        }

        output_path = ALIGNMENT_DIR / f"alignment_{lang}_corpus_first.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {output_path}")

        # 生成稿件
        script_lines = [
            f"保加利亚语朗读稿件（按时间排序）",
            f"生成时间: 2026-02-23",
            f"匹配句子: {len(matched)}",
            f"覆盖时长: {total_dur/60:.1f}分钟",
            ""
        ]

        current_museum = None
        for i, m in enumerate(matched, 1):
            if m["museum"] != current_museum:
                current_museum = m["museum"]
                script_lines.append(f"\n{'='*50}")
                script_lines.append(f"  {current_museum}")
                script_lines.append(f"{'='*50}\n")

            script_lines.append(f"[{i}] {m['corpus_text']}")
            script_lines.append(f"    # 匹配分: {m['match_score']:.2f}, 时间: {m['start_time']:.1f}s-{m['end_time']:.1f}s\n")

        script_path = PROJECT_ROOT / "交付" / "朗读稿件" / f"reading_script_{lang}_corpus_first.txt"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines))
        print(f"稿件已保存: {script_path}")


if __name__ == "__main__":
    main()

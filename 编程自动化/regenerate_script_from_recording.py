#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Whisper转录生成与录音匹配的朗读稿件

当朗读者没有按照原稿件顺序朗读时，使用此脚本：
1. 运行Whisper转录录音
2. 将转录文本与语料库匹配
3. 生成与录音实际顺序一致的稿件
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

# 添加父目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

# 语言配置
LANGUAGE_CONFIG = {
    "zh": {"name": "中文"},
    "en": {"name": "英文"},
    "ja": {"name": "日语"},
    "es": {"name": "西班牙语"},
    "bg": {"name": "保加利亚语"},
}

# 目录配置
PROJECT_ROOT = config.PROJECT_ROOT
RECORDING_DIR = PROJECT_ROOT / "交付" / "录音"
PROCESSED_DIR = RECORDING_DIR / "处理后"
ALIGNMENT_DIR = RECORDING_DIR / "对齐"
CORPUS_PATH = PROJECT_ROOT / "交付" / "multilingual_corpus.json"


def normalize_text(text: str) -> str:
    """标准化文本用于匹配"""
    text = text.lower()
    # 移除标点符号（保留字母、数字、空格和西里尔字母）
    text = re.sub(r'[^\w\s\u0400-\u04FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_corpus_sentences(lang: str) -> List[Dict]:
    """从语料库加载指定语言的所有句子"""
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
                        "en": sent.get("en", ""),
                        "ja": sent.get("ja", ""),
                        "es": sent.get("es", ""),
                        "bg": sent.get("bg", ""),
                    })

    print(f"从语料库加载了 {len(sentences)} 个{LANGUAGE_CONFIG[lang]['name']}句子")
    return sentences


def transcribe_with_whisper(
    audio_path: Path,
    lang: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16"
) -> List[Dict]:
    """使用Whisper转录音频，返回segment列表"""
    from faster_whisper import WhisperModel

    print(f"加载Whisper模型: {model_size}")
    print(f"设备: {device}, 计算类型: {compute_type}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    print(f"转录音频: {audio_path}")

    segments, info = model.transcribe(
        str(audio_path),
        language=lang,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    print(f"音频时长: {info.duration:.1f}秒")

    # 收集所有segment
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "normalized": normalize_text(segment.text.strip()),
        })

    print(f"转录完成: {len(results)} 个片段")
    return results


def match_segments_to_corpus(
    whisper_segments: List[Dict],
    corpus_sentences: List[Dict],
    similarity_threshold: float = 0.5  # 降低阈值以增加覆盖率
) -> List[Dict]:
    """将Whisper转录的segment与语料库句子匹配"""

    matched_results = []
    used_corpus_indices = set()

    for seg in whisper_segments:
        seg_text = seg["normalized"]
        if len(seg_text) < 10:  # 跳过太短的segment
            continue

        best_match = None
        best_score = 0
        best_idx = -1

        for idx, sent in enumerate(corpus_sentences):
            if idx in used_corpus_indices:
                continue

            corpus_text = sent["normalized"]

            # 计算相似度
            score = SequenceMatcher(None, seg_text, corpus_text).ratio()

            # 也检查是否一个包含另一个
            if seg_text in corpus_text or corpus_text in seg_text:
                score = max(score, 0.8)

            if score > best_score:
                best_score = score
                best_match = sent
                best_idx = idx

        if best_score >= similarity_threshold and best_match:
            used_corpus_indices.add(best_idx)
            matched_results.append({
                "whisper_start": seg["start"],
                "whisper_end": seg["end"],
                "whisper_text": seg["text"],
                "matched_corpus": best_match,
                "match_score": best_score,
            })

    print(f"匹配完成: {len(matched_results)} / {len(whisper_segments)} 个片段匹配")
    return matched_results


def generate_matched_reading_script(
    matched_results: List[Dict],
    lang: str,
    output_path: Path
):
    """生成与录音匹配的朗读稿件"""

    lines = []
    lines.append(f"多语种朗读稿件（与录音匹配） - {LANGUAGE_CONFIG[lang]['name']}")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"句子总数: {len(matched_results)}句")
    lines.append(f"来源: 根据录音内容自动匹配")
    lines.append("")
    lines.append("=" * 60)
    lines.append("说明：此稿件根据实际录音内容生成，顺序与录音一致")
    lines.append("=" * 60)
    lines.append("")

    current_museum = None
    total_duration = 0

    for i, result in enumerate(matched_results, 1):
        corpus = result["matched_corpus"]
        museum = corpus.get("museum", "")

        # 新博物馆，添加分隔
        if museum != current_museum:
            current_museum = museum
            lines.append("")
            lines.append("=" * 60)
            lines.append(f"  {museum}")
            lines.append("=" * 60)
            lines.append("")

        # 计算时长
        duration = result["whisper_end"] - result["whisper_start"]
        total_duration += duration

        # 添加句子
        lines.append(f"[{i}] {corpus.get(lang, '')}")
        lines.append(f"    # 匹配分数: {result['match_score']:.2f}, 时长: {duration:.1f}s")
        lines.append("")

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"稿件已保存: {output_path}")
    print(f"总时长: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
    return matched_results


def generate_new_alignment(
    matched_results: List[Dict],
    lang: str,
    audio_path: Path,
    output_path: Path
):
    """生成新的对齐文件"""

    sentences = []
    for i, result in enumerate(matched_results, 1):
        corpus = result["matched_corpus"]
        duration = result["whisper_end"] - result["whisper_start"]

        sentences.append({
            "index": i,
            "text": corpus.get(lang, ""),
            "start_time": round(result["whisper_start"], 3),
            "end_time": round(result["whisper_end"], 3),
            "duration": round(duration, 3),
            "match_score": round(result["match_score"], 3),
            "museum": corpus.get("museum", ""),
            "board_id": corpus.get("board_id", ""),
        })

    alignment = {
        "language": lang,
        "language_name": LANGUAGE_CONFIG[lang]["name"],
        "audio_file": audio_path.name,
        "duration_seconds": round(matched_results[-1]["whisper_end"], 2) if matched_results else 0,
        "alignment_time": datetime.now().isoformat(),
        "alignment_method": "whisper_matched",
        "total_sentences": len(sentences),
        "sentences": sentences,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)

    print(f"对齐文件已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从录音生成匹配的朗读稿件")
    parser.add_argument("--lang", "-l", required=True, help="语言代码")
    parser.add_argument("--model", "-m", default="large-v3", help="Whisper模型")
    parser.add_argument("--device", "-d", default="cuda", help="设备")
    parser.add_argument("--skip-transcribe", action="store_true", help="跳过转录，使用已有结果")
    args = parser.parse_args()

    # 设置缓存目录（避免占用C盘）
    os.environ["HF_HOME"] = str(PROJECT_ROOT / "cache" / "huggingface")

    # 音频路径
    audio_path = PROCESSED_DIR / args.lang / f"{args.lang}_full.wav"
    if not audio_path.exists():
        print(f"音频文件不存在: {audio_path}")
        return

    # 加载语料库句子
    corpus_sentences = load_corpus_sentences(args.lang)

    # 转录或加载
    if args.skip_transcribe:
        # 尝试从转录JSON文件加载
        transcript_path = ALIGNMENT_DIR / f"whisper_transcript_{args.lang}.json"
        if transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                raw_segments = json.load(f)
            whisper_segments = [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "normalized": normalize_text(s["text"])
                }
                for s in raw_segments
            ]
            print(f"从转录文件加载 {len(whisper_segments)} 个片段")
        else:
            print(f"没有找到转录文件: {transcript_path}")
            return
    else:
        # 运行Whisper转录
        whisper_segments = transcribe_with_whisper(
            audio_path, args.lang, args.model, args.device
        )

    # 匹配
    matched_results = match_segments_to_corpus(
        whisper_segments, corpus_sentences
    )

    # 生成稿件
    output_dir = PROJECT_ROOT / "交付" / "朗读稿件"
    script_path = output_dir / f"reading_script_{args.lang}_matched.txt"
    generate_matched_reading_script(matched_results, args.lang, script_path)

    # 生成新的对齐文件
    new_alignment_path = ALIGNMENT_DIR / f"alignment_{args.lang}_matched.json"
    generate_new_alignment(matched_results, args.lang, audio_path, new_alignment_path)


if __name__ == "__main__":
    main()

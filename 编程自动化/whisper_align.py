#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper音频对齐工具

使用faster-whisper对音频进行精确的音频-文本对齐。
比静音检测方法更精确，能正确匹配朗读内容与时间戳。

使用方法：
    python whisper_align.py --lang es
    python whisper_align.py --lang bg --model medium
    python whisper_align.py --lang es --model large-v3 --device cuda
"""

import json
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

# 添加父目录到路径以导入 config
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

# 语言配置
LANGUAGE_CONFIG = {
    "zh": {"name": "中文", "whisper_lang": "zh"},
    "en": {"name": "英文", "whisper_lang": "en"},
    "ja": {"name": "日语", "whisper_lang": "ja"},
    "es": {"name": "西班牙语", "whisper_lang": "es"},
    "bg": {"name": "保加利亚语", "whisper_lang": "bg"},
}

# 目录配置
PROJECT_ROOT = config.PROJECT_ROOT
RECORDING_DIR = PROJECT_ROOT / "交付" / "录音"
PROCESSED_DIR = RECORDING_DIR / "处理后"
ALIGNMENT_DIR = RECORDING_DIR / "对齐"
READING_SCRIPT_DIR = PROJECT_ROOT / "交付" / "朗读稿件"


def get_ffmpeg_path() -> str:
    """获取ffmpeg可执行文件路径"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def load_reading_script(lang: str) -> List[Dict]:
    """
    加载朗读稿件中的句子列表

    Args:
        lang: 语言代码

    Returns:
        句子列表 [{"index": 1, "text": "..."}, ...]
    """
    script_path = READING_SCRIPT_DIR / f"reading_script_{lang}.txt"

    if not script_path.exists():
        raise FileNotFoundError(f"朗读稿件不存在: {script_path}")

    sentences = []
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析朗读稿件，提取句子
    # 格式: [数字] 句子内容
    pattern = r'\[(\d+)\]\s*(.+?)(?=\n\[|\n==|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for num, text in matches:
        text = text.strip()
        if text:
            sentences.append({
                "index": int(num),
                "text": text
            })

    print(f"从朗读稿件加载了 {len(sentences)} 个句子")
    return sentences


def normalize_text(text: str) -> str:
    """标准化文本用于匹配"""
    # 转小写
    text = text.lower()
    # 移除标点符号（保留字母、数字、空格）
    text = re.sub(r'[^\w\s]', '', text)
    # 多个空格合并为一个
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def transcribe_with_whisper(
    audio_path: Path,
    lang: str,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16"
) -> List[Dict]:
    """
    使用faster-whisper转录音频并获取word-level时间戳

    Args:
        audio_path: 音频文件路径
        lang: 语言代码
        model_size: 模型大小 (tiny, base, small, medium, large-v2, large-v3)
        device: 设备 (cuda, cpu)
        compute_type: 计算类型 (float16, int8, float32)

    Returns:
        转录结果列表 [{"start": 0.0, "end": 5.0, "text": "..."}, ...]
    """
    from faster_whisper import WhisperModel

    whisper_lang = LANGUAGE_CONFIG.get(lang, {}).get("whisper_lang", lang)

    print(f"加载Whisper模型: {model_size}")
    print(f"设备: {device}, 计算类型: {compute_type}")
    print(f"语言: {whisper_lang}")

    # 加载模型
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    print(f"开始转录音频: {audio_path}")
    print("这可能需要几分钟...")

    # 转录音频
    segments, info = model.transcribe(
        str(audio_path),
        language=whisper_lang,
        word_timestamps=True,
        vad_filter=True,  # 使用VAD过滤静音
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    print(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")
    print(f"音频时长: {info.duration:.1f}秒")

    # 收集所有segment和word
    results = []
    word_list = []

    for segment in segments:
        segment_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": []
        }

        if segment.words:
            for word in segment.words:
                word_data = {
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability
                }
                word_list.append(word_data)
                segment_data["words"].append(word_data)

        results.append(segment_data)
        print(f"  [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()[:50]}...")

    print(f"\n转录完成: {len(results)} 个片段, {len(word_list)} 个词")

    return results, word_list, info


def find_sentence_boundaries(
    whisper_segments: List[Dict],
    word_list: List[Dict],
    reference_sentences: List[Dict]
) -> List[Dict]:
    """
    通过文本匹配找到每个参考句子在Whisper转录中的边界

    Args:
        whisper_segments: Whisper转录的segment列表
        word_list: Whisper转录的词列表（带时间戳）
        reference_sentences: 参考句子列表

    Returns:
        对齐后的句子列表（带时间戳）
    """
    print("\n进行文本匹配...")

    # 将所有Whisper词按顺序组合
    whisper_text = " ".join([w["word"] for w in word_list])
    whisper_normalized = normalize_text(whisper_text)
    whisper_words_normalized = [normalize_text(w["word"]) for w in word_list]

    aligned_sentences = []
    current_word_idx = 0

    for sent in reference_sentences:
        sent_idx = sent["index"]
        sent_text = sent["text"]
        sent_normalized = normalize_text(sent_text)
        sent_words = sent_normalized.split()

        if not sent_words:
            continue

        # 在Whisper词序列中寻找匹配
        best_start_idx = current_word_idx
        best_end_idx = current_word_idx
        best_score = 0

        # 搜索窗口：从当前位置开始，最多往后看100个词
        search_window = 100

        for start_offset in range(min(50, len(word_list) - current_word_idx)):
            start_idx = current_word_idx + start_offset

            # 尝试不同长度的匹配
            for length in range(len(sent_words) - 2, len(sent_words) + 10):
                end_idx = start_idx + length
                if end_idx > len(word_list):
                    break

                # 提取候选文本
                candidate_words = whisper_words_normalized[start_idx:end_idx]
                candidate_text = " ".join(candidate_words)

                # 计算相似度
                score = SequenceMatcher(None, sent_normalized, candidate_text).ratio()

                if score > best_score:
                    best_score = score
                    best_start_idx = start_idx
                    best_end_idx = end_idx

                # 如果找到很好的匹配，提前退出
                if score > 0.9:
                    break

            if best_score > 0.9:
                break

        # 获取时间戳
        if best_start_idx < len(word_list) and best_end_idx <= len(word_list):
            start_time = word_list[best_start_idx]["start"]
            end_time = word_list[min(best_end_idx - 1, len(word_list) - 1)]["end"]

            aligned_sentences.append({
                "index": sent_idx,
                "text": sent_text,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "match_score": round(best_score, 3),
                "word_range": [best_start_idx, best_end_idx]
            })

            # 更新当前位置（允许一些重叠）
            current_word_idx = max(best_end_idx - 2, current_word_idx + 1)
        else:
            # 如果找不到匹配，使用上一个句子的结束时间
            if aligned_sentences:
                prev_end = aligned_sentences[-1]["end_time"]
                aligned_sentences.append({
                    "index": sent_idx,
                    "text": sent_text,
                    "start_time": round(prev_end + 0.5, 3),
                    "end_time": round(prev_end + 5.0, 3),
                    "duration": 4.5,
                    "match_score": 0.0,
                    "word_range": [-1, -1],
                    "warning": "未找到匹配"
                })
            print(f"  警告: 句子 {sent_idx} 未找到匹配")

        if sent_idx % 20 == 0:
            print(f"  已处理: {sent_idx}/{len(reference_sentences)} 句")

    return aligned_sentences


def align_with_whisper(
    audio_path: Path,
    lang: str,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16"
) -> Dict:
    """
    使用Whisper进行完整的音频-文本对齐

    Args:
        audio_path: 音频文件路径
        lang: 语言代码
        model_size: Whisper模型大小
        device: 设备
        compute_type: 计算类型

    Returns:
        对齐结果字典
    """
    # 加载参考句子
    reference_sentences = load_reading_script(lang)

    # 使用Whisper转录
    whisper_segments, word_list, info = transcribe_with_whisper(
        audio_path, lang, model_size, device, compute_type
    )

    # 进行文本匹配
    aligned_sentences = find_sentence_boundaries(
        whisper_segments, word_list, reference_sentences
    )

    # 构建完整结果
    result = {
        "language": lang,
        "language_name": LANGUAGE_CONFIG.get(lang, {}).get("name", lang),
        "audio_file": audio_path.name,
        "duration_seconds": round(info.duration, 2),
        "alignment_time": datetime.now().isoformat(),
        "alignment_method": f"whisper_{model_size}",
        "whisper_info": {
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3),
            "transcription_duration": round(info.duration_after_vad, 2)
        },
        "total_sentences": len(aligned_sentences),
        "sentences": aligned_sentences
    }

    # 统计匹配质量
    good_matches = sum(1 for s in aligned_sentences if s.get("match_score", 0) > 0.7)
    avg_score = sum(s.get("match_score", 0) for s in aligned_sentences) / len(aligned_sentences)

    print(f"\n对齐完成:")
    print(f"  总句子数: {len(aligned_sentences)}")
    print(f"  高质量匹配(>0.7): {good_matches} ({good_matches/len(aligned_sentences)*100:.1f}%)")
    print(f"  平均匹配分数: {avg_score:.3f}")

    return result


def split_audio_by_alignment(
    audio_path: Path,
    alignment_data: Dict,
    output_dir: Path
) -> List[Path]:
    """
    根据对齐时间戳切分音频

    Args:
        audio_path: 完整音频文件路径
        alignment_data: 对齐数据
        output_dir: 输出目录

    Returns:
        切分后的音频文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lang = alignment_data["language"]
    ffmpeg_path = get_ffmpeg_path()

    output_files = []
    total = len(alignment_data["sentences"])

    print(f"\n切分音频文件...")

    for i, sent in enumerate(alignment_data["sentences"]):
        index = sent["index"]
        start = sent["start_time"]
        end = sent["end_time"]
        duration = end - start

        # 生成输出文件名
        output_file = output_dir / f"{lang}_{index:04d}.wav"

        # 使用ffmpeg切分
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(duration),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            output_files.append(output_file)
        else:
            print(f"警告: 切分句子 {index} 失败")

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{total}")

    print(f"切分完成: {len(output_files)} 个音频文件")
    return output_files


def generate_metadata(
    alignment_data: Dict,
    audio_dir: Path,
    output_path: Path
) -> Dict:
    """
    生成符合HuggingFace格式的元数据
    """
    lang = alignment_data["language"]

    records = []
    for sent in alignment_data["sentences"]:
        index = sent["index"]
        audio_file = audio_dir / f"{lang}_{index:04d}.wav"

        record = {
            "audio_path": str(audio_file.name) if audio_file.exists() else None,
            "text": sent["text"],
            "language": lang,
            "language_name": LANGUAGE_CONFIG.get(lang, {}).get("name", lang),
            "sentence_index": index,
            "start_time": sent["start_time"],
            "end_time": sent["end_time"],
            "duration_seconds": sent.get("duration", 0),
        }

        if "match_score" in sent:
            record["match_score"] = sent["match_score"]

        records.append(record)

    metadata = {
        "dataset_info": {
            "name": f"museum_reading_corpus_{lang}",
            "description": f"博物馆解说词{LANGUAGE_CONFIG.get(lang, {}).get('name', lang)}朗读语料",
            "version": "2.0.0",
            "created_at": datetime.now().isoformat(),
            "language": lang,
            "alignment_method": alignment_data["alignment_method"],
            "total_samples": len(records),
            "total_duration_seconds": alignment_data["duration_seconds"],
            "audio_format": {
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "format": "wav"
            }
        },
        "samples": records
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"元数据已保存: {output_path}")
    return metadata


def process_language(
    lang: str,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    skip_split: bool = False
):
    """
    处理单个语言的录音

    Args:
        lang: 语言代码
        model_size: Whisper模型大小
        device: 设备
        compute_type: 计算类型
        skip_split: 是否跳过音频切分
    """
    print(f"\n{'='*60}")
    print(f"处理语言: {LANGUAGE_CONFIG.get(lang, {}).get('name', lang)}")
    print(f"{'='*60}")

    # 查找音频文件
    audio_path = PROCESSED_DIR / lang / f"{lang}_full.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    print(f"音频文件: {audio_path}")

    # 进行Whisper对齐
    alignment_data = align_with_whisper(audio_path, lang, model_size, device, compute_type)

    # 保存对齐结果
    alignment_path = ALIGNMENT_DIR / f"alignment_{lang}_whisper.json"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(alignment_data, f, ensure_ascii=False, indent=2)
    print(f"对齐结果已保存: {alignment_path}")

    if not skip_split:
        # 切分音频
        split_dir = PROCESSED_DIR / lang / "sentences_whisper"
        split_files = split_audio_by_alignment(audio_path, alignment_data, split_dir)

        # 生成元数据
        metadata_path = RECORDING_DIR / f"speech_corpus_metadata_{lang}_whisper.json"
        generate_metadata(alignment_data, split_dir, metadata_path)

    return alignment_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Whisper音频对齐工具")
    parser.add_argument(
        "--lang", "-l",
        type=str,
        required=True,
        choices=list(LANGUAGE_CONFIG.keys()),
        help="语言代码"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper模型大小 (默认: medium)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备 (默认: cuda)"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        choices=["float16", "int8", "float32"],
        help="计算类型 (默认: float16)"
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="跳过音频切分"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="只显示音频信息，不进行对齐"
    )

    args = parser.parse_args()

    # 检查CUDA是否可用并调整计算类型
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("警告: CUDA不可用，切换到CPU")
                args.device = "cpu"
                args.compute_type = "int8"
            else:
                print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("警告: PyTorch未安装，无法检测CUDA，使用CPU")
            args.device = "cpu"
            args.compute_type = "int8"

    # CPU不支持float16，自动切换到int8
    if args.device == "cpu" and args.compute_type == "float16":
        print("注意: CPU不支持float16，自动使用int8")
        args.compute_type = "int8"

    # 处理
    try:
        result = process_language(
            args.lang,
            args.model,
            args.device,
            args.compute_type,
            args.skip_split
        )
        print(f"\n处理完成!")
        print(f"  对齐方法: {result['alignment_method']}")
        print(f"  总句子数: {result['total_sentences']}")
        print(f"  音频时长: {result['duration_seconds']:.1f}秒")
    except Exception as e:
        print(f"处理失败: {e}")
        raise


if __name__ == "__main__":
    main()

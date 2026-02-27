#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
录音处理与双模态语料库构建工具

功能：
1. convert_to_wav() - 将M4A录音转换为WAV格式（16kHz, 16bit, mono）
2. align_audio_text() - 使用aeneas进行音频-文本对齐
3. split_by_sentence() - 按句切分音频
4. generate_metadata() - 生成符合HuggingFace格式的元数据

使用方法：
    python process_audio_recording.py --input 交付/录音/原始/es_recording.m4a --lang es
    python process_audio_recording.py --batch  # 处理所有语言
"""

import json
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 添加父目录到路径以导入 config
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

# ==================== 配置参数 ====================
AUDIO_CONFIG = {
    "sample_rate": 16000,     # 16kHz - 语音识别标准采样率
    "channels": 1,            # 单声道
    "bit_depth": 16,          # 16-bit
    "format": "wav",          # 输出格式
}

# 支持的语言配置
SUPPORTED_LANGUAGES = {
    "zh": {"name": "中文", "aeneas_lang": "cmn-Hans-CN"},
    "en": {"name": "英文", "aeneas_lang": "eng-GBR"},
    "ja": {"name": "日语", "aeneas_lang": "jpn-JPN"},
    "es": {"name": "西班牙语", "aeneas_lang": "spa-ESP"},
    "bg": {"name": "保加利亚语", "aeneas_lang": "bul-BGR"},
}

# 目录配置
RECORDING_DIR = config.PROJECT_ROOT / "交付" / "录音"
ORIGINAL_DIR = RECORDING_DIR / "原始"
PROCESSED_DIR = RECORDING_DIR / "处理后"
ALIGNMENT_DIR = RECORDING_DIR / "对齐"
READING_SCRIPT_DIR = config.PROJECT_ROOT / "交付" / "朗读稿件"


def get_ffmpeg_path() -> str:
    """获取ffmpeg可执行文件路径"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # fallback to system ffmpeg


def check_ffmpeg() -> bool:
    """检查ffmpeg是否可用"""
    try:
        ffmpeg_path = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_aeneas() -> bool:
    """检查aeneas是否可用"""
    try:
        import aeneas
        return True
    except ImportError:
        return False


def convert_to_wav(
    input_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    将音频文件转换为WAV格式

    Args:
        input_path: 输入音频文件路径（M4A, MP3等）
        output_path: 输出WAV文件路径（可选，默认同名.wav）
        sample_rate: 采样率（默认16000）
        channels: 声道数（默认1，单声道）

    Returns:
        输出WAV文件路径

    Raises:
        RuntimeError: 如果ffmpeg不可用或转换失败
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg不可用，请先安装ffmpeg并添加到PATH")

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 生成输出路径
    if output_path is None:
        output_path = input_path.with_suffix(".wav")

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建ffmpeg命令
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",                    # 覆盖已存在的文件
        "-i", str(input_path),   # 输入文件
        "-ar", str(sample_rate), # 采样率
        "-ac", str(channels),    # 声道数
        "-acodec", "pcm_s16le",  # 16-bit PCM编码
        str(output_path)         # 输出文件
    ]

    print(f"转换音频: {input_path.name} -> {output_path.name}")
    print(f"  采样率: {sample_rate}Hz, 声道: {channels}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=300  # 5分钟超时
    )

    if result.returncode != 0:
        raise RuntimeError(f"音频转换失败: {result.stderr}")

    print(f"  完成: {output_path}")
    return output_path


def merge_audio_files(
    input_files: List[Path],
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    合并多个音频文件为一个

    Args:
        input_files: 输入音频文件列表
        output_path: 输出文件路径
        sample_rate: 采样率
        channels: 声道数

    Returns:
        输出文件路径
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg不可用")

    # 按文件名排序
    input_files = sorted(input_files)

    # 创建文件列表
    list_file = output_path.parent / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for input_file in input_files:
            # 使用相对路径或绝对路径
            f.write(f"file '{input_file.absolute()}'\n")

    # 使用ffmpeg合并
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-acodec", "pcm_s16le",
        str(output_path)
    ]

    print(f"合并音频文件: {[f.name for f in input_files]}")
    print(f"  -> {output_path.name}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=300
    )

    # 清理临时文件
    if list_file.exists():
        list_file.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"音频合并失败: {result.stderr}")

    print(f"  完成: {output_path}")
    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """
    获取音频文件时长（秒）

    Args:
        audio_path: 音频文件路径

    Returns:
        音频时长（秒）
    """
    # 使用 pydub 获取音频时长
    try:
        from pydub import AudioSegment
        from pydub.utils import mediainfo

        # 设置 ffmpeg 路径
        import os
        os.environ["FFMPEG_BINARY"] = get_ffmpeg_path()

        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0  # 转换为秒
    except Exception as e:
        # 备用方案：使用 ffmpeg 命令
        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path,
            "-i", str(audio_path),
            "-f", "null", "-"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # 从 stderr 中解析时长
        import re
        match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', result.stderr)
        if match:
            hours, minutes, seconds, centiseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100

        raise RuntimeError(f"无法获取音频时长: {e}")


def load_reading_script(lang: str) -> List[str]:
    """
    加载朗读稿件中的句子列表

    Args:
        lang: 语言代码（zh, en, ja, es, bg）

    Returns:
        句子列表
    """
    script_path = READING_SCRIPT_DIR / f"reading_script_{lang}.txt"

    if not script_path.exists():
        raise FileNotFoundError(f"朗读稿件不存在: {script_path}")

    sentences = []
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析朗读稿件，提取句子
    # 格式: [数字] 句子内容
    import re
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


def load_reading_script_info() -> Dict:
    """加载朗读稿件信息JSON"""
    info_path = READING_SCRIPT_DIR / "reading_script_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"朗读稿件信息不存在: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_audio_text(
    audio_path: Path,
    sentences: List[Dict],
    lang: str,
    output_path: Optional[Path] = None,
    use_silence_detection: bool = True
) -> Dict:
    """
    进行音频-文本对齐

    Args:
        audio_path: 音频文件路径
        sentences: 句子列表 [{"index": 1, "text": "..."}, ...]
        lang: 语言代码
        output_path: 对齐结果输出路径（可选）
        use_silence_detection: 当aeneas不可用时是否使用静音检测

    Returns:
        对齐结果字典 {
            "language": "es",
            "audio_file": "es_full.wav",
            "duration_seconds": 3124.5,
            "sentences": [
                {"index": 1, "text": "...", "start_time": 0.0, "end_time": 5.23},
                ...
            ]
        }
    """
    # 获取音频时长
    duration = get_audio_duration(audio_path)
    print(f"音频时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")

    if check_aeneas():
        return _align_with_aeneas(audio_path, sentences, lang, output_path, duration)
    elif use_silence_detection:
        print("警告: aeneas不可用，使用静音检测作为备选方案")
        return _align_with_silence_detection(audio_path, sentences, lang, output_path, duration)
    else:
        print("警告: aeneas不可用，使用均匀分布作为备选方案")
        return _align_uniform(audio_path, sentences, lang, output_path, duration)


def _align_with_aeneas(
    audio_path: Path,
    sentences: List[Dict],
    lang: str,
    output_path: Optional[Path],
    duration: float
) -> Dict:
    """使用aeneas进行精确对齐"""
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    from aeneas.runtimeconfiguration import RuntimeConfiguration
    from aeneas.logger import Logger as AeneasLogger

    print(f"音频时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")

    # 获取aeneas语言代码
    aeneas_lang = SUPPORTED_LANGUAGES.get(lang, {}).get("aeneas_lang", "eng-GBR")
    print(f"对齐语言: {aeneas_lang}")

    # 构建文本字符串（aeneas格式：每行一个片段）
    text_content = "\n".join([s["text"] for s in sentences])

    # 创建临时文本文件
    temp_text_path = audio_path.parent / f"temp_text_{lang}.txt"
    with open(temp_text_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    try:
        # 创建aeneas任务
        task = Task()
        task.audio_file_path_absolute = str(audio_path)
        task.text_file_path_absolute = str(temp_text_path)
        task.text_file_format = "txt"  # 纯文本，每行一个片段
        task.sync_map_file_format = "json"  # 输出JSON格式

        # 配置运行时参数
        config_string = f"task_language={aeneas_lang}|os_task_file_format=json|is_text_type=plain"
        task.configuration_string = config_string

        # 执行对齐
        print("正在进行音频-文本对齐...")
        logger = AeneasLogger(tee=False)
        executor = ExecuteTask(task, logger=logger)
        executor.execute()

        # 获取对齐结果
        sync_map = task.sync_map

        # 解析结果
        aligned_sentences = []
        for i, (sentence, fragment) in enumerate(zip(sentences, sync_map)):
            aligned_sentences.append({
                "index": sentence["index"],
                "text": sentence["text"],
                "start_time": fragment.start,
                "end_time": fragment.end,
                "duration": fragment.end - fragment.start
            })

        # 构建完整结果
        result = {
            "language": lang,
            "language_name": SUPPORTED_LANGUAGES.get(lang, {}).get("name", lang),
            "audio_file": audio_path.name,
            "duration_seconds": round(duration, 2),
            "alignment_time": datetime.now().isoformat(),
            "total_sentences": len(aligned_sentences),
            "sentences": aligned_sentences
        }

        # 保存结果
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"对齐结果已保存: {output_path}")

        return result

    finally:
        # 清理临时文件
        if temp_text_path.exists():
            temp_text_path.unlink()


def _align_with_silence_detection(
    audio_path: Path,
    sentences: List[Dict],
    lang: str,
    output_path: Optional[Path],
    duration: float
) -> Dict:
    """使用静音检测进行对齐（备选方案）"""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    import os

    # 设置 ffmpeg 路径
    os.environ["FFMPEG_BINARY"] = get_ffmpeg_path()

    print(f"加载音频文件进行静音检测...")
    audio = AudioSegment.from_file(str(audio_path))

    # 检测非静音段落
    print("检测静音段落...")
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=500,  # 最小静音长度500ms
        silence_thresh=audio.dBFS - 16,  # 静音阈值
        seek_step=100
    )

    print(f"检测到 {len(nonsilent_ranges)} 个非静音段落")

    # 将非静音段落映射到句子
    aligned_sentences = []

    if len(nonsilent_ranges) >= len(sentences):
        # 如果检测到的段落数 >= 句子数，取前N个段落
        for i, sentence in enumerate(sentences):
            start_ms, end_ms = nonsilent_ranges[i]
            aligned_sentences.append({
                "index": sentence["index"],
                "text": sentence["text"],
                "start_time": start_ms / 1000.0,
                "end_time": end_ms / 1000.0,
                "duration": (end_ms - start_ms) / 1000.0
            })
    else:
        # 如果段落数 < 句子数，回退到均匀分布
        print(f"警告: 检测到的段落数({len(nonsilent_ranges)})少于句子数({len(sentences)})，使用均匀分布")
        return _align_uniform(audio_path, sentences, lang, output_path, duration)

    result = {
        "language": lang,
        "language_name": SUPPORTED_LANGUAGES.get(lang, {}).get("name", lang),
        "audio_file": audio_path.name,
        "duration_seconds": round(duration, 2),
        "alignment_time": datetime.now().isoformat(),
        "alignment_method": "silence_detection",
        "total_sentences": len(aligned_sentences),
        "sentences": aligned_sentences
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"对齐结果已保存: {output_path}")

    return result


def _align_uniform(
    audio_path: Path,
    sentences: List[Dict],
    lang: str,
    output_path: Optional[Path],
    duration: float
) -> Dict:
    """使用均匀分布进行对齐（最简单的备选方案）"""
    n_sentences = len(sentences)
    avg_duration = duration / n_sentences if n_sentences > 0 else 0

    print(f"使用均匀分布对齐: {n_sentences} 句, 平均每句 {avg_duration:.1f} 秒")

    aligned_sentences = []
    for i, sentence in enumerate(sentences):
        start_time = i * avg_duration
        end_time = (i + 1) * avg_duration
        aligned_sentences.append({
            "index": sentence["index"],
            "text": sentence["text"],
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(avg_duration, 3)
        })

    result = {
        "language": lang,
        "language_name": SUPPORTED_LANGUAGES.get(lang, {}).get("name", lang),
        "audio_file": audio_path.name,
        "duration_seconds": round(duration, 2),
        "alignment_time": datetime.now().isoformat(),
        "alignment_method": "uniform_distribution",
        "note": "这是备选方案，时间戳可能不准确。建议安装aeneas获得更精确的对齐。",
        "total_sentences": len(aligned_sentences),
        "sentences": aligned_sentences
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"对齐结果已保存: {output_path}")

    return result


def split_by_sentence(
    audio_path: Path,
    alignment_data: Dict,
    output_dir: Path
) -> List[Path]:
    """
    根据对齐时间戳切分音频

    Args:
        audio_path: 完整音频文件路径
        alignment_data: 对齐数据（来自align_audio_text）
        output_dir: 输出目录

    Returns:
        切分后的音频文件路径列表
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg不可用")

    output_dir.mkdir(parents=True, exist_ok=True)
    lang = alignment_data["language"]

    output_files = []

    for sent in alignment_data["sentences"]:
        index = sent["index"]
        start = sent["start_time"]
        end = sent["end_time"]
        duration = end - start

        # 生成输出文件名
        output_file = output_dir / f"{lang}_{index:04d}.wav"

        # 使用ffmpeg切分
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(duration),
            "-acodec", "copy",  # 避免重新编码
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            output_files.append(output_file)
        else:
            print(f"警告: 切分句子 {index} 失败: {result.stderr}")

    print(f"切分完成: {len(output_files)} 个音频文件")
    return output_files


def generate_metadata(
    alignment_data: Dict,
    audio_dir: Path,
    output_path: Path
) -> Dict:
    """
    生成符合HuggingFace datasets格式的元数据

    Args:
        alignment_data: 对齐数据
        audio_dir: 音频文件目录
        output_path: 元数据输出路径

    Returns:
        元数据字典
    """
    lang = alignment_data["language"]

    # 构建数据集记录
    records = []
    for sent in alignment_data["sentences"]:
        index = sent["index"]
        audio_file = audio_dir / f"{lang}_{index:04d}.wav"

        record = {
            "audio_path": str(audio_file.name) if audio_file.exists() else None,
            "text": sent["text"],
            "language": lang,
            "language_name": SUPPORTED_LANGUAGES.get(lang, {}).get("name", lang),
            "sentence_index": index,
            "start_time": sent["start_time"],
            "end_time": sent["end_time"],
            "duration_seconds": round(sent["duration"], 3),
        }
        records.append(record)

    # 构建完整元数据
    metadata = {
        "dataset_info": {
            "name": f"museum_reading_corpus_{lang}",
            "description": f"博物馆解说词{SUPPORTED_LANGUAGES.get(lang, {}).get('name', lang)}朗读语料",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "language": lang,
            "total_samples": len(records),
            "total_duration_seconds": alignment_data["duration_seconds"],
            "audio_format": {
                "sample_rate": AUDIO_CONFIG["sample_rate"],
                "channels": AUDIO_CONFIG["channels"],
                "bit_depth": AUDIO_CONFIG["bit_depth"],
                "format": AUDIO_CONFIG["format"]
            }
        },
        "samples": records
    }

    # 保存元数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"元数据已保存: {output_path}")
    return metadata


def process_language(
    lang: str,
    input_file: Optional[Path] = None,
    input_files: Optional[List[Path]] = None,
    skip_conversion: bool = False
) -> Dict:
    """
    处理单个语言的录音

    Args:
        lang: 语言代码
        input_file: 输入录音文件（可选，用于单文件）
        input_files: 输入录音文件列表（可选，用于多文件合并）
        skip_conversion: 是否跳过格式转换

    Returns:
        处理结果
    """
    print(f"\n{'='*60}")
    print(f"处理语言: {SUPPORTED_LANGUAGES.get(lang, {}).get('name', lang)}")
    print(f"{'='*60}")

    # 准备输出目录
    lang_output_dir = PROCESSED_DIR / lang
    lang_output_dir.mkdir(parents=True, exist_ok=True)

    # 查找输入文件
    if input_files is None and input_file is None:
        # 在原始目录中查找匹配的文件
        input_files = list(ORIGINAL_DIR.glob(f"{lang}_recording*.*"))
        if not input_files:
            raise FileNotFoundError(f"未找到 {lang} 的录音文件")

        # 如果有多个文件，需要合并
        if len(input_files) > 1:
            print(f"发现多个录音文件: {[f.name for f in sorted(input_files)]}")
        else:
            input_file = input_files[0]
            input_files = None

    if input_file:
        print(f"输入文件: {input_file}")
    elif input_files:
        print(f"输入文件: {[f.name for f in sorted(input_files)]}")

    # 步骤1: 格式转换/合并
    wav_path = lang_output_dir / f"{lang}_full.wav"

    if input_files and len(input_files) > 1:
        # 多文件：先转换再合并
        converted_files = []
        for i, f in enumerate(sorted(input_files)):
            if f.suffix.lower() != ".wav":
                temp_wav = lang_output_dir / f"temp_{i:02d}.wav"
                convert_to_wav(f, temp_wav)
                converted_files.append(temp_wav)
            else:
                converted_files.append(f)

        # 合并所有文件
        merge_audio_files(converted_files, wav_path)

        # 清理临时文件
        for f in converted_files:
            if f.name.startswith("temp_"):
                f.unlink()

    elif not skip_conversion and input_file and input_file.suffix.lower() != ".wav":
        convert_to_wav(input_file, wav_path)
    else:
        if input_file:
            wav_path = input_file if input_file.suffix.lower() == ".wav" else lang_output_dir / f"{lang}_full.wav"
            if not wav_path.exists() and input_file.suffix.lower() == ".wav":
                shutil.copy(input_file, wav_path)

    # 步骤2: 加载朗读稿件
    sentences = load_reading_script(lang)

    # 步骤3: 音频-文本对齐
    alignment_path = ALIGNMENT_DIR / f"alignment_{lang}.json"
    alignment_data = align_audio_text(wav_path, sentences, lang, alignment_path)

    # 步骤4: 按句切分（可选）
    split_dir = lang_output_dir / "sentences"
    split_files = split_by_sentence(wav_path, alignment_data, split_dir)

    # 步骤5: 生成元数据
    metadata_path = RECORDING_DIR / f"speech_corpus_metadata_{lang}.json"
    metadata = generate_metadata(alignment_data, split_dir, metadata_path)

    return {
        "language": lang,
        "wav_file": str(wav_path),
        "alignment_file": str(alignment_path),
        "split_count": len(split_files),
        "metadata_file": str(metadata_path)
    }


def process_all_languages():
    """处理所有支持的语言"""
    results = {}

    for lang in SUPPORTED_LANGUAGES:
        try:
            result = process_language(lang)
            results[lang] = result
        except FileNotFoundError as e:
            print(f"跳过 {lang}: {e}")
        except Exception as e:
            print(f"处理 {lang} 时出错: {e}")
            results[lang] = {"error": str(e)}

    # 保存处理摘要
    summary_path = RECORDING_DIR / "processing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "processed_at": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n处理摘要已保存: {summary_path}")
    return results


def create_directory_structure():
    """创建录音目录结构"""
    dirs = [
        ORIGINAL_DIR,
        PROCESSED_DIR,
        ALIGNMENT_DIR,
    ]

    # 为每种语言创建子目录
    for lang in SUPPORTED_LANGUAGES:
        dirs.append(PROCESSED_DIR / lang)
        dirs.append(PROCESSED_DIR / lang / "sentences")

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {d}")

    print("\n目录结构创建完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="录音处理与双模态语料库构建工具")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="输入录音文件路径"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="语言代码"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量处理所有语言"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="创建目录结构"
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="跳过音频格式转换"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查依赖是否安装"
    )

    args = parser.parse_args()

    # 检查依赖
    if args.check_deps:
        print("检查依赖...")
        print(f"  ffmpeg: {'已安装' if check_ffmpeg() else '未安装'}")
        print(f"  aeneas: {'已安装' if check_aeneas() else '未安装'}")
        return

    # 创建目录结构
    if args.init:
        create_directory_structure()
        return

    # 批量处理
    if args.batch:
        process_all_languages()
        return

    # 单语言处理
    if args.lang:
        process_language(args.lang, args.input, args.skip_conversion)
        return

    # 无参数时显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()

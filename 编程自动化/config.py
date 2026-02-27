#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目集中配置模块
所有路径和阈值在此统一管理，其他脚本通过 import config 使用
"""

import sys
import io
from pathlib import Path

# ==================== 统一设置编码（只执行一次） ====================
_ENCODING_SET = getattr(sys, '_museum_encoding_set', False)
if not _ENCODING_SET:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# ==================== 项目根目录 ====================
# 自动检测：config.py 在 编程自动化/ 下，项目根目录是其父目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================== 目录路径 ====================
RAW_IMAGE_DIR = PROJECT_ROOT / "原始语料"          # 原始展板图片
INTERMEDIATE_DIR = PROJECT_ROOT / "中间结果"        # OCR中间结果
OCR_OUTPUT_DIR = PROJECT_ROOT / "ocr输出"           # 早期OCR输出（parallel_corpus等）
FINAL_OUTPUT_DIR = PROJECT_ROOT / "输出"            # 最终导出
REVIEW_RESULT_DIR = INTERMEDIATE_DIR / "审核结果"   # 人工审核结果
DICT_DIR = PROJECT_ROOT / "词典"                    # OCR纠错词典

# ==================== 关键文件路径 ====================
OCR_RESULTS_FILE = INTERMEDIATE_DIR / "ocr_results.json"           # 主OCR结果（ocr_processor格式）
OCR_RESULTS_ALL_FILE = INTERMEDIATE_DIR / "ocr_results_all.json"   # 合并后的完整OCR结果
REVIEW_QUEUE_FILE = INTERMEDIATE_DIR / "review_queue.json"         # 需审核队列
REVIEWED_RESULTS_FILE = REVIEW_RESULT_DIR / "reviewed_results.json"
PARALLEL_CORPUS_FILE = OCR_OUTPUT_DIR / "parallel_corpus.json"     # 平行语料库（rapid格式）
OCR_ERRORS_DICT_FILE = DICT_DIR / "common_ocr_errors.json"

# ==================== 图片格式 ====================
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ==================== OCR 配置 ====================
# 测试模式
TEST_MODE = False
TEST_MODE_LIMIT = 10

# 选择特定博物馆（空列表 = 处理所有）
SELECTED_MUSEUMS = []  # 例如: ["国家博物馆", "故宫", "首都博物馆"]

# 质量阈值
QUALITY_CONFIG = {
    "ocr_confidence_threshold": 0.85,  # 低于此值标记需审核
    "low_confidence_threshold": 0.8,   # 低置信度分界线
}

# 分级标准
GRADE_CONFIG = {
    "A": {"min_ocr_confidence": 0.9},
    "B": {"min_ocr_confidence": 0.8},
    "C": {"min_ocr_confidence": 0.0},
}

# ==================== 审核界面配置 ====================
REVIEW_SERVER_PORT = 7865
REVIEW_SERVER_NAME = "127.0.0.1"

# ==================== 语料清洗配置 ====================
CLEANER_SERVER_PORT = 7866        # 避免与审核界面 7865 冲突
CLEANER_SERVER_NAME = "127.0.0.1"

# ==================== LLM 配置 ====================
import os
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-85f213a647cc4028ba8844621f628f39")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TIMEOUT = 120       # API调用超时（秒）- 长展板需要更多时间
LLM_MAX_RETRIES = 3     # API调用最大重试次数

# ==================== LLM 增强配置 ====================
ENHANCED_DIR = INTERMEDIATE_DIR / "enhanced"       # 增强结果目录
ENHANCED_CORPUS_FILE = ENHANCED_DIR / "enhanced_corpus.json"
ENHANCED_PROGRESS_FILE = ENHANCED_DIR / "progress.json"

LLM_ENHANCE_CONFIG = {
    "rate_limit_delay": 0.5,     # API 调用间隔（秒）- DeepSeek V3 限速较宽松
    "max_tokens_per_call": 8192, # 单次最大输出 token - 长展板需要更多空间
    "short_text_threshold": 100, # 短文本阈值（字符数），低于此用简化 prompt
    "content_ratio_min": 0.60,   # 输出/输入最小字数比（校验用）- 标题提取后可能偏低
    "content_ratio_max": 1.15,   # 输出/输入最大字数比（校验用）- 收紧以防LLM添加内容
}

# ==================== 多模态模型配置 ====================
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-cdf8f27980b24974bf927c459b90adb8")
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.5-plus"

KIMI_API_KEY = os.getenv("KIMI_API_KEY", "sk-4PQTAz7f2VlLFz5vD0B2C297A50440B9B2E7D60145AeFb55")
KIMI_API_URL = "https://aihubmix.com/v1"
KIMI_MODEL = "kimi-k2.5"

MULTIMODAL_DIR = ENHANCED_DIR / "multimodal"             # 多模态增强结果目录
MULTIMODAL_PROGRESS_FILE = MULTIMODAL_DIR / "progress.json"

MULTIMODAL_CONFIG = {
    "image_max_size": 1280,      # 图片长边最大像素
    "image_quality": 85,         # JPEG 压缩质量
    "max_workers": 3,            # 并发线程数
    "max_tokens_per_call": 4096, # 单次最大输出 token
    "content_ratio_min": 0.50,   # 输出/输入最小字数比（多模态纠错可能删掉更多）
    "content_ratio_max": 1.60,   # 输出/输入最大字数比（多模态看图可能纠正 OCR 漏字，比原文长）
}


# ==================== 302.ai 中转配置（统一接入 Gemini / Claude 等国际模型）====================
AI302_API_KEY = os.getenv("AI302_API_KEY", "sk-4PQTAz7f2VlLFz5vD0B2C297A50440B9B2E7D60145AeFb55")
AI302_API_URL = "https://aihubmix.com/v1"             # OpenAI 兼容接口（aihubmix）

# ==================== 翻译用模型标识（按 302.ai 实际支持名称填写）====================
GEMINI3_FLASH_MODEL  = os.getenv("GEMINI3_FLASH_MODEL",  "gemini-3-flash-preview")  # JA 翻译A/裁判, ES 翻译B/裁判, BG 翻译B
CLAUDE_SONNET_MODEL  = os.getenv("CLAUDE_SONNET_MODEL",  "claude-sonnet-4-6")       # BG 翻译A/裁判

# Qwen-MT（翻译专用，CJK 优化）— 复用已有 QWEN_API_KEY
QWEN_MT_MODEL = os.getenv("QWEN_MT_MODEL", "qwen-mt-turbo")   # JA 翻译B

# Kimi K2.5（ES 翻译A）— 复用已有 KIMI_API_KEY
KIMI_TRANSLATE_MODEL = os.getenv("KIMI_TRANSLATE_MODEL", "kimi-k2.5")

# ==================== 多语种翻译配置 ====================
MULTILINGUAL_DIR             = ENHANCED_DIR                                    # 与 enhanced 同目录
MULTILINGUAL_CORPUS_FILE     = MULTILINGUAL_DIR / "multilingual_corpus.json"
TRANSLATION_PROGRESS_FILE    = MULTILINGUAL_DIR / "translation_progress.json"
MUSEUM_GLOSSARY_FILE         = DICT_DIR / "museum_glossary.json"

TRANSLATION_CONFIG = {
    "max_workers":        5,      # 并发线程数（段落级）
    "max_tokens":      4096,      # 单次最大输出 token
    "retry_max":          3,      # 单次 API 最大重试
    "retry_base_wait":    2,      # 指数退避基础等待（秒）
    "rate_limit_delay": 0.3,      # API 调用间隔（秒）
    "timeout":          120,      # API 调用超时（秒）
}

# 各语言翻译模型分配（按计划 §2.2）
TRANSLATION_MODELS = {
    "ja": {
        "translator_a": {"provider": "302ai", "model": GEMINI3_FLASH_MODEL,  "label": "Gemini3-Flash"},
        "translator_b": {"provider": "qwen",  "model": QWEN_MT_MODEL,        "label": "Qwen-MT-Turbo"},
        "judge":         {"provider": "302ai", "model": GEMINI3_FLASH_MODEL,  "label": "Gemini3-Flash"},
    },
    "es": {
        "translator_a": {"provider": "kimi",  "model": KIMI_TRANSLATE_MODEL, "label": "Kimi-K2.5"},
        "translator_b": {"provider": "302ai", "model": GEMINI3_FLASH_MODEL,  "label": "Gemini3-Flash"},
        "judge":         {"provider": "302ai", "model": GEMINI3_FLASH_MODEL,  "label": "Gemini3-Flash"},
    },
    "bg": {
        "translator_a": {"provider": "302ai", "model": CLAUDE_SONNET_MODEL,  "label": "Claude-Sonnet"},
        "translator_b": {"provider": "302ai", "model": GEMINI3_FLASH_MODEL,  "label": "Gemini3-Flash"},
        "judge":         {"provider": "302ai", "model": GEMINI3_FLASH_MODEL,  "label": "Gemini3-Flash"},
    },
}

# 各语言专项说明（注入 prompt）
LANG_INSTRUCTIONS = {
    "ja": (
        "Use polite neutral form (丁寧語, です/ます体). "
        "Transliterate Chinese proper nouns into established Japanese kanji/katakana conventions."
    ),
    "es": (
        "Use neutral Castilian Spanish (español estándar). "
        "Avoid regional variations and voseo. "
        "Keep Chinese dynasty names as commonly used in Spanish academic contexts."
    ),
    "bg": (
        "Use standard Bulgarian literary language (книжовен български). "
        "Transliterate Chinese names phonetically using Cyrillic script per established Bulgarian conventions."
    ),
}

LANG_NAMES = {
    "ja": "Japanese",
    "es": "Spanish",
    "bg": "Bulgarian",
}


def ensure_dirs():
    """确保所有输出目录存在"""
    for d in [INTERMEDIATE_DIR, OCR_OUTPUT_DIR, FINAL_OUTPUT_DIR, REVIEW_RESULT_DIR, DICT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    """打印当前配置（用于调试）"""
    print("=" * 60)
    print("项目配置")
    print("=" * 60)
    print(f"项目根目录:   {PROJECT_ROOT}")
    print(f"原始图片:     {RAW_IMAGE_DIR}")
    print(f"中间结果:     {INTERMEDIATE_DIR}")
    print(f"最终输出:     {FINAL_OUTPUT_DIR}")
    print(f"OCR结果文件:  {OCR_RESULTS_FILE}")
    print(f"测试模式:     {TEST_MODE}")
    if SELECTED_MUSEUMS:
        print(f"选定博物馆:   {', '.join(SELECTED_MUSEUMS)}")
    print("=" * 60)

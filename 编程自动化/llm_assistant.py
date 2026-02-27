#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM辅助纠错模块（可选增强）
功能：
1. OCR纠错：对低置信度文本进行修正建议
2. 对齐验证：验证中英文本对是否匹配
3. 内容理解：判断文本类型（标题/正文/说明）

使用DeepSeek-V3 API（性价比高）
"""

import sys
import io
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

# 设置编码（如果 config 模块未设置过）
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# ==================== 配置 ====================
try:
    import config as cfg
    DEEPSEEK_API_KEY = cfg.DEEPSEEK_API_KEY
    DEEPSEEK_API_URL = cfg.DEEPSEEK_API_URL
    DEEPSEEK_MODEL = cfg.DEEPSEEK_MODEL
    LLM_TIMEOUT = cfg.LLM_TIMEOUT
    LLM_MAX_RETRIES = cfg.LLM_MAX_RETRIES
except ImportError:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"
    LLM_TIMEOUT = 30
    LLM_MAX_RETRIES = 2

# Qwen-VL配置（用于复杂布局，按需调用）
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")


@dataclass
class OCRCorrectionResult:
    """OCR纠错结果"""
    original_text: str
    corrected_text: str
    confidence: float  # LLM对修正的置信度
    explanation: str  # 修正说明


@dataclass
class AlignmentValidationResult:
    """对齐验证结果"""
    is_aligned: bool
    confidence: float
    explanation: str


@dataclass
class ContentTypeResult:
    """内容类型判断结果"""
    content_type: str  # title, description, instruction, poetry, other
    confidence: float


class LLMAssistant:
    """LLM辅助助手"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.enabled = bool(self.api_key)

        if self.enabled:
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=DEEPSEEK_API_URL,
                    timeout=LLM_TIMEOUT
                )
                print("[OK] LLM Assistant initialized with DeepSeek")
            except ImportError:
                print("[WARNING] openai package not installed, LLM features disabled")
                print("  Install with: pip install openai")
                self.enabled = False
            except Exception as e:
                print(f"[WARNING] Failed to initialize LLM: {e}")
                self.enabled = False
        else:
            print("[INFO] LLM Assistant disabled (no API key)")
            print("  Set DEEPSEEK_API_KEY environment variable to enable")

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从LLM响应中提取JSON对象（支持嵌套大括号）"""
        import re
        # 尝试方法1：找 ```json ... ``` 代码块
        code_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试方法2：找最外层 { ... }
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        start = None  # 继续找下一个

        return None

    def _call_api(self, messages: List[Dict], max_tokens: int = 500) -> Optional[str]:
        """调用API，支持重试"""
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < LLM_MAX_RETRIES:
                    import time
                    wait = 2 ** attempt
                    print(f"[RETRY] API call failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[ERROR] API call failed after {LLM_MAX_RETRIES + 1} attempts: {e}")
        return None

    def correct_ocr_text(self, text: str, language: str = "zh",
                         context: Optional[str] = None) -> Optional[OCRCorrectionResult]:
        """
        对OCR识别的文本进行纠错

        Args:
            text: OCR识别的文本
            language: 语言 (zh/en)
            context: 上下文（可选）

        Returns:
            OCRCorrectionResult or None
        """
        if not self.enabled:
            return None

        lang_name = "中文" if language == "zh" else "English"
        context_str = f"\n上下文：{context}" if context else ""

        prompt = f"""你是一位专业的OCR纠错专家。请修正以下{lang_name}文本中的OCR识别错误。

原始文本：{text}{context_str}

请分析并回答：
1. 文本中是否有明显的OCR错误（如形近字、拼写错误等）？
2. 修正后的文本是什么？
3. 你对修正的置信度是多少（0-1）？
4. 请简要说明修正理由。

请以JSON格式输出：
{{
    "has_error": true/false,
    "corrected_text": "修正后的文本",
    "confidence": 0.95,
    "explanation": "修正说明"
}}"""

        try:
            content = self._call_api([
                {"role": "system", "content": "你是一个专业的OCR纠错专家，擅长识别和修正OCR错误。"},
                {"role": "user", "content": prompt}
            ], max_tokens=500)

            if not content:
                return None

            result = self._extract_json(content)
            if result:

                if result.get("has_error"):
                    return OCRCorrectionResult(
                        original_text=text,
                        corrected_text=result.get("corrected_text", text),
                        confidence=result.get("confidence", 0.5),
                        explanation=result.get("explanation", "")
                    )
                else:
                    return OCRCorrectionResult(
                        original_text=text,
                        corrected_text=text,
                        confidence=1.0,
                        explanation="No errors detected"
                    )

        except Exception as e:
            print(f"[ERROR] OCR correction failed: {e}")

        return None

    def validate_alignment(self, zh_text: str, en_text: str) -> Optional[AlignmentValidationResult]:
        """
        验证中英文本对是否属于同一内容段落

        Args:
            zh_text: 中文文本
            en_text: 英文文本

        Returns:
            AlignmentValidationResult or None
        """
        if not self.enabled:
            return None

        prompt = f"""你是一位博物馆文本对齐验证专家。请判断以下中英文本是否是同一内容的翻译。

中文：{zh_text}

English：{en_text}

请分析并回答：
1. 这两段文本是否表达相同的内容？（是/否/不确定）
2. 你的置信度是多少（0-1）？
3. 简要说明理由。

请以JSON格式输出：
{{
    "is_aligned": true/false,
    "confidence": 0.95,
    "explanation": "判断理由"
}}"""

        try:
            content = self._call_api([
                {"role": "system", "content": "你是一个博物馆文本对齐验证专家，擅长判断中英文本是否为对应翻译。"},
                {"role": "user", "content": prompt}
            ], max_tokens=300)

            if not content:
                return None

            result = self._extract_json(content)
            if result:

                return AlignmentValidationResult(
                    is_aligned=result.get("is_aligned", False),
                    confidence=result.get("confidence", 0.5),
                    explanation=result.get("explanation", "")
                )

        except Exception as e:
            print(f"[ERROR] Alignment validation failed: {e}")

        return None

    def classify_content_type(self, text: str, language: str = "zh") -> Optional[ContentTypeResult]:
        """
        判断文本内容类型

        Args:
            text: 文本内容
            language: 语言 (zh/en)

        Returns:
            ContentTypeResult or None
        """
        if not self.enabled:
            return None

        prompt = f"""你是一位博物馆文本分类专家。请判断以下文本的类型。

文本：{text}

可能的类型：
- title: 标题（文物名称、展区名称等）
- description: 描述/说明（文物介绍、历史背景等）
- poetry: 诗词/铭文
- instruction: 指示/说明（出口、请勿触摸等）
- other: 其他

请以JSON格式输出：
{{
    "content_type": "title/description/poetry/instruction/other",
    "confidence": 0.95
}}"""

        try:
            content = self._call_api([
                {"role": "system", "content": "你是一个博物馆文本分类专家。"},
                {"role": "user", "content": prompt}
            ], max_tokens=200)

            if not content:
                return None

            result = self._extract_json(content)
            if result:

                return ContentTypeResult(
                    content_type=result.get("content_type", "other"),
                    confidence=result.get("confidence", 0.5)
                )

        except Exception as e:
            print(f"[ERROR] Content classification failed: {e}")

        return None

    def batch_process_low_confidence(self, aligned_pairs: List[Dict],
                                     ocr_threshold: float = 0.85) -> List[Dict]:
        """
        批量处理低置信度的文本对

        Args:
            aligned_pairs: 对齐的文本对列表
            ocr_threshold: OCR置信度阈值

        Returns:
            添加了LLM建议的结果列表
        """
        if not self.enabled:
            return aligned_pairs

        results = []

        for pair in aligned_pairs:
            result = pair.copy()
            result["llm_suggestions"] = {}

            zh = pair.get("zh")
            en = pair.get("en")

            # 检查低置信度OCR
            if zh and zh.get("ocr_confidence", 1) < ocr_threshold:
                correction = self.correct_ocr_text(zh.get("text", ""), "zh")
                if correction:
                    result["llm_suggestions"]["zh_correction"] = {
                        "original": correction.original_text,
                        "suggested": correction.corrected_text,
                        "confidence": correction.confidence,
                        "explanation": correction.explanation
                    }

            if en and en.get("ocr_confidence", 1) < ocr_threshold:
                correction = self.correct_ocr_text(en.get("text", ""), "en")
                if correction:
                    result["llm_suggestions"]["en_correction"] = {
                        "original": correction.original_text,
                        "suggested": correction.corrected_text,
                        "confidence": correction.confidence,
                        "explanation": correction.explanation
                    }

            # 验证对齐
            if zh and en:
                validation = self.validate_alignment(zh.get("text", ""), en.get("text", ""))
                if validation:
                    result["llm_suggestions"]["alignment_validation"] = {
                        "is_aligned": validation.is_aligned,
                        "confidence": validation.confidence,
                        "explanation": validation.explanation
                    }

            results.append(result)

        return results


def demo():
    """演示LLM辅助功能"""
    assistant = LLMAssistant()

    if not assistant.enabled:
        print("\nLLM Assistant is disabled. Set DEEPSEEK_API_KEY to enable.")
        return

    # 示例1: OCR纠错
    print("\n=== OCR Correction Demo ===")
    sample_text = "这件青銅器出士于河南安陽"
    result = assistant.correct_ocr_text(sample_text, "zh")
    if result:
        print(f"Original: {result.original_text}")
        print(f"Corrected: {result.corrected_text}")
        print(f"Confidence: {result.confidence}")
        print(f"Explanation: {result.explanation}")

    # 示例2: 对齐验证
    print("\n=== Alignment Validation Demo ===")
    zh = "这件青铜器出土于河南安阳"
    en = "This bronze ware was unearthed in Anyang, Henan"
    result = assistant.validate_alignment(zh, en)
    if result:
        print(f"Chinese: {zh}")
        print(f"English: {en}")
        print(f"Is Aligned: {result.is_aligned}")
        print(f"Confidence: {result.confidence}")
        print(f"Explanation: {result.explanation}")

    # 示例3: 内容分类
    print("\n=== Content Classification Demo ===")
    sample = "清乾隆青花缠枝莲纹瓶"
    result = assistant.classify_content_type(sample, "zh")
    if result:
        print(f"Text: {sample}")
        print(f"Type: {result.content_type}")
        print(f"Confidence: {result.confidence}")


if __name__ == "__main__":
    demo()

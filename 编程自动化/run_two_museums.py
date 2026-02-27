#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理两个指定博物馆的OCR：抗日纪念馆和党史馆
"""

import sys
import io
from pathlib import Path

# 设置编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入OCR处理器
from ocr_processor import MuseumOCRProcessor, INPUT_DIR, OUTPUT_DIR
import ocr_processor as ocr_module


def main():
    """运行指定两个博物馆的OCR处理"""
    # 设置只处理抗日纪念馆和党史馆
    ocr_module.SELECTED_MUSEUMS = ["抗日纪念馆", "党史馆"]
    ocr_module.TEST_MODE = False  # 非测试模式，处理全部

    # 运行处理
    processor = MuseumOCRProcessor(INPUT_DIR, OUTPUT_DIR)
    processor.process_all()


if __name__ == "__main__":
    main()

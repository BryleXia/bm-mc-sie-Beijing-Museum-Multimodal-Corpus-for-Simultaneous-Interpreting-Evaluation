#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量OCR处理脚本
用于处理所有图片（非测试模式）
"""

import sys
import io
from pathlib import Path

# 设置编码
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# 导入配置
try:
    import config as cfg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as cfg

from ocr_processor import MuseumOCRProcessor


def main():
    """运行完整的OCR批处理"""
    print("=" * 70)
    print("博物馆平行语料 - 批量OCR处理")
    print("=" * 70)
    print(f"输入目录: {cfg.RAW_IMAGE_DIR}")
    print(f"输出目录: {cfg.INTERMEDIATE_DIR}")
    print("=" * 70)

    # 确认
    response = input("\n这将处理所有图片（约605张），可能需要较长时间。\n确认开始? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("已取消")
        return

    # 确保非测试模式
    cfg.TEST_MODE = False

    # 运行处理
    processor = MuseumOCRProcessor(str(cfg.RAW_IMAGE_DIR), str(cfg.INTERMEDIATE_DIR))
    processor.process_all()

    print("\n" + "=" * 70)
    print("处理完成！")
    print(f"结果保存在: {cfg.INTERMEDIATE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

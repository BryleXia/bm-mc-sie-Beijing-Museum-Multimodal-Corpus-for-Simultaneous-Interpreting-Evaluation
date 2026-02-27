#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 预览脚本 - 使用 EasyOCR (更稳定)
"""

import sys
import io
from pathlib import Path

# 设置编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_DIR = r"E:\ai知识库\nlp大赛\原始语料"
SAMPLE_SIZE = 5  # 预览图片数量

def main():
    print("=" * 60)
    print("OCR Preview Mode with EasyOCR (First {} images)".format(SAMPLE_SIZE))
    print("=" * 60)

    # 导入并初始化OCR
    print("\nInitializing EasyOCR engine...")
    print("(First run will download models, please wait...)")
    import easyocr
    # 支持中英文
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    print("[OK] OCR engine ready\n")

    # 找图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = []

    for ext in image_extensions:
        all_images.extend(Path(INPUT_DIR).rglob(f"*{ext}"))

    if not all_images:
        print("[ERROR] No image files found")
        return

    print(f"Found {len(all_images)} images total, testing first {SAMPLE_SIZE}\n")

    # 测试前N张
    for idx, img_path in enumerate(all_images[:SAMPLE_SIZE], 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{SAMPLE_SIZE}] Image: {img_path.name}")
        print(f"Path: {img_path}")
        print(f"{'='*60}")

        try:
            result = reader.readtext(str(img_path))

            if result:
                print(f"\nDetected {len(result)} text regions:\n")

                for i, detection in enumerate(result, 1):
                    bbox = detection[0]  # 文本框坐标
                    text = detection[1]   # 文本内容
                    confidence = detection[2]  # 置信度

                    # 简单判断语言
                    lang_hint = "ZH" if any('\u4e00' <= c <= '\u9fff' for c in text) else "EN"

                    print(f"  [{i}] [{lang_hint}] Conf: {confidence:.3f} | Text: {text}")
            else:
                print("  (No text detected)")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n\n{'='*60}")
    print("Preview complete!")
    print("\nNext steps:")
    print("  1. If OCR quality looks good, run the full processor:")
    print("     python ocr_processor_easy.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

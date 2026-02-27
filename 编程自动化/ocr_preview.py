#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 预览脚本 - 快速测试几张图片的效果
"""

import sys
import io
from pathlib import Path
from paddleocr import PaddleOCR

# 设置编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_DIR = r"E:\ai知识库\nlp大赛\原始语料"
SAMPLE_SIZE = 5  # 预览图片数量

def main():
    print("=" * 60)
    print("OCR Preview Mode (First {} images)".format(SAMPLE_SIZE))
    print("=" * 60)

    # 初始化OCR
    print("\nInitializing OCR engine...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
    print("[OK] OCR engine ready\n")

    # 找图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = []

    for ext in image_extensions:
        all_images.extend(Path(INPUT_DIR).rglob(f"*{ext}"))

    if not all_images:
        print("[ERROR] No image files found")
        return

    # 测试前N张
    for img_path in all_images[:SAMPLE_SIZE]:
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"Path: {img_path}")
        print(f"{'='*60}")

        result = ocr.predict(str(img_path))

        if result and result[0]:
            print(f"\nDetected {len(result[0])} text regions:\n")

            for i, line in enumerate(result[0], 1):
                if line:
                    text = line[1][0]
                    confidence = line[1][1]
                    print(f"  [{i}] Confidence: {confidence:.3f} | Text: {text}")
        else:
            print("  (No text detected)")

    print(f"\n\n{'='*60}")
    print("Preview complete! To process all images, run:")
    print("  python ocr_processor.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

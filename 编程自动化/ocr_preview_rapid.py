#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 预览脚本 - 使用 RapidOCR (国内优化，无需下载模型)
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
    print("OCR Preview Mode with RapidOCR (First {} images)".format(SAMPLE_SIZE))
    print("=" * 60)

    # 导入并初始化OCR
    print("\nInitializing RapidOCR engine...")
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
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
            result = engine(str(img_path))
            # RapidOCR 返回: (results, times)
            # results 是列表，每个元素是 [box, text, score]

            if result and result[0]:
                texts = result[0]
                print(f"\nDetected {len(texts)} text regions:\n")

                for i, item in enumerate(texts[:10], 1):  # 只显示前10条
                    box = item[0]      # 文本框坐标
                    text = item[1]      # 文本内容
                    score = float(item[2])     # 置信度

                    # 简单判断语言
                    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
                    lang_hint = "ZH" if has_chinese else "EN"

                    print(f"  [{i}] [{lang_hint}] Conf: {score:.3f} | Text: {text}")

                if len(texts) > 10:
                    print(f"  ... and {len(texts) - 10} more regions")
            else:
                print("  (No text detected)")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'='*60}")
    print("Preview complete!")
    print("\nNext steps:")
    print("  1. If OCR quality looks good, run the full processor:")
    print("     python ocr_processor_rapid.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

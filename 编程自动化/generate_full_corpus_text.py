#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成完整的人类可读语料文本
"""

import json
from pathlib import Path
from datetime import datetime

INPUT_FILE = Path(r"E:\ai知识库\nlp大赛\中间结果\ocr_results.json")
OUTPUT_FILE = Path(r"E:\ai知识库\nlp大赛\中间结果\corpus_text_all.txt")

def main():
    print("Loading OCR results...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        results = data.get('results', [])

    print(f"Generating readable text for {len(results)} images...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("博物馆平行语料 - 段落级对齐文本\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n")
        f.write(f"总计: {len(results)} 张图片\n")
        f.write("=" * 70 + "\n\n")

        for result in results:
            if "error" in result:
                continue

            f.write(f"\n{'='*70}\n")
            f.write(f"图片ID: {result['image_id']}\n")
            f.write(f"博物馆: {result['source']['museum']}\n")
            f.write(f"图片名: {result['source']['image_name']}\n")

            quality = result['quality']
            f.write(f"质量等级: {quality['grade']}, ")
            f.write(f"中文{quality['zh_block_count']}块, 英文{quality['en_block_count']}块, ")
            f.write(f"平均OCR置信度{quality['avg_ocr_confidence']}\n")

            if result.get('needs_review'):
                f.write(f"需审核: {result.get('review_reason', '')}\n")

            f.write(f"{'='*70}\n\n")

            # 输出完整段落
            f.write(f"【中文】\n{result['zh_text']}\n\n")
            f.write(f"【English】\n{result['en_text']}\n\n")

    print(f"[OK] Generated: {OUTPUT_FILE}")
    print(f"Total entries: {len(results)}")

if __name__ == "__main__":
    main()

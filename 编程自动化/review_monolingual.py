#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单语图片查看与配对工具
用于查看纯中文和纯英文的图片OCR结果，并人工判断对应关系
"""

import json
import sys
import io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_FILE = r"E:\ai知识库\nlp大赛\ocr输出\ocr_results.json"
OUTPUT_FILE = r"E:\ai知识库\nlp大赛\ocr输出\monolingual_review.json"

def main():
    # 读取OCR结果
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data['entries']

    # 筛选单语图片
    chinese_only = []
    english_only = []

    for entry in entries:
        if 'error' in entry:
            continue

        img_type = entry['classification']['image_type']

        if img_type == 'chinese_only':
            chinese_only.append(entry)
        elif img_type == 'english_only':
            english_only.append(entry)

    print("=" * 70)
    print("单语图片查看与配对")
    print("=" * 70)
    print(f"\n发现 {len(chinese_only)} 张纯中文图片")
    print(f"发现 {len(english_only)} 张纯英文图片\n")

    # 按博物馆分组展示
    print("-" * 70)
    print("【纯中文图片】")
    print("-" * 70)

    for i, entry in enumerate(chinese_only, 1):
        print(f"\n[{i}] {entry['entry_id']}")
        print(f"    图片: {entry['source']['image_name']}")
        print(f"    内容:")
        content = entry['image_content']['zh_all']
        # 限制显示长度
        lines = content.split('\n')[:5]
        for line in lines:
            print(f"      {line}")
        if len(content.split('\n')) > 5:
            print(f"      ... (还有 {len(content.split('\n')) - 5} 行)")

    print("\n" + "-" * 70)
    print("【纯英文图片】")
    print("-" * 70)

    for i, entry in enumerate(english_only, 1):
        print(f"\n[{i}] {entry['entry_id']}")
        print(f"    图片: {entry['source']['image_name']}")
        print(f"    内容:")
        content = entry['image_content']['en_all']
        # 限制显示长度
        lines = content.split('\n')[:5]
        for line in lines:
            print(f"      {line}")
        if len(content.split('\n')) > 5:
            print(f"      ... (还有 {len(content.split('\n')) - 5} 行)")

    # 保存单语图片信息供后续处理
    review_data = {
        "metadata": {
            "total_chinese_only": len(chinese_only),
            "total_english_only": len(english_only),
            "note": "请人工检查这些单语图片，判断它们是否互为对应翻译"
        },
        "chinese_only": [
            {
                "entry_id": e['entry_id'],
                "image_name": e['source']['image_name'],
                "museum": e['source']['museum'],
                "content": e['image_content']['zh_all'],
                "potential_match": None  # 待填写
            }
            for e in chinese_only
        ],
        "english_only": [
            {
                "entry_id": e['entry_id'],
                "image_name": e['source']['image_name'],
                "museum": e['source']['museum'],
                "content": e['image_content']['en_all'],
                "potential_match": None  # 待填写
            }
            for e in english_only
        ],
        "pairing_suggestions": []  # 待填写
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("查看完成！")
    print("=" * 70)
    print(f"\n单语图片信息已保存到: {OUTPUT_FILE}")
    print("\n建议操作:")
    print("  1. 打开上述JSON文件，查看所有单语图片的完整内容")
    print("  2. 对比文件名和拍摄时间，判断哪些是同一展品的对应文本")
    print("  3. 在 pairing_suggestions 字段中填写配对建议")
    print("\n可能的配对线索:")
    print("  - 文件名接近（如 IMG_001.jpg 和 IMG_002.jpg）")
    print("  - 同一博物馆的连续拍摄")
    print("  - 内容主题相关")

if __name__ == "__main__":
    main()

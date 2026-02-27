#!/usr/bin/env python3
"""
从中英双语数据提取黄金标准语料库

从 parallel_corpus_5lang.json 中提取中英对照部分，
生成 zh_en_parallel.json 作为黄金标准。
"""

import json
from datetime import datetime
from pathlib import Path

def extract_zh_en_parallel():
    """提取中英双语平行语料"""

    # 读取五语种语料库
    input_file = Path(__file__).parent.parent / "交付" / "parallel_corpus_5lang.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # 构建黄金标准语料库
    gold_standard = {
        "corpus_id": "bisu-museum-gold-standard-2026",
        "name": "中国博物馆解说词黄金标准语料库（中英对照）",
        "version": "1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "languages": ["zh", "en"],
        "language_names": {
            "zh": "中文",
            "en": "English"
        },
        "metadata": {
            "source": "故宫、国家博物馆、首都博物馆、党史馆、抗日纪念馆展板",
            "domain": "博物馆解说词",
            "translation_type": "博物馆官方翻译（非机翻）",
            "total_units": len(corpus["alignment_units"]),
            "coverage": {
                "zh": len(corpus["alignment_units"]),
                "en": len(corpus["alignment_units"])
            },
            "museums": {}
        },
        "alignment_units": []
    }

    # 统计各博物馆数量
    museum_counts = {}

    # 提取中英对照数据
    for unit in corpus["alignment_units"]:
        zh_en_unit = {
            "id": unit["id"],
            "source": unit["source"],
            "translations": {
                "zh": unit["translations"]["zh"],
                "en": unit["translations"]["en"]
            }
        }
        gold_standard["alignment_units"].append(zh_en_unit)

        # 统计博物馆
        museum = unit["source"]["museum"]
        museum_counts[museum] = museum_counts.get(museum, 0) + 1

    gold_standard["metadata"]["museums"] = museum_counts

    # 保存黄金标准文件
    output_dir = Path(__file__).parent.parent / "黄金标准"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "zh_en_parallel.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gold_standard, f, ensure_ascii=False, indent=2)

    print(f"黄金标准语料库已生成:")
    print(f"  文件: {output_file}")
    print(f"  总条目: {len(gold_standard['alignment_units'])}")
    print(f"  博物馆分布:")
    for museum, count in museum_counts.items():
        print(f"    - {museum}: {count} 条")

    return gold_standard

if __name__ == "__main__":
    extract_zh_en_parallel()

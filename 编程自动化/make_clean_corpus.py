# -*- coding: utf-8 -*-
"""
生成简洁版五语种平行语料库（符合行业标准）
结构：corpus_id + languages + alignment_units[]
每条包含 id、source（博物馆+图片）、translations（五语种）
"""
import json
import os
import sys
import io
from datetime import datetime

# 编码设置
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
        sys._museum_encoding_set = True
    except Exception:
        pass

SRC = r"E:\ai知识库\nlp大赛\中间结果\enhanced\multilingual_corpus.json"
DST = r"E:\ai知识库\nlp大赛\交付\parallel_corpus_5lang.json"

with open(SRC, encoding="utf-8") as f:
    raw = json.load(f)

units = []
sent_counter = 0

for board in raw.get("boards", []):
    board_id = board.get("board_id", "unknown")
    src = board.get("source", {})
    museum = src.get("museum", "")
    image = src.get("image_name", "")
    title = board.get("board_title", {})

    for para in board.get("paragraphs", []):
        para_idx = para.get("para_index", 0)
        for sent in para.get("sentences", []):
            sent_idx = sent.get("sent_index", 0)
            sent_counter += 1

            # 五语种文本，缺任何一个语种就跳过整条
            translations = {}
            for lang in ("zh", "en", "ja", "es", "bg"):
                txt = sent.get(lang, "") or ""
                if txt.strip():
                    translations[lang] = txt

            if len(translations) != 5:  # 必须五语种齐全
                continue

            unit = {
                "id": f"{board_id}_p{para_idx}s{sent_idx}",
                "source": {
                    "museum": museum,
                    "image": image,
                    "title_zh": title.get("zh", ""),
                    "title_en": title.get("en", ""),
                },
                "translations": translations
            }
            units.append(unit)

# 统计各语种覆盖率
lang_counts = {lang: 0 for lang in ("zh", "en", "ja", "es", "bg")}
for u in units:
    for lang in u["translations"]:
        lang_counts[lang] = lang_counts[lang] + 1

out = {
    "corpus_id": "bisu-museum-parallel-2026",
    "name": "中国博物馆多语种解说词平行语料库",
    "version": "1.0",
    "created": datetime.now().strftime("%Y-%m-%d"),
    "languages": ["zh", "en", "ja", "es", "bg"],
    "language_names": {
        "zh": "中文", "en": "English", "ja": "日本語",
        "es": "Español", "bg": "Български"
    },
    "metadata": {
        "source": "故宫、国家博物馆、首都博物馆、党史馆、抗日纪念馆展板",
        "domain": "博物馆解说词",
        "total_units": len(units),
        "coverage": lang_counts,
    },
    "alignment_units": units
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

size_mb = os.path.getsize(DST) / 1024 / 1024
print(f"[OK] 生成完成！")
print(f"   文件: {DST}")
print(f"   大小: {size_mb:.2f} MB")
print(f"   对齐单元: {len(units)}")
print(f"   覆盖率: {lang_counts}")

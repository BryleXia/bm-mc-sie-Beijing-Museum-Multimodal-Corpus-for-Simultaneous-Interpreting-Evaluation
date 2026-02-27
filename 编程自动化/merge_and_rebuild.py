# -*- coding: utf-8 -*-
"""
合并4对单语展板（一板两拍），重新生成五语种语料库
"""
import json
import os
from datetime import datetime

# 4对匹配关系：(中文板ID, 英文板ID)
PAIRS = [
    ("国家博物馆_IMG20260121114840", "国家博物馆_IMG20260121114842"),  # 宋代其他名窑
    ("国家博物馆_IMG20260121121509", "国家博物馆_IMG20260121121512"),  # 前言农民画
    ("故宫_IMG20260115144710", "故宫_IMG20260115144715"),              # 前言陶瓷
    ("首都博物馆_IMG20260120153150", "首都博物馆_IMG20260120153157"),  # 人类文明黄金
]

SRC = r"E:\ai知识库\nlp大赛\中间结果\enhanced\multilingual_corpus.json"
DST = r"E:\ai知识库\nlp大赛\交付\parallel_corpus_5lang.json"

with open(SRC, encoding="utf-8") as f:
    raw = json.load(f)

boards = raw.get("boards", [])
board_dict = {b["board_id"]: b for b in boards}

merged_count = 0

for zh_id, en_id in PAIRS:
    zh_board = board_dict.get(zh_id)
    en_board = board_dict.get(en_id)

    if not zh_board or not en_board:
        print(f"Skip: {zh_id} or {en_id} not found")
        continue

    # 从中文板取 zh，从英文板取 en + 翻译
    zh_paras = zh_board.get("paragraphs", [])
    en_paras = en_board.get("paragraphs", [])

    if not zh_paras or not en_paras:
        print(f"Skip: {zh_id} or {en_id} has no paragraphs")
        continue

    # 合并策略：用英文板的结构（已有翻译），把每句的 zh 替换成中文板的对应内容
    # 假设段落和句子数量相同（同一展板的两种语言）
    new_paras = []
    for i, en_para in enumerate(en_paras):
        zh_para = zh_paras[i] if i < len(zh_paras) else None

        new_sents = []
        for j, en_sent in enumerate(en_para.get("sentences", [])):
            zh_sent = zh_para.get("sentences", [])[j] if zh_para and j < len(zh_para.get("sentences", [])) else None

            # 构建合并后的句子
            merged_sent = {
                "sent_index": en_sent.get("sent_index", j),
                "zh": (zh_sent.get("zh", "") if zh_sent else ""),
                "en": en_sent.get("en", ""),
                "ja": en_sent.get("ja", ""),
                "es": en_sent.get("es", ""),
                "bg": en_sent.get("bg", ""),
            }
            new_sents.append(merged_sent)

        new_paras.append({
            "para_index": en_para.get("para_index", i),
            "zh": zh_para.get("zh", "") if zh_para else "",
            "en": en_para.get("en", ""),
            "sentences": new_sents,
        })

    # 更新英文板为合并板
    en_board["paragraphs"] = new_paras
    en_board["board_id"] = en_id  # 保持原ID
    en_board["_merged_from"] = zh_id  # 标记合并来源

    # 删除中文板
    if zh_id in board_dict:
        del board_dict[zh_id]

    merged_count += 1
    print(f"Merged: {zh_id} + {en_id}")

# 保存更新后的 multilingual_corpus（可选，备份）
backup_path = SRC.replace(".json", f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
os.rename(SRC, backup_path)
with open(SRC, "w", encoding="utf-8") as f:
    json.dump(raw, f, ensure_ascii=False, indent=2)
print(f"\nSaved merged corpus (backup: {os.path.basename(backup_path)})")

# ========== 生成简洁版五语种语料库 ==========
units = []
for board in raw.get("boards", []):
    board_id = board.get("board_id", "")
    src = board.get("source", {})
    museum = src.get("museum", "")
    image = src.get("image_name", "")
    title = board.get("board_title", {})

    for para in board.get("paragraphs", []):
        para_idx = para.get("para_index", 0)
        for sent in para.get("sentences", []):
            sent_idx = sent.get("sent_index", 0)

            # 五语种必须齐全
            translations = {}
            for lang in ("zh", "en", "ja", "es", "bg"):
                txt = (sent.get(lang, "") or "").strip()
                if txt:
                    translations[lang] = txt

            if len(translations) != 5:
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

# 统计
lang_counts = {lang: 0 for lang in ("zh", "en", "ja", "es", "bg")}
for u in units:
    for lang in u["translations"]:
        lang_counts[lang] += 1

out = {
    "corpus_id": "bisu-museum-parallel-2026",
    "name": "中国博物馆多语种解说词平行语料库",
    "version": "1.1",
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
        "merged_pairs": merged_count,
    },
    "alignment_units": units
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

size_mb = os.path.getsize(DST) / 1024 / 1024
print(f"\n=== Done ===")
print(f"  Merged pairs: {merged_count}")
print(f"  Total units: {len(units)}")
print(f"  Coverage: {lang_counts}")
print(f"  File: {DST}")
print(f"  Size: {size_mb:.2f} MB")

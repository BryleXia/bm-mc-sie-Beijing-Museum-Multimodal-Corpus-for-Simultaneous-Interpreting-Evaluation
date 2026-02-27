# -*- coding: utf-8 -*-
"""
完整修复：从OCR捞回被过滤的英文板，与中文板合并，补全翻译
"""
import json
import os
import copy
from datetime import datetime

# 4对匹配关系
PAIRS = [
    ("国家博物馆_IMG20260121114840", "国家博物馆_IMG20260121114842"),  # 宋代其他名窑
    ("国家博物馆_IMG20260121121509", "国家博物馆_IMG20260121121512"),  # 前言农民画
    ("故宫_IMG20260115144710", "故宫_IMG20260115144715"),              # 前言陶瓷
    ("首都博物馆_IMG20260120153150", "首都博物馆_IMG20260120153157"),  # 人类文明黄金
]

ENHANCED_PATH = r"E:\ai知识库\nlp大赛\中间结果\enhanced\enhanced_corpus.json"
OCR_PATH = r"E:\ai知识库\nlp大赛\中间结果\ocr_results.json"
MULTI_PATH = r"E:\ai知识库\nlp大赛\中间结果\enhanced\multilingual_corpus.json"

# 加载数据
with open(ENHANCED_PATH, encoding="utf-8") as f:
    enhanced = json.load(f)
with open(OCR_PATH, encoding="utf-8") as f:
    ocr = json.load(f)
with open(MULTI_PATH, encoding="utf-8") as f:
    multi = json.load(f)

enhanced_dict = {b["board_id"]: b for b in enhanced.get("boards", [])}
multi_dict = {b["board_id"]: b for b in multi.get("boards", [])}
ocr_dict = {r["image_id"]: r for r in ocr.get("results", [])}

def get_ocr_text(img_id, lang):
    """从OCR结果获取文本"""
    r = ocr_dict.get(img_id)
    if not r:
        return ""
    return (r.get(f"{lang}_text") or "").strip()

def create_board_from_ocr(img_id, lang):
    """从OCR创建一个简单的板结构"""
    r = ocr_dict.get(img_id)
    if not r:
        return None
    text = (r.get(f"{lang}_text") or "").strip()
    if not text:
        return None

    src = r.get("source", {})
    return {
        "board_id": img_id,
        "source": src,
        "board_title": {},
        "paragraphs": [{
            "para_index": 0,
            lang: text,
            "sentences": [{
                "sent_index": 0,
                lang: text
            }]
        }]
    }

merged_count = 0
fixed_in_multi = 0

for zh_id, en_id in PAIRS:
    zh_board_enhanced = enhanced_dict.get(zh_id)
    en_board_enhanced = enhanced_dict.get(en_id)
    zh_board_multi = multi_dict.get(zh_id)
    en_board_multi = multi_dict.get(en_id)

    # 情况1：两个都在 multi_corpus（故宫那对）
    if zh_board_multi and en_board_multi:
        # 合并到 en_board，删除 zh_board
        zh_paras = zh_board_multi.get("paragraphs", [])
        en_paras = en_board_multi.get("paragraphs", [])

        for i, en_para in enumerate(en_paras):
            zh_para = zh_paras[i] if i < len(zh_paras) else None
            en_sents = en_para.get("sentences", [])
            zh_sents = zh_para.get("sentences", []) if zh_para else []

            for j, en_sent in enumerate(en_sents):
                zh_sent = zh_sents[j] if j < len(zh_sents) else None
                if zh_sent and zh_sent.get("zh"):
                    en_sent["zh"] = zh_sent["zh"]
                # en/翻译已经在 en_sent 里

            # 更新段落级 zh
            if zh_para and zh_para.get("zh"):
                en_para["zh"] = zh_para["zh"]

        # 标记删除 zh_board
        zh_board_multi["_delete"] = True
        merged_count += 1
        print(f"[Multi] Merged: {zh_id} + {en_id}")
        continue

    # 情况2：zh 在 multi，en 不在（需要从 enhanced 或 OCR 恢复）
    if zh_board_multi and not en_board_multi:
        # 从 enhanced 或 OCR 获取 en 文本
        if en_board_enhanced:
            en_paras = en_board_enhanced.get("paragraphs", [])
        else:
            # 从 OCR 创建
            en_text = get_ocr_text(en_id, "en")
            if en_text:
                en_paras = [{
                    "para_index": 0,
                    "en": en_text,
                    "sentences": [{"sent_index": 0, "en": en_text}]
                }]
            else:
                en_paras = []

        if not en_paras:
            print(f"[Skip] No EN text for {en_id}")
            continue

        zh_paras = zh_board_multi.get("paragraphs", [])

        # 合并：把 zh 填到 en 结构里
        for i, en_para in enumerate(en_paras):
            zh_para = zh_paras[i] if i < len(zh_paras) else None
            en_sents = en_para.get("sentences", [])
            zh_sents = zh_para.get("sentences", []) if zh_para else []

            for j, en_sent in enumerate(en_sents):
                zh_sent = zh_sents[j] if j < len(zh_sents) else None
                if zh_sent and zh_sent.get("zh"):
                    en_sent["zh"] = zh_sent["zh"]
                # 翻译字段暂时为空，需要后续补充

            if zh_para and zh_para.get("zh"):
                en_para["zh"] = zh_para["zh"]

        # 创建新的合并板
        src = ocr_dict.get(en_id, {}).get("source", {})
        new_board = {
            "board_id": en_id,
            "source": src,
            "board_title": {},
            "paragraphs": en_paras,
            "_merged_from": zh_id,
            "_needs_translation": True  # 标记需要补充翻译
        }
        multi["boards"].append(new_board)
        zh_board_multi["_delete"] = True
        merged_count += 1
        fixed_in_multi += 1
        print(f"[Recovered] Merged: {zh_id} + {en_id} (needs translation)")
        continue

    print(f"[Skip] {zh_id} + {en_id} - zh not in multi")

# 删除标记的板
multi["boards"] = [b for b in multi.get("boards", []) if not b.get("_delete")]

# 更新 metadata
multi["metadata"]["total_boards"] = len(multi["boards"])

# 保存
backup = MULTI_PATH.replace(".json", f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
if os.path.exists(MULTI_PATH):
    os.rename(MULTI_PATH, backup)
with open(MULTI_PATH, "w", encoding="utf-8") as f:
    json.dump(multi, f, ensure_ascii=False, indent=2)

print(f"\n=== Done ===")
print(f"Merged: {merged_count}")
print(f"Fixed from OCR/enhanced: {fixed_in_multi}")
print(f"Backup: {os.path.basename(backup)}")
print(f"\nNote: {fixed_in_multi} boards need translation (marked _needs_translation)")
print("Run multilingual_translate.py to fill translations")

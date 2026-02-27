#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工增强处理工具 - Gradio 界面
用于处理 LLM 增强失败的条目（27条）

功能：
  1. 浏览原始图片 + OCR 文本
  2. 人工修正中英文文本
  3. 用空行分割段落，自动切句对齐
  4. 保存结果并合并到 enhanced_corpus.json

用法:
  python manual_enhance.py              # 启动审核界面
  python manual_enhance.py --merge      # 将手动结果合并到增强语料库
  python manual_enhance.py --export     # 导出失败条目报告

操作指南：
  - 查看左侧图片，对照右侧 OCR 文本
  - 修正 OCR 错误（粘连单词、错字等）
  - 用一个空行分隔不同段落（中英文分别操作）
  - 中文段落数必须等于英文段落数（1:1 对齐）
  - 点击"保存并下一条"自动切句、保存结果
  - 无法修复的条目点击"跳过（排除）"
"""

import json
import re
import sys
import io
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 编码
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ==================== 数据加载 ====================

def load_failed_entries() -> List[Dict]:
    """加载失败条目，返回含原始文本和图片路径的列表"""
    # 1. 读取 progress.json 获取 failed 列表
    if not cfg.ENHANCED_PROGRESS_FILE.exists():
        print("[ERROR] progress.json 不存在，请先运行 llm_enhance.py")
        return []

    with open(cfg.ENHANCED_PROGRESS_FILE, 'r', encoding='utf-8') as f:
        progress = json.load(f)

    failed_ids = set(progress.get("failed", []))
    if not failed_ids:
        print("[INFO] 没有失败条目")
        return []

    # 2. 读取 OCR 结果
    ocr_map = {}
    if cfg.OCR_RESULTS_FILE.exists():
        with open(cfg.OCR_RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data.get("results", []):
                image_id = entry.get("image_id", "")
                if image_id in failed_ids:
                    ocr_map[image_id] = entry

    # 3. 读取审核结果（优先级更高）
    review_map = {}
    if cfg.REVIEWED_RESULTS_FILE.exists():
        with open(cfg.REVIEWED_RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for r in data.get("results", []):
                rid = r.get("image_id", "")
                if rid in failed_ids:
                    review_map[rid] = r

    # 4. 读取已有手动处理结果
    manual_file = cfg.ENHANCED_DIR / "manual_results.json"
    manual_done = {}
    if manual_file.exists():
        with open(manual_file, 'r', encoding='utf-8') as f:
            manual_data = json.load(f)
            for entry in manual_data.get("entries", []):
                manual_done[entry["board_id"]] = entry

    # 5. 组装条目列表
    entries = []
    for image_id in sorted(failed_ids):
        ocr = ocr_map.get(image_id, {})
        review = review_map.get(image_id, {})

        # 获取文本（审核修正 > 原始 OCR）
        zh = ocr.get("zh_text", "")
        en = ocr.get("en_text", "")

        if review.get("review_status") == "corrected":
            czh = review.get("corrected_zh", "")
            cen = review.get("corrected_en", "")
            if czh and czh.strip() != "[删除]":
                zh = czh
            elif czh and czh.strip() == "[删除]":
                zh = ""
            if cen and cen.strip() != "[删除]":
                en = cen
            elif cen and cen.strip() == "[删除]":
                en = ""

        # 图片路径
        source = ocr.get("source", {})
        museum = source.get("museum", image_id.split("_")[0])
        image_name = source.get("image_name", "")
        if not image_name:
            # 从 image_id 推断
            parts = image_id.split("_", 1)
            if len(parts) == 2:
                image_name = parts[1] + ".jpg"
        image_path = cfg.RAW_IMAGE_DIR / museum / image_name

        entry = {
            "image_id": image_id,
            "museum": museum,
            "image_name": image_name,
            "image_path": str(image_path),
            "image_exists": image_path.exists(),
            "zh_text": zh,
            "en_text": en,
            "zh_len": len(zh),
            "en_len": len(en),
            "quality_grade": ocr.get("quality", {}).get("grade", "?"),
            "ocr_confidence": ocr.get("quality", {}).get("confidence", 0),
            "already_done": image_id in manual_done,
            "manual_result": manual_done.get(image_id),
        }
        entries.append(entry)

    return entries


# ==================== 文本处理 ====================

def split_paragraphs(text: str) -> List[str]:
    """按空行分段"""
    if not text or not text.strip():
        return []
    # 用连续换行分割
    paras = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paras if p.strip()]


def split_sentences_zh(text: str) -> List[str]:
    """中文分句"""
    if not text.strip():
        return []
    # 按 。！？；分句，保留标点
    parts = re.split(r'(。|！|？|；)', text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if part in ('。', '！', '？', '；'):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def split_sentences_en(text: str) -> List[str]:
    """英文分句"""
    if not text.strip():
        return []
    # 先处理常见缩写避免误分
    text_clean = text
    for abbr in ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'etc.', 'vs.', 'i.e.', 'e.g.', 'No.', 'St.', 'Jr.', 'Sr.', 'Ltd.', 'Corp.', 'Inc.', 'U.S.', 'U.K.', 'B.C.', 'A.D.']:
        text_clean = text_clean.replace(abbr, abbr.replace('.', '<<DOT>>'))

    # 按 . ! ? 分句
    parts = re.split(r'([.!?])\s+', text_clean)
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if part in ('.', '!', '?'):
            restored = current.strip().replace('<<DOT>>', '.')
            if restored:
                sentences.append(restored)
            current = ""
    if current.strip():
        restored = current.strip().replace('<<DOT>>', '.')
        sentences.append(restored)
    return sentences


def align_sentences(zh_sents: List[str], en_sents: List[str]) -> List[Dict]:
    """对齐中英句子（尽量 1:1，数量不等时合并末尾）"""
    if not zh_sents and not en_sents:
        return []

    pairs = []
    n_zh = len(zh_sents)
    n_en = len(en_sents)

    if n_zh == n_en:
        # 完美对齐
        for z, e in zip(zh_sents, en_sents):
            pairs.append({"zh": z, "en": e})
    elif n_zh > n_en:
        # 中文多，英文少 → 末尾中文合并
        for i in range(n_en - 1):
            pairs.append({"zh": zh_sents[i], "en": en_sents[i]})
        # 剩余中文合并到最后一个英文
        remaining_zh = "".join(zh_sents[n_en - 1:])
        pairs.append({"zh": remaining_zh, "en": en_sents[-1]})
    else:
        # 英文多，中文少 → 末尾英文合并
        for i in range(n_zh - 1):
            pairs.append({"zh": zh_sents[i], "en": en_sents[i]})
        remaining_en = " ".join(en_sents[n_zh - 1:])
        pairs.append({"zh": zh_sents[-1], "en": remaining_en})

    return pairs


def build_board_from_text(image_id: str, source: Dict,
                          zh_text: str, en_text: str,
                          title_zh: str = "", title_en: str = "") -> Dict:
    """从人工修正的文本构建 board 结构"""
    zh_paras = split_paragraphs(zh_text)
    en_paras = split_paragraphs(en_text)

    # 段落数对齐：如果不一致，尝试合并末尾
    if len(zh_paras) != len(en_paras):
        n_min = min(len(zh_paras), len(en_paras))
        if n_min == 0:
            # 一边完全没有段落
            if not zh_paras:
                zh_paras = [""] * len(en_paras)
            else:
                en_paras = [""] * len(zh_paras)
        else:
            if len(zh_paras) > len(en_paras):
                # 合并多余中文段落到最后
                merged = "\n".join(zh_paras[n_min - 1:])
                zh_paras = zh_paras[:n_min - 1] + [merged]
            else:
                merged = " ".join(en_paras[n_min - 1:])
                en_paras = en_paras[:n_min - 1] + [merged]

    paragraphs = []
    for i, (zp, ep) in enumerate(zip(zh_paras, en_paras)):
        zh_sents = split_sentences_zh(zp)
        en_sents = split_sentences_en(ep)
        sent_pairs = align_sentences(zh_sents, en_sents)

        para = {
            "para_index": i,
            "zh": zp,
            "en": ep,
            "sentences": [
                {"sent_index": j, "zh": s["zh"], "en": s["en"]}
                for j, s in enumerate(sent_pairs)
            ],
        }
        paragraphs.append(para)

    return {
        "board_id": image_id,
        "source": source,
        "board_title": {"zh": title_zh, "en": title_en},
        "corrections": {"zh_changes": [], "en_changes": [], "note": "人工修正"},
        "paragraphs": paragraphs,
        "manual_processed": True,
        "processed_at": datetime.now().isoformat(),
    }


# ==================== 保存 ====================

def save_manual_result(board: Dict):
    """保存一条手动处理结果"""
    manual_file = cfg.ENHANCED_DIR / "manual_results.json"
    cfg.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    data = {"metadata": {"updated_at": "", "total": 0}, "entries": []}
    if manual_file.exists():
        with open(manual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # 替换或追加
    entries = data.get("entries", [])
    found = False
    for i, e in enumerate(entries):
        if e.get("board_id") == board["board_id"]:
            entries[i] = board
            found = True
            break
    if not found:
        entries.append(board)

    data["entries"] = entries
    data["metadata"]["updated_at"] = datetime.now().isoformat()
    data["metadata"]["total"] = len(entries)

    with open(manual_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(entries)


def save_skip(image_id: str):
    """标记跳过"""
    manual_file = cfg.ENHANCED_DIR / "manual_results.json"
    cfg.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    data = {"metadata": {"updated_at": "", "total": 0}, "entries": [], "skipped": []}
    if manual_file.exists():
        with open(manual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    if "skipped" not in data:
        data["skipped"] = []
    if image_id not in data["skipped"]:
        data["skipped"].append(image_id)

    data["metadata"]["updated_at"] = datetime.now().isoformat()

    with open(manual_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== 合并到增强语料库 ====================

def merge_to_corpus():
    """将手动结果合并到 enhanced_corpus.json"""
    manual_file = cfg.ENHANCED_DIR / "manual_results.json"
    if not manual_file.exists():
        print("[ERROR] manual_results.json 不存在")
        return

    with open(manual_file, 'r', encoding='utf-8') as f:
        manual_data = json.load(f)

    manual_entries = {e["board_id"]: e for e in manual_data.get("entries", [])}
    manual_skipped = set(manual_data.get("skipped", []))

    if not manual_entries and not manual_skipped:
        print("[INFO] 没有手动处理结果")
        return

    # 读取增强语料库
    if not cfg.ENHANCED_CORPUS_FILE.exists():
        print("[ERROR] enhanced_corpus.json 不存在")
        return

    with open(cfg.ENHANCED_CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    existing_ids = {b["board_id"] for b in corpus.get("boards", [])}

    # 添加手动结果
    added = 0
    for board_id, board in manual_entries.items():
        if board_id not in existing_ids:
            corpus["boards"].append(board)
            added += 1
            existing_ids.add(board_id)

    # 更新统计
    total_paras = sum(len(b.get("paragraphs", [])) for b in corpus["boards"])
    total_sents = sum(
        len(p.get("sentences", []))
        for b in corpus["boards"]
        for p in b.get("paragraphs", [])
    )

    corpus["metadata"]["total_boards"] = len(corpus["boards"])
    corpus["metadata"]["total_paragraphs"] = total_paras
    corpus["metadata"]["total_sentence_pairs"] = total_sents
    corpus["metadata"]["manual_processed"] = len(manual_entries)
    corpus["metadata"]["manual_skipped"] = len(manual_skipped)
    corpus["metadata"]["merged_at"] = datetime.now().isoformat()

    # 更新 progress.json 中的 failed 列表（移除已处理的）
    if cfg.ENHANCED_PROGRESS_FILE.exists():
        with open(cfg.ENHANCED_PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        old_failed = set(progress.get("failed", []))
        processed = set(manual_entries.keys()) | manual_skipped
        new_failed = [fid for fid in old_failed if fid not in processed]
        progress["failed"] = new_failed
        progress["metadata"]["last_updated"] = datetime.now().isoformat()

        # 把手动处理的加入 completed
        for board_id, board in manual_entries.items():
            if board_id not in progress.get("completed", {}):
                progress["completed"][board_id] = {
                    "processed_at": board.get("processed_at", datetime.now().isoformat()),
                    "input_source": "manual",
                    "result": board,
                }
        progress["metadata"]["total_completed"] = len(progress["completed"])

        # 把跳过的加入 skipped
        for sid in manual_skipped:
            if sid not in progress.get("skipped", []):
                progress["skipped"].append(sid)

        with open(cfg.ENHANCED_PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        print(f"[OK] progress.json 已更新: failed {len(old_failed)} → {len(new_failed)}")

    # 保存
    with open(cfg.ENHANCED_CORPUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已合并 {added} 条手动结果到 enhanced_corpus.json")
    print(f"     跳过 {len(manual_skipped)} 条")
    print(f"     语料库: {corpus['metadata']['total_boards']} 展板, "
          f"{total_paras} 段落, {total_sents} 句对")


# ==================== 导出报告 ====================

def export_report():
    """导出失败条目报告（方便离线查看）"""
    entries = load_failed_entries()
    if not entries:
        return

    report_file = cfg.ENHANCED_DIR / "failed_entries_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"LLM 增强失败条目报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"共 {len(entries)} 条\n")
        f.write("=" * 80 + "\n\n")

        for i, e in enumerate(entries, 1):
            status = "✅ 已处理" if e["already_done"] else "❌ 待处理"
            f.write(f"{'─' * 80}\n")
            f.write(f"[{i:02d}] {e['image_id']}  {status}\n")
            f.write(f"     博物馆: {e['museum']}  |  图片: {e['image_name']}\n")
            f.write(f"     质量: {e['quality_grade']}级  |  置信度: {e['ocr_confidence']:.2f}\n")
            f.write(f"     中文: {e['zh_len']}字  |  英文: {e['en_len']}字\n")
            f.write(f"     图片: {'存在' if e['image_exists'] else '缺失'} - {e['image_path']}\n")
            f.write(f"\n  【中文 OCR】\n")
            f.write(f"  {e['zh_text'][:500]}\n")
            if e['zh_len'] > 500:
                f.write(f"  ... (省略 {e['zh_len'] - 500} 字)\n")
            f.write(f"\n  【英文 OCR】\n")
            f.write(f"  {e['en_text'][:500]}\n")
            if e['en_len'] > 500:
                f.write(f"  ... (省略 {e['en_len'] - 500} 字)\n")
            f.write("\n")

    print(f"[OK] 报告已导出: {report_file}")


# ==================== Gradio 界面 ====================

def build_gradio_app():
    """构建 Gradio 界面"""
    import gradio as gr

    entries = load_failed_entries()
    if not entries:
        print("[ERROR] 没有需要处理的条目")
        return None

    print(f"[OK] 加载 {len(entries)} 条失败条目")

    # 统计
    done_count = sum(1 for e in entries if e["already_done"])
    todo_count = len(entries) - done_count

    current_idx = [0]  # 用列表使其在闭包中可变

    def get_entry_info(idx):
        """获取当前条目信息"""
        if idx < 0 or idx >= len(entries):
            return None
        return entries[idx]

    def load_entry(idx):
        """加载第 idx 条的所有界面元素"""
        if idx < 0:
            idx = 0
        if idx >= len(entries):
            idx = len(entries) - 1
        current_idx[0] = idx

        e = entries[idx]
        status = "✅ 已处理" if e["already_done"] else "⏳ 待处理"
        header = (f"**[{idx+1}/{len(entries)}]** `{e['image_id']}`  {status}\n\n"
                  f"博物馆: {e['museum']}  |  质量: {e['quality_grade']}级  |  "
                  f"置信度: {e['ocr_confidence']:.2f}  |  "
                  f"中文 {e['zh_len']}字 / 英文 {e['en_len']}字")

        img = e["image_path"] if e["image_exists"] else None

        # 如果已有手动结果，加载它
        zh_text = e["zh_text"]
        en_text = e["en_text"]
        title_zh = ""
        title_en = ""

        if e["manual_result"]:
            mr = e["manual_result"]
            title_zh = mr.get("board_title", {}).get("zh", "")
            title_en = mr.get("board_title", {}).get("en", "")
            # 从 paragraphs 重建分段文本
            zh_paras = [p.get("zh", "") for p in mr.get("paragraphs", [])]
            en_paras = [p.get("en", "") for p in mr.get("paragraphs", [])]
            if zh_paras:
                zh_text = "\n\n".join(zh_paras)
            if en_paras:
                en_text = "\n\n".join(en_paras)

        return (
            header,   # info_md
            img,      # image
            zh_text,  # zh_textbox
            en_text,  # en_textbox
            title_zh, # title_zh
            title_en, # title_en
            idx,      # slider
        )

    def on_save(zh_text, en_text, title_zh, title_en):
        """保存当前条目"""
        idx = current_idx[0]
        e = entries[idx]

        zh_paras = split_paragraphs(zh_text)
        en_paras = split_paragraphs(en_text)

        # 验证
        if not zh_text.strip() and not en_text.strip():
            return '⚠️ 中英文都为空，请填写至少一种语言的文本，或点击「跳过」'

        if zh_paras and en_paras and len(zh_paras) != len(en_paras):
            return (f"⚠️ 段落数不匹配：中文 {len(zh_paras)} 段 vs 英文 {len(en_paras)} 段\n"
                    f"请用空行分隔段落，确保中英文段落数一致。\n"
                    f"（如果无法对齐，系统会自动合并末尾段落）")

        # 构建 board
        source = {
            "museum": e["museum"],
            "image_name": e["image_name"],
            "image_path": e["image_path"],
        }
        board = build_board_from_text(
            e["image_id"], source,
            zh_text, en_text,
            title_zh.strip(), title_en.strip()
        )

        n_para = len(board["paragraphs"])
        n_sent = sum(len(p["sentences"]) for p in board["paragraphs"])

        count = save_manual_result(board)
        entries[idx]["already_done"] = True
        entries[idx]["manual_result"] = board

        return (f"✅ 已保存 `{e['image_id']}`\n"
                f"   {n_para} 段落, {n_sent} 句对\n"
                f"   总计已处理: {count} 条")

    def on_skip():
        """跳过当前条目"""
        idx = current_idx[0]
        e = entries[idx]
        save_skip(e["image_id"])
        entries[idx]["already_done"] = True
        return f"⏭️ 已跳过 `{e['image_id']}`"

    def on_preview(zh_text, en_text, title_zh, title_en):
        """预览分段分句结果"""
        zh_paras = split_paragraphs(zh_text)
        en_paras = split_paragraphs(en_text)

        lines = []
        lines.append(f"**标题**: zh=「{title_zh.strip()}」 en=「{title_en.strip()}」\n")
        lines.append(f"**段落数**: 中文 {len(zh_paras)} 段, 英文 {len(en_paras)} 段\n")

        if zh_paras and en_paras and len(zh_paras) != len(en_paras):
            lines.append(f"⚠️ **段落数不一致！** 保存时会自动合并末尾段落\n")

        # 模拟对齐
        n = max(len(zh_paras), len(en_paras))
        for i in range(n):
            lines.append(f"---\n### 段落 {i+1}")
            zp = zh_paras[i] if i < len(zh_paras) else "(无)"
            ep = en_paras[i] if i < len(en_paras) else "(无)"

            zh_sents = split_sentences_zh(zp) if zp != "(无)" else []
            en_sents = split_sentences_en(ep) if ep != "(无)" else []
            pairs = align_sentences(zh_sents, en_sents) if zh_sents or en_sents else []

            lines.append(f"句对数: {len(pairs)}\n")
            for j, pair in enumerate(pairs):
                lines.append(f"**S{j+1}** ZH: {pair['zh']}")
                lines.append(f"**S{j+1}** EN: {pair['en']}\n")

        return "\n".join(lines)

    def go_next():
        idx = min(current_idx[0] + 1, len(entries) - 1)
        return load_entry(idx)

    def go_prev():
        idx = max(current_idx[0] - 1, 0)
        return load_entry(idx)

    def go_to(idx):
        return load_entry(int(idx))

    # ---------- 构建界面 ----------
    with gr.Blocks(
        title="人工增强处理",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            f"# 📝 人工增强处理工具\n"
            f"共 **{len(entries)}** 条失败条目 | "
            f"已处理 **{done_count}** | 待处理 **{todo_count}**\n\n"
            f"操作：修正文本 → 用**空行**分段 → 预览 → 保存"
        )

        with gr.Row():
            prev_btn = gr.Button("⬅️ 上一条", scale=1)
            slider = gr.Slider(
                minimum=0, maximum=len(entries) - 1, step=1, value=0,
                label="条目索引", scale=4,
            )
            next_btn = gr.Button("➡️ 下一条", scale=1)

        info_md = gr.Markdown("")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image = gr.Image(label="原始图片", type="filepath", height=500)

            with gr.Column(scale=1):
                with gr.Row():
                    title_zh_box = gr.Textbox(label="标题(中)", placeholder="可留空", scale=1)
                    title_en_box = gr.Textbox(label="标题(英)", placeholder="可留空", scale=1)
                zh_box = gr.Textbox(
                    label="中文文本（用空行分段）",
                    lines=12, max_lines=30,
                    placeholder="修正 OCR 错误后粘贴中文文本...\n\n用空行分隔不同段落",
                )
                en_box = gr.Textbox(
                    label="英文文本（用空行分段）",
                    lines=12, max_lines=30,
                    placeholder="修正 OCR 错误后粘贴英文文本...\n\n用空行分隔不同段落",
                )

        with gr.Row():
            save_btn = gr.Button("💾 保存并处理", variant="primary", scale=2)
            preview_btn = gr.Button("👁️ 预览分段分句", scale=1)
            skip_btn = gr.Button("⏭️ 跳过（排除此条）", variant="stop", scale=1)

        status_md = gr.Markdown("")
        preview_md = gr.Markdown("")

        # 输出组件列表
        load_outputs = [info_md, image, zh_box, en_box, title_zh_box, title_en_box, slider]

        # 事件绑定
        slider.change(go_to, inputs=[slider], outputs=load_outputs)
        prev_btn.click(go_prev, outputs=load_outputs)
        next_btn.click(go_next, outputs=load_outputs)

        save_btn.click(
            on_save,
            inputs=[zh_box, en_box, title_zh_box, title_en_box],
            outputs=[status_md],
        )
        skip_btn.click(on_skip, outputs=[status_md])
        preview_btn.click(
            on_preview,
            inputs=[zh_box, en_box, title_zh_box, title_en_box],
            outputs=[preview_md],
        )

        # 初始加载
        app.load(lambda: load_entry(0), outputs=load_outputs)

    return app


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="人工增强处理工具")
    parser.add_argument("--merge", action="store_true",
                        help="将手动结果合并到 enhanced_corpus.json")
    parser.add_argument("--export", action="store_true",
                        help="导出失败条目报告")
    parser.add_argument("--port", type=int, default=7866,
                        help="Gradio 服务端口（默认7866）")
    args = parser.parse_args()

    if args.merge:
        merge_to_corpus()
        return

    if args.export:
        export_report()
        return

    app = build_gradio_app()
    if app:
        print(f"\n启动人工增强界面: http://127.0.0.1:{args.port}")
        print("操作完成后运行: python manual_enhance.py --merge")
        app.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            show_error=True,
        )


if __name__ == "__main__":
    main()

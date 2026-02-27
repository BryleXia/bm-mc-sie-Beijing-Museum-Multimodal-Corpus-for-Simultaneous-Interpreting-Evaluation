#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建平行语料库
整合：
1. 双语图片（图片级对齐）
2. 人工配对的单语图片
生成最终的平行语料库格式

支持两种输入格式：
  - ocr_processor.py 格式: zh_text, en_text, quality.grade
  - ocr_processor_rapid.py 格式: image_content.zh_all, classification.image_type

修复记录 (2026-02):
- 使用集中配置模块
- 自动检测并适配两种输入数据格式
- 添加输入验证
"""

import json
import sys
import io
from pathlib import Path
from collections import defaultdict

if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# ==================== 配置 ====================
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as cfg
    # 默认读 ocr输出 目录（rapid格式），如果不存在则读中间结果（paragraph格式）
    _ocr_output_file = cfg.OCR_OUTPUT_DIR / "ocr_results.json"
    _intermediate_file = cfg.OCR_RESULTS_FILE
    if _ocr_output_file.exists():
        INPUT_FILE = str(_ocr_output_file)
    else:
        INPUT_FILE = str(_intermediate_file)
    OUTPUT_DIR = str(cfg.OCR_OUTPUT_DIR)
except ImportError:
    _project_root = Path(__file__).resolve().parent.parent
    INPUT_FILE = str(_project_root / "ocr输出" / "ocr_results.json")
    OUTPUT_DIR = str(_project_root / "ocr输出")

# 高置信度人工配对
MANUAL_PAIRS = [
    {
        "pair_id": "manual_001",
        "zh_entry_id": "国家博物馆_IMG20260121114840",
        "en_entry_id": "国家博物馆_IMG20260121114842",
        "topic": "宋代其他名窑",
        "confidence": "high"
    },
    {
        "pair_id": "manual_002",
        "zh_entry_id": "国家博物馆_IMG20260121121509",
        "en_entry_id": "国家博物馆_IMG20260121121512",
        "topic": "农民画前言",
        "confidence": "high"
    },
    {
        "pair_id": "manual_003",
        "zh_entry_id": "首都博物馆_IMG20260120153150",
        "en_entry_id": "首都博物馆_IMG20260120153157",
        "topic": "人类文明交流结论",
        "confidence": "high"
    }
]


def detect_data_format(entries):
    """
    自动检测数据格式
    返回: "rapid" (ocr_processor_rapid) 或 "paragraph" (ocr_processor)
    """
    if not entries:
        return "unknown"

    sample = entries[0]
    if 'image_content' in sample and 'classification' in sample:
        return "rapid"
    elif 'zh_text' in sample:
        return "paragraph"
    else:
        return "unknown"


def get_entry_id(entry, fmt):
    """获取条目ID"""
    if fmt == "rapid":
        return entry.get('entry_id', '')
    else:
        return entry.get('image_id', '')


def get_zh_text(entry, fmt):
    """获取中文文本"""
    if fmt == "rapid":
        return entry.get('image_content', {}).get('zh_all', '')
    else:
        return entry.get('zh_text', '')


def get_en_text(entry, fmt):
    """获取英文文本"""
    if fmt == "rapid":
        return entry.get('image_content', {}).get('en_all', '')
    else:
        return entry.get('en_text', '')


def is_bilingual(entry, fmt):
    """判断是否为双语图片"""
    if fmt == "rapid":
        return entry.get('classification', {}).get('image_type') == 'bilingual'
    else:
        # paragraph 格式：同时有中文和英文文本
        return bool(entry.get('zh_text')) and bool(entry.get('en_text'))


def get_quality_score(entry, fmt):
    """获取质量分数"""
    if fmt == "rapid":
        return entry.get('quality', {}).get('avg_confidence', 0)
    else:
        return entry.get('quality', {}).get('avg_ocr_confidence', 0)


def get_block_count(entry, fmt):
    """获取文本块数量"""
    if fmt == "rapid":
        return entry.get('classification', {}).get('text_block_count', 0)
    else:
        return entry.get('quality', {}).get('total_blocks', 0)


def create_parallel_entry(entry, fmt):
    """从单条OCR记录创建平行语料条目（兼容两种格式）"""
    entry_id = get_entry_id(entry, fmt)
    zh = get_zh_text(entry, fmt)
    en = get_en_text(entry, fmt)

    return {
        "parallel_id": entry_id,
        "source": {
            "museum": entry.get('source', {}).get('museum', ''),
            "image_name": entry.get('source', {}).get('image_name', ''),
            "image_path": entry.get('source', {}).get('image_path', '')
        },
        "parallel_texts": {
            "zh": zh,
            "en": en
        },
        "metadata": {
            "alignment_type": "image_level",
            "quality_score": get_quality_score(entry, fmt),
            "text_block_count": get_block_count(entry, fmt)
        },
        # 预留翻译字段
        "translations": {
            "targets": {},  # { "ja": "...", "fr": "...", "de": "..." }
            "status": "pending"
        }
    }


def create_manual_pair_entry(pair_info, entries_dict, fmt):
    """创建人工配对的平行语料条目"""
    zh_entry = entries_dict.get(pair_info['zh_entry_id'])
    en_entry = entries_dict.get(pair_info['en_entry_id'])

    if not zh_entry or not en_entry:
        return None

    return {
        "parallel_id": pair_info['pair_id'],
        "source": {
            "museum": zh_entry.get('source', {}).get('museum', ''),
            "zh_image": zh_entry.get('source', {}).get('image_name', ''),
            "en_image": en_entry.get('source', {}).get('image_name', ''),
            "pairing_method": "manual",
            "pairing_confidence": pair_info['confidence'],
            "topic": pair_info['topic']
        },
        "parallel_texts": {
            "zh": get_zh_text(zh_entry, fmt),
            "en": get_en_text(en_entry, fmt)
        },
        "metadata": {
            "alignment_type": "manual_paired",
            "zh_quality": get_quality_score(zh_entry, fmt),
            "en_quality": get_quality_score(en_entry, fmt),
            "notes": "人工配对：文件名接近且内容匹配"
        },
        "translations": {
            "targets": {},
            "status": "pending"
        }
    }


def main():
    print("=" * 70)
    print("创建多语种平行语料库")
    print("=" * 70)

    # 读取OCR结果
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {INPUT_FILE}")
        return

    print(f"读取: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容两种顶层结构: "entries" (rapid) 或 "results" (paragraph)
    entries = data.get('entries', data.get('results', []))
    if not entries:
        print("[ERROR] 数据为空")
        return

    # 自动检测格式
    fmt = detect_data_format(entries)
    print(f"[OK] 检测到数据格式: {fmt} ({len(entries)} 条记录)")

    if fmt == "unknown":
        print("[ERROR] 无法识别数据格式，请检查输入文件")
        return

    # 构建ID索引
    entries_dict = {}
    for e in entries:
        if 'error' not in e:
            eid = get_entry_id(e, fmt)
            entries_dict[eid] = e

    parallel_corpus = []

    # 1. 处理双语图片
    print("\n[Step 1] 处理双语图片...")
    bilingual_count = 0

    for entry in entries:
        if 'error' in entry:
            continue

        if is_bilingual(entry, fmt):
            parallel_entry = create_parallel_entry(entry, fmt)
            parallel_corpus.append(parallel_entry)
            bilingual_count += 1

    print(f"  -> 添加 {bilingual_count} 条双语图片记录")

    # 2. 处理人工配对
    print("\n[Step 2] 处理人工配对...")
    manual_count = 0

    for pair in MANUAL_PAIRS:
        entry = create_manual_pair_entry(pair, entries_dict, fmt)
        if entry:
            parallel_corpus.append(entry)
            manual_count += 1
            print(f"  [OK] 配对: {pair['topic']} ({pair['pair_id']})")
        else:
            # 可能ID不存在于当前数据中，不是错误
            print(f"  [SKIP] 配对未找到: {pair['pair_id']} (可能不在当前数据集中)")

    # 3. 统计信息
    print("\n[Step 3] 生成统计...")

    stats = {
        "total_parallel_entries": len(parallel_corpus),
        "bilingual_image_entries": bilingual_count,
        "manual_paired_entries": manual_count,
        "by_museum": dict(defaultdict(int))
    }

    museum_counts = defaultdict(int)
    for entry in parallel_corpus:
        museum_counts[entry['source']['museum']] += 1
    stats['by_museum'] = dict(museum_counts)

    # 4. 保存平行语料库
    print("\n[Step 4] 保存结果...")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 主文件
    corpus_file = output_dir / "parallel_corpus.json"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "name": "中国博物馆多语种解说词平行语料库",
                "version": "1.0",
                "created_at": data.get('metadata', {}).get('created_at', ''),
                "input_format": fmt,
                "alignment_type": "image_level + manual_paired",
                "source_languages": ["zh", "en"],
                "target_languages": [],
                "statistics": stats
            },
            "entries": parallel_corpus
        }, f, ensure_ascii=False, indent=2)

    print(f"  [OK] 平行语料库: {corpus_file}")

    # 5. 生成纯文本对照版
    text_file = output_dir / "parallel_texts.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("中国博物馆多语种解说词平行语料库 - 文本对照版\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总条目数: {len(parallel_corpus)}\n")
        f.write(f"  - 双语图片: {bilingual_count}\n")
        f.write(f"  - 人工配对: {manual_count}\n\n")

        for i, entry in enumerate(parallel_corpus, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"条目 [{i}/{len(parallel_corpus)}] ID: {entry['parallel_id']}\n")
            f.write(f"来源: {entry['source']['museum']}\n")

            if entry['metadata']['alignment_type'] == 'manual_paired':
                f.write(f"配对方式: 人工配对 ({entry['source'].get('topic', '')})\n")
            else:
                f.write(f"配对方式: 图片级对齐\n")

            f.write(f"{'='*80}\n\n")

            zh_text = entry['parallel_texts']['zh']
            if zh_text:
                f.write("【中文】\n")
                f.write(zh_text)
                f.write("\n\n")

            en_text = entry['parallel_texts']['en']
            if en_text:
                f.write("【English】\n")
                f.write(en_text)
                f.write("\n\n")

    print(f"  [OK] 文本对照版: {text_file}")

    # 6. 按博物馆分文件
    for museum in stats['by_museum']:
        museum_entries = [e for e in parallel_corpus if e['source']['museum'] == museum]
        museum_file = output_dir / f"{museum}_parallel.json"

        with open(museum_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "museum": museum,
                    "entry_count": len(museum_entries)
                },
                "entries": museum_entries
            }, f, ensure_ascii=False, indent=2)

        print(f"  [OK] {museum}: {len(museum_entries)} 条")

    # 打印统计
    print("\n" + "=" * 70)
    print("平行语料库创建完成！")
    print("=" * 70)
    print(f"\n总条目数: {len(parallel_corpus)}")
    print(f"  - 双语图片对齐: {bilingual_count}")
    print(f"  - 人工配对对齐: {manual_count}")
    print(f"\n按博物馆分布:")
    for museum, count in sorted(stats['by_museum'].items()):
        print(f"  - {museum}: {count} 条")

    print("\n输出文件:")
    print(f"  - parallel_corpus.json (主文件)")
    print(f"  - parallel_texts.txt (纯文本对照)")
    for museum in stats['by_museum']:
        print(f"  - {museum}_parallel.json")
    print("\n下一步: 翻译 (zh/en -> ja/fr/de/...)")

if __name__ == "__main__":
    main()

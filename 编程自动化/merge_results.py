#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并两次OCR处理的结果
第一次：党史馆 + 抗日纪念馆 (137张)
第二次：国家博物馆 + 故宫 + 首都博物馆 (446张)
总计：583张

修复记录 (2026-02):
- 使用集中配置模块
- 修复备份文件名查找逻辑
- 合并前不再覆盖原始文件（改为保存到 _all 文件）
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
try:
    import config as cfg
    INPUT_DIR = cfg.INTERMEDIATE_DIR
except ImportError:
    INPUT_DIR = Path(__file__).resolve().parent.parent / "中间结果"

# 可能的第一批结果文件名（按优先级查找）
FIRST_BATCH_CANDIDATES = [
    "ocr_results_two_museums.json.backup",
    "ocr_results_two_museums.json",
    "ocr_results_batch1.json",
]

SECOND_BATCH_FILE = "ocr_results.json"


def find_first_batch_file(input_dir: Path) -> Path:
    """查找第一批结果文件"""
    for name in FIRST_BATCH_CANDIDATES:
        f = input_dir / name
        if f.exists():
            print(f"[OK] 找到第一批结果: {f.name}")
            return f
    raise FileNotFoundError(
        f"找不到第一批OCR结果文件，尝试过: {', '.join(FIRST_BATCH_CANDIDATES)}"
    )


def main():
    print("=" * 60)
    print("合并两次OCR处理结果")
    print("=" * 60)

    # 读取第一批结果
    first_file = find_first_batch_file(INPUT_DIR)
    print(f"\nLoading first batch: {first_file.name}")
    with open(first_file, 'r', encoding='utf-8') as f:
        first_batch = json.load(f)
        first_results = first_batch.get('results', [])

    # 读取第二批结果
    second_file = INPUT_DIR / SECOND_BATCH_FILE
    if not second_file.exists():
        print(f"[ERROR] 第二批结果文件不存在: {second_file}")
        return

    print(f"Loading second batch: {second_file.name}")
    with open(second_file, 'r', encoding='utf-8') as f:
        second_batch = json.load(f)
        second_results = second_batch.get('results', [])

    # 检查重复 image_id
    first_ids = {r.get('image_id') for r in first_results}
    second_ids = {r.get('image_id') for r in second_results}
    duplicates = first_ids & second_ids
    if duplicates:
        print(f"[WARNING] 发现 {len(duplicates)} 个重复 image_id，将保留第二批的版本:")
        for d in list(duplicates)[:5]:
            print(f"  - {d}")
        # 去重：移除第一批中的重复项
        first_results = [r for r in first_results if r.get('image_id') not in duplicates]

    # 合并结果
    all_results = first_results + second_results

    print(f"\nFirst batch:  {len(first_results)} images")
    print(f"Second batch: {len(second_results)} images")
    print(f"Total merged: {len(all_results)} images")

    # 统计
    stats = {
        "total_images": len(all_results),
        "processed": len(all_results),
        "failed": 0,
        "needs_review": 0,
        "by_museum": {},
        "by_grade": {"A": 0, "B": 0, "C": 0}
    }

    for r in all_results:
        if r.get('needs_review'):
            stats["needs_review"] += 1
        museum = r.get('source', {}).get('museum', 'unknown')
        stats["by_museum"][museum] = stats["by_museum"].get(museum, 0) + 1
        grade = r.get('quality', {}).get('grade', 'C')
        stats["by_grade"][grade] = stats["by_grade"].get(grade, 0) + 1

    print(f"\n质量分级统计:")
    print(f"  A级: {stats['by_grade']['A']} 张")
    print(f"  B级: {stats['by_grade']['B']} 张")
    print(f"  C级: {stats['by_grade']['C']} 张")
    print(f"\n按博物馆分布:")
    for museum, count in sorted(stats["by_museum"].items()):
        print(f"  {museum}: {count} 张")

    # 保存合并结果
    print("\nSaving merged results...")

    # 1. 保存到 _all 文件（不覆盖原始文件）
    merged_file = INPUT_DIR / "ocr_results_all.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_images": stats["total_images"],
                "processed": stats["processed"],
                "failed": stats["failed"],
                "needs_review": stats["needs_review"],
                "processor": "RapidOCR",
                "alignment_method": "paragraph_level",
                "batches": ["党史馆+抗日纪念馆", "国家博物馆+故宫+首都博物馆"]
            },
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] Merged results: {merged_file} ({len(all_results)} items)")

    # 2. 按分级保存
    for grade in ["A", "B", "C"]:
        grade_results = [r for r in all_results if r.get('quality', {}).get('grade') == grade]
        grade_file = INPUT_DIR / f"grade_{grade}_all.json"
        with open(grade_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "grade": grade,
                    "count": len(grade_results),
                    "description": {
                        "A": "High quality: OCR confidence > 0.9",
                        "B": "Medium quality: OCR confidence 0.8-0.9",
                        "C": "Low quality: OCR confidence < 0.8"
                    }[grade]
                },
                "results": grade_results
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Grade {grade}: {grade_file} ({len(grade_results)} items)")

    # 3. 需审核队列
    review_list = [r for r in all_results if r.get('needs_review')]
    review_file = INPUT_DIR / "review_queue_all.json"
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump({
            "count": len(review_list),
            "items": [{"image_id": r["image_id"],
                       "reason": r.get("review_reason", ""),
                       "image_path": r.get("source", {}).get("image_path", "")}
                      for r in review_list]
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] Review queue: {review_file} ({len(review_list)} items)")

    # 4. 用合并结果更新主文件（备份原始文件先）
    main_file = INPUT_DIR / "ocr_results.json"
    backup_file = INPUT_DIR / "ocr_results_before_merge.json.bak"
    if main_file.exists():
        shutil.copy(main_file, backup_file)
        print(f"[OK] Backed up original: {backup_file.name}")
    shutil.copy(merged_file, main_file)
    print(f"[OK] Updated ocr_results.json with merged data")

    print("\n" + "=" * 60)
    print("合并完成!")
    print(f"总计: {len(all_results)} 张图片")
    print(f"高质量(A级): {stats['by_grade']['A']} 张")
    print(f"中等质量(B级): {stats['by_grade']['B']} 张")
    print(f"需关注(C级): {stats['by_grade']['C']} 张")
    print(f"需人工审核: {stats['needs_review']} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()

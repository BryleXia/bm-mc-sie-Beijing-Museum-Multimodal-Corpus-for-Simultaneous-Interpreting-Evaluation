#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
博物馆语料批量OCR处理器 - RapidOCR版本
核心原则：严格按图片分组，保留中英文对应关系
"""

import sys
import io
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 设置编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==================== 配置 ====================
INPUT_DIR = r"E:\ai知识库\nlp大赛\原始语料"
OUTPUT_DIR = r"E:\ai知识库\nlp大赛\ocr输出"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class MuseumOCRProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 统计数据
        self.stats = {
            "total_images": 0,
            "processed": 0,
            "failed": 0,
            "bilingual_images": 0,  # 同时含中英文的图片数
            "chinese_only": 0,
            "english_only": 0,
            "by_museum": defaultdict(int)
        }

    def init_engine(self):
        """初始化OCR引擎"""
        print("Initializing RapidOCR engine...")
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()

    def find_images(self):
        """按博物馆分组查找所有图片"""
        museum_images = defaultdict(list)

        for ext in IMAGE_EXTENSIONS:
            for img_path in self.input_dir.rglob(f"*{ext}"):
                # 确定博物馆名称
                relative = img_path.relative_to(self.input_dir)
                museum = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"
                museum_images[museum].append(img_path)

        # 排序
        for museum in museum_images:
            museum_images[museum].sort()

        return dict(museum_images)

    def detect_language(self, text):
        """检测文本语言"""
        if not text:
            return "unknown"

        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())

        if chinese_chars > 0 and english_chars > 0:
            return "mixed"
        elif chinese_chars > 0:
            return "zh"
        elif english_chars > 0:
            return "en"
        else:
            return "other"

    def process_single_image(self, img_path, museum_name, engine):
        """处理单张图片 - 核心函数"""
        try:
            # OCR识别
            result = engine(str(img_path))

            if not result or not result[0]:
                return None

            raw_results = result[0]

            # 按顺序整理所有文本块
            text_blocks = []
            zh_texts = []
            en_texts = []
            mixed_texts = []

            for item in raw_results:
                if len(item) >= 3:
                    box = item[0]  # 坐标
                    text = item[1]  # 文字
                    score = float(item[2])  # 置信度

                    lang = self.detect_language(text)

                    block = {
                        "text": text,
                        "language": lang,
                        "confidence": round(score, 4),
                        "bbox": box
                    }
                    text_blocks.append(block)

                    # 分类存储
                    if lang == "zh":
                        zh_texts.append(text)
                    elif lang == "en":
                        en_texts.append(text)
                    else:
                        mixed_texts.append(text)

            # 判断图片类型
            has_zh = len(zh_texts) > 0
            has_en = len(en_texts) > 0

            if has_zh and has_en:
                image_type = "bilingual"
                self.stats["bilingual_images"] += 1
            elif has_zh:
                image_type = "chinese_only"
                self.stats["chinese_only"] += 1
            elif has_en:
                image_type = "english_only"
                self.stats["english_only"] += 1
            else:
                image_type = "other"

            # 构建结果 - 严格按图片分组
            entry = {
                "entry_id": f"{museum_name}_{img_path.stem}",
                "source": {
                    "museum": museum_name,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "processed_at": datetime.now().isoformat()
                },
                "image_content": {
                    # 原始顺序的文本块（保留位置信息）
                    "text_blocks_in_order": text_blocks,

                    # 按语言分类（但不打乱图片内对应关系）
                    "zh_all": "\n".join(zh_texts),
                    "en_all": "\n".join(en_texts),
                    "mixed": mixed_texts,

                    # 原始拼接（按OCR检测顺序）
                    "full_text_raw": "\n".join([b["text"] for b in text_blocks])
                },
                "classification": {
                    "image_type": image_type,
                    "has_chinese": has_zh,
                    "has_english": has_en,
                    "text_block_count": len(text_blocks),
                    "zh_block_count": len(zh_texts),
                    "en_block_count": len(en_texts)
                },
                "quality": {
                    "avg_confidence": round(
                        sum(b["confidence"] for b in text_blocks) / len(text_blocks), 4
                    ) if text_blocks else 0
                },
                # 预留字段给后续翻译和对齐
                "alignment": {
                    "status": "image_level",  # 图片级对齐
                    "parallel_pair_id": None,  # 后续可能配对的中文/英文图对
                    "translation_status": "pending"
                },
                "translation": {
                    "targets": {},  # { "ja": "...", "fr": "..." }
                    "notes": ""
                }
            }

            return entry

        except Exception as e:
            print(f"    [ERROR] Failed to process {img_path.name}: {e}")
            return {
                "entry_id": f"{museum_name}_{img_path.stem}",
                "error": str(e),
                "source": {
                    "museum": museum_name,
                    "image_name": img_path.name
                }
            }

    def save_results(self, all_entries, museum_entries):
        """保存结果到文件"""
        print("\n" + "="*60)
        print("Saving results...")
        print("="*60)

        # 1. 保存总结果
        output_file = self.output_dir / "ocr_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_images": self.stats["total_images"],
                    "processed": self.stats["processed"],
                    "failed": self.stats["failed"],
                    "bilingual_images": self.stats["bilingual_images"],
                    "chinese_only": self.stats["chinese_only"],
                    "english_only": self.stats["english_only"],
                    "processor": "RapidOCR",
                    "alignment_type": "image_level"
                },
                "entries": all_entries
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Total results: {output_file}")

        # 2. 按博物馆分文件
        for museum, entries in museum_entries.items():
            museum_file = self.output_dir / f"{museum}_ocr.json"
            with open(museum_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "museum": museum,
                        "entry_count": len(entries),
                        "created_at": datetime.now().isoformat()
                    },
                    "entries": entries
                }, f, ensure_ascii=False, indent=2)
            print(f"[OK] {museum}: {museum_file}")

        # 3. 保存人类可读的文本版本
        text_file = self.output_dir / "extracted_texts.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("博物馆语料 OCR 提取文本\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            for entry in all_entries:
                if "error" in entry:
                    continue

                f.write(f"\n{'='*70}\n")
                f.write(f"条目ID: {entry['entry_id']}\n")
                f.write(f"博物馆: {entry['source']['museum']}\n")
                f.write(f"图片名: {entry['source']['image_name']}\n")
                f.write(f"类型: {entry['classification']['image_type']}\n")
                f.write(f"质量: {entry['quality']['avg_confidence']}\n")
                f.write(f"{'='*70}\n\n")

                content = entry['image_content']

                # 分别输出中文和英文
                if content['zh_all']:
                    f.write("【中文内容】\n")
                    f.write(content['zh_all'])
                    f.write("\n\n")

                if content['en_all']:
                    f.write("【English Content】\n")
                    f.write(content['en_all'])
                    f.write("\n\n")

                # 如果是混合类型，标注
                if entry['classification']['image_type'] == 'bilingual':
                    f.write("[注: 此图同时包含中英文，已按图片级对齐]\n\n")

        print(f"[OK] Human-readable: {text_file}")

    def process_all(self):
        """主处理流程"""
        print("=" * 70)
        print("博物馆语料 OCR 批量处理器")
        print("核心原则: 严格按图片分组，保留中英文对应关系")
        print("=" * 70)

        # 初始化OCR
        engine = self.init_engine()

        # 查找图片
        museum_images = self.find_images()

        if not museum_images:
            print("[ERROR] No images found!")
            return

        # 统计总数
        total = sum(len(imgs) for imgs in museum_images.values())
        self.stats["total_images"] = total

        print(f"\n发现 {len(museum_images)} 个博物馆，共 {total} 张图片:\n")
        for museum, imgs in sorted(museum_images.items()):
            print(f"  - {museum}: {len(imgs)} 张")

        print("\n" + "-" * 70)
        print("开始处理...")
        print("-" * 70 + "\n")

        all_entries = []
        museum_entries = defaultdict(list)

        # 逐个博物馆处理
        for museum, images in sorted(museum_images.items()):
            print(f"\n[{museum}] 处理 {len(images)} 张图片...")

            for idx, img_path in enumerate(images, 1):
                print(f"  [{idx}/{len(images)}] {img_path.name}", end=" ")

                entry = self.process_single_image(img_path, museum, engine)

                if entry:
                    all_entries.append(entry)
                    museum_entries[museum].append(entry)
                    self.stats["processed"] += 1
                    self.stats["by_museum"][museum] += 1

                    # 显示简要信息
                    if "error" in entry:
                        print("[FAILED]")
                        self.stats["failed"] += 1
                    else:
                        img_type = entry['classification']['image_type']
                        blocks = entry['classification']['text_block_count']
                        print(f"[{img_type}, {blocks} blocks]")
                else:
                    print("[NO TEXT]")

        # 保存结果
        self.save_results(all_entries, museum_entries)

        # 打印统计
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("处理完成 - 统计信息")
        print("=" * 70)
        print(f"总图片数: {self.stats['total_images']}")
        print(f"成功处理: {self.stats['processed']}")
        print(f"处理失败: {self.stats['failed']}")
        print(f"\n图片类型分布:")
        print(f"  - 中英双语图: {self.stats['bilingual_images']} (可直接用于平行语料)")
        print(f"  - 仅中文图: {self.stats['chinese_only']}")
        print(f"  - 仅英文图: {self.stats['english_only']}")
        print(f"\n按博物馆分布:")
        for museum, count in sorted(self.stats['by_museum'].items()):
            print(f"  - {museum}: {count}")
        print("\n" + "=" * 70)
        print(f"所有结果已保存到: {self.output_dir}")
        print("=" * 70)


def main():
    processor = MuseumOCRProcessor(INPUT_DIR, OUTPUT_DIR)
    processor.process_all()


if __name__ == "__main__":
    main()

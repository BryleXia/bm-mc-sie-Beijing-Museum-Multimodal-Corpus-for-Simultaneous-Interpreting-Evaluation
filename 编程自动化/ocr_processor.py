#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
博物馆平行语料OCR处理器 - 段落级对齐版
核心功能：
1. 使用RapidOCR提取文本块（含位置bbox和置信度）
2. 按字符集分离中英文，按y坐标排序后合并为完整段落
3. 一张图片 = 一个完整的中英文段落对
4. 输出结构化JSON，标记需人工审核项
"""

import sys
import io
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# 设置编码（如果 config 模块未设置过）
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# ==================== 配置 ====================
# 优先使用集中配置
try:
    import config as cfg
    INPUT_DIR = str(cfg.RAW_IMAGE_DIR)
    OUTPUT_DIR = str(cfg.INTERMEDIATE_DIR)
    IMAGE_EXTENSIONS = cfg.IMAGE_EXTENSIONS
    TEST_MODE = cfg.TEST_MODE
    TEST_MODE_LIMIT = cfg.TEST_MODE_LIMIT
    SELECTED_MUSEUMS = cfg.SELECTED_MUSEUMS
    QUALITY_CONFIG = cfg.QUALITY_CONFIG
except ImportError:
    # 回退：使用项目相对路径
    _project_root = Path(__file__).resolve().parent.parent
    INPUT_DIR = str(_project_root / "原始语料")
    OUTPUT_DIR = str(_project_root / "中间结果")
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    TEST_MODE = False
    TEST_MODE_LIMIT = 10
    SELECTED_MUSEUMS = []  # 空列表 = 处理所有博物馆
    QUALITY_CONFIG = {
        "ocr_confidence_threshold": 0.85,
        "low_confidence_threshold": 0.8,
    }


class TextBlock:
    """文本块数据结构"""
    def __init__(self, text: str, bbox: List, confidence: float, lang: str):
        self.text = text
        self.bbox = bbox  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        self.confidence = confidence
        self.lang = lang

    @property
    def center_y(self) -> float:
        """计算文本框中心点Y坐标"""
        return (self.bbox[0][1] + self.bbox[2][1]) / 2

    @property
    def center_x(self) -> float:
        """计算文本框中心点X坐标"""
        return (self.bbox[0][0] + self.bbox[2][0]) / 2

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "lang": self.lang
        }


class MuseumOCRProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 统计
        self.stats = {
            "total_images": 0,
            "processed": 0,
            "failed": 0,
            "needs_review": 0,
            "by_museum": defaultdict(int)
        }

    def init_engine(self):
        """初始化RapidOCR引擎"""
        print("Initializing RapidOCR engine...")
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()

    def find_images(self) -> Dict[str, List[Path]]:
        """按博物馆分组查找所有图片"""
        museum_images = defaultdict(list)

        for ext in IMAGE_EXTENSIONS:
            for img_path in self.input_dir.rglob(f"*{ext}"):
                relative = img_path.relative_to(self.input_dir)
                museum = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"
                museum_images[museum].append(img_path)

        # 排序
        for museum in museum_images:
            museum_images[museum].sort()

        return dict(museum_images)

    def detect_language(self, text: str) -> str:
        """检测文本语言：基于字符集简单区分"""
        if not text:
            return "unknown"

        # 检测CJK字符（中文）
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))
        # 检测拉丁字母（英文）
        has_latin = bool(re.search(r'[a-zA-Z]', text))

        if has_cjk:
            return "zh"
        elif has_latin:
            return "en"
        else:
            return "other"

    def extract_text_blocks(self, img_path: Path, engine) -> List[TextBlock]:
        """提取图片中的文本块"""
        result = engine(str(img_path))

        if not result or not result[0]:
            return []

        blocks = []
        for item in result[0]:
            if len(item) >= 3:
                box = item[0]  # 坐标
                text = item[1]  # 文字
                score = float(item[2])  # 置信度

                lang = self.detect_language(text)
                if lang in ["zh", "en"]:  # 只保留中英文
                    block = TextBlock(text, box, score, lang)
                    blocks.append(block)

        return blocks

    def merge_blocks_to_paragraph(self, blocks: List[TextBlock]) -> str:
        """
        将多个文本块按y坐标排序后合并为完整段落
        保持阅读顺序（从上到下）
        """
        if not blocks:
            return ""

        # 按y坐标排序（从上到下）
        sorted_blocks = sorted(blocks, key=lambda b: b.center_y)

        # 合并文本，用空格连接英文，直接连接中文
        texts = [b.text for b in sorted_blocks]

        # 检测语言类型来决定连接方式
        if sorted_blocks[0].lang == "zh":
            # 中文：直接连接
            return "".join(texts)
        else:
            # 英文：用空格连接
            return " ".join(texts)

    def process_single_image(self, img_path: Path, museum_name: str,
                            engine) -> Optional[Dict]:
        """处理单张图片"""
        try:
            # 提取文本块
            blocks = self.extract_text_blocks(img_path, engine)

            if not blocks:
                return None

            # 分离中英文
            zh_blocks = [b for b in blocks if b.lang == "zh"]
            en_blocks = [b for b in blocks if b.lang == "en"]

            # 合并为完整段落
            zh_text = self.merge_blocks_to_paragraph(zh_blocks)
            en_text = self.merge_blocks_to_paragraph(en_blocks)

            # 计算整体OCR置信度
            all_confidences = [b.confidence for b in blocks]
            avg_ocr_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

            # 识别低置信度文本块
            low_confidence_blocks = [
                {
                    "text": b.text[:50] + "..." if len(b.text) > 50 else b.text,
                    "lang": b.lang,
                    "confidence": round(b.confidence, 4),
                    "bbox": b.bbox
                }
                for b in blocks
                if b.confidence < QUALITY_CONFIG["ocr_confidence_threshold"]
            ]

            # 确定是否需要审核
            needs_review = len(low_confidence_blocks) > 0 or avg_ocr_confidence < QUALITY_CONFIG["low_confidence_threshold"]

            # 确定质量等级
            if avg_ocr_confidence > 0.9:
                quality_grade = "A"
            elif avg_ocr_confidence >= 0.8:
                quality_grade = "B"
            else:
                quality_grade = "C"
                needs_review = True

            # 构建结果
            result = {
                "image_id": f"{museum_name}_{img_path.stem}",
                "source": {
                    "museum": museum_name,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "processed_at": datetime.now().isoformat()
                },
                "zh_text": zh_text,
                "en_text": en_text,
                "text_blocks": [b.to_dict() for b in blocks],
                "quality": {
                    "grade": quality_grade,
                    "zh_block_count": len(zh_blocks),
                    "en_block_count": len(en_blocks),
                    "total_blocks": len(blocks),
                    "avg_ocr_confidence": round(avg_ocr_confidence, 4),
                    "low_confidence_blocks": low_confidence_blocks
                },
                "needs_review": needs_review,
                "review_reason": self._get_review_reason(avg_ocr_confidence, low_confidence_blocks, zh_text, en_text)
            }

            return result

        except Exception as e:
            print(f"    [ERROR] Failed to process {img_path.name}: {e}")
            return {
                "image_id": f"{museum_name}_{img_path.stem}",
                "error": str(e),
                "source": {
                    "museum": museum_name,
                    "image_name": img_path.name
                },
                "needs_review": True,
                "review_reason": "processing_error"
            }

    def _get_review_reason(self, avg_confidence: float,
                          low_conf_blocks: List[Dict],
                          zh_text: str, en_text: str) -> str:
        """获取需要审核的原因"""
        reasons = []

        # 检查低OCR置信度
        if low_conf_blocks:
            reasons.append(f"low_ocr_confidence({len(low_conf_blocks)}blocks)")

        # 检查整体置信度
        if avg_confidence < QUALITY_CONFIG["low_confidence_threshold"]:
            reasons.append(f"low_avg_confidence({avg_confidence:.2f})")

        # 检查是否缺少中文或英文
        if not zh_text:
            reasons.append("missing_chinese")
        if not en_text:
            reasons.append("missing_english")

        return ";".join(reasons) if reasons else ""

    def save_results(self, all_results: List[Dict]):
        """保存结果到文件"""
        print("\n" + "="*60)
        print("Saving results...")
        print("="*60)

        # 1. 保存完整结果
        output_file = self.output_dir / "ocr_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_images": self.stats["total_images"],
                    "processed": self.stats["processed"],
                    "failed": self.stats["failed"],
                    "needs_review": self.stats["needs_review"],
                    "processor": "RapidOCR",
                    "alignment_method": "paragraph_level"
                },
                "results": all_results
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] OCR results: {output_file}")

        # 2. 保存需审核列表
        review_list = [r for r in all_results if r.get("needs_review")]
        review_file = self.output_dir / "review_queue.json"
        with open(review_file, 'w', encoding='utf-8') as f:
            json.dump({
                "count": len(review_list),
                "items": [{"image_id": r["image_id"],
                          "reason": r.get("review_reason", ""),
                          "image_path": r["source"]["image_path"]}
                         for r in review_list]
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Review queue: {review_file} ({len(review_list)} items)")

        # 3. 按质量分级保存
        for grade in ["A", "B", "C"]:
            grade_results = [r for r in all_results
                           if r.get("quality", {}).get("grade") == grade and "error" not in r]
            grade_file = self.output_dir / f"grade_{grade}_corpus.json"
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
            print(f"[OK] Grade {grade} corpus: {grade_file} ({len(grade_results)} items)")

        # 4. 保存人类可读的文本版本
        text_file = self.output_dir / "corpus_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("博物馆平行语料 - 段落级对齐文本\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            for result in all_results:
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

        print(f"[OK] Human-readable text: {text_file}")

        # 5. 保存TSV格式（便于导入其他工具）
        tsv_file = self.output_dir / "corpus.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("image_id\tmuseum\tzh_text\ten_text\tquality_grade\tneeds_review\n")
            for result in all_results:
                if "error" in result:
                    continue
                zh = result['zh_text'].replace('\t', ' ').replace('\n', ' ')
                en = result['en_text'].replace('\t', ' ').replace('\n', ' ')
                grade = result.get('quality', {}).get('grade', 'C')
                review = '1' if result.get('needs_review') else '0'
                f.write(f"{result['image_id']}\t{result['source']['museum']}\t")
                f.write(f"{zh}\t{en}\t{grade}\t{review}\n")

        print(f"[OK] TSV format: {tsv_file}")

    def process_all(self):
        """主处理流程"""
        print("=" * 70)
        print("博物馆平行语料OCR处理器 - 段落级对齐版")
        print("核心：一张图片 = 一个完整的中英文段落对")
        if TEST_MODE:
            print(f"[测试模式] 仅处理前 {TEST_MODE_LIMIT} 张图片")
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

        all_results = []
        processed_count = 0

        # 逐个博物馆处理
        for museum, images in sorted(museum_images.items()):
            # 检查是否只处理选定的博物馆
            if SELECTED_MUSEUMS and museum not in SELECTED_MUSEUMS:
                continue

            # 测试模式：限制处理数量
            if TEST_MODE and processed_count >= TEST_MODE_LIMIT:
                print(f"\n[测试模式] 已达到限制 {TEST_MODE_LIMIT} 张，停止处理")
                break

            print(f"\n[{museum}] 处理 {len(images)} 张图片...")

            for idx, img_path in enumerate(images, 1):
                # 测试模式：限制处理数量
                if TEST_MODE and processed_count >= TEST_MODE_LIMIT:
                    break
                print(f"  [{idx}/{len(images)}] {img_path.name}", end=" ")

                result = self.process_single_image(img_path, museum, engine)

                if result:
                    all_results.append(result)
                    self.stats["processed"] += 1
                    self.stats["by_museum"][museum] += 1
                    processed_count += 1

                    if result.get("needs_review"):
                        self.stats["needs_review"] += 1
                        print(f"[REVIEW]")
                    elif "error" in result:
                        self.stats["failed"] += 1
                        print(f"[FAILED]")
                    else:
                        grade = result['quality']['grade']
                        zh_len = len(result['zh_text'])
                        en_len = len(result['en_text'])
                        print(f"[OK, Grade {grade}, zh:{zh_len}chars, en:{en_len}chars]")
                else:
                    print("[NO TEXT]")

        # 保存结果
        self.save_results(all_results)

        # 打印统计
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        if TEST_MODE:
            print("处理完成 - 统计信息 [测试模式]")
        else:
            print("处理完成 - 统计信息")
        print("=" * 70)
        if TEST_MODE:
            print(f"[测试模式] 限制处理: {TEST_MODE_LIMIT} 张")
        if SELECTED_MUSEUMS:
            print(f"[选定博物馆] {', '.join(SELECTED_MUSEUMS)}")
        print(f"总图片数: {self.stats['total_images']}")
        print(f"成功处理: {self.stats['processed']}")
        print(f"处理失败: {self.stats['failed']}")
        print(f"需人工审核: {self.stats['needs_review']}")
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

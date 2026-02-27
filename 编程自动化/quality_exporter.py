#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量分级与导出器 - 段落级对齐版
功能：
1. 读取OCR结果和审核结果
2. 按A/B/C三级分级（基于OCR置信度和人工审核状态）
3. 导出最终语料库（JSON、TSV、TMX等格式）
"""

import sys
import io
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional

# 设置编码（如果 config 模块未设置过）
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# ==================== 配置 ====================
try:
    import config as cfg
    INPUT_DIR = str(cfg.INTERMEDIATE_DIR)
    OUTPUT_DIR = str(cfg.FINAL_OUTPUT_DIR)
    GRADE_CONFIG = cfg.GRADE_CONFIG
except ImportError:
    _project_root = Path(__file__).resolve().parent.parent
    INPUT_DIR = str(_project_root / "中间结果")
    OUTPUT_DIR = str(_project_root / "输出")
    GRADE_CONFIG = {
        "A": {"min_ocr_confidence": 0.9},
        "B": {"min_ocr_confidence": 0.8},
        "C": {"min_ocr_confidence": 0.0},
    }


class QualityExporter:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.ocr_results = []
        self.review_results = {}

        # 统计
        self.stats = {
            "total_images": 0,
            "grade_a": 0,
            "grade_b": 0,
            "grade_c": 0,
            "by_museum": defaultdict(lambda: defaultdict(int))
        }

    def load_data(self):
        """加载OCR结果和审核结果"""
        # 加载OCR结果
        ocr_file = self.input_dir / "ocr_results.json"
        if ocr_file.exists():
            with open(ocr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.ocr_results = data.get("results", [])
            print(f"[OK] Loaded OCR results: {len(self.ocr_results)} images")
        else:
            print(f"[ERROR] OCR results file not found: {ocr_file}")
            return False

        # 加载审核结果（如果存在）
        review_file = self.input_dir / "审核结果" / "reviewed_results.json"
        if review_file.exists():
            with open(review_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get("results", []):
                    self.review_results[item["image_id"]] = item
            print(f"[OK] Loaded review results: {len(self.review_results)} images")
        else:
            print("[INFO] No review results found, will use OCR data only")

        return True

    def calculate_grade(self, result: Dict) -> str:
        """
        计算语料质量等级

        分级标准：
        - A级：平均OCR置信度 > 0.9，人工已确认
        - B级：平均OCR置信度 0.8-0.9，或未经人工审核
        - C级：平均OCR置信度 < 0.8
        """
        quality = result.get("quality", {})
        avg_confidence = quality.get("avg_ocr_confidence", 0)

        # 检查人工审核状态
        image_id = result.get("image_id")
        review = self.review_results.get(image_id)
        human_confirmed = review and review.get("review_status") == "confirmed"
        human_corrected = review and review.get("review_status") == "corrected"

        # 如果有修正，使用修正后的文本，质量等级提升
        if human_corrected:
            return "A"  # 人工修正后视为最高质量

        # A级：高置信度 + 人工确认
        if avg_confidence >= GRADE_CONFIG["A"]["min_ocr_confidence"] and human_confirmed:
            return "A"

        # B级：中等置信度，或未经人工审核的高置信度
        if avg_confidence >= GRADE_CONFIG["B"]["min_ocr_confidence"]:
            return "B"

        # C级：低置信度
        return "C"

    def get_final_text(self, result: Dict) -> tuple:
        """获取最终的中英文文本（使用修正后的如果有）"""
        image_id = result.get("image_id")
        review = self.review_results.get(image_id)

        zh_text = result.get("zh_text", "")
        en_text = result.get("en_text", "")

        if review:
            # 使用修正后的文本
            corrected_zh = review.get("corrected_zh", "")
            corrected_en = review.get("corrected_en", "")

            # [删除] 标记：显式清空该语种文本
            if corrected_zh.strip() == "[删除]":
                zh_text = ""
            elif corrected_zh:
                zh_text = corrected_zh

            if corrected_en.strip() == "[删除]":
                en_text = ""
            elif corrected_en:
                en_text = corrected_en

        return zh_text, en_text

    def process_all(self):
        """处理所有数据并分级"""
        print("\n" + "="*60)
        print("Processing and grading...")
        print("="*60)

        # 分级结果
        grade_a = []
        grade_b = []
        grade_c = []

        for result in self.ocr_results:
            if "error" in result:
                continue

            image_id = result["image_id"]
            museum = result["source"]["museum"]

            self.stats["total_images"] += 1

            # 检查是否被人工跳过（标记为不需要的语料）
            review = self.review_results.get(image_id)
            if review and review.get("review_status") == "skipped":
                self.stats.setdefault("skipped", 0)
                self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                continue

            # 计算等级
            grade = self.calculate_grade(result)

            # 获取最终文本
            zh_text, en_text = self.get_final_text(result)

            # 跳过没有文本的
            if not zh_text and not en_text:
                continue

            # 构建语料条目
            corpus_entry = {
                "image_id": image_id,
                "museum": museum,
                "zh": zh_text,
                "en": en_text,
                "metadata": {
                    "original_quality": result.get("quality", {}),
                    "review_status": self.review_results.get(image_id, {}).get("review_status", "unreviewed"),
                    "review_comment": self.review_results.get(image_id, {}).get("review_comment", "")
                }
            }

            # 添加到对应等级
            if grade == "A":
                grade_a.append(corpus_entry)
                self.stats["grade_a"] += 1
                self.stats["by_museum"][museum]["A"] += 1
            elif grade == "B":
                grade_b.append(corpus_entry)
                self.stats["grade_b"] += 1
                self.stats["by_museum"][museum]["B"] += 1
            else:
                grade_c.append(corpus_entry)
                self.stats["grade_c"] += 1
                self.stats["by_museum"][museum]["C"] += 1

        return grade_a, grade_b, grade_c

    def save_corpus(self, grade_a: List[Dict], grade_b: List[Dict], grade_c: List[Dict]):
        """保存分级语料库"""
        print("\n" + "="*60)
        print("Saving corpus files...")
        print("="*60)

        # 1. A级语料
        a_file = self.output_dir / "corpus_level_a.json"
        with open(a_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "level": "A",
                    "description": "高质量语料 - OCR置信度>0.9且人工已确认，或人工修正",
                    "count": len(grade_a),
                    "created_at": datetime.now().isoformat()
                },
                "entries": grade_a
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Level A corpus: {a_file} ({len(grade_a)} entries)")

        # 2. B级语料
        b_file = self.output_dir / "corpus_level_b.json"
        with open(b_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "level": "B",
                    "description": "中等质量语料 - OCR置信度0.8-0.9",
                    "count": len(grade_b),
                    "created_at": datetime.now().isoformat()
                },
                "entries": grade_b
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Level B corpus: {b_file} ({len(grade_b)} entries)")

        # 3. C级语料
        c_file = self.output_dir / "corpus_level_c.json"
        with open(c_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "level": "C",
                    "description": "需关注语料 - OCR置信度<0.8",
                    "count": len(grade_c),
                    "created_at": datetime.now().isoformat()
                },
                "entries": grade_c
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Level C corpus: {c_file} ({len(grade_c)} entries)")

        # 4. 合并语料（所有等级）
        all_file = self.output_dir / "corpus_all.json"
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "description": "完整语料库（含所有等级）",
                    "total_count": len(grade_a) + len(grade_b) + len(grade_c),
                    "grade_a": len(grade_a),
                    "grade_b": len(grade_b),
                    "grade_c": len(grade_c),
                    "created_at": datetime.now().isoformat()
                },
                "entries": {
                    "A": grade_a,
                    "B": grade_b,
                    "C": grade_c
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Complete corpus: {all_file}")

        # 5. TSV格式（便于导入Excel等）
        tsv_file = self.output_dir / "corpus.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("image_id\tmuseum\tzh_text\ten_text\tgrade\n")
            for entry in grade_a + grade_b + grade_c:
                zh = entry['zh'].replace('\t', ' ').replace('\n', ' ')
                en = entry['en'].replace('\t', ' ').replace('\n', ' ')
                grade = self.calculate_grade_from_entry(entry)
                f.write(f"{entry['image_id']}\t{entry['museum']}\t{zh}\t{en}\t{grade}\n")
        print(f"[OK] TSV format: {tsv_file}")

        # 6. TMX格式（翻译记忆库）
        tmx_file = self.output_dir / "corpus.tmx"
        self._save_tmx(tmx_file, grade_a + grade_b)
        print(f"[OK] TMX format: {tmx_file}")

        # 7. 纯文本平行语料
        txt_file = self.output_dir / "corpus_parallel.txt"
        self._save_parallel_text(txt_file, grade_a + grade_b + grade_c)
        print(f"[OK] Parallel text: {txt_file}")

        # 8. 统计报告
        stats_file = self.output_dir / "corpus_stats.json"
        self._save_stats(stats_file)
        print(f"[OK] Statistics: {stats_file}")

    def calculate_grade_from_entry(self, entry: Dict) -> str:
        """从条目推断等级"""
        review_status = entry.get("metadata", {}).get("review_status", "")
        if review_status == "corrected":
            return "A"
        quality = entry.get("metadata", {}).get("original_quality", {})
        conf = quality.get("avg_ocr_confidence", 0)
        if conf >= 0.9 and review_status == "confirmed":
            return "A"
        elif conf >= 0.8:
            return "B"
        else:
            return "C"

    def _save_tmx(self, filepath: Path, entries: List[Dict]):
        """保存为TMX格式（Translation Memory eXchange）"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<tmx version="1.4">\n')
            f.write('  <header\n')
            f.write('    creationtool="MuseumCorpusBuilder"\n')
            f.write(f'    creationdate="{datetime.now().strftime("%Y%m%dT%H%M%SZ")}"\n')
            f.write('    srclang="zh-CN"\n')
            f.write('    adminlang="zh-CN"\n')
            f.write('    datatype="plaintext"\n')
            f.write('    o-tmf="ABCTransMem"\n')
            f.write('    segtype="paragraph"\n')
            f.write('  />\n')
            f.write('  <body>\n')

            for entry in entries:
                f.write('    <tu>\n')
                f.write(f'      <prop type="image_id">{entry["image_id"]}</prop>\n')
                f.write(f'      <prop type="museum">{entry["museum"]}</prop>\n')
                f.write(f'      <tuv xml:lang="zh-CN">\n')
                f.write(f'        <seg>{self._escape_xml(entry["zh"])}</seg>\n')
                f.write('      </tuv>\n')
                f.write(f'      <tuv xml:lang="en">\n')
                f.write(f'        <seg>{self._escape_xml(entry["en"])}</seg>\n')
                f.write('      </tuv>\n')
                f.write('    </tu>\n')

            f.write('  </body>\n')
            f.write('</tmx>\n')

    def _escape_xml(self, text: str) -> str:
        """转义XML特殊字符"""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))

    def _save_parallel_text(self, filepath: Path, entries: List[Dict]):
        """保存为平行文本格式"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("博物馆平行语料库\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            for entry in entries:
                f.write(f"# Image: {entry['image_id']}\n")
                f.write(f"# Museum: {entry['museum']}\n")
                f.write(f"[ZH] {entry['zh']}\n")
                f.write(f"[EN] {entry['en']}\n")
                f.write("\n")

    def _save_stats(self, filepath: Path):
        """保存统计报告"""
        total = self.stats["total_images"]
        stats = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_images": total,
                "grade_a": self.stats["grade_a"],
                "grade_b": self.stats["grade_b"],
                "grade_c": self.stats["grade_c"],
                "high_quality_ratio": round(
                    (self.stats["grade_a"] + self.stats["grade_b"]) / max(total, 1), 4
                )
            },
            "by_museum": dict(self.stats["by_museum"]),
            "grade_criteria": {
                "A": "OCR置信度>0.9且人工已确认，或人工修正",
                "B": "OCR置信度0.8-0.9",
                "C": "OCR置信度<0.8"
            },
            "recommendations": []
        }

        # 生成建议
        if self.stats["grade_c"] > total * 0.3:
            stats["recommendations"].append(
                "C级语料占比超过30%，建议增加人工审核或改进OCR质量"
            )

        if self.stats["grade_a"] < total * 0.1:
            stats["recommendations"].append(
                "A级高质量语料较少，建议进行更多人工审核以提升语料质量"
            )

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def print_statistics(self):
        """打印统计信息"""
        total = self.stats["total_images"]
        print("\n" + "=" * 70)
        print("语料库统计")
        print("=" * 70)
        print(f"总图片数: {total}")
        skipped = self.stats.get('skipped', 0)
        if skipped > 0:
            print(f"人工跳过: {skipped} (已从语料库中排除)")
        print(f"\n质量分级:")
        print(f"  A级 (高质量): {self.stats['grade_a']} ({self.stats['grade_a']/max(total,1)*100:.1f}%)")
        print(f"  B级 (中等):   {self.stats['grade_b']} ({self.stats['grade_b']/max(total,1)*100:.1f}%)")
        print(f"  C级 (需关注): {self.stats['grade_c']} ({self.stats['grade_c']/max(total,1)*100:.1f}%)")

        high_quality = self.stats['grade_a'] + self.stats['grade_b']
        print(f"\n高质量语料占比: {high_quality/max(total,1)*100:.1f}%")

        print(f"\n按博物馆分布:")
        for museum, counts in sorted(self.stats['by_museum'].items()):
            total_museum = counts.get('A', 0) + counts.get('B', 0) + counts.get('C', 0)
            print(f"  {museum}:")
            print(f"    A: {counts.get('A', 0)}, B: {counts.get('B', 0)}, C: {counts.get('C', 0)} (总计: {total_museum})")

        print("\n" + "=" * 70)
        print(f"所有文件已保存到: {self.output_dir}")
        print("=" * 70)

    def run(self):
        """运行完整流程"""
        print("=" * 70)
        print("博物馆平行语料库 - 质量分级与导出")
        print("=" * 70)

        # 加载数据
        if not self.load_data():
            return

        # 处理并分级
        grade_a, grade_b, grade_c = self.process_all()

        # 保存结果
        self.save_corpus(grade_a, grade_b, grade_c)

        # 打印统计
        self.print_statistics()


def main():
    exporter = QualityExporter(INPUT_DIR, OUTPUT_DIR)
    exporter.run()


if __name__ == "__main__":
    main()

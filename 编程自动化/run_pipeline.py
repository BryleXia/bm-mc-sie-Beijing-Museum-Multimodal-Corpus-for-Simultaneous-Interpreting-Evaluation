#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流程脚本
运行整个博物馆平行语料库构建流程

Usage:
    python run_pipeline.py --stage ocr      # 只运行OCR
    python run_pipeline.py --stage review   # 只运行审核界面
    python run_pipeline.py --stage export   # 只运行导出
    python run_pipeline.py --stage all      # 运行完整流程

修复记录 (2026-02):
- 使用集中配置模块
- 修复审核界面路径和工作目录问题
- 添加阶段前置检查
"""

import sys
import io
import argparse
from pathlib import Path

# 设置编码
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# 导入配置
try:
    import config as cfg
except ImportError:
    # 确保能找到 config 模块
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as cfg


def run_ocr(test_mode: bool = False):
    """运行OCR处理"""
    print("\n" + "=" * 70)
    print("阶段一：OCR处理")
    print("=" * 70)

    from ocr_processor import MuseumOCRProcessor

    # 设置模式
    cfg.TEST_MODE = test_mode
    if not test_mode:
        response = input("\n这将处理所有图片，可能需要较长时间。\n确认开始? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("已取消")
            return False

    processor = MuseumOCRProcessor(str(cfg.RAW_IMAGE_DIR), str(cfg.INTERMEDIATE_DIR))
    processor.process_all()
    return True


def run_review():
    """运行审核界面"""
    print("\n" + "=" * 70)
    print("阶段二：人工审核")
    print("=" * 70)

    # 检查 OCR 结果是否存在
    if not cfg.OCR_RESULTS_FILE.exists():
        print(f"[ERROR] OCR结果文件不存在: {cfg.OCR_RESULTS_FILE}")
        print("请先运行: python run_pipeline.py --stage ocr")
        return False

    print(f"启动审核界面: http://{cfg.REVIEW_SERVER_NAME}:{cfg.REVIEW_SERVER_PORT}")
    print("按 Ctrl+C 停止")
    print("=" * 70 + "\n")

    import subprocess
    review_app_path = Path(__file__).resolve().parent.parent / "审核界面" / "review_app.py"
    review_app_dir = review_app_path.parent

    if not review_app_path.exists():
        print(f"[ERROR] 审核界面脚本不存在: {review_app_path}")
        return False

    try:
        subprocess.run(
            [sys.executable, str(review_app_path)],
            cwd=str(review_app_dir),
            check=True
        )
    except KeyboardInterrupt:
        print("\n审核界面已停止")
    except subprocess.CalledProcessError as e:
        print(f"审核界面异常退出 (code {e.returncode})")
        return False

    return True


def run_export():
    """运行导出"""
    print("\n" + "=" * 70)
    print("阶段三：质量分级与导出")
    print("=" * 70)

    # 检查 OCR 结果是否存在
    if not cfg.OCR_RESULTS_FILE.exists():
        print(f"[ERROR] OCR结果文件不存在: {cfg.OCR_RESULTS_FILE}")
        print("请先运行OCR阶段")
        return False

    from quality_exporter import QualityExporter

    exporter = QualityExporter(str(cfg.INTERMEDIATE_DIR), str(cfg.FINAL_OUTPUT_DIR))
    exporter.run()
    return True


def main():
    parser = argparse.ArgumentParser(description='博物馆平行语料库构建流程')
    parser.add_argument('--stage', choices=['ocr', 'review', 'export', 'all'],
                        default='all', help='运行阶段')
    parser.add_argument('--test', action='store_true',
                        help='测试模式（只处理10张图片）')

    args = parser.parse_args()

    print("=" * 70)
    print("博物馆平行语料库构建系统")
    cfg.print_config()

    if args.stage in ['ocr', 'all']:
        if not run_ocr(test_mode=args.test):
            return

    if args.stage in ['review', 'all']:
        if not run_review():
            return

    if args.stage in ['export', 'all']:
        if not run_export():
            return

    print("\n" + "=" * 70)
    print("流程完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

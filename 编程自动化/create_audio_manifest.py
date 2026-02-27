#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建音频清单文件 - 符合 Hugging Face 标准

输入：
- reading_script_info.json (index → board_id, text)
- alignment_*.json (时间戳)
输出：
- audio_manifest.jsonl
"""

import json
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(r"E:\ai知识库\nlp大赛")
INFO_FILE = PROJECT_ROOT / "归档_非交付/交付_旧版/朗读稿件/reading_script_info.json"
ALIGNMENT_FILES = {
    "bg": PROJECT_ROOT / "归档_非交付/交付_旧版/录音/对齐/alignment_bg_v3.json",
    "es": PROJECT_ROOT / "归档_非交付/交付_旧版/录音/对齐/alignment_es_v3.json",
    "ja": PROJECT_ROOT / "归档_非交付/交付_旧版/录音/对齐/alignment_ja_llm.json",
}
OUTPUT_FILE = PROJECT_ROOT / "语料成果包/03_音频语料/audio_manifest.jsonl"

def main():
    # 1. 加载朗读稿件信息
    print(f"读取朗读稿件信息: {INFO_FILE}")
    with open(INFO_FILE, 'r', encoding='utf-8') as f:
        info = json.load(f)

    # 建立 index → sentence 映射
    index_to_sent = {s['index']: s for s in info['sentences']}
    print(f"  - 共 {len(index_to_sent)} 条朗读稿件")

    # 2. 加载各语言对齐数据
    alignments = {}
    for lang, path in ALIGNMENT_FILES.items():
        print(f"读取对齐文件 [{lang}]: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        alignments[lang] = {s['index']: s for s in data['sentences']}

        # 统计有效对齐数量
        valid = sum(1 for s in data['sentences'] if s.get('start_time') is not None)
        print(f"  - 共 {len(alignments[lang])} 条, 有效对齐 {valid} 条")

    # 3. 生成清单
    records = []
    stats = {"bg": 0, "es": 0, "ja": 0}

    for lang in ["bg", "es", "ja"]:
        for idx in range(1, 213):
            if idx not in index_to_sent:
                print(f"警告: index {idx} 不在朗读稿件中")
                continue
            if idx not in alignments[lang]:
                print(f"警告: index {idx} 不在 {lang} 对齐文件中")
                continue

            sent = index_to_sent[idx]
            align = alignments[lang][idx]

            # 跳过没有时间戳的记录
            if align.get('start_time') is None:
                continue

            record = {
                "audio_path": f"{lang}/{lang}_{idx:04d}.wav",
                "text": sent.get(lang, ""),
                "start_time": round(align.get('start_time', 0), 2),
                "end_time": round(align.get('end_time', 0), 2),
                "duration": round(align.get('duration', 0), 2),
                "language": lang,
                "index": idx,
                "museum": sent.get('museum', ''),
                "board_id": sent.get('board_id', ''),
            }

            # 可选字段
            if 'match_score' in align:
                record['match_score'] = round(align['match_score'], 3)
            if 'method' in align:
                record['method'] = align['method']

            records.append(record)
            stats[lang] += 1

    # 按语言和index排序
    records.sort(key=lambda x: ({"bg": 0, "es": 1, "ja": 2}[x['language']], x['index']))

    # 4. 写入 JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n生成完成!")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"总记录数: {len(records)}")
    print(f"  - 保加利亚语 (bg): {stats['bg']} 条")
    print(f"  - 西班牙语 (es): {stats['es']} 条")
    print(f"  - 日语 (ja): {stats['ja']} 条")

if __name__ == "__main__":
    main()

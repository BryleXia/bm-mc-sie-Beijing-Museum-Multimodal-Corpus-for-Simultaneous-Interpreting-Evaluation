#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语种朗读稿件生成器

从多语种语料库中选取适合朗读的句子，生成各语言的朗读稿件。

策略：
- 目标时长：30分钟（按英语约140词/分钟计算，约4200词）
- 按博物馆顺序排列，每个馆的句子数量明确
- 按句子长度分层：短句20%, 中句50%, 长句30%
- 优先选择翻译完整的句子
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 添加父目录到路径以导入 config
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

# ==================== 配置参数 ====================
# 核心原则：不同语言朗读速度不同，以最慢的语言（保加利亚语）为基准
#
# 各语言朗读速度（博物馆讲解类文本）：
# - 保加利亚语：约110词/分钟（辅音群多，专有名词复杂）
# - 日语：约130词/分钟（按英语词数折算）
# - 英语：约140词/分钟
#
# 目标：保加利亚语约25分钟 → 英语词数约 25 * 110 = 2750词（约200句）

TARGET_MINUTES_BG = 25          # 保加利亚语目标朗读时长（分钟）
BG_WORDS_PER_MINUTE = 110       # 保加利亚语朗读速度（用于目标计算）
EN_WORDS_PER_MINUTE = 140       # 英语朗读速度（用于时长显示）
JA_WORDS_PER_MINUTE = 130       # 日语朗读速度（按英语词数折算）

TARGET_WORDS = TARGET_MINUTES_BG * BG_WORDS_PER_MINUTE  # 目标：2750英语词

# 博物馆排序顺序（朗读顺序）
MUSEUM_ORDER = ["国家博物馆", "故宫", "抗日纪念馆", "首都博物馆", "党史馆"]

# 博物馆名称映射（处理可能的变体）
MUSEUM_NAME_MAP = {
    "国家博物馆": "国家博物馆",
    "故宫": "故宫",
    "故宫博物院": "故宫",
    "抗日纪念馆": "抗日纪念馆",
    "中国人民抗日战争纪念馆": "抗日纪念馆",
    "首都博物馆": "首都博物馆",
    "党史馆": "党史馆",
    "中国共产党历史展览馆": "党史馆",
}

# 句子长度分类（按英文词数）
SHORT_THRESHOLD = 15   # 短句：<15词
LONG_THRESHOLD = 35    # 长句：>35词

# 长度分布目标（默认）
LENGTH_DISTRIBUTION = {
    "short": 0.20,   # 20%
    "medium": 0.50,  # 50%
    "long": 0.30,    # 30%
}

# 博物馆特定配置（针对有问题的博物馆调整）
MUSEUM_CONFIG = {
    "抗日纪念馆": {
        "min_words": 40,           # 提高最小词数阈值，只保留大段介绍
        "length_distribution": {   # 只要中长句
            "short": 0.0,          # 不要短句
            "medium": 0.40,
            "long": 0.60,          # 多选长句
        },
        "weight_multiplier": 0.5,  # 减少总体比例
    },
    "党史馆": {
        "weight_multiplier": 2.0,  # 增加总体比例
    }
}

# 语言配置
LANGUAGES = {
    "zh": {"name": "中文", "filename": "reading_script_zh.txt", "for_tts": True},
    "en": {"name": "英文", "filename": "reading_script_en.txt", "for_tts": True},
    "ja": {"name": "日语", "filename": "reading_script_ja.txt", "for_tts": False},
    "es": {"name": "西班牙语", "filename": "reading_script_es.txt", "for_tts": False},
    "bg": {"name": "保加利亚语", "filename": "reading_script_bg.txt", "for_tts": False},
}


def normalize_museum_name(name: str) -> str:
    """标准化博物馆名称"""
    return MUSEUM_NAME_MAP.get(name, name)


def is_valid_japanese(text: str) -> bool:
    """检查文本是否包含日语特有字符（平假名或片假名）"""
    for char in text:
        # 平假名范围: U+3040-U+309F
        # 片假名范围: U+30A0-U+30FF
        if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF':
            return True
    return False


def is_valid_spanish(text: str) -> bool:
    """检查文本是否包含西班牙语特有字符"""
    # 西班牙语特有字符: ñ, á, é, í, ó, ú, ü, ¿, ¡
    spanish_chars = 'ñáéíóúü¿¡'
    for char in text.lower():
        if char in spanish_chars:
            return True
    # 如果没有特有字符，至少不能全是英文常见词
    # 简单检查：如果有de, la, el, que等西语高频词
    lower_text = text.lower()
    spanish_words = [' de ', ' la ', ' el ', ' que ', ' con ', ' por ', ' del ']
    for word in spanish_words:
        if word in lower_text:
            return True
    return False


def is_valid_bulgarian(text: str) -> bool:
    """检查文本是否包含保加利亚语特有字符（西里尔字母）"""
    for char in text:
        # 西里尔字母范围: U+0400-U+04FF
        if '\u0400' <= char <= '\u04FF':
            return True
    return False


def classify_sentence_length(en_text: str) -> str:
    """根据英文词数分类句子长度"""
    word_count = len(en_text.split())
    if word_count < SHORT_THRESHOLD:
        return "short"
    elif word_count > LONG_THRESHOLD:
        return "long"
    else:
        return "medium"


def load_corpus(corpus_path: Path) -> Dict:
    """加载多语种语料库"""
    print(f"加载语料库: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_all_sentences(corpus: Dict) -> List[Dict]:
    """
    从语料库中提取所有句子，附带元数据

    返回: [
        {
            "museum": "国家博物馆",
            "board_id": "xxx",
            "board_title_zh": "...",
            "board_title_en": "...",
            "zh": "...",
            "en": "...",
            "ja": "...",
            "es": "...",
            "bg": "...",
            "length_class": "medium",
            "word_count": 25,
        },
        ...
    ]
    """
    all_sentences = []

    for board in corpus.get("boards", []):
        museum = normalize_museum_name(board.get("source", {}).get("museum", "未知"))
        board_id = board.get("board_id", "")
        board_title_zh = board.get("board_title", {}).get("zh", "")
        board_title_en = board.get("board_title", {}).get("en", "")

        for para in board.get("paragraphs", []):
            for sent in para.get("sentences", []):
                zh = sent.get("zh", "").strip()
                en = sent.get("en", "").strip()

                # 跳过空句子
                if not zh or not en:
                    continue

                # 检查是否有完整的五语种翻译
                ja = sent.get("ja", "").strip()
                es = sent.get("es", "").strip()
                bg = sent.get("bg", "").strip()

                # 如果缺少任何翻译，跳过
                if not ja or not es or not bg:
                    continue

                # 验证翻译是否是正确的语言（过滤掉假翻译）
                if not is_valid_japanese(ja):
                    continue
                if not is_valid_spanish(es):
                    continue
                if not is_valid_bulgarian(bg):
                    continue

                length_class = classify_sentence_length(en)
                word_count = len(en.split())

                all_sentences.append({
                    "museum": museum,
                    "board_id": board_id,
                    "board_title_zh": board_title_zh,
                    "board_title_en": board_title_en,
                    "zh": zh,
                    "en": en,
                    "ja": ja,
                    "es": es,
                    "bg": bg,
                    "length_class": length_class,
                    "word_count": word_count,
                })

    return all_sentences


def group_by_museum_and_length(sentences: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    按博物馆和句子长度分组

    返回: {
        "国家博物馆": {
            "short": [...],
            "medium": [...],
            "long": [...]
        },
        ...
    }
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for sent in sentences:
        museum = sent["museum"]
        length = sent["length_class"]
        grouped[museum][length].append(sent)

    return grouped


def select_sentences_by_duration(
    grouped: Dict[str, Dict[str, List[Dict]]],
    target_words: int,
    museum_order: List[str]
) -> Tuple[List[Dict], Dict]:
    """
    按目标词数选择句子，按博物馆顺序排列

    1. 计算各博物馆可用句子的总词数
    2. 按比例分配目标词数
    3. 各博物馆按长度分层选择
    4. 按博物馆顺序排列输出
    """
    # 计算各博物馆可用总词数（应用权重调整）
    museum_word_counts = {}
    for museum, lengths in grouped.items():
        total = sum(sum(s["word_count"] for s in sents) for sents in lengths.values())
        # 应用权重调整
        weight = MUSEUM_CONFIG.get(museum, {}).get("weight_multiplier", 1.0)
        museum_word_counts[museum] = total * weight

    total_available_words = sum(museum_word_counts.values())

    # 按比例分配目标词数
    museum_word_targets = {}
    for museum, available in museum_word_counts.items():
        ratio = available / total_available_words if total_available_words > 0 else 0
        museum_word_targets[museum] = int(target_words * ratio)

    selected = []
    selection_stats = {}
    current_word_total = 0

    print(f"\n目标词数: {target_words} 词 (约 {target_words // EN_WORDS_PER_MINUTE} 分钟)")
    print("\n各博物馆分配：")

    for museum in museum_order:
        if museum not in grouped:
            continue

        museum_sentences = grouped[museum]

        # 获取博物馆特定配置
        museum_cfg = MUSEUM_CONFIG.get(museum, {})
        min_words = museum_cfg.get("min_words", 0)
        length_dist = museum_cfg.get("length_distribution", LENGTH_DISTRIBUTION)

        # 过滤掉词数太少的句子（如照片说明）
        filtered_sentences = {
            length: [s for s in sents if s["word_count"] >= min_words]
            for length, sents in museum_sentences.items()
        }

        available_counts = {
            length: len(sents)
            for length, sents in filtered_sentences.items()
        }
        total_available = sum(available_counts.values())

        if total_available == 0:
            print(f"  {museum}: 过滤后无可用句子（min_words={min_words}）")
            continue

        # 计算该博物馆的目标词数
        word_target = museum_word_targets.get(museum, 0)

        # 按长度分层选择，累计达到词数目标
        museum_selected = []
        museum_word_count = 0

        # 使用博物馆特定的长度分布
        for length, ratio in length_dist.items():
            sentences = filtered_sentences.get(length, [])
            if not sentences:
                continue

            # 该长度类别的目标词数
            length_word_target = word_target * ratio

            # 随机打乱后选择
            random.shuffle(sentences)

            for sent in sentences:
                if museum_word_count >= word_target:
                    break
                museum_selected.append(sent)
                museum_word_count += sent["word_count"]

        # 记录统计
        stats = {"count": len(museum_selected), "words": museum_word_count}
        stats.update({length: sum(1 for s in museum_selected if s["length_class"] == length)
                      for length in ["short", "medium", "long"]})
        selection_stats[museum] = stats

        selected.extend(museum_selected)
        current_word_total += museum_word_count

        # 显示过滤信息
        filter_info = f" (过滤<{min_words}词)" if min_words > 0 else ""
        print(f"  {museum}: {len(museum_selected)}句, {museum_word_count}词{filter_info}")

    print(f"  ─────────────────")
    print(f"  总计: {len(selected)}句, {current_word_total}词 (约 {current_word_total / EN_WORDS_PER_MINUTE:.1f} 分钟)")

    return selected, selection_stats


def format_reading_text(sentences: List[Dict], lang: str, selection_stats: Dict) -> str:
    """
    格式化朗读稿件（按博物馆分组显示）

    格式：
    ========== 国家博物馆（25句）==========
    [1] 句子内容
    [2] 句子内容
    ...
    ========== 故宫（18句）==========
    ...
    """
    lines = []
    lines.append(f"多语种朗读稿件 - {LANGUAGES[lang]['name']}")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_words = sum(s["word_count"] for s in sentences)

    # 根据语言使用不同的语速计算时长
    if lang == "bg":
        minutes = total_words / BG_WORDS_PER_MINUTE
    elif lang == "ja":
        minutes = total_words / JA_WORDS_PER_MINUTE
    else:
        minutes = total_words / EN_WORDS_PER_MINUTE

    lines.append(f"句子总数: {len(sentences)}句 | 总词数: {total_words}词 | 预估时长: {minutes:.1f}分钟")
    lines.append("")
    lines.append("=" * 60)
    lines.append("朗读说明：")
    lines.append("- 请按顺序朗读，每个博物馆的句子数量已明确标注")
    lines.append("- 请以正常语速朗读，每句之间稍作停顿")
    lines.append("- 如遇到不熟悉的词汇，可跳过或标注")
    lines.append("=" * 60)
    lines.append("")

    # 按博物馆分组显示
    current_museum = None
    sentence_num = 0

    for sent in sentences:
        museum = sent.get("museum", "")

        # 新博物馆，添加分隔标题
        if museum != current_museum:
            current_museum = museum
            stats = selection_stats.get(museum, {})
            count = stats.get("count", 0)
            words = stats.get("words", 0)
            lines.append("")
            lines.append(f"{'=' * 60}")
            lines.append(f"  {museum}（{count}句，{words}词）")
            lines.append(f"{'=' * 60}")
            lines.append("")

        sentence_num += 1
        text = sent.get(lang, "")
        lines.append(f"[{sentence_num}] {text}")
        lines.append("")

    return "\n".join(lines)


def format_reading_text_with_context(sentences: List[Dict], lang: str, selection_stats: Dict) -> str:
    """
    格式化朗读稿件（带上下文信息，用于TTS或审核，按博物馆分组）

    格式：
    ========== 国家博物馆（25句）==========
    # [1] 展板标题
    句子内容
    ...
    """
    lines = []
    lines.append(f"多语种朗读稿件 - {LANGUAGES[lang]['name']}")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_words = sum(s["word_count"] for s in sentences)

    # 根据语言使用不同的语速计算时长
    if lang == "bg":
        minutes = total_words / BG_WORDS_PER_MINUTE
    elif lang == "ja":
        minutes = total_words / JA_WORDS_PER_MINUTE
    else:
        minutes = total_words / EN_WORDS_PER_MINUTE

    lines.append(f"句子总数: {len(sentences)}句 | 总词数: {total_words}词 | 预估时长: {minutes:.1f}分钟")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    current_museum = None
    sentence_num = 0

    for sent in sentences:
        museum = sent.get("museum", "")

        # 新博物馆，添加分隔标题
        if museum != current_museum:
            current_museum = museum
            stats = selection_stats.get(museum, {})
            count = stats.get("count", 0)
            words = stats.get("words", 0)
            lines.append("")
            lines.append(f"{'=' * 60}")
            lines.append(f"  {museum}（{count}句，{words}词）")
            lines.append(f"{'=' * 60}")
            lines.append("")

        sentence_num += 1
        text = sent.get(lang, "")
        title_zh = sent.get("board_title_zh", "")

        # 添加来源信息
        lines.append(f"# [{sentence_num}] {title_zh}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def generate_info_json(
    sentences: List[Dict],
    output_path: Path
) -> Dict:
    """
    生成选句信息和统计JSON
    """
    # 统计信息
    museum_stats = defaultdict(lambda: {"count": 0, "short": 0, "medium": 0, "long": 0})
    word_count_total = defaultdict(int)

    for sent in sentences:
        museum = sent["museum"]
        length = sent["length_class"]
        museum_stats[museum]["count"] += 1
        museum_stats[museum][length] += 1
        word_count_total["en"] += sent["word_count"]

        # 统计其他语言的字符/词数
        word_count_total["zh"] += len(sent["zh"])
        word_count_total["ja"] += len(sent["ja"])
        word_count_total["es"] += len(sent["es"].split())
        word_count_total["bg"] += len(sent["bg"].split())

    # 生成句子详情
    sentence_details = []
    for i, sent in enumerate(sentences, 1):
        sentence_details.append({
            "index": i,
            "museum": sent["museum"],
            "board_id": sent["board_id"],
            "board_title_zh": sent["board_title_zh"],
            "length_class": sent["length_class"],
            "word_count_en": sent["word_count"],
            "zh": sent["zh"],
            "en": sent["en"],
            "ja": sent["ja"],
            "es": sent["es"],
            "bg": sent["bg"],
        })

    info = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_sentences": len(sentences),
            "target_minutes_bg": TARGET_MINUTES_BG,
            "target_words": TARGET_WORDS,
            "generation_strategy": "duration_based_selection",
        },
        "museum_distribution": dict(museum_stats),
        "length_distribution": LENGTH_DISTRIBUTION,
        "word_count_stats": dict(word_count_total),
        "estimated_reading_time": {
            "all_languages_minutes": round(word_count_total["en"] / EN_WORDS_PER_MINUTE, 1),
            "note": "各语言朗读时间相近（信息量相同）"
        },
        "sentences": sentence_details,
    }

    # 保存JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n信息文件已保存: {output_path}")

    return info


def main():
    """主函数"""
    print("=" * 60)
    print("多语种朗读稿件生成器")
    print("=" * 60)

    # 设置随机种子以保证可复现
    random.seed(42)

    # 路径配置
    corpus_path = config.PROJECT_ROOT / "交付" / "multilingual_corpus.json"
    output_dir = config.PROJECT_ROOT / "交付" / "朗读稿件"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查语料库文件
    if not corpus_path.exists():
        print(f"错误：语料库文件不存在: {corpus_path}")
        return

    # 加载语料库
    corpus = load_corpus(corpus_path)
    print(f"语料库加载成功，共 {corpus['metadata']['total_boards']} 个展板")

    # 提取所有句子
    print("\n提取句子中...")
    all_sentences = extract_all_sentences(corpus)
    print(f"共提取 {len(all_sentences)} 个完整五语种句子")

    # 统计原始分布
    print("\n原始分布统计：")
    museum_counts = defaultdict(int)
    for sent in all_sentences:
        museum_counts[sent["museum"]] += 1
    for museum, count in sorted(museum_counts.items(), key=lambda x: -x[1]):
        print(f"  {museum}: {count}句")

    # 按博物馆和长度分组
    print("\n按博物馆和句子长度分组...")
    grouped = group_by_museum_and_length(all_sentences)

    # 按目标时长选择句子（30分钟 ≈ 4200词）
    print("\n按目标时长选择句子...")
    selected_sentences, selection_stats = select_sentences_by_duration(
        grouped, TARGET_WORDS, MUSEUM_ORDER
    )

    # 生成各语言的朗读稿件
    print("\n生成朗读稿件...")
    for lang, lang_info in LANGUAGES.items():
        output_path = output_dir / lang_info["filename"]

        # 母语者朗读稿件使用简洁格式，TTS稿件使用带上下文格式
        if lang_info["for_tts"]:
            text = format_reading_text_with_context(selected_sentences, lang, selection_stats)
        else:
            text = format_reading_text(selected_sentences, lang, selection_stats)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  {lang_info['name']}: {output_path}")

    # 生成信息JSON
    info_path = output_dir / "reading_script_info.json"
    info = generate_info_json(selected_sentences, info_path)

    # 打印预估朗读时长（各语言不同）
    total_words = sum(s["word_count"] for s in selected_sentences)
    print("\n预估朗读时长（基于各语言实际语速）：")
    print(f"  保加利亚语: 约 {total_words / BG_WORDS_PER_MINUTE:.1f} 分钟")
    print(f"  日语: 约 {total_words / JA_WORDS_PER_MINUTE:.1f} 分钟")
    print(f"  英语: 约 {total_words / EN_WORDS_PER_MINUTE:.1f} 分钟")

    print("\n" + "=" * 60)
    print("完成！朗读稿件已生成到:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

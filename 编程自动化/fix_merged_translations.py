# -*- coding: utf-8 -*-
"""
修复已合并但缺少翻译的展板
针对 _needs_translation=True 的展板，补全 ja/es/bg 翻译

使用与 multilingual_translate.py 相同的模型配置（多模型博弈 + LLM裁判）
"""
import json
import os
import sys
import io
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

# 编码设置
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
        sys._museum_encoding_set = True
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

MULTI_PATH = r"E:\ai知识库\nlp大赛\中间结果\enhanced\multilingual_corpus.json"

# 需要翻译的语言
TARGET_LANGS = ["ja", "es", "bg"]

# Prompt 模板
TRANSLATOR_SYSTEM_PROMPT = """You are a professional museum exhibit translator.
Translate from Chinese (primary) or English (fallback) to the target language.
Use formal museum style. Output ONLY JSON: {"translations": ["text1", "text2", ...]}
Number of translations MUST match input exactly."""

TRANSLATOR_USER_TEMPLATE = """Museum: {museum}
Target language: {target_lang}

Sentences to translate:
{sentences_block}

Special instructions: {lang_instruction}

Output JSON with {n} translations:"""

JUDGE_SYSTEM_PROMPT = """You are a translation quality evaluator for museum texts.
Select the better translation or synthesize an improved version.
Output ONLY JSON: {"translations": ["best1", "best2", ...]}"""

JUDGE_USER_TEMPLATE = """Original:
ZH: {zh_text}
EN ref: {en_text}

Candidate A ({label_a}):
{trans_a}

Candidate B ({label_b}):
{trans_b}

Select better or synthesize. Output JSON with 1 translation:"""


def make_client(provider: str):
    """创建 API 客户端"""
    try:
        import openai
    except ImportError:
        raise RuntimeError("请安装 openai: pip install openai")

    timeout = cfg.TRANSLATION_CONFIG["timeout"]

    if provider == "302ai":
        return openai.OpenAI(api_key=cfg.AI302_API_KEY, base_url=cfg.AI302_API_URL, timeout=timeout)
    elif provider == "qwen":
        return openai.OpenAI(api_key=cfg.QWEN_API_KEY, base_url=cfg.QWEN_API_URL, timeout=timeout)
    elif provider == "kimi":
        return openai.OpenAI(api_key=cfg.KIMI_API_KEY, base_url=cfg.KIMI_API_URL, timeout=timeout)
    else:
        raise ValueError(f"未知 provider: {provider}")


def extract_json(text: str) -> Optional[Dict]:
    """从响应中提取 JSON"""
    if not text:
        return None
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def call_api(client, model: str, system: str, user: str, provider: str, temperature: float = 0.3) -> Optional[str]:
    """调用 API"""
    if "kimi" in model.lower():
        temperature = 1.0

    if provider == "qwen":
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [Error] API failed: {e}")
                return None
    return None


def translate_sentences_batch(
    sentences: List[Dict],
    lang: str,
    museum: str,
    clients: Dict[str, Any]
) -> List[str]:
    """批量翻译句子（使用多模型博弈）"""
    # 过滤有 zh 或 en 的句子
    valid_sents = [(i, s) for i, s in enumerate(sentences) if s.get("zh", "").strip() or s.get("en", "").strip()]
    if not valid_sents:
        return []

    lang_cfg = cfg.TRANSLATION_MODELS[lang]
    lang_name = cfg.LANG_NAMES[lang]
    lang_instr = cfg.LANG_INSTRUCTIONS[lang]

    # 构建句子列表
    sent_lines = []
    for idx, (i, s) in enumerate(valid_sents):
        zh = s.get("zh", "") or ""
        en = s.get("en", "") or ""
        sent_lines.append(f"{idx+1}. ZH: {zh} | EN: {en}")

    user_prompt = TRANSLATOR_USER_TEMPLATE.format(
        museum=museum,
        target_lang=lang_name,
        sentences_block="\n".join(sent_lines),
        lang_instruction=lang_instr,
        n=len(valid_sents)
    )

    # 并行调用两个翻译模型
    def run_translator(model_cfg: Dict, label: str) -> Optional[List[str]]:
        client = clients[model_cfg["provider"]]
        raw = call_api(client, model_cfg["model"], TRANSLATOR_SYSTEM_PROMPT, user_prompt, model_cfg["provider"])
        if not raw:
            return None
        parsed = extract_json(raw)
        if not parsed or "translations" not in parsed:
            return None
        trans = parsed["translations"]
        if len(trans) != len(valid_sents):
            return None
        return trans

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_translator, lang_cfg["translator_a"], "A")
        fut_b = pool.submit(run_translator, lang_cfg["translator_b"], "B")
        trans_a = fut_a.result()
        trans_b = fut_b.result()

    # 如果两个模型都失败
    if trans_a is None and trans_b is None:
        print(f"    [FAIL] 两个翻译模型均失败")
        return []

    # 如果只有一个成功，直接使用
    if trans_a is None:
        return trans_b
    if trans_b is None:
        return trans_a

    # 两个都成功，用裁判模型选择
    judge_cfg = lang_cfg["judge"]
    final_trans = []

    for idx, (i, s) in enumerate(valid_sents):
        zh = s.get("zh", "") or ""
        en = s.get("en", "") or ""

        judge_user = JUDGE_USER_TEMPLATE.format(
            zh_text=zh,
            en_text=en,
            label_a=lang_cfg["translator_a"]["label"],
            trans_a=trans_a[idx],
            label_b=lang_cfg["translator_b"]["label"],
            trans_b=trans_b[idx]
        )

        client = clients[judge_cfg["provider"]]
        raw = call_api(client, judge_cfg["model"], JUDGE_SYSTEM_PROMPT, judge_user, judge_cfg["provider"])
        if raw:
            parsed = extract_json(raw)
            if parsed and "translations" in parsed and parsed["translations"]:
                final_trans.append(parsed["translations"][0])
                continue

        # 裁判失败，优先使用 A
        final_trans.append(trans_a[idx])

    return final_trans


def main():
    print("=" * 60)
    print("修复已合并展板的翻译")
    print("=" * 60)

    # 加载数据
    with open(MULTI_PATH, encoding="utf-8") as f:
        data = json.load(f)

    boards = data.get("boards", [])

    # 找到需要翻译的展板
    needs_translation_boards = [b for b in boards if b.get("_needs_translation")]

    print(f"\n找到 {len(needs_translation_boards)} 个需要翻译的展板:")
    for b in needs_translation_boards:
        board_id = b.get("board_id", "unknown")
        n_paras = len(b.get("paragraphs", []))
        n_sents = sum(len(p.get("sentences", [])) for p in b.get("paragraphs", []))
        print(f"  - {board_id}: {n_paras} 段落, {n_sents} 句子")

    if not needs_translation_boards:
        print("\n没有需要翻译的展板。")
        return

    # 初始化客户端
    print("\n初始化 API 客户端...")
    clients = {}
    providers = set()
    for lang in TARGET_LANGS:
        lang_cfg = cfg.TRANSLATION_MODELS[lang]
        for role in ["translator_a", "translator_b", "judge"]:
            providers.add(lang_cfg[role]["provider"])

    for provider in providers:
        try:
            clients[provider] = make_client(provider)
            print(f"  ✓ {provider}")
        except Exception as e:
            print(f"  ✗ {provider}: {e}")
            return

    total_translated = {lang: 0 for lang in TARGET_LANGS}

    for board in needs_translation_boards:
        board_id = board.get("board_id", "unknown")
        museum = board.get("source", {}).get("museum", "Museum")
        print(f"\n{'='*60}")
        print(f"处理展板: {board_id}")
        print(f"博物馆: {museum}")

        for para in board.get("paragraphs", []):
            para_idx = para.get("para_index", 0)
            sentences = para.get("sentences", [])

            # 检查哪些语言需要翻译
            for lang in TARGET_LANGS:
                missing_sents = [s for s in sentences if not s.get(lang, "").strip() and (s.get("zh", "").strip() or s.get("en", "").strip())]
                if not missing_sents:
                    continue

                print(f"\n  段落 {para_idx}: 翻译 {len(missing_sents)} 句到 {lang}...")

                translations = translate_sentences_batch(sentences, lang, museum, clients)

                if translations:
                    # 填充翻译结果
                    valid_indices = [i for i, s in enumerate(sentences) if s.get("zh", "").strip() or s.get("en", "").strip()]
                    for trans_idx, sent_idx in enumerate(valid_indices):
                        if trans_idx < len(translations):
                            sentences[sent_idx][lang] = translations[trans_idx]
                            total_translated[lang] += 1
                    print(f"    ✓ 完成 {len(translations)} 条翻译")
                else:
                    print(f"    ✗ 翻译失败")

                time.sleep(0.5)  # 避免 API 限流

        # 完成后移除标记
        if "_needs_translation" in board:
            del board["_needs_translation"]
        print(f"\n  ✓ 展板 {board_id} 处理完成")

    # 更新元数据
    data["metadata"]["updated_at"] = datetime.now().isoformat()

    # 统计
    print("\n" + "=" * 60)
    print("翻译统计:")
    for lang, count in total_translated.items():
        print(f"  {lang}: {count} 条")

    # 保存
    backup = MULTI_PATH.replace(".json", f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(MULTI_PATH):
        import shutil
        shutil.copy(MULTI_PATH, backup)
        print(f"\n备份: {os.path.basename(backup)}")

    with open(MULTI_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存: {MULTI_PATH}")
    print("\n下一步: 运行 python make_clean_corpus.py 生成最终语料库")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
补全缺失的单语种翻译
针对只缺一种语言的句子，补全翻译
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


def translate_single(zh_text: str, en_text: str, target_lang: str, clients: Dict) -> str:
    """翻译单个句子"""
    lang_cfg = cfg.TRANSLATION_MODELS[target_lang]
    lang_name = cfg.LANG_NAMES[target_lang]
    lang_instr = cfg.LANG_INSTRUCTIONS[target_lang]

    # 优先使用 zh，否则用 en
    source_text = zh_text if zh_text.strip() else en_text
    source_lang = "Chinese" if zh_text.strip() else "English"

    system = f"You are a professional museum exhibit translator. Translate to {lang_name}. Use formal museum style. Output ONLY the translation, no explanations."
    user = f"""Source ({source_lang}): {source_text}
Reference English: {en_text}

Instructions: {lang_instr}

Translation:"""

    model_cfg = lang_cfg["translator_a"]  # 使用第一个翻译模型
    client = clients[model_cfg["provider"]]

    # 构建消息
    if model_cfg["provider"] == "qwen":
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(3):
        try:
            temp = 1.0 if "kimi" in model_cfg["model"].lower() else 0.3
            resp = client.chat.completions.create(
                model=model_cfg["model"],
                messages=messages,
                max_tokens=1024,
                temperature=temp,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    [Error] Translation failed: {e}")
                return ""
    return ""


def main():
    print("=" * 60)
    print("补全缺失的单语种翻译")
    print("=" * 60)

    # 加载数据
    with open(MULTI_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # 找出缺失单语种的句子
    missing_sentences = []  # (board_id, para_idx, sent_idx, sent, missing_lang)

    for board in data.get("boards", []):
        board_id = board.get("board_id", "unknown")
        for para in board.get("paragraphs", []):
            para_idx = para.get("para_index", 0)
            for sent in para.get("sentences", []):
                sent_idx = sent.get("sent_index", 0)

                has_zh = bool(sent.get("zh", "").strip())
                has_en = bool(sent.get("en", "").strip())
                has_ja = bool(sent.get("ja", "").strip())
                has_es = bool(sent.get("es", "").strip())
                has_bg = bool(sent.get("bg", "").strip())

                # 检查是否只缺一种语言
                missing = []
                if not has_zh: missing.append("zh")
                if not has_en: missing.append("en")
                if not has_ja: missing.append("ja")
                if not has_es: missing.append("es")
                if not has_bg: missing.append("bg")

                # 只缺一种语言，且有 zh 或 en 作为源
                if len(missing) == 1 and missing[0] in ["ja", "es", "bg"] and (has_zh or has_en):
                    missing_sentences.append((board_id, para_idx, sent_idx, sent, missing[0]))

    print(f"\n找到 {len(missing_sentences)} 个句子只缺一种语言:")
    by_lang = {}
    for _, _, _, _, lang in missing_sentences:
        by_lang[lang] = by_lang.get(lang, 0) + 1
    for lang, count in sorted(by_lang.items()):
        print(f"  缺 {lang}: {count}")

    if not missing_sentences:
        print("\n没有需要翻译的句子。")
        return

    # 初始化客户端
    print("\n初始化 API 客户端...")
    clients = {}
    providers = set()
    for lang in ["ja", "es", "bg"]:
        lang_cfg = cfg.TRANSLATION_MODELS[lang]
        providers.add(lang_cfg["translator_a"]["provider"])

    for provider in providers:
        try:
            clients[provider] = make_client(provider)
            print(f"  [OK] {provider}")
        except Exception as e:
            print(f"  [FAIL] {provider}: {e}")
            return

    # 翻译
    success_count = 0
    for i, (board_id, para_idx, sent_idx, sent, target_lang) in enumerate(missing_sentences):
        zh = sent.get("zh", "") or ""
        en = sent.get("en", "") or ""

        print(f"[{i+1}/{len(missing_sentences)}] {board_id} p{para_idx}s{sent_idx} -> {target_lang}...", end=" ")

        translation = translate_single(zh, en, target_lang, clients)
        if translation:
            sent[target_lang] = translation
            print("[OK]")
            success_count += 1
        else:
            print("[FAIL]")

        time.sleep(0.3)  # 避免 API 限流

    # 更新元数据
    data["metadata"]["updated_at"] = datetime.now().isoformat()

    # 保存
    print("\n" + "=" * 60)
    print(f"翻译完成: {success_count}/{len(missing_sentences)}")

    backup = MULTI_PATH.replace(".json", f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(MULTI_PATH):
        import shutil
        shutil.copy(MULTI_PATH, backup)
        print(f"备份: {os.path.basename(backup)}")

    with open(MULTI_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存: {MULTI_PATH}")
    print("\n下一步: 运行 python make_clean_corpus.py 生成最终语料库")


if __name__ == "__main__":
    main()

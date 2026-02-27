#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语种翻译模块：多模型博弈翻译（Multi-Model Ensemble + LLM-as-Judge）

为 enhanced_corpus.json 中的每个句对添加日语（ja）、保加利亚语（bg）、西班牙语（es）翻译，
构成五语种平行语料库。

架构：
  每种目标语言 → 2 个翻译模型独立翻译 → 1 个裁判模型综合选出最优或合成更好版本

  zh_sentence ──┬──► Translator A ──► translation_A ──┐
                └──► Translator B ──► translation_B ──┤──► Judge ──► final_translation
                (+ en_ref for context)                 └────────────────────────────────┘

模型分配（按排行榜 §2.2）：
  日语（JA）：翻译A=Gemini3-Flash, 翻译B=Qwen-MT-Turbo, 裁判=Gemini3-Flash
  西班牙语（ES）：翻译A=Kimi-K2.5, 翻译B=Gemini3-Flash, 裁判=Gemini3-Flash
  保加利亚语（BG）：翻译A=Claude-Sonnet, 翻译B=Gemini3-Flash, 裁判=Claude-Sonnet

用法：
  python multilingual_translate.py --lang ja            # 翻译日语
  python multilingual_translate.py --lang es            # 翻译西班牙语
  python multilingual_translate.py --lang bg            # 翻译保加利亚语
  python multilingual_translate.py --lang all           # 翻译全部三种语言
  python multilingual_translate.py --lang ja --test 5   # 测试前5个段落
  python multilingual_translate.py --stats              # 查看进度
  python multilingual_translate.py --assemble           # 从进度文件组装输出
"""

import json
import sys
import io
import re
import os
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

# 编码
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
        sys._museum_encoding_set = True
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ==================== Prompt 模板 ====================

TRANSLATOR_SYSTEM_PROMPT = """\
You are a professional museum exhibit translator specializing in Chinese cultural heritage texts.

PRIMARY SOURCE: Chinese text. Translate FROM Chinese.
REFERENCE ONLY: English text (museum's own simplified translation; use for proper noun guidance only).
Register: Formal, neutral museum-guide style.
Output: ONLY the translation JSON, no explanations.
Sentence count in output MUST match input exactly.\
"""

JUDGE_SYSTEM_PROMPT = """\
You are a professional translation evaluator for museum exhibit texts.
Your task: given the original texts and two candidate translations, select the better one
or synthesize an improved version combining the strengths of both.
Output ONLY the final translation JSON — no explanations, no commentary.\
"""

TRANSLATOR_USER_TEMPLATE = """\
Museum: {museum_name} | Board: {board_title_zh}

== PARAGRAPH CONTEXT ==
ZH paragraph: {paragraph_zh}
EN reference: {paragraph_en}

== TASK: Translate each sentence to {target_language} ==

Glossary (use consistently):
{glossary_snippet}

Special instruction: {lang_instruction}

Sentences to translate:
{sentences_block}

Output JSON (sentence count MUST be {n_sentences}):
{{"translations": ["translation_1", "translation_2", ...]}}\
"""

JUDGE_USER_TEMPLATE = """\
Museum: {museum_name} | Board: {board_title_zh}

== SOURCE TEXTS ==
ZH paragraph: {paragraph_zh}
EN reference: {paragraph_en}

== CANDIDATE TRANSLATIONS ({target_language}) ==
[Candidate A — {label_a}]
{candidate_a_block}

[Candidate B — {label_b}]
{candidate_b_block}

== EVALUATION CRITERIA ==
1. Accuracy (faithfulness to Chinese source)
2. Fluency (natural {target_language})
3. Register (formal museum style)
4. Terminology (proper nouns handled correctly)

Select the better translation for each sentence, or synthesize an improved version.
Output ONLY a JSON with exactly {n_sentences} strings:
{{"translations": ["best_1", "best_2", ...]}}\
"""


# ==================== JSON 提取 ====================

def extract_json(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON（兼容 ```json 代码块和裸 JSON）"""
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


def extract_translations(response_text: str, expected_count: int) -> Optional[List[str]]:
    """从响应中提取翻译列表，校验数量是否匹配"""
    parsed = extract_json(response_text)
    if not parsed:
        return None
    translations = parsed.get("translations", [])
    if not isinstance(translations, list):
        return None
    if len(translations) != expected_count:
        return None
    if not all(isinstance(t, str) and t.strip() for t in translations):
        return None
    return [t.strip() for t in translations]


# ==================== API 客户端工厂 ====================

def make_client(provider: str):
    """根据 provider 创建 OpenAI 兼容客户端"""
    try:
        import openai
    except ImportError:
        raise RuntimeError("请安装 openai: pip install openai")

    timeout = cfg.TRANSLATION_CONFIG["timeout"]

    if provider == "302ai":
        if not cfg.AI302_API_KEY:
            raise RuntimeError("AI302_API_KEY 未设置（请设置环境变量 AI302_API_KEY 或在 config.py 填写）")
        return openai.OpenAI(
            api_key=cfg.AI302_API_KEY,
            base_url=cfg.AI302_API_URL,
            timeout=timeout,
        )
    elif provider == "qwen":
        return openai.OpenAI(
            api_key=cfg.QWEN_API_KEY,
            base_url=cfg.QWEN_API_URL,
            timeout=timeout,
        )
    elif provider == "kimi":
        return openai.OpenAI(
            api_key=cfg.KIMI_API_KEY,
            base_url=cfg.KIMI_API_URL,
            timeout=timeout,
        )
    else:
        raise ValueError(f"未知 provider: {provider}")


# ==================== 术语表加载 ====================

def load_glossary() -> Dict:
    """加载 museum_glossary.json"""
    gf = cfg.MUSEUM_GLOSSARY_FILE
    if gf.exists():
        with open(gf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("glossary", {})
    return {}


def build_glossary_snippet(glossary: Dict, lang: str, max_terms: int = 30) -> str:
    """将术语表转换为适合注入 prompt 的字符串"""
    lines = []
    for category, terms in glossary.items():
        for zh_key, translations in terms.items():
            target = translations.get(lang, "")
            en = translations.get("en", "")
            if target:
                lines.append(f"  {zh_key} → {target} (EN: {en})")
            if len(lines) >= max_terms:
                break
        if len(lines) >= max_terms:
            break
    return "\n".join(lines) if lines else "  (no glossary terms)"


# ==================== 单次 API 调用 ====================

def call_api_once(client, model: str, system: str, user: str,
                  max_tokens: int = 4096, temperature: float = 0.3,
                  provider: str = "") -> Optional[str]:
    """调用 API，带指数退避重试，返回原始文本"""
    # Kimi K2.5 强制 temperature=1
    if "kimi" in model.lower():
        temperature = 1.0

    # Qwen 系列不支持 system 角色，将 system 并入 user 消息
    if provider == "qwen":
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    for attempt in range(cfg.TRANSLATION_CONFIG["retry_max"]):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = cfg.TRANSLATION_CONFIG["retry_base_wait"] ** attempt
            print(f"    [RETRY {attempt+1}/{cfg.TRANSLATION_CONFIG['retry_max']}] {e}, 等待{wait}s...",
                  flush=True)
            time.sleep(wait)
    return None


# ==================== EN→target 兜底翻译（空ZH句专用）====================

def translate_en_fallback(
    en_texts: List[str],
    lang: str,
    lang_name: str,
    lang_instr: str,
    client,
    model_cfg: Dict,
) -> List[str]:
    """对 ZH 为空的句子，用 EN 参考译文翻译到目标语言（EN→target）。
    返回与 en_texts 等长的译文列表；失败时返回原英文作为占位。
    """
    if not en_texts:
        return []
    n = len(en_texts)
    lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(en_texts))
    system = (
        f"You are a professional museum exhibit translator. "
        f"Translate the following English sentences to {lang_name}. "
        f"Special instruction: {lang_instr} "
        f"Output ONLY JSON: {{\"translations\": [\"trans_1\", ...]}}, "
        f"exactly {n} items."
    )
    user = f"Sentences to translate to {lang_name}:\n{lines}\n\nOutput JSON (exactly {n} items):"
    provider = model_cfg["provider"]
    for _ in range(2):
        raw = call_api_once(client, model_cfg["model"], system, user, provider=provider)
        if raw:
            result = extract_translations(raw, n)
            if result:
                return result
    # 兜底：返回原英文（保证不缺位）
    return list(en_texts)


# ==================== 段落级翻译 ====================

def translate_paragraph(
    para: Dict,
    lang: str,
    board_info: Dict,
    glossary: Dict,
    clients: Dict[str, Any],
) -> Optional[Dict]:
    """翻译单个段落（段落内所有句子批量处理）。

    Returns:
        {
            "final": ["译文1", "译文2", ...],
            "_candidates": {"a": [...], "b": [...]},
            "_judge_model": "...",
            "_translator_a": "...",
            "_translator_b": "...",
        }
        或 None（失败）
    """
    sentences = para.get("sentences", [])
    if not sentences:
        return None

    n = len(sentences)

    # ── 预处理：分离空ZH句 ──────────────────────────────────────────
    active_idx   = [i for i, s in enumerate(sentences) if s.get("zh", "").strip()]
    empty_zh_idx = [i for i, s in enumerate(sentences) if not s.get("zh", "").strip()]

    if not active_idx:
        return None  # 全空，无法翻译

    active_sents = [sentences[i] for i in active_idx]
    n_active = len(active_sents)  # 实际用于主翻译的句数
    # ────────────────────────────────────────────────────────────────

    lang_cfg = cfg.TRANSLATION_MODELS[lang]
    lang_name = cfg.LANG_NAMES[lang]
    lang_instr = cfg.LANG_INSTRUCTIONS[lang]

    museum = board_info.get("museum", "")
    board_title_zh = board_info.get("board_title_zh", "")
    para_zh = para.get("zh", "")
    para_en = para.get("en", "")
    glossary_snippet = build_glossary_snippet(glossary, lang)

    # 构造句子列表文本（只含有ZH内容的句子）
    sent_lines = []
    for i, s in enumerate(active_sents):
        zh_text = s.get("zh", "")
        en_text = s.get("en", "")
        sent_lines.append(f"{i+1}. ZH: {zh_text} | EN ref: {en_text}")
    sentences_block = "\n".join(sent_lines)

    # Translator prompt（n_sentences 用 n_active，而非原始 n）
    translator_user = TRANSLATOR_USER_TEMPLATE.format(
        museum_name=museum,
        board_title_zh=board_title_zh,
        paragraph_zh=para_zh,
        paragraph_en=para_en,
        target_language=lang_name,
        glossary_snippet=glossary_snippet,
        lang_instruction=lang_instr,
        sentences_block=sentences_block,
        n_sentences=n_active,
    )

    # ---- 并行调用 Translator A + B ----
    model_a_cfg = lang_cfg["translator_a"]
    model_b_cfg = lang_cfg["translator_b"]

    def run_translator(model_cfg: Dict, label: str) -> Optional[List[str]]:
        client = clients[model_cfg["provider"]]
        model = model_cfg["model"]
        provider = model_cfg["provider"]
        for attempt in range(2):
            raw = call_api_once(client, model, TRANSLATOR_SYSTEM_PROMPT, translator_user,
                                provider=provider)
            if not raw:
                continue
            result = extract_translations(raw, n_active)
            if result:
                return result
            # 数量不对，追加修正消息后再试
            if attempt == 0:
                print(f"      [WARN] {label} 翻译数量不匹配（期望{n_active}），尝试修正...", flush=True)
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_translator, model_a_cfg, "TransA")
        fut_b = pool.submit(run_translator, model_b_cfg, "TransB")
        trans_a = fut_a.result()
        trans_b = fut_b.result()

    # 处理 A/B 失败情况
    if trans_a is None and trans_b is None:
        print("      [FAIL] 两个翻译模型均失败", flush=True)
        return None

    def _apply_merge(result: Dict) -> Dict:
        """将空ZH句的EN兜底译文合并回结果（按原始位置）"""
        if not empty_zh_idx:
            return result
        empty_en_texts = [sentences[i].get("en", "") for i in empty_zh_idx]
        model_b_cfg_fb = lang_cfg["translator_b"]
        empty_translations = translate_en_fallback(
            empty_en_texts, lang, lang_name, lang_instr,
            clients[model_b_cfg_fb["provider"]], model_b_cfg_fb,
        )
        merged_final = [""] * n
        for pos, trans in zip(active_idx, result["final"]):
            merged_final[pos] = trans
        for pos, trans in zip(empty_zh_idx, empty_translations):
            merged_final[pos] = trans
        result["final"] = merged_final
        if result["_candidates"]["a"]:
            merged_a = [""] * n
            for pos, t in zip(active_idx, result["_candidates"]["a"]):
                merged_a[pos] = t
            result["_candidates"]["a"] = merged_a
        if result["_candidates"]["b"]:
            merged_b = [""] * n
            for pos, t in zip(active_idx, result["_candidates"]["b"]):
                merged_b[pos] = t
            result["_candidates"]["b"] = merged_b
        return result

    if trans_a is None:
        print("      [WARN] TransA 失败，直接使用 TransB 结果", flush=True)
        return _apply_merge({
            "final": trans_b,
            "_candidates": {"a": None, "b": trans_b},
            "_judge_model": "skipped",
            "_translator_a": model_a_cfg["label"],
            "_translator_b": model_b_cfg["label"],
        })
    if trans_b is None:
        print("      [WARN] TransB 失败，直接使用 TransA 结果", flush=True)
        return _apply_merge({
            "final": trans_a,
            "_candidates": {"a": trans_a, "b": None},
            "_judge_model": "skipped",
            "_translator_a": model_a_cfg["label"],
            "_translator_b": model_b_cfg["label"],
        })

    # ---- 调用 Judge ----
    judge_cfg = lang_cfg["judge"]
    judge_client = clients[judge_cfg["provider"]]

    def fmt_candidate(translations: List[str]) -> str:
        return "\n".join(f"{i+1}. {t}" for i, t in enumerate(translations))

    judge_user = JUDGE_USER_TEMPLATE.format(
        museum_name=museum,
        board_title_zh=board_title_zh,
        paragraph_zh=para_zh,
        paragraph_en=para_en,
        target_language=lang_name,
        label_a=model_a_cfg["label"],
        candidate_a_block=fmt_candidate(trans_a),
        label_b=model_b_cfg["label"],
        candidate_b_block=fmt_candidate(trans_b),
        n_sentences=n_active,
    )

    final_translations = None
    for attempt in range(2):
        raw_judge = call_api_once(judge_client, judge_cfg["model"],
                                  JUDGE_SYSTEM_PROMPT, judge_user,
                                  provider=judge_cfg["provider"])
        if raw_judge:
            final_translations = extract_translations(raw_judge, n_active)
            if final_translations:
                break
        print(f"      [WARN] Judge 失败/数量不匹配 (attempt {attempt+1})", flush=True)

    if not final_translations:
        # Judge 失败：用 trans_a 作为保底
        print("      [WARN] Judge 失败，使用 TransA 结果作为保底", flush=True)
        final_translations = trans_a

    result = {
        "final": final_translations,
        "_candidates": {"a": trans_a, "b": trans_b},
        "_judge_model": judge_cfg["label"],
        "_translator_a": model_a_cfg["label"],
        "_translator_b": model_b_cfg["label"],
    }

    return _apply_merge(result)


# ==================== 主处理类 ====================

class MultilingualTranslator:
    def __init__(self, langs: List[str], test_limit: int = 0,
                 workers: int = 5, assemble_only: bool = False):
        self.langs = langs
        self.test_mode = test_limit > 0
        self.test_limit = test_limit
        self.workers = workers
        self.assemble_only = assemble_only
        self._lock = threading.Lock()

        self.corpus: Dict = {}          # enhanced_corpus.json 全量数据
        self.glossary: Dict = {}        # museum_glossary.json
        self.progress: Dict = {}        # translation_progress.json
        self.clients: Dict = {}         # provider → openai.OpenAI 客户端

        self.stats: Dict[str, Dict] = {lang: {"ok": 0, "fail": 0} for lang in langs}

    # ---------- 初始化 ----------

    def setup_clients(self) -> bool:
        """初始化所有需要的 API 客户端"""
        needed_providers: set = set()
        for lang in self.langs:
            lang_cfg = cfg.TRANSLATION_MODELS[lang]
            for role in ["translator_a", "translator_b", "judge"]:
                needed_providers.add(lang_cfg[role]["provider"])

        print(f"[INFO] 需要的 provider: {needed_providers}")

        for provider in needed_providers:
            try:
                client = make_client(provider)
                self.clients[provider] = client
                print(f"[OK] {provider} 客户端已初始化")
            except Exception as e:
                print(f"[ERROR] {provider} 客户端初始化失败: {e}")
                if provider == "302ai":
                    print("       → 请设置环境变量 AI302_API_KEY 或在 config.py 中填写 AI302_API_KEY")
                    return False
                # 非 302ai 提供商失败时继续（允许部分降级）
        return True

    def load_data(self) -> bool:
        """加载 enhanced_corpus.json 和 museum_glossary.json"""
        if not cfg.ENHANCED_CORPUS_FILE.exists():
            print(f"[ERROR] enhanced_corpus.json 不存在: {cfg.ENHANCED_CORPUS_FILE}")
            return False
        with open(cfg.ENHANCED_CORPUS_FILE, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        n_boards = len(self.corpus.get("boards", []))
        print(f"[OK] 加载语料库: {n_boards} 个展板")

        self.glossary = load_glossary()
        print(f"[OK] 加载术语表: {sum(len(v) for v in self.glossary.values())} 条术语")
        return True

    def load_progress(self):
        """加载断点进度"""
        cfg.MULTILINGUAL_DIR.mkdir(parents=True, exist_ok=True)
        if cfg.TRANSLATION_PROGRESS_FILE.exists():
            with open(cfg.TRANSLATION_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            done = sum(
                len(self.progress.get("completed", {}).get(lang, {}))
                for lang in self.langs
            )
            print(f"[RESUME] 各语言已完成段落: {done} 个")
        else:
            self.progress = {
                "metadata": {
                    "started_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "langs": self.langs,
                },
                "completed": {lang: {} for lang in ["ja", "es", "bg"]},
                "failed":    {lang: [] for lang in ["ja", "es", "bg"]},
            }

    def save_progress(self):
        """保存进度（原子写入）"""
        self.progress["metadata"]["last_updated"] = datetime.now().isoformat()
        tmp = cfg.MULTILINGUAL_DIR / "translation_progress.tmp.json"
        final = cfg.TRANSLATION_PROGRESS_FILE
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)
        if final.exists():
            final.unlink()
        tmp.rename(final)

    # ---------- 队列构建 ----------

    def build_queue(self, lang: str) -> List[Dict]:
        """构建待翻译队列（段落级）"""
        completed_ids = set(self.progress.get("completed", {}).get(lang, {}).keys())
        failed_ids    = set(self.progress.get("failed", {}).get(lang, []))
        queue = []

        for board in self.corpus.get("boards", []):
            board_id = board.get("board_id", "")
            museum   = board.get("source", {}).get("museum", "")
            title_zh = board.get("board_title", {})
            if isinstance(title_zh, dict):
                title_zh = title_zh.get("zh", "")

            for para in board.get("paragraphs", []):
                para_index = para.get("para_index", 0)
                item_id = f"{board_id}__p{para_index}"

                if item_id in completed_ids or item_id in failed_ids:
                    continue

                sentences = para.get("sentences", [])
                if not sentences:
                    continue

                queue.append({
                    "item_id":        item_id,
                    "board_id":       board_id,
                    "para_index":     para_index,
                    "museum":         museum,
                    "board_title_zh": title_zh,
                    "paragraph_zh":   para.get("zh", ""),
                    "paragraph_en":   para.get("en", ""),
                    "sentences":      sentences,
                })

        return queue

    # ---------- 处理单条（线程池使用）----------

    def _process_one(self, i: int, total: int, item: Dict, lang: str) -> Tuple[str, Optional[Dict]]:
        item_id = item["item_id"]
        museum  = item["museum"]
        n_sents = len(item["sentences"])
        print(f"  [{i+1}/{total}] {item_id} ({museum}) [{n_sents}句]", flush=True)

        time.sleep(cfg.TRANSLATION_CONFIG["rate_limit_delay"])

        para = {
            "zh":        item["paragraph_zh"],
            "en":        item["paragraph_en"],
            "sentences": item["sentences"],
        }
        board_info = {
            "museum":         item["museum"],
            "board_title_zh": item["board_title_zh"],
        }

        result = translate_paragraph(para, lang, board_info, self.glossary, self.clients)
        return item_id, result

    # ---------- 语言处理主流程 ----------

    def run_lang(self, lang: str):
        """翻译单种目标语言"""
        lang_name = cfg.LANG_NAMES[lang]
        lang_models = cfg.TRANSLATION_MODELS[lang]

        print(f"\n{'='*70}")
        print(f"翻译语言: {lang_name} ({lang.upper()})")
        print(f"  TransA: {lang_models['translator_a']['label']}")
        print(f"  TransB: {lang_models['translator_b']['label']}")
        print(f"  Judge:  {lang_models['judge']['label']}")
        print(f"{'='*70}")

        queue = self.build_queue(lang)
        total = min(len(queue), self.test_limit) if self.test_mode else len(queue)
        print(f"待处理: {total} 个段落" + (" (测试模式)" if self.test_mode else ""))

        if total == 0:
            print("[OK] 无需处理")
            return

        items = queue[:total]
        failed_this_run = []

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._process_one, i, total, item, lang): item
                for i, item in enumerate(items)
            }

            for future in as_completed(futures):
                try:
                    item_id, result = future.result()
                except Exception as e:
                    item = futures[future]
                    item_id = item["item_id"]
                    print(f"  [ERROR] {item_id}: {e}", flush=True)
                    result = None

                with self._lock:
                    if result:
                        self.progress["completed"].setdefault(lang, {})[item_id] = {
                            "translated_at": datetime.now().isoformat(),
                            "result": result,
                        }
                        self.stats[lang]["ok"] += 1
                        n_final = len(result["final"])
                        print(f"  [OK] {item_id}: {n_final}句 "
                              f"A={result['_translator_a']} "
                              f"B={result['_translator_b']} "
                              f"J={result['_judge_model']}", flush=True)
                    else:
                        if item_id not in self.progress.get("failed", {}).get(lang, []):
                            self.progress.setdefault("failed", {}).setdefault(lang, []).append(item_id)
                        failed_this_run.append(item_id)
                        self.stats[lang]["fail"] += 1
                        print(f"  [FAIL] {item_id}", flush=True)

                    self.save_progress()

        print(f"\n[{lang.upper()}] 完成: {self.stats[lang]['ok']}, 失败: {self.stats[lang]['fail']}")
        if failed_this_run:
            print(f"[{lang.upper()}] 失败列表: {failed_this_run}")

    # ---------- 组装输出 ----------

    def assemble_corpus(self):
        """从进度文件组装 multilingual_corpus.json"""
        print(f"\n{'='*70}")
        print("组装五语种平行语料库...")

        completed = self.progress.get("completed", {})
        total_sents_with_translation = {lang: 0 for lang in ["ja", "es", "bg"]}
        total_sents = 0

        # 深拷贝 boards，向 sentences 注入翻译
        import copy
        output_boards = []

        for board in self.corpus.get("boards", []):
            board_id = board.get("board_id", "")
            new_board = copy.deepcopy(board)

            for para in new_board.get("paragraphs", []):
                para_index = para.get("para_index", 0)
                item_id = f"{board_id}__p{para_index}"
                sentences = para.get("sentences", [])
                total_sents += len(sentences)

                # 初始化 _translation_meta（段落级）
                para.setdefault("_translation_meta", {})

                for lang in ["ja", "es", "bg"]:
                    lang_completed = completed.get(lang, {})
                    if item_id in lang_completed:
                        result = lang_completed[item_id]["result"]
                        final_list = result.get("final", [])
                        candidates = result.get("_candidates", {})
                        judge_model = result.get("_judge_model", "")
                        trans_a_label = result.get("_translator_a", "")
                        trans_b_label = result.get("_translator_b", "")

                        # 向每个 sentence 注入对应的翻译
                        for j, sent in enumerate(sentences):
                            if j < len(final_list):
                                sent[lang] = final_list[j]
                                # 附候选（可供人工验证）
                                if "_candidates" not in sent:
                                    sent["_candidates"] = {}
                                sent["_candidates"][lang] = {
                                    "a": candidates.get("a", [None])[j] if candidates.get("a") and j < len(candidates["a"]) else None,
                                    "b": candidates.get("b", [None])[j] if candidates.get("b") and j < len(candidates["b"]) else None,
                                }
                                total_sents_with_translation[lang] += 1
                            else:
                                sent[lang] = ""

                        # 段落级 meta
                        para["_translation_meta"][lang] = {
                            "translator_a": trans_a_label,
                            "translator_b": trans_b_label,
                            "judge": judge_model,
                        }
                    else:
                        # 未翻译：留空
                        for sent in sentences:
                            sent.setdefault(lang, "")

            output_boards.append(new_board)

        # 构造输出文件
        original_meta = self.corpus.get("metadata", {})
        output = {
            "metadata": {
                "name": "中国博物馆多语种解说词平行语料库（五语种）",
                "version": "5.0",
                "created_at": datetime.now().isoformat(),
                "processor": "multilingual_translate.py — Multi-Model Ensemble + LLM-as-Judge",
                "source_version": original_meta.get("version", ""),
                "total_boards": len(output_boards),
                "total_paragraphs": original_meta.get("total_paragraphs", 0),
                "total_sentence_pairs": original_meta.get("total_sentence_pairs", 0),
                "languages": ["zh", "en", "ja", "es", "bg"],
                "translation_coverage": {
                    lang: total_sents_with_translation[lang]
                    for lang in ["ja", "es", "bg"]
                },
                "translation_models": {
                    lang: {
                        "translator_a": cfg.TRANSLATION_MODELS[lang]["translator_a"]["label"],
                        "translator_b": cfg.TRANSLATION_MODELS[lang]["translator_b"]["label"],
                        "judge":        cfg.TRANSLATION_MODELS[lang]["judge"]["label"],
                    }
                    for lang in ["ja", "es", "bg"]
                },
                "failed": {
                    lang: len(self.progress.get("failed", {}).get(lang, []))
                    for lang in ["ja", "es", "bg"]
                },
            },
            "boards": output_boards,
        }

        cfg.MULTILINGUAL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cfg.MULTILINGUAL_DIR / "multilingual_corpus.tmp.json"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        if cfg.MULTILINGUAL_CORPUS_FILE.exists():
            backup = cfg.MULTILINGUAL_DIR / f"multilingual_corpus.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cfg.MULTILINGUAL_CORPUS_FILE.rename(backup)
            print(f"[OK] 备份旧版: {backup.name}")
        tmp.rename(cfg.MULTILINGUAL_CORPUS_FILE)

        print(f"[OK] 五语种语料库已保存: {cfg.MULTILINGUAL_CORPUS_FILE}")
        print(f"     展板: {len(output_boards)}, 句对: {original_meta.get('total_sentence_pairs', 0)}")
        print(f"     翻译覆盖: JA={total_sents_with_translation['ja']}, "
              f"ES={total_sents_with_translation['es']}, "
              f"BG={total_sents_with_translation['bg']}")

    # ---------- 统计 ----------

    def print_stats(self):
        print(f"\n{'='*70}")
        print("翻译统计")
        print(f"{'='*70}")
        completed = self.progress.get("completed", {})
        failed    = self.progress.get("failed", {})
        for lang in ["ja", "es", "bg"]:
            n_done = len(completed.get(lang, {}))
            n_fail = len(failed.get(lang, []))
            lang_name = cfg.LANG_NAMES.get(lang, lang)
            print(f"  {lang.upper()} ({lang_name}): 完成 {n_done}, 失败 {n_fail}")
        print(f"{'='*70}")

    # ---------- 主入口 ----------

    def run(self):
        print("=" * 70)
        print("多语种翻译：Multi-Model Ensemble + LLM-as-Judge")
        print(f"目标语言: {', '.join(self.langs)}")
        print(f"并发线程: {self.workers}")
        print("=" * 70)

        if not self.load_data():
            return
        if not self.setup_clients():
            return
        self.load_progress()

        if self.assemble_only:
            self.assemble_corpus()
            self.print_stats()
            return

        for lang in self.langs:
            self.run_lang(lang)

        self.assemble_corpus()
        self.print_stats()


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="多语种翻译：Multi-Model Ensemble + LLM-as-Judge"
    )
    parser.add_argument(
        "--lang", type=str, default="all",
        choices=["ja", "es", "bg", "all"],
        help="目标语言（ja/es/bg/all，默认 all）",
    )
    parser.add_argument(
        "--test", type=int, metavar="N", default=0,
        help="测试模式：只处理前 N 个段落",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="并发线程数（默认 5）",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="显示翻译进度统计",
    )
    parser.add_argument(
        "--assemble", action="store_true",
        help="仅从进度文件组装输出，不发起新翻译",
    )
    args = parser.parse_args()

    langs = ["ja", "es", "bg"] if args.lang == "all" else [args.lang]

    translator = MultilingualTranslator(
        langs=langs,
        test_limit=args.test,
        workers=args.workers,
        assemble_only=args.assemble,
    )

    if args.stats:
        translator.load_data()
        translator.load_progress()
        translator.print_stats()
        return

    translator.run()


if __name__ == "__main__":
    main()

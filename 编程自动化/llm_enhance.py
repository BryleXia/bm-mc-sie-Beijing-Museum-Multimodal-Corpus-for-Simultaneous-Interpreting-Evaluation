#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM增强模块：OCR纠错 + 段落分割 + 句对齐
使用 DeepSeek V3 API 一次调用完成：
  1. 纠正中英文 OCR 错误（词语粘连、字母错误、形近字）
  2. 将展板文本分割为段落
  3. 对齐中英段落
  4. 段落内句子对齐
  5. 输出层级结构: board → paragraphs → sentences

用法:
  python llm_enhance.py                # 处理全部
  python llm_enhance.py --test 3       # 测试前3条
  python llm_enhance.py --resume       # 从断点恢复
  python llm_enhance.py --stats        # 查看进度
  python llm_enhance.py --assemble     # 只从 progress 组装最终文件
  python llm_enhance.py --rerun-uncleaned  # 重跑未人工清洗的展板（保留已清洗的）
"""

import json
import sys
import io
import re
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# 编码
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys._museum_encoding_set = True
    except Exception:
        pass

# 配置
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# ==================== Prompt 模板 ====================

SYSTEM_PROMPT = """You are a corpus cleaning assistant for museum exhibit board text captured by OCR.

Your job is STRICTLY LIMITED to:
1. Fixing clear OCR errors (and ONLY OCR errors)
2. Splitting the text into paragraphs and aligning Chinese-English paragraph pairs
3. Splitting paragraphs into sentences and aligning Chinese-English sentence pairs

You MUST output valid JSON and nothing else. No markdown, no explanations.

## ABSOLUTE RULES — TEXT PRESERVATION (VIOLATION = TOTAL FAILURE)
- This is a CORPUS CLEANING task, NOT translation, NOT completion, NOT rewriting.
- NEVER add any text that does not appear in the OCR input. This is the #1 rule.
- NEVER translate Chinese to fill in "missing" English text, or vice versa.
- NEVER expand fragments or incomplete sentences. If the OCR captured a fragment, keep it as a fragment.
- The English text on museum exhibit boards is often SHORTER and more concise than the Chinese. This is NORMAL and INTENTIONAL — do NOT "complete" it.
- "missing_text" is NOT a valid correction type. You CANNOT add text the OCR didn't capture.
- The output text must be CHARACTER-FOR-CHARACTER identical to the input, EXCEPT for genuine OCR error fixes.
- Every single character-level difference between input and output MUST be listed in "corrections".
- If you are not confident something is an OCR error, leave it UNCHANGED.

## OCR Error Patterns (fix ONLY these — nothing else)
- English: concatenated words ("theParty" → "the Party") → type: word_split
- English: letter substitutions from OCR misread ("thitough" → "through") → type: letter_substitution
- English: case errors from OCR ("cpC" → "CPC") → type: case_error
- Chinese: form-similar character confusion (土↔士, 己↔已, 未↔末, 人↔入) → type: form_similar
- Chinese: OCR-specific museum vocabulary errors (出士→出土, 博勿院→博物院) → type: ocr_specific

## WHAT IS NOT AN OCR ERROR (NEVER "fix" these)
- Missing sentences or paragraphs — the board may genuinely have less English text
- Incomplete translations — Chinese museum boards often have abridged English versions
- Abrupt endings — the OCR captured what was on the physical board
- Grammatical issues in the original text — not your job to fix

## PARAGRAPH AND SENTENCE ALIGNMENT
- Split into logical paragraphs (most boards have 1-6 paragraphs).
- Each Chinese paragraph pairs with exactly one English paragraph (1:1).
- If English is shorter, some later paragraphs may have en: "" (empty string). This is correct.
- Within each paragraph, split into sentence pairs. Each Chinese sentence pairs with one English sentence.
- If a Chinese sentence has NO corresponding English, set en: "" for that sentence pair. Do NOT fabricate English.
- Sentence boundary: 。！？ (Chinese) or . ! ? (English).

## board_title
- Extract ONLY if there is an obvious standalone title/header clearly separate from body text.
- If no clear title, set both zh and en to empty strings.
- Title text must come verbatim from the input — do not invent titles."""

OCR_HINT = """Chinese OCR confusions: 土↔士, 己↔已↔巳, 未↔末, 人↔入, 天↔无↔夫, 大↔太↔犬, 今↔令, 出士→出土, 青銅→青铜, 展覧→展览
English OCR confusions: word concatenation (split them), bronse→bronze, ceramicc→ceramic, dinasty→dynasty, peroid→period, exhibtion→exhibition"""


def build_full_prompt(zh_text: str, en_text: str) -> str:
    return f"""Clean and structure this museum exhibit board text.
Task: (1) Fix ONLY clear OCR errors. (2) Split into paragraphs and align zh-en pairs. (3) Split each paragraph into sentence pairs.

CRITICAL REMINDERS:
- Keep ALL text VERBATIM — only fix OCR errors (character-level typos, concatenated words, etc.)
- Do NOT add, invent, translate, or complete any text. The English side is often shorter than Chinese — that is NORMAL.
- If Chinese has more content than English, pair excess Chinese sentences with en: "" (empty string).
- Every OCR fix must appear in corrections. If no errors, corrections lists should be empty.
- "missing_text" is NOT allowed as a correction type.

## OCR error hints:
{OCR_HINT}

## Chinese text (OCR raw):
{zh_text}

## English text (OCR raw):
{en_text}

## Output (strict JSON, no other text):
Note: Do NOT include paragraph-level zh/en text — only output sentences. Paragraph text will be reconstructed by joining sentences.
{{
  "board_title": {{"zh": "标题(如有，否则空字符串)", "en": "Title (if any, else empty string)"}},
  "corrections": {{
    "zh_changes": [{{"from": "OCR错字", "to": "正确", "type": "form_similar/missing_char/..."}}],
    "en_changes": [{{"from": "OCRerror", "to": "corrected", "type": "word_split/letter_substitution/case_error"}}]
  }},
  "paragraphs": [
    {{
      "sentences": [
        {{"zh": "中文句子1。", "en": "English sentence 1."}},
        {{"zh": "中文句子2，无对应英文。", "en": ""}}
      ]
    }}
  ]
}}"""


def build_short_prompt(zh_text: str, en_text: str) -> str:
    return f"""This is a short museum exhibit label/title. ONLY fix clear OCR errors. Do NOT rephrase or add any text.

Chinese: {zh_text}
English: {en_text}

Output (strict JSON, no other text):
{{
  "board_title": {{"zh": "Chinese with only OCR fixes", "en": "English with only OCR fixes"}},
  "corrections": {{
    "zh_changes": [{{"from": "...", "to": "...", "type": "..."}}],
    "en_changes": [{{"from": "...", "to": "...", "type": "..."}}]
  }},
  "paragraphs": []
}}"""


def build_mono_prompt(text: str, lang: str) -> str:
    lang_name = "Chinese" if lang == "zh" else "English"
    other = "en" if lang == "zh" else "zh"
    return f"""Clean and structure this museum exhibit board text (single language: {lang_name}). ONLY fix clear OCR errors, then segment into paragraphs and sentences.

CRITICAL: Do NOT add, translate, or invent any text. Keep verbatim except OCR fixes.

## OCR error hints:
{OCR_HINT}

## {lang_name} text (OCR raw — preserve verbatim except OCR fixes):
{text}

## Output (strict JSON, no other text):
Note: Only output sentences, no paragraph-level text.
{{
  "board_title": {{"{lang}": "", "{other}": ""}},
  "corrections": {{
    "{lang}_changes": [{{"from": "...", "to": "...", "type": "..."}}]
  }},
  "paragraphs": [
    {{
      "sentences": [
        {{"{lang}": "句子。", "{other}": ""}}
      ]
    }}
  ]
}}"""


# ==================== JSON 提取 ====================

def extract_json(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON（兼容 ```json 代码块和裸 JSON）"""
    if not text:
        return None

    # 方法1：```json ... ```
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 方法2：最外层大括号（支持嵌套）
    depth = 0
    start = None
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


# ==================== 主处理类 ====================

class LLMEnhancer:
    def __init__(self, test_limit: int = 0, workers: int = 5):
        self.test_mode = test_limit > 0
        self.test_limit = test_limit
        self.workers = workers
        self.client = None
        self._lock = threading.Lock()  # 保护 progress 读写

        # 数据
        self.ocr_results: List[Dict] = []
        self.review_map: Dict[str, Dict] = {}
        self.progress: Dict = {}

        # rerun 模式保留数据
        self._cleaned_boards: List[Dict] = []  # 已人工清洗的展板
        self._deleted_boards: List[Dict] = []  # 已删除的展板

        # 统计
        self.stats = {
            "processed": 0, "skipped": 0, "failed": 0,
            "total_input_tokens": 0, "total_output_tokens": 0,
        }

    def setup(self):
        """初始化 API 客户端"""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=cfg.DEEPSEEK_API_KEY,
                base_url=cfg.DEEPSEEK_API_URL,
                timeout=cfg.LLM_TIMEOUT,
            )
            print(f"[OK] DeepSeek API 已连接 (model: {cfg.DEEPSEEK_MODEL})")
            return True
        except ImportError:
            print("[ERROR] 请安装 openai: pip install openai")
            return False

    def load_inputs(self):
        """加载 OCR 结果和审核结果"""
        # OCR 结果
        if not cfg.OCR_RESULTS_FILE.exists():
            print(f"[ERROR] OCR 结果不存在: {cfg.OCR_RESULTS_FILE}")
            return False
        with open(cfg.OCR_RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.ocr_results = data.get("results", [])
        print(f"[OK] 加载 OCR 结果: {len(self.ocr_results)} 条")

        # 审核结果
        review_file = cfg.REVIEWED_RESULTS_FILE
        if review_file.exists():
            with open(review_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for r in data.get("results", []):
                    self.review_map[r["image_id"]] = r
            print(f"[OK] 加载审核结果: {len(self.review_map)} 条")
        return True

    def load_progress(self):
        """加载断点进度"""
        cfg.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        if cfg.ENHANCED_PROGRESS_FILE.exists():
            with open(cfg.ENHANCED_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            done = len(self.progress.get("completed", {}))
            fail = len(self.progress.get("failed", []))
            print(f"[RESUME] 已完成 {done} 条, 失败 {fail} 条")
        else:
            self.progress = {
                "metadata": {"started_at": datetime.now().isoformat(), "version": "1.0"},
                "completed": {},
                "skipped": [],
                "failed": [],
                "token_usage": {"total_input": 0, "total_output": 0},
            }

    def save_progress(self):
        """保存进度（原子写入）"""
        self.progress["metadata"]["last_updated"] = datetime.now().isoformat()
        self.progress["metadata"]["total_completed"] = len(self.progress["completed"])

        tmp = cfg.ENHANCED_DIR / "progress.tmp.json"
        final = cfg.ENHANCED_PROGRESS_FILE

        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

        if final.exists():
            final.unlink()
        tmp.rename(final)

    # ---------- 文本解析 ----------

    def resolve_text(self, entry: Dict) -> Tuple[str, str, str]:
        """获取最佳可用文本：人工修正 > 人工确认 > 原始 OCR"""
        zh = entry.get("zh_text", "")
        en = entry.get("en_text", "")
        source = "raw_ocr"

        review = self.review_map.get(entry.get("image_id", ""))
        if not review:
            return zh, en, source

        status = review.get("review_status", "")
        if status == "skipped":
            return "", "", "skipped"

        if status == "confirmed":
            return zh, en, "reviewed_confirmed"

        if status == "corrected":
            czh = review.get("corrected_zh", "")
            cen = review.get("corrected_en", "")

            if czh.strip() == "[删除]":
                zh = ""
            elif czh:
                zh = czh

            if cen.strip() == "[删除]":
                en = ""
            elif cen:
                en = cen

            source = "reviewed_corrected"

        return zh, en, source

    # ---------- API 调用 ----------

    def call_api(self, messages: List[Dict], max_tokens: int = 4096) -> Optional[str]:
        """调用 DeepSeek API，带重试"""
        for attempt in range(cfg.LLM_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=cfg.DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                # 统计 token（线程安全）
                if hasattr(resp, 'usage') and resp.usage:
                    with self._lock:
                        self.progress["token_usage"]["total_input"] += resp.usage.prompt_tokens
                        self.progress["token_usage"]["total_output"] += resp.usage.completion_tokens
                return resp.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [RETRY {attempt+1}/{cfg.LLM_MAX_RETRIES}] {e}, 等待 {wait}s...")
                time.sleep(wait)
        return None

    # ---------- 校验 ----------

    def validate_result(self, result: Dict, zh_in: str, en_in: str) -> Tuple[bool, str]:
        """校验 LLM 输出结构和内容完整性"""
        # 结构校验
        if "paragraphs" not in result:
            return False, "缺少 paragraphs 字段"
        if not isinstance(result["paragraphs"], list):
            return False, "paragraphs 不是数组"
        for i, p in enumerate(result["paragraphs"]):
            if "sentences" not in p:
                return False, f"paragraphs[{i}] 缺少 sentences"
            if not isinstance(p["sentences"], list):
                return False, f"paragraphs[{i}].sentences 不是数组"
            for j, s in enumerate(p["sentences"]):
                if "zh" not in s or "en" not in s:
                    return False, f"paragraphs[{i}].sentences[{j}] 缺少 zh/en"

        # ===== 纠错类型校验：拒绝 missing_text 等伪造内容 =====
        corrections = result.get("corrections", {})
        forbidden_types = {"missing_text", "added_text", "translation", "completion"}
        for lang_key in ["zh_changes", "en_changes"]:
            for c in corrections.get(lang_key, []):
                ctype = c.get("type", "").lower()
                if any(ft in ctype for ft in forbidden_types):
                    return False, f"禁止的纠错类型: {c.get('type')} (不允许添加内容)"
                # 单条纠错的 to 不应比 from 长太多（>3倍 且 多出>30字符 = 可疑添加）
                from_text = c.get("from", "")
                to_text = c.get("to", "")
                if len(from_text) > 0 and len(to_text) > len(from_text) * 3 and len(to_text) - len(from_text) > 30:
                    return False, f"可疑的大量添加: '{from_text[:30]}...' → '{to_text[:30]}...' (长度 {len(from_text)}→{len(to_text)})"

        # 标题可选
        board_title = result.get("board_title", {})
        title_zh = board_title.get("zh", "") if isinstance(board_title, dict) else ""
        title_en = board_title.get("en", "") if isinstance(board_title, dict) else ""

        # 从 sentences 重建段落文本用于内容完整性检查
        conf = cfg.LLM_ENHANCE_CONFIG
        recon_zh = title_zh
        recon_en = title_en
        for p in result["paragraphs"]:
            # 段落可能直接有 zh/en 字段（兼容旧格式），或从 sentences 拼接
            if "zh" in p and p["zh"]:
                recon_zh += p["zh"]
            else:
                recon_zh += "".join(s.get("zh", "") for s in p.get("sentences", []))
            if "en" in p and p["en"]:
                recon_en += " " + p["en"]
            else:
                recon_en += " " + " ".join(s.get("en", "") for s in p.get("sentences", []))
        recon_en = recon_en.strip()

        if zh_in and len(zh_in) > 20:
            ratio = len(recon_zh) / len(zh_in)
            if ratio < conf["content_ratio_min"] or ratio > conf["content_ratio_max"]:
                return False, f"中文字数比 {ratio:.2f} 超出范围 [{conf['content_ratio_min']}-{conf['content_ratio_max']}]"

        if en_in and len(en_in) > 20:
            ratio = len(recon_en) / len(en_in)
            if ratio < conf["content_ratio_min"] or ratio > conf["content_ratio_max"]:
                return False, f"英文字数比 {ratio:.2f} 超出范围 [{conf['content_ratio_min']}-{conf['content_ratio_max']}]"

        return True, "ok"

    # ---------- 单条处理 ----------

    def process_single(self, image_id: str, zh: str, en: str, source_info: Dict) -> Optional[Dict]:
        """处理单条展板"""
        has_zh = bool(zh and zh.strip())
        has_en = bool(en and en.strip())
        conf = cfg.LLM_ENHANCE_CONFIG

        if not has_zh and not has_en:
            return None

        # 选择 prompt
        total_len = len(zh or "") + len(en or "")
        if has_zh and has_en and total_len < conf["short_text_threshold"]:
            prompt = build_short_prompt(zh, en)
            max_tokens = 800
        elif has_zh and has_en:
            prompt = build_full_prompt(zh, en)
            max_tokens = min(conf["max_tokens_per_call"], max(1500, int(total_len * 3)))
        else:
            text = zh if has_zh else en
            lang = "zh" if has_zh else "en"
            prompt = build_mono_prompt(text, lang)
            max_tokens = min(conf["max_tokens_per_call"], max(1000, int(len(text) * 3)))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # 尝试最多 3 次（含 JSON 修复重试）
        for attempt in range(3):
            raw = self.call_api(messages, max_tokens)
            if not raw:
                continue

            parsed = extract_json(raw)
            if not parsed:
                # JSON 解析失败，要求重新输出
                if attempt < 2:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content":
                        "Your response was not valid JSON. Please output ONLY the JSON object, nothing else."})
                continue

            # 校验
            ok, reason = self.validate_result(parsed, zh, en)
            if ok:
                return self._build_board(image_id, source_info, zh, en, parsed)
            else:
                print(f"  [WARN] 校验失败: {reason}")
                if attempt < 2:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content":
                        f"The JSON structure is incorrect: {reason}. Please fix and output valid JSON."})

        return None

    def _build_board(self, image_id: str, source_info: Dict,
                     zh_in: str, en_in: str, llm_result: Dict) -> Dict:
        """组装最终的 board 结构（从 sentences 重建 paragraph 文本）"""
        board_title = llm_result.get("board_title", {"zh": "", "en": ""})
        if not isinstance(board_title, dict):
            board_title = {"zh": "", "en": ""}

        corrections = llm_result.get("corrections", {})
        paragraphs = []
        for i, p in enumerate(llm_result.get("paragraphs", [])):
            sentences = []
            for j, s in enumerate(p.get("sentences", [])):
                sentences.append({
                    "sent_index": j,
                    "zh": s.get("zh", ""),
                    "en": s.get("en", ""),
                })

            # 从 sentences 重建段落文本（如果 LLM 没有提供段落级文本）
            para_zh = p.get("zh", "")
            para_en = p.get("en", "")
            if not para_zh and sentences:
                para_zh = "".join(s["zh"] for s in sentences)
            if not para_en and sentences:
                para_en = " ".join(s["en"] for s in sentences).strip()

            para = {
                "para_index": i,
                "zh": para_zh,
                "en": para_en,
                "sentences": sentences,
            }
            paragraphs.append(para)

        return {
            "board_id": image_id,
            "source": source_info,
            "board_title": board_title,
            "corrections": corrections,
            "paragraphs": paragraphs,
        }

    # ---------- 主流程 ----------

    def build_queue(self) -> List[Dict]:
        """构建处理队列"""
        completed_ids = set(self.progress.get("completed", {}).keys())
        skipped_ids = set(self.progress.get("skipped", []))
        failed_ids = set(self.progress.get("failed", []))
        queue = []

        for entry in self.ocr_results:
            if "error" in entry:
                continue
            image_id = entry.get("image_id", "")
            if image_id in completed_ids or image_id in skipped_ids or image_id in failed_ids:
                continue

            zh, en, src = self.resolve_text(entry)
            if src == "skipped":
                self.progress["skipped"].append(image_id)
                continue
            if not zh and not en:
                self.progress["skipped"].append(image_id)
                continue

            queue.append({
                "image_id": image_id,
                "source": entry.get("source", {}),
                "quality": entry.get("quality", {}),
                "zh_text": zh,
                "en_text": en,
                "input_source": src,
            })

        return queue

    def _process_one_item(self, i: int, total: int, item: Dict) -> tuple:
        """处理单条（供线程池调用），返回 (image_id, result_or_None, item)"""
        image_id = item["image_id"]
        museum = item["source"].get("museum", "?")
        has_zh = "ZH" if item["zh_text"] else "--"
        has_en = "EN" if item["en_text"] else "--"
        print(f"[{i+1}/{total}] {image_id} ({museum}) [{has_zh}|{has_en}]")

        result = self.process_single(
            image_id, item["zh_text"], item["en_text"], item["source"]
        )
        return image_id, result, item

    def run(self):
        """运行完整流程（并发版）"""
        print("=" * 70)
        print("LLM 增强处理：OCR 纠错 + 段落分割 + 句对齐")
        print(f"并发线程数: {self.workers}")
        print("=" * 70)

        if not self.setup():
            return
        if not self.load_inputs():
            return
        self.load_progress()

        queue = self.build_queue()
        total = min(len(queue), self.test_limit) if self.test_mode else len(queue)
        print(f"\n待处理: {total} 条" + (" (测试模式)" if self.test_mode else ""))

        if total == 0:
            print("[OK] 没有需要处理的条目")
            self.assemble_corpus(
                cleaned_boards=self._cleaned_boards or None,
                deleted_boards=self._deleted_boards or None
            )
            return

        failed_this_run = []
        items = queue[:total]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._process_one_item, i, total, item): item
                for i, item in enumerate(items)
            }

            for future in as_completed(futures):
                try:
                    image_id, result, item = future.result()
                except Exception as e:
                    item = futures[future]
                    image_id = item["image_id"]
                    print(f"  [ERROR] {image_id}: {e}")
                    result = None

                with self._lock:
                    if result:
                        n_para = len(result["paragraphs"])
                        n_sent = sum(len(p["sentences"]) for p in result["paragraphs"])
                        n_zh_corr = len(result.get("corrections", {}).get("zh_changes", []))
                        n_en_corr = len(result.get("corrections", {}).get("en_changes", []))
                        print(f"  [OK] {image_id}: {n_para} 段, {n_sent} 句对, 纠错 zh={n_zh_corr} en={n_en_corr}")

                        self.progress["completed"][image_id] = {
                            "processed_at": datetime.now().isoformat(),
                            "input_source": item["input_source"],
                            "result": result,
                        }
                        self.stats["processed"] += 1
                    else:
                        print(f"  [FAIL] {image_id}")
                        if image_id not in self.progress["failed"]:
                            self.progress["failed"].append(image_id)
                        failed_this_run.append(image_id)
                        self.stats["failed"] += 1

                    self.save_progress()

        # 完成后组装
        print("\n" + "=" * 70)
        self.assemble_corpus(
            cleaned_boards=self._cleaned_boards or None,
            deleted_boards=self._deleted_boards or None
        )
        self.print_stats()

        if failed_this_run:
            print(f"\n[WARN] 本次失败 {len(failed_this_run)} 条:")
            for fid in failed_this_run:
                print(f"  - {fid}")

    def assemble_corpus(self, cleaned_boards: List[Dict] = None,
                        deleted_boards: List[Dict] = None):
        """从 progress 组装最终的 enhanced_corpus.json

        Args:
            cleaned_boards: 已人工清洗的展板列表（rerun-uncleaned 模式下传入，原样保留）
            deleted_boards: 已删除的展板列表（rerun-uncleaned 模式下传入，原样保留）
        """
        completed = self.progress.get("completed", {})
        if not completed and not cleaned_boards:
            print("[INFO] 无已完成数据，跳过组装")
            return

        boards = []
        total_paras = 0
        total_sents = 0

        # 先加入已人工清洗的展板（保留用户编辑）
        if cleaned_boards:
            for b in cleaned_boards:
                boards.append(b)
                for p in b.get("paragraphs", []):
                    total_paras += 1
                    total_sents += len(p.get("sentences", []))
            print(f"[OK] 保留 {len(cleaned_boards)} 个已清洗展板")

        # 再加入 LLM 处理结果（排除已清洗的）
        cleaned_ids = {b["board_id"] for b in (cleaned_boards or [])}
        for image_id, data in completed.items():
            if image_id in cleaned_ids:
                continue
            result = data.get("result", {})
            boards.append(result)
            for p in result.get("paragraphs", []):
                total_paras += 1
                total_sents += len(p.get("sentences", []))

        corpus = {
            "metadata": {
                "name": "中国博物馆多语种解说词平行语料库（LLM增强版）",
                "version": "3.0",
                "created_at": datetime.now().isoformat(),
                "processor": f"llm_enhance.py + {cfg.DEEPSEEK_MODEL}",
                "total_boards": len(boards),
                "total_paragraphs": total_paras,
                "total_sentence_pairs": total_sents,
                "token_usage": self.progress.get("token_usage", {}),
                "skipped": len(self.progress.get("skipped", [])),
                "failed": len(self.progress.get("failed", [])),
            },
            "boards": boards,
        }

        # 保留已删除展板
        if deleted_boards:
            corpus["deleted_boards"] = deleted_boards

        cfg.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        with open(cfg.ENHANCED_CORPUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        print(f"[OK] 增强语料库已保存: {cfg.ENHANCED_CORPUS_FILE}")
        print(f"     展板: {len(boards)}, 段落: {total_paras}, 句对: {total_sents}")

    def print_stats(self):
        """打印统计"""
        usage = self.progress.get("token_usage", {})
        inp = usage.get("total_input", 0)
        out = usage.get("total_output", 0)
        # DeepSeek V3 pricing
        cost_in = inp / 1_000_000 * 0.27
        cost_out = out / 1_000_000 * 1.10
        cost = cost_in + cost_out

        print(f"\n{'='*70}")
        print("统计")
        print(f"{'='*70}")
        print(f"已完成: {len(self.progress.get('completed', {}))}")
        print(f"已跳过: {len(self.progress.get('skipped', []))}")
        print(f"失败:   {len(self.progress.get('failed', []))}")
        print(f"Token:  输入 {inp:,} + 输出 {out:,} = {inp+out:,}")
        print(f"费用:   ~${cost:.3f} (￥{cost*7.2:.2f})")
        print(f"{'='*70}")

    def prepare_rerun_uncleaned(self):
        """准备重跑未清洗的展板：
        1. 读取 enhanced_corpus.json，提取已清洗 / 已删除的展板
        2. 从 progress.json 中删除未清洗展板的 completed 记录和 failed 记录
        3. 后续 run() 会自动重新处理这些展板
        """
        corpus_file = cfg.ENHANCED_CORPUS_FILE
        if not corpus_file.exists():
            print(f"[ERROR] 增强语料不存在: {corpus_file}")
            return False

        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        boards = corpus_data.get("boards", [])
        self._deleted_boards = corpus_data.get("deleted_boards", [])

        # 分离已清洗 vs 未清洗
        self._cleaned_boards = [b for b in boards if b.get("_cleaned")]
        uncleaned_boards = [b for b in boards if not b.get("_cleaned")]

        cleaned_ids = {b["board_id"] for b in self._cleaned_boards}
        uncleaned_ids = {b["board_id"] for b in uncleaned_boards}

        print(f"[INFO] 已清洗（保留）: {len(self._cleaned_boards)} 个展板")
        print(f"[INFO] 未清洗（重跑）: {len(uncleaned_ids)} 个展板")
        print(f"[INFO] 已删除（保留）: {len(self._deleted_boards)} 个展板")

        # 从 progress 中移除未清洗展板的 completed 记录
        # 同时清除所有 failed 记录（失败的展板也需要重跑）
        self.load_progress()

        removed_completed = 0
        for uid in list(self.progress.get("completed", {}).keys()):
            if uid not in cleaned_ids:
                del self.progress["completed"][uid]
                removed_completed += 1

        removed_failed = len(self.progress.get("failed", []))
        self.progress["failed"] = []  # 清空全部失败记录，让它们全部重跑

        print(f"[INFO] 从 progress 移除: {removed_completed} completed, {removed_failed} failed")
        self.save_progress()

        return True


def main():
    parser = argparse.ArgumentParser(description="LLM增强：OCR纠错 + 段落/句子对齐")
    parser.add_argument("--test", type=int, metavar="N", default=0,
                        help="测试模式：只处理前N条")
    parser.add_argument("--workers", type=int, default=5,
                        help="并发线程数（默认5）")
    parser.add_argument("--resume", action="store_true",
                        help="从断点恢复（默认行为，已自动启用）")
    parser.add_argument("--assemble", action="store_true",
                        help="只从 progress 组装最终文件")
    parser.add_argument("--stats", action="store_true",
                        help="显示处理进度统计")
    parser.add_argument("--rerun-uncleaned", action="store_true",
                        help="重跑未人工清洗的展板（保留 _cleaned 的展板，用更严格的 prompt 重新处理其余展板）")
    args = parser.parse_args()

    enhancer = LLMEnhancer(test_limit=args.test, workers=args.workers)

    if args.stats:
        enhancer.load_progress()
        enhancer.print_stats()
        return

    if args.assemble:
        enhancer.load_inputs()
        enhancer.load_progress()
        enhancer.assemble_corpus()
        enhancer.print_stats()
        return

    if args.rerun_uncleaned:
        print("=" * 70)
        print("重跑模式：保留已清洗展板，重新处理未清洗展板")
        print("=" * 70)
        if not enhancer.prepare_rerun_uncleaned():
            return
        # 继续执行正常的 run()，progress 已清理，会重新处理未清洗的展板

    enhancer.run()


if __name__ == "__main__":
    main()

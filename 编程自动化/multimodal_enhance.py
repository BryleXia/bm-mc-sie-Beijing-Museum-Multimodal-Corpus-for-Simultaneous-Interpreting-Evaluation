#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态增强模块：基于 VLM 看图校验 + OCR纠错 + 保守对齐
使用 Qwen 3.5 Plus (多模态) 看展板原图，一次调用完成：
  1. 对照原图纠正 OCR 错误（看得到原图，修正看不到原图时的残留错误）
  2. 基于视觉布局划分段落
  3. 保守句子对齐（不确定时整段配对）
  4. 输出层级结构: board → paragraphs → sentences

审核拦截的展板自动 fallback 到 Kimi K2.5（无政治审核）

用法:
  python multimodal_enhance.py                    # 处理全部
  python multimodal_enhance.py --test 3           # 测试前3条
  python multimodal_enhance.py --provider kimi    # 用 Kimi 处理
  python multimodal_enhance.py --only-failed      # 只处理之前失败的
  python multimodal_enhance.py --stats            # 查看进度
  python multimodal_enhance.py --assemble         # 只从 progress 组装最终文件
"""

import json
import sys
import io
import re
import os
import base64
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# 编码 + 关闭缓冲（后台运行时也能实时看输出）
if not getattr(sys, '_museum_encoding_set', False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
        sys._museum_encoding_set = True
    except Exception:
        pass

# 配置
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ==================== Prompt 模板 ====================

MULTIMODAL_PROMPT_TEMPLATE = """你是博物馆双语展板校验助手。

我提供：1. 展板原始照片  2. 当前已清洗的中英文文本

## 任务 1：文本校验
看照片上的文字，与"当前文本"逐字比较：
- 修正与图片不符的错误（OCR误识别的字符、错字、漏字）
- 删除图片上不存在的文字（OCR幻觉产生的多余内容）
- 绝不添加照片上没有的任何内容
- 如果当前文本与照片完全一致，corrections 列表留空

## 任务 2：段落划分
根据图片上的视觉布局（间距、分隔线、标题等）分段：
- 中英文 1:1 配对
- 如果英文段落数少于中文，多出的中文段落对应 en: ""
- 段落顺序与照片上从上到下的布局一致

## 任务 3：保守句子对齐
在每个段落内部尝试句子级对齐：
- 非常确定中英句子一一对应时，拆分为句子对，标记 alignment_level: "sentence"
- 不确定时（英文是意译/缩写/顺序不同），整段配对，标记 alignment_level: "paragraph"
- 宁可粗对齐（paragraph），不要错对齐
- 句子边界：中文用 。！？ 分割，英文用 . ! ? 分割

## 当前中文文本：
{zh_text}

## 当前英文文本：
{en_text}

## 输出要求
输出严格 JSON，不要任何其他文字：
{{"board_title":{{"zh":"展板标题(如有,否则空字符串)","en":"Board title(if any, else empty string)"}},
"corrections":{{"zh_changes":[{{"from":"原文错误","to":"修正后","type":"ocr_error/form_similar/extra_text"}}],"en_changes":[{{"from":"original error","to":"corrected","type":"ocr_error/word_split/extra_text"}}]}},
"paragraphs":[{{"alignment_level":"sentence","sentences":[{{"zh":"中文句子。","en":"English sentence."}}]}}]}}"""

MULTIMODAL_PROMPT_FAILED = """你是博物馆双语展板文本提取助手。

我提供展板原始照片。请从照片中提取所有中英文文字。

## 任务
1. 识别照片上的所有中文文字和英文文字
2. 根据视觉布局分段，中英 1:1 配对
3. 段落内尝试句子级对齐（不确定时整段配对）
4. 如有明显的标题/题头，提取到 board_title

## 输出要求
输出严格 JSON，不要任何其他文字：
{{"board_title":{{"zh":"","en":""}},
"corrections":{{"zh_changes":[],"en_changes":[]}},
"paragraphs":[{{"alignment_level":"sentence|paragraph","sentences":[{{"zh":"中文句子。","en":"English sentence."}}]}}]}}"""


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

class MultimodalEnhancer:
    def __init__(self, provider: str = "qwen", test_limit: int = 0,
                 workers: int = 3, only_failed: bool = False):
        self.provider = provider
        self.test_mode = test_limit > 0
        self.test_limit = test_limit
        self.workers = workers
        self.only_failed = only_failed
        self.client = None
        self._lock = threading.Lock()

        # 数据
        self.enhanced_corpus: Dict = {}        # enhanced_corpus.json 全量数据
        self.ocr_results: List[Dict] = []      # ocr_results.json
        self.ocr_map: Dict[str, Dict] = {}     # image_id → ocr entry
        self.enhanced_boards: Dict[str, Dict] = {}  # image_id → board from corpus
        self.cleaned_ids: set = set()           # 已人工清洗的 board_id
        self.progress: Dict = {}

        # 统计
        self.stats = {
            "processed": 0, "skipped": 0, "failed": 0,
            "fallback_kimi": 0,
        }

    def setup_client(self, provider: str = None):
        """初始化 API 客户端"""
        provider = provider or self.provider
        try:
            import openai
        except ImportError:
            print("[ERROR] 请安装 openai: pip install openai")
            return False

        if provider == "qwen":
            self.client = openai.OpenAI(
                api_key=cfg.QWEN_API_KEY,
                base_url=cfg.QWEN_API_URL,
                timeout=180,  # 多模态调用需要更长超时
            )
            self._current_model = cfg.QWEN_MODEL
            print(f"[OK] Qwen API 已连接 (model: {cfg.QWEN_MODEL})")
        elif provider == "kimi":
            self.client = openai.OpenAI(
                api_key=cfg.KIMI_API_KEY,
                base_url=cfg.KIMI_API_URL,
                timeout=180,
            )
            self._current_model = cfg.KIMI_MODEL
            print(f"[OK] Kimi API 已连接 (model: {cfg.KIMI_MODEL})")
        else:
            print(f"[ERROR] 未知 provider: {provider}")
            return False

        return True

    def _get_kimi_client(self):
        """获取 Kimi 客户端（用于 fallback）"""
        import openai
        return openai.OpenAI(
            api_key=cfg.KIMI_API_KEY,
            base_url=cfg.KIMI_API_URL,
            timeout=180,
        )

    def load_data(self):
        """加载 enhanced_corpus.json + ocr_results.json"""
        # 加载 OCR 结果
        if not cfg.OCR_RESULTS_FILE.exists():
            print(f"[ERROR] OCR 结果不存在: {cfg.OCR_RESULTS_FILE}")
            return False
        with open(cfg.OCR_RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.ocr_results = data.get("results", [])
        for r in self.ocr_results:
            self.ocr_map[r.get("image_id", "")] = r
        print(f"[OK] 加载 OCR 结果: {len(self.ocr_results)} 条")

        # 加载 enhanced_corpus
        if cfg.ENHANCED_CORPUS_FILE.exists():
            with open(cfg.ENHANCED_CORPUS_FILE, 'r', encoding='utf-8') as f:
                self.enhanced_corpus = json.load(f)
            boards = self.enhanced_corpus.get("boards", [])
            for b in boards:
                bid = b.get("board_id", "")
                self.enhanced_boards[bid] = b
                if b.get("_cleaned"):
                    self.cleaned_ids.add(bid)
            print(f"[OK] 加载增强语料: {len(boards)} 条 (已清洗: {len(self.cleaned_ids)})")
        else:
            print("[WARN] enhanced_corpus.json 不存在，将从 OCR 结果开始处理")

        return True

    def load_progress(self):
        """加载断点进度"""
        cfg.MULTIMODAL_DIR.mkdir(parents=True, exist_ok=True)
        if cfg.MULTIMODAL_PROGRESS_FILE.exists():
            with open(cfg.MULTIMODAL_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            done = len(self.progress.get("completed", {}))
            fail = len(self.progress.get("failed", []))
            print(f"[RESUME] 已完成 {done} 条, 失败 {fail} 条")
        else:
            self.progress = {
                "metadata": {
                    "started_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "provider": self.provider,
                },
                "completed": {},
                "skipped": [],
                "failed": [],
                "token_usage": {"total_input": 0, "total_output": 0},
            }

    def save_progress(self):
        """保存进度（原子写入）"""
        self.progress["metadata"]["last_updated"] = datetime.now().isoformat()
        self.progress["metadata"]["total_completed"] = len(self.progress["completed"])

        tmp = cfg.MULTIMODAL_DIR / "progress.tmp.json"
        final = cfg.MULTIMODAL_PROGRESS_FILE

        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

        if final.exists():
            final.unlink()
        tmp.rename(final)

    # ---------- 图片处理 ----------

    def prepare_image(self, image_path: str) -> Optional[str]:
        """读取图片并缩放，返回 base64 编码的 JPEG"""
        try:
            from PIL import Image
        except ImportError:
            print("[ERROR] 请安装 Pillow: pip install Pillow")
            return None

        path = Path(image_path)
        if not path.exists():
            print(f"  [WARN] 图片不存在: {image_path}")
            return None

        try:
            img = Image.open(path)

            # 转换为 RGB（处理 RGBA 或其他模式）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 缩放长边到 max_size
            max_size = cfg.MULTIMODAL_CONFIG["image_max_size"]
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            # 编码为 JPEG base64
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format='JPEG', quality=cfg.MULTIMODAL_CONFIG["image_quality"])
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return b64
        except Exception as e:
            print(f"  [ERROR] 图片处理失败 {image_path}: {e}")
            return None

    # ---------- Prompt 构建 ----------

    def build_prompt(self, zh_text: str, en_text: str, is_failed: bool = False) -> str:
        """构建多模态 prompt"""
        if is_failed:
            # 之前因审核拦截而失败的展板，没有已清洗文本，直接从图片提取
            return MULTIMODAL_PROMPT_FAILED

        return MULTIMODAL_PROMPT_TEMPLATE.format(
            zh_text=zh_text or "(无中文文本)",
            en_text=en_text or "(无英文文本)",
        )

    # ---------- API 调用 ----------

    def call_api(self, image_b64: str, prompt: str,
                 client=None, model: str = None) -> Optional[Dict]:
        """调用多模态 API，返回解析后的 JSON"""
        client = client or self.client
        model = model or self._current_model
        conf = cfg.MULTIMODAL_CONFIG

        # Kimi K2.5 只允许 temperature=1
        is_kimi = model == cfg.KIMI_MODEL
        temperature = 1.0 if is_kimi else 0.2

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        for attempt in range(cfg.LLM_MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=conf["max_tokens_per_call"],
                )
                # 统计 token
                if hasattr(resp, 'usage') and resp.usage:
                    with self._lock:
                        self.progress["token_usage"]["total_input"] += resp.usage.prompt_tokens
                        self.progress["token_usage"]["total_output"] += resp.usage.completion_tokens

                raw = resp.choices[0].message.content
                if not raw:
                    print(f"  [WARN] 空响应 (attempt {attempt+1})")
                    continue

                # 检查是否被审核拦截
                if self._is_content_filtered(raw):
                    return {"_filtered": True, "_raw": raw}

                parsed = extract_json(raw)
                if parsed:
                    return parsed

                # JSON 解析失败，重试时加入纠正消息
                if attempt < cfg.LLM_MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": "你的回复不是合法 JSON。请只输出 JSON 对象，不要其他文字。"}
                    ]})
                    print(f"  [RETRY] JSON 解析失败，重试 (attempt {attempt+1})")

            except Exception as e:
                wait = 2 ** attempt
                err_str = str(e)
                # 审核拦截的异常特征 — 立即返回，不再重试
                if any(kw in err_str.lower() for kw in [
                    'content_filter', 'sensitive', 'security', 'policy',
                    'data_inspection_failed', 'inappropriate',
                ]):
                    return {"_filtered": True, "_raw": err_str}
                print(f"  [RETRY {attempt+1}/{cfg.LLM_MAX_RETRIES}] {e}, 等待 {wait}s...", flush=True)
                time.sleep(wait)

        return None

    def _is_content_filtered(self, response: str) -> bool:
        """检测响应是否被内容审核拦截"""
        filter_keywords = [
            "无法", "不能", "抱歉", "违规", "敏感",
            "content policy", "unable to process",
            "安全", "审核",
        ]
        # 只在短响应中检测（正常响应会很长）
        if len(response) < 200:
            lower = response.lower()
            return any(kw in lower for kw in filter_keywords)
        return False

    # ---------- 校验 ----------

    def validate_result(self, result: Dict, original_zh: str, original_en: str,
                        is_failed_board: bool = False) -> Tuple[bool, str]:
        """校验多模态输出的结构和内容"""
        # 被审核拦截
        if not isinstance(result, dict):
            return False, f"结果不是 dict 类型: {type(result).__name__}"
        if result.get("_filtered"):
            return False, "内容审核拦截"

        # 结构校验
        if "paragraphs" not in result:
            return False, "缺少 paragraphs 字段"
        if not isinstance(result["paragraphs"], list):
            return False, "paragraphs 不是数组"
        if len(result["paragraphs"]) == 0:
            return False, "paragraphs 为空"

        for i, p in enumerate(result["paragraphs"]):
            if "sentences" not in p:
                return False, f"paragraphs[{i}] 缺少 sentences"
            if not isinstance(p["sentences"], list):
                return False, f"paragraphs[{i}].sentences 不是数组"
            if len(p["sentences"]) == 0:
                return False, f"paragraphs[{i}].sentences 为空"
            for j, s in enumerate(p["sentences"]):
                if "zh" not in s or "en" not in s:
                    return False, f"paragraphs[{i}].sentences[{j}] 缺少 zh/en"

            # alignment_level 校验
            level = p.get("alignment_level", "")
            if level and level not in ("sentence", "paragraph"):
                # 自动修正常见拼写
                pass  # 不做强制校验，后续处理时默认 paragraph

        # 纠错类型校验：拒绝 missing_text 等
        corrections = result.get("corrections", {})
        forbidden_types = {"missing_text", "added_text", "translation", "completion"}
        for lang_key in ["zh_changes", "en_changes"]:
            for c in corrections.get(lang_key, []):
                ctype = c.get("type", "").lower()
                if any(ft in ctype for ft in forbidden_types):
                    return False, f"禁止的纠错类型: {c.get('type')}"

        # 内容完整性检查（字数比）— 仅警告不拒绝
        # 多模态模型看的是真实图片，字数差异可能是正确纠错
        conf = cfg.MULTIMODAL_CONFIG
        ratio_min = conf["content_ratio_min"]
        ratio_max = conf["content_ratio_max"]

        board_title = result.get("board_title", {})
        title_zh = board_title.get("zh", "") if isinstance(board_title, dict) else ""
        title_en = board_title.get("en", "") if isinstance(board_title, dict) else ""

        recon_zh = title_zh
        recon_en = title_en
        for p in result["paragraphs"]:
            recon_zh += "".join(s.get("zh", "") for s in p.get("sentences", []))
            recon_en += " " + " ".join(s.get("en", "") for s in p.get("sentences", []))
        recon_en = recon_en.strip()

        warnings = []
        if original_zh and len(original_zh) > 20:
            ratio = len(recon_zh) / len(original_zh)
            if ratio < ratio_min or ratio > ratio_max:
                warnings.append(f"中文字数比 {ratio:.2f}")
                # 极端偏差仍然拒绝（可能是模型幻觉）
                if ratio < 0.10 or ratio > 5.0:
                    return False, f"中文字数比 {ratio:.2f} 极端异常"

        if original_en and len(original_en) > 20:
            ratio = len(recon_en) / len(original_en)
            if ratio < ratio_min or ratio > ratio_max:
                warnings.append(f"英文字数比 {ratio:.2f}")
                if ratio < 0.10 or ratio > 5.0:
                    return False, f"英文字数比 {ratio:.2f} 极端异常"

        if warnings:
            return True, f"ok (warn: {'; '.join(warnings)})"
        return True, "ok"

    # ---------- 单条处理 ----------

    def process_board(self, board_info: Dict) -> Optional[Dict]:
        """处理单个展板：prepare_image → build_prompt → call_api → validate
        Qwen 失败时自动 fallback 到 Kimi"""
        image_id = board_info["image_id"]
        image_path = board_info["image_path"]
        zh_text = board_info.get("zh_text", "")
        en_text = board_info.get("en_text", "")
        is_failed = board_info.get("is_failed_board", False)

        # 准备图片
        image_b64 = self.prepare_image(image_path)
        if not image_b64:
            return None

        # 构建 prompt
        prompt = self.build_prompt(zh_text, en_text, is_failed=is_failed)

        # 调用主 provider
        result = self.call_api(image_b64, prompt)

        # 校验
        if result and isinstance(result, dict) and not result.get("_filtered"):
            ok, reason = self.validate_result(result, zh_text, en_text, is_failed_board=is_failed)
            if ok:
                if "warn" in reason:
                    print(f"  [WARN] {image_id} {reason}", flush=True)
                return self._build_board(image_id, board_info, zh_text, en_text, result)
            else:
                print(f"  [WARN] {image_id} 校验失败: {reason}", flush=True)

        # Fallback 到 Kimi（仅当主 provider 是 qwen 时）
        if self.provider == "qwen":
            filtered = result and isinstance(result, dict) and result.get("_filtered")
            if filtered:
                print(f"  [FALLBACK] {image_id} 被审核拦截，切换 Kimi")
            else:
                print(f"  [FALLBACK] {image_id} Qwen 失败，尝试 Kimi")

            try:
                kimi_client = self._get_kimi_client()
                result = self.call_api(image_b64, prompt,
                                       client=kimi_client, model=cfg.KIMI_MODEL)
                if result and isinstance(result, dict) and not result.get("_filtered"):
                    ok, reason = self.validate_result(result, zh_text, en_text, is_failed_board=is_failed)
                    if ok:
                        self.stats["fallback_kimi"] += 1
                        board = self._build_board(image_id, board_info, zh_text, en_text, result)
                        board["_provider"] = "kimi"
                        return board
                    else:
                        print(f"  [WARN] {image_id} Kimi 也校验失败: {reason}")
            except Exception as e:
                print(f"  [ERROR] {image_id} Kimi fallback 失败: {e}")

        return None

    def _build_board(self, image_id: str, source_info: Dict,
                     zh_in: str, en_in: str, llm_result: Dict) -> Dict:
        """组装最终的 board 结构"""
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

            # 从 sentences 重建段落文本
            para_zh = "".join(s["zh"] for s in sentences)
            para_en = " ".join(s["en"] for s in sentences).strip()

            # alignment_level 默认为 paragraph
            alignment_level = p.get("alignment_level", "paragraph")
            if alignment_level not in ("sentence", "paragraph"):
                alignment_level = "paragraph"

            para = {
                "para_index": i,
                "zh": para_zh,
                "en": para_en,
                "alignment_level": alignment_level,
                "sentences": sentences,
            }
            paragraphs.append(para)

        return {
            "board_id": image_id,
            "source": source_info.get("source", {}),
            "board_title": board_title,
            "corrections": corrections,
            "paragraphs": paragraphs,
            "_provider": self.provider,
        }

    # ---------- 构建处理队列 ----------

    def build_queue(self) -> List[Dict]:
        """构建处理队列"""
        completed_ids = set(self.progress.get("completed", {}).keys())
        skipped_ids = set(self.progress.get("skipped", []))
        progress_failed = set(self.progress.get("failed", []))
        queue = []

        if self.only_failed:
            # 只处理之前 LLM 增强失败的展板（13个）
            # 从 llm_enhance.py 的 progress 中获取失败列表
            llm_progress_file = cfg.ENHANCED_PROGRESS_FILE
            llm_failed_ids = set()
            if llm_progress_file.exists():
                with open(llm_progress_file, 'r', encoding='utf-8') as f:
                    llm_progress = json.load(f)
                llm_failed_ids = set(llm_progress.get("failed", []))
            print(f"[INFO] LLM 增强失败展板: {len(llm_failed_ids)} 个")

            # 同时包含不在 enhanced_corpus 中的展板
            enhanced_ids = set(self.enhanced_boards.keys())
            all_ocr_ids = set(self.ocr_map.keys())
            missing_ids = all_ocr_ids - enhanced_ids - set(
                self.enhanced_corpus.get("metadata", {}).get("skipped_ids", [])
            )

            target_ids = llm_failed_ids | missing_ids
            # 排除已完成的
            target_ids -= completed_ids

            for image_id in sorted(target_ids):
                if image_id in skipped_ids:
                    continue
                ocr_entry = self.ocr_map.get(image_id)
                if not ocr_entry:
                    continue
                image_path = ocr_entry.get("source", {}).get("image_path", "")
                if not image_path or not Path(image_path).exists():
                    print(f"  [SKIP] {image_id}: 图片不存在")
                    continue

                queue.append({
                    "image_id": image_id,
                    "source": ocr_entry.get("source", {}),
                    "image_path": image_path,
                    "zh_text": ocr_entry.get("zh_text", ""),
                    "en_text": ocr_entry.get("en_text", ""),
                    "is_failed_board": True,
                })
            print(f"[INFO] 失败/缺失展板待处理: {len(queue)} 个")
        else:
            # 处理所有非人工清洗的展板
            for ocr_entry in self.ocr_results:
                image_id = ocr_entry.get("image_id", "")
                if not image_id:
                    continue

                # 跳过已完成
                if image_id in completed_ids or image_id in skipped_ids:
                    continue

                # 跳过已人工清洗的展板
                if image_id in self.cleaned_ids:
                    with self._lock:
                        if image_id not in self.progress.get("skipped", []):
                            self.progress.setdefault("skipped", []).append(image_id)
                    continue

                image_path = ocr_entry.get("source", {}).get("image_path", "")
                if not image_path or not Path(image_path).exists():
                    continue

                # 获取当前最佳文本（enhanced > ocr）
                enhanced = self.enhanced_boards.get(image_id)
                if enhanced:
                    zh_text = self._get_board_text(enhanced, "zh")
                    en_text = self._get_board_text(enhanced, "en")
                else:
                    zh_text = ocr_entry.get("zh_text", "")
                    en_text = ocr_entry.get("en_text", "")

                if not zh_text and not en_text:
                    continue

                queue.append({
                    "image_id": image_id,
                    "source": ocr_entry.get("source", {}),
                    "image_path": image_path,
                    "zh_text": zh_text,
                    "en_text": en_text,
                    "is_failed_board": image_id not in self.enhanced_boards,
                })

        return queue

    def _get_board_text(self, board: Dict, lang: str) -> str:
        """从 board 结构中提取完整文本"""
        parts = []
        title = board.get("board_title", {})
        if isinstance(title, dict) and title.get(lang):
            parts.append(title[lang])
        for p in board.get("paragraphs", []):
            if p.get(lang):
                parts.append(p[lang])
            else:
                # 从 sentences 拼接
                sents = p.get("sentences", [])
                if lang == "zh":
                    parts.append("".join(s.get("zh", "") for s in sents))
                else:
                    parts.append(" ".join(s.get("en", "") for s in sents))
        joiner = "\n" if lang == "zh" else "\n"
        return joiner.join(parts).strip()

    # ---------- 线程处理 ----------

    def _process_one_item(self, i: int, total: int, item: Dict) -> Tuple[str, Optional[Dict], Dict]:
        """处理单条（供线程池调用）"""
        image_id = item["image_id"]
        museum = item["source"].get("museum", "?")
        is_failed = "失败补" if item.get("is_failed_board") else "校验"
        print(f"[{i+1}/{total}] {image_id} ({museum}) [{is_failed}]", flush=True)

        # 限速：避免 API 被打满
        time.sleep(1.0)

        result = self.process_board(item)
        return image_id, result, item

    # ---------- 主流程 ----------

    def run(self):
        """运行完整流程"""
        print("=" * 70)
        print("多模态增强处理：VLM 看图校验 + OCR纠错 + 保守对齐")
        print(f"Provider: {self.provider} | 并发线程数: {self.workers}")
        print("=" * 70)

        if not self.setup_client():
            return
        if not self.load_data():
            return
        self.load_progress()

        queue = self.build_queue()
        total = min(len(queue), self.test_limit) if self.test_mode else len(queue)
        print(f"\n待处理: {total} 条" + (" (测试模式)" if self.test_mode else ""))

        if total == 0:
            print("[OK] 没有需要处理的条目")
            self.assemble_corpus()
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
                        provider = result.get("_provider", self.provider)
                        alignment_levels = [p.get("alignment_level", "?") for p in result["paragraphs"]]
                        level_str = ",".join(set(alignment_levels))
                        print(f"  [OK] {image_id}: {n_para}段 {n_sent}句对 "
                              f"纠错zh={n_zh_corr} en={n_en_corr} "
                              f"对齐={level_str} provider={provider}", flush=True)

                        self.progress["completed"][image_id] = {
                            "processed_at": datetime.now().isoformat(),
                            "provider": provider,
                            "result": result,
                        }
                        self.stats["processed"] += 1
                    else:
                        print(f"  [FAIL] {image_id}", flush=True)
                        if image_id not in self.progress.get("failed", []):
                            self.progress.setdefault("failed", []).append(image_id)
                        failed_this_run.append(image_id)
                        self.stats["failed"] += 1

                    self.save_progress()

        # 完成后组装
        print("\n" + "=" * 70)
        self.assemble_corpus()
        self.print_stats()

        if failed_this_run:
            print(f"\n[WARN] 本次失败 {len(failed_this_run)} 条:")
            for fid in failed_this_run:
                print(f"  - {fid}")

    def assemble_corpus(self):
        """从 progress 组装最终的 enhanced_corpus.json
        - 30 个已清洗展板：原样保留
        - 其余展板：使用多模态校验结果（如有），否则保留 LLM 增强结果
        """
        completed = self.progress.get("completed", {})
        if not completed and not self.enhanced_boards:
            print("[INFO] 无已完成数据，跳过组装")
            return

        boards = []
        total_paras = 0
        total_sents = 0
        multimodal_count = 0
        kept_llm_count = 0
        kept_cleaned_count = 0

        # 收集所有应出现的 board_id（保持顺序）
        all_board_ids = []
        seen = set()
        # 先从 enhanced_corpus 中获取已有的 board 顺序
        for b in self.enhanced_corpus.get("boards", []):
            bid = b.get("board_id", "")
            if bid and bid not in seen:
                all_board_ids.append(bid)
                seen.add(bid)
        # 再加入多模态处理的新板（之前失败的）
        for image_id in completed:
            if image_id not in seen:
                all_board_ids.append(image_id)
                seen.add(image_id)

        for bid in all_board_ids:
            if bid in self.cleaned_ids:
                # 已人工清洗：原样保留
                board = self.enhanced_boards[bid]
                boards.append(board)
                kept_cleaned_count += 1
            elif bid in completed:
                # 有多模态结果：使用多模态版本
                board = completed[bid]["result"]
                boards.append(board)
                multimodal_count += 1
            elif bid in self.enhanced_boards:
                # 没有多模态结果但有 LLM 增强结果：保留
                board = self.enhanced_boards[bid]
                boards.append(board)
                kept_llm_count += 1

            # 统计段落/句子
            b = boards[-1] if boards and boards[-1].get("board_id") == bid else None
            if b:
                for p in b.get("paragraphs", []):
                    total_paras += 1
                    total_sents += len(p.get("sentences", []))

        # 保留 deleted_boards
        deleted_boards = self.enhanced_corpus.get("deleted_boards", [])

        corpus = {
            "metadata": {
                "name": "中国博物馆多语种解说词平行语料库（多模态增强版）",
                "version": "4.0",
                "created_at": datetime.now().isoformat(),
                "processor": f"multimodal_enhance.py + {cfg.QWEN_MODEL}",
                "total_boards": len(boards),
                "total_paragraphs": total_paras,
                "total_sentence_pairs": total_sents,
                "composition": {
                    "cleaned_manual": kept_cleaned_count,
                    "multimodal_enhanced": multimodal_count,
                    "llm_enhanced_kept": kept_llm_count,
                },
                "token_usage": self.progress.get("token_usage", {}),
                "skipped": len(self.progress.get("skipped", [])),
                "failed": len(self.progress.get("failed", [])),
            },
            "boards": boards,
        }

        if deleted_boards:
            corpus["deleted_boards"] = deleted_boards

        cfg.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        # 写入到 enhanced_corpus.json（覆盖）
        tmp = cfg.ENHANCED_DIR / "enhanced_corpus.tmp.json"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        if cfg.ENHANCED_CORPUS_FILE.exists():
            # 先备份
            backup = cfg.ENHANCED_DIR / f"enhanced_corpus.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cfg.ENHANCED_CORPUS_FILE.rename(backup)
            print(f"[OK] 已备份旧版: {backup.name}")

        tmp.rename(cfg.ENHANCED_CORPUS_FILE)

        print(f"[OK] 增强语料库已保存: {cfg.ENHANCED_CORPUS_FILE}")
        print(f"     展板: {len(boards)} (清洗={kept_cleaned_count}, "
              f"多模态={multimodal_count}, LLM保留={kept_llm_count})")
        print(f"     段落: {total_paras}, 句对: {total_sents}")

    def print_stats(self):
        """打印统计"""
        usage = self.progress.get("token_usage", {})
        inp = usage.get("total_input", 0)
        out = usage.get("total_output", 0)
        # Qwen 3.5 Plus 定价：0.8元/百万tokens (input+output 统一价)
        cost_qwen = (inp + out) / 1_000_000 * 0.8
        # Kimi 补充的大约成本
        cost_kimi = self.stats.get("fallback_kimi", 0) * 5000 / 1_000_000 * 8  # ~8元/百万

        print(f"\n{'='*70}")
        print("统计")
        print(f"{'='*70}")
        print(f"已完成:      {len(self.progress.get('completed', {}))}")
        print(f"已跳过:      {len(self.progress.get('skipped', []))}")
        print(f"失败:        {len(self.progress.get('failed', []))}")
        print(f"Kimi fallback: {self.stats.get('fallback_kimi', 0)}")
        print(f"Token:       输入 {inp:,} + 输出 {out:,} = {inp+out:,}")
        print(f"预估费用:    Qwen ~¥{cost_qwen:.2f} + Kimi ~¥{cost_kimi:.2f} "
              f"= ~¥{cost_qwen+cost_kimi:.2f}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="多模态增强：VLM 看图校验 + OCR纠错 + 保守对齐"
    )
    parser.add_argument("--test", type=int, metavar="N", default=0,
                        help="测试模式：只处理前N条")
    parser.add_argument("--workers", type=int, default=3,
                        help="并发线程数（默认3）")
    parser.add_argument("--provider", type=str, default="qwen",
                        choices=["qwen", "kimi"],
                        help="主模型提供商（默认qwen）")
    parser.add_argument("--only-failed", action="store_true",
                        help="只处理之前失败/缺失的展板")
    parser.add_argument("--assemble", action="store_true",
                        help="只从 progress 组装最终文件")
    parser.add_argument("--stats", action="store_true",
                        help="显示处理进度统计")
    args = parser.parse_args()

    enhancer = MultimodalEnhancer(
        provider=args.provider,
        test_limit=args.test,
        workers=args.workers,
        only_failed=args.only_failed,
    )

    if args.stats:
        enhancer.load_data()
        enhancer.load_progress()
        enhancer.print_stats()
        return

    if args.assemble:
        enhancer.load_data()
        enhancer.load_progress()
        enhancer.assemble_corpus()
        enhancer.print_stats()
        return

    enhancer.run()


if __name__ == "__main__":
    main()

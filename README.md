# 面向AI同声传译评测的北京博物馆多模态平行语料库构建

> 北京第二外国语学院第一届语料库技术大赛 - 语料库构建类（主赛道）- 文旅与社会传播类
>
> 主题："语言数据 · 智能应用 · 创新表达"

### 团队信息

| 字段 | 内容 |
|------|------|
| **团队名称** | 博古通译队 |
| **指导教师** | 刘斌 |
| **选题分类** | 文旅与社会传播类 |

**团队成员**：

| 姓名 | 专业 | 角色 |
|------|------|------|
| 夏启诚 | 西班牙语 | 负责人 |
| 关海依 | 新闻 | 成员 |
| 卢米拉 | 新闻 | 成员 |
| 李刘飞儿 | 保加利亚语 | 成员 |
| 孔帅豪 | 中文 | 成员 |
| 徐紫嫣 | 日语 | 成员 |

---

## 一、项目概述

本项目以中国博物馆展板解说词为语料来源，通过手机实地拍摄、OCR自动识别、人工审核校对、**LLM自动纠错与多级对齐**的技术流程，构建了一个**句子级对齐的五语种平行语料库**（中、英、日、西班牙、保加利亚语）。

### 核心特点

- **金标准中英对照**：中文和英文原文来自博物馆实地展板，由专业翻译团队提供，是绝对可靠的参考译文，可作为AI评测的黄金标准
- **五语种平行**：中、英、日、西班牙、保加利亚语，1,959 条五语种齐全的句子对
- **句子级对齐**：展板 → 段落 → 句子，三级层次结构完整保留从属关系
- **多模型博弈翻译**：每种目标语言使用 2 个翻译模型 + 1 个裁判模型，确保翻译质量
- **LLM 自动纠错**：DeepSeek V3 + Qwen VL 自动修复 OCR 错误
- **质量三级分级**：基于 OCR 置信度和人工审核状态，将语料分为 A/B/C 三级
- **全流程自动化**：从原始图片到结构化语料库，提供一键式处理管道
- **人工审核闭环**：内置 Web 审核界面，支持确认、修正、跳过操作
- **多格式导出**：JSON、TSV、TMX（翻译记忆库）、纯文本平行语料

### 数据规模

| 指标 | 数值 |
|------|------|
| 原始图片总数 | **605 张** |
| 覆盖博物馆 | 5 个（故宫、国家博物馆、首都博物馆、党史馆、抗日纪念馆） |
| OCR 成功处理 | 583 张 |
| **最终展板数** | **556 条** |
| **段落级对齐** | **834 对** |
| **句子级对齐** | **2,678 对** |
| **五语种齐全** | **1,959 对** |
| 完美双语展板 | 393 (70.6%) |
| 含单语段落 | 72 (12.9%) |
| 疑似简略翻译 | 112 (20.1%) |

### 五语种覆盖率

| 语言 | 完成数 | 缺失数 | 覆盖率 |
|------|--------|--------|--------|
| 中文 (zh) | 2,307 | 371 | 86.1% |
| 英文 (en) | 2,355 | 323 | 87.9% |
| 日语 (ja) | 2,628 | 50 | 98.1% |
| 西班牙语 (es) | 2,619 | 59 | 97.8% |
| 保加利亚语 (bg) | 2,658 | 20 | 99.3% |

---

## 二、技术架构

### 2.1 工作流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 原始图片  │ ─→ │ OCR 提取  │ ─→ │ 人工审核  │ ─→ │ LLM 增强     │ ─→ │ 多模态校验 │ ─→ │ 多语种翻译  │ ─→ │ 数据修复     │ ─→ │ 导出语料  │
│ 605张     │    │ RapidOCR │    │ Gradio   │    │ DeepSeek V3  │    │ Qwen VL   │    │ ja/bg/es    │    │ 合并+补全    │    │ 多格式   │
└──────────┘    └──────────┘    └──────────┘    │ OCR纠错      │    └───────────┘    │ 多模型博弈  │    └──────────────┘    └──────────┘
  阶段一           阶段二           阶段三       │ 段落分割+对齐 │      阶段五        └──────────────┘         阶段七            阶段八
                                                │ 句子分割+对齐 │                     阶段六
                                                └──────────────┘
                                                   阶段四
```

### 2.2 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| OCR 引擎 | RapidOCR (onnxruntime) | 离线运行，无需联网 |
| 语言检测 | Unicode 字符集正则匹配 | CJK 范围检测中文，Latin 检测英文 |
| 审核界面 | Gradio 6.x Web UI | 浏览器内操作，支持图片预览 |
| **LLM 增强** | **DeepSeek V3 API** | **OCR 纠错 + 段落/句子级对齐，5线程并发** |
| **多语种翻译** | **Gemini 3 Flash + Kimi K2.5 + Claude Sonnet（302.ai中转）** | **多模型并行 + LLM裁判，3语种×2翻译模型** |
| 数据格式 | JSON / TSV / TMX | 兼容主流翻译工具和 NLP 框架 |

### 2.3 模型选择依据

本项目基于 **Text Arena**（arena.ai）排行榜选择翻译模型，每个目标语言采用 **2个翻译模型 + 1个裁判模型** 的博弈机制：

| 目标语言 | 翻译模型A | 翻译模型B | 裁判模型 | 说明 |
|----------|-----------|-----------|----------|------|
| 日语 | Gemini 3 Flash | Qwen-MT-Turbo | Gemini 3 Flash | 排行榜日语榜前排，通用性能稳定 |
| 西班牙语 | Kimi K2.5 | Gemini 3 Flash | Gemini 3 Flash | 排行榜西班牙语榜前排 |
| 保加利亚语 | Claude Sonnet 4.6 | Gemini 3 Flash | Gemini 3 Flash | WMT翻译冠军模型 |
| OCR 纠错 | DeepSeek V3 | - | - | 国产开源模型，性价比极高 |
| 多模态校验 | Qwen 3.5 Plus | - | - | 国产多模态最强，可直接读取展板图片校对 |

> **选择逻辑**：通过 302.ai 中转统一接入多个模型，使用双翻译 + LLM 裁判机制选择最佳译文。

### 2.4 质量分级标准

| 等级 | 条件 | 数量 | 占比 |
|------|------|------|------|
| **A 级** | OCR 置信度 > 0.9 **且** 人工已确认，或人工修正 | 0 | 0%（待审核） |
| **B 级** | OCR 置信度 0.8 ~ 0.9 | 550 | 94.3% |
| **C 级** | OCR 置信度 < 0.8 | 33 | 5.7% |

> A 级语料当前为 0 是因为尚未完成人工审核阶段。完成审核后，预计大部分 B 级语料将升为 A 级。

---

## 三、目录结构

```
nlp大赛/
│
├── 编程自动化/                          ← 全部处理脚本（21个 Python 文件）
│   ├── config.py                       ⭐ 集中配置模块（所有路径和参数）
│   ├── ocr_processor.py                ⭐ OCR 处理器（段落级对齐）
│   ├── llm_enhance.py                  ⭐ LLM 增强（OCR纠错 + 段落/句子对齐，5线程并发）
│   ├── multimodal_enhance.py           ⭐ 多模态增强（Qwen VL 看图校验 + Kimi fallback）
│   ├── multilingual_translate.py       ⭐ 多语种翻译脚本（多模型博弈，ja/bg/es）
│   ├── quality_exporter.py             ⭐ 质量分级与多格式导出
│   ├── create_parallel_corpus.py       ⭐ 平行语料库生成（兼容两种数据格式）
│   ├── full_merge_fix.py               ⭐ 合并一板两拍展板
│   ├── fix_merged_translations.py      ⭐ 为合并后展板补全翻译
│   ├── fill_missing_translations.py    ⭐ 补全缺失的单语种翻译
│   ├── make_clean_corpus.py            ⭐ 生成简洁版五语种语料库
│   ├── ocr_processor_rapid.py             OCR 处理器（图片级分组，早期版本）
│   ├── merge_results.py                   合并多批次 OCR 结果
│   ├── llm_assistant.py                   LLM 辅助纠错（早期版本，已被 llm_enhance.py 取代）
│   ├── run_pipeline.py                    完整流程编排脚本
│   ├── run_ocr_batch.py                   批量 OCR 处理脚本
│   ├── review_monolingual.py              单语图片审核模块
│   ├── ocr_preview.py                     OCR 预览工具
│   ├── ocr_preview_easy.py                简化版预览
│   ├── ocr_preview_rapid.py               快速预览
│   └── test_rapidocr.py                   OCR 引擎测试
│
├── 审核界面/                            ← 人工审核 Web 界面
│   ├── review_app.py                   ⭐ Gradio 审核应用（主入口）
│   └── start_review.bat                   Windows 一键启动脚本
│
├── 原始语料/                            ← 手机拍摄的博物馆展板图片（605张）
│   ├── 故宫/                              113 张
│   ├── 国家博物馆/                         288 张
│   ├── 首都博物馆/                         67 张
│   ├── 党史馆/                            18 张
│   └── 抗日纪念馆/                        119 张
│
├── 中间结果/                            ← OCR 处理和审核的中间产物
│   ├── ocr_results.json                ⭐ 主 OCR 结果（583条，8.3MB）
│   ├── ocr_results_all.json               合并后的完整 OCR 结果
│   ├── review_queue.json                  需人工审核队列（505条）
│   ├── grade_A_all.json                   A 级语料（合并版）
│   ├── grade_B_all.json                   B 级语料（合并版）
│   ├── grade_C_all.json                   C 级语料（合并版）
│   ├── corpus.tsv                         TSV 格式中间结果
│   ├── corpus_text.txt                    可读文本格式
│   ├── enhanced/                        ⭐ LLM 增强结果
│   │   ├── enhanced_corpus.json        ⭐ 增强后语料库（556条，含句子级对齐）
│   │   ├── multilingual_corpus.json    ⭐ 五语种平行语料库（1,959条五语种齐全）
│   │   ├── translation_progress.json   ⭐ 翻译进度（断点续传）
│   │   └── progress.json                  LLM增强处理进度（断点续传用）
│   └── 审核结果/                          人工审核输出目录
│       ├── reviewed_results.json          完整审核结果
│       ├── confirmed_corpus.json          确认的语料
│       ├── corrected_corpus.json          修正后的语料
│       └── skipped_list.json              跳过列表
│
├── ocr输出/                             ← 平行语料库（早期 OCR 输出）
│   ├── parallel_corpus.json            ⭐ 平行语料库主文件（433条）
│   ├── parallel_texts.txt                 纯文本对照版
│   ├── 故宫_parallel.json                 各馆分文件
│   ├── 国家博物馆_parallel.json
│   └── 首都博物馆_parallel.json
│
├── 输出/                                ← 最终导出文件（8个文件，共 6.7MB）
│   ├── corpus_level_a.json                A 级语料（JSON）
│   ├── corpus_level_b.json                B 级语料（JSON，1.9MB，550条）
│   ├── corpus_level_c.json                C 级语料（JSON，504KB，33条）
│   ├── corpus_all.json                    全部语料（JSON，2.5MB）
│   ├── corpus.tsv                         表格格式（584行）
│   ├── corpus.tmx                         TMX 翻译记忆库格式（812KB）
│   ├── corpus_parallel.txt                纯文本平行语料（2920行）
│   └── corpus_stats.json                  语料库统计报告
│
├── 交付/                                ⭐ 核心交付物
│   ├── parallel_corpus_5lang.json      ⭐ 五语种平行语料库（1,959对，2.75MB）
│   ├── multilingual_corpus.json          完整五语种语料（含候选译文）
│   ├── enhanced_corpus.json              增强后中英双语语料
│   ├── museum_glossary.json              五语种术语表
│   └── corpus.tmx                         TMX 翻译记忆库
│
├── 词典/                                ← OCR 纠错词典 + 多语种术语表
│   ├── common_ocr_errors.json             100+ 条中英文常见 OCR 错误映射
│   └── museum_glossary.json            ⭐ 五语种博物馆术语表（110+条，ja/bg/es/en/zh）
│
├── README.md                            ← 本文件
├── PROJECT_STATUS.md                       项目状态文档
└── 项目进度报告.md                         进度报告
```

---

## 四、快速开始

### 4.1 环境准备

```bash
# Python 3.10+ 推荐
pip install rapidocr-onnxruntime gradio

# 可选：LLM 辅助纠错
pip install openai
set DEEPSEEK_API_KEY=your_api_key    # Windows
```

### 4.2 一键运行完整流程

```bash
cd 编程自动化
python run_pipeline.py --stage all        # 完整流程：OCR → 审核 → 导出
python run_pipeline.py --stage all --test # 测试模式（仅处理10张）
```

### 4.3 分阶段运行

#### 阶段一：OCR 提取

```bash
cd 编程自动化

# 测试模式（处理10张）
python ocr_processor.py

# 批量处理（处理全部605张）
python run_ocr_batch.py
```

输出到 `中间结果/`：
- `ocr_results.json` — 完整 OCR 结果（核心文件）
- `review_queue.json` — 需人工审核队列
- `grade_A/B/C_corpus.json` — 按质量分级
- `corpus.tsv` — 表格格式
- `corpus_text.txt` — 人类可读文本

#### 阶段二：人工审核

```bash
cd 审核界面
python review_app.py
# 浏览器自动打开 http://127.0.0.1:7865
```

或使用 Windows 一键启动：
```
双击 审核界面/start_review.bat
```

**审核界面功能说明：**

| 区域 | 内容 |
|------|------|
| 左侧面板 | 原始展板图片 + 图片元信息（博物馆、质量等级、OCR置信度） |
| 右侧「OCR结果」标签 | 自动提取的中文和英文文本（只读） |
| 右侧「修正」标签 | 可输入修正后的中/英文本 |
| 质量警告 | 显示低置信度的文本块及其置信度数值 |
| 审核备注 | 可输入任意备注信息 |

**操作按钮：**

| 按钮 | 作用 | 适用场景 |
|------|------|---------|
| ✅ 确认正确 | OCR 结果无误，标记为已确认 | OCR 质量好、无需修改 |
| 💾 保存修正 | 保存修正后的文本 | OCR 有错误，已在「修正」标签中改正 |
| ⏭️ 跳过 | 跳过此图片 | 图片质量差或不适合入库 |
| 🔍 跳到下一个未审核 | 自动跳转到下一个未审核的条目 | 快速定位待审核项 |
| ⬅️ / ➡️ | 前后翻页 | 逐条浏览 |

审核结果自动保存到 `中间结果/审核结果/`。

#### 阶段三：质量分级与导出

```bash
cd 编程自动化
python quality_exporter.py
```

输出到 `输出/`，共 8 个文件：
- `corpus_level_a.json` — A 级高质量语料
- `corpus_level_b.json` — B 级中等质量语料
- `corpus_level_c.json` — C 级需关注语料
- `corpus_all.json` — 全部语料（含所有等级）
- `corpus.tsv` — 表格格式（便于 Excel 导入）
- `corpus.tmx` — TMX 翻译记忆库格式
- `corpus_parallel.txt` — 纯文本平行语料
- `corpus_stats.json` — 统计报告

#### 阶段四：生成平行语料库（可选）

```bash
cd 编程自动化
python create_parallel_corpus.py
```

输出到 `ocr输出/`：
- `parallel_corpus.json` — 多语种平行语料库主文件（预留翻译字段）
- `parallel_texts.txt` — 纯文本对照版
- `{博物馆名}_parallel.json` — 按博物馆分文件

---

## 五、配置说明

所有配置集中在 `编程自动化/config.py` 中，无需修改各脚本中的硬编码路径。

### 5.1 路径配置

```python
# config.py 自动检测项目根目录（无需手动修改）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGE_DIR   = PROJECT_ROOT / "原始语料"     # 原始图片
INTERMEDIATE_DIR = PROJECT_ROOT / "中间结果"     # OCR 中间结果
FINAL_OUTPUT_DIR = PROJECT_ROOT / "输出"         # 最终导出
```

### 5.2 OCR 配置

```python
TEST_MODE = False         # True = 仅处理前10张
TEST_MODE_LIMIT = 10
SELECTED_MUSEUMS = []     # 空列表 = 处理所有博物馆
                          # 例如: ["国家博物馆", "故宫"]

QUALITY_CONFIG = {
    "ocr_confidence_threshold": 0.85,  # 低于此值标记需审核
    "low_confidence_threshold": 0.8,   # 低置信度分界线
}
```

### 5.3 审核界面配置

```python
REVIEW_SERVER_PORT = 7865
REVIEW_SERVER_NAME = "127.0.0.1"    # 本地访问
```

### 5.4 LLM 增强配置

```python
# DeepSeek V3 API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-85f2...8f39")  # 已内置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TIMEOUT = 120       # API 超时（秒）
LLM_MAX_RETRIES = 3     # 单次 API 调用最大重试

# 增强处理参数
LLM_ENHANCE_CONFIG = {
    "rate_limit_delay": 0.5,     # API 调用间隔（秒）
    "max_tokens_per_call": 8192, # 单次最大输出 token
    "short_text_threshold": 100, # 短文本阈值（字符数）
    "content_ratio_min": 0.70,   # 输出/输入最小字数比
    "content_ratio_max": 1.30,   # 输出/输入最大字数比
}
```

---

## 六、数据格式

### 6.1 OCR 结果格式（`中间结果/ocr_results.json`）

```json
{
  "metadata": {
    "total_images": 583,
    "processed": 583,
    "needs_review": 505,
    "processor": "RapidOCR",
    "alignment_method": "paragraph_level"
  },
  "results": [
    {
      "image_id": "党史馆_IMG20260129131237",
      "source": {
        "museum": "党史馆",
        "image_name": "IMG20260129131237.jpg",
        "image_path": "E:\\...\\原始语料\\党史馆\\IMG20260129131237.jpg",
        "processed_at": "2026-02-09T16:56:58"
      },
      "zh_text": "完整的中文段落...",
      "en_text": "Complete English paragraph...",
      "text_blocks": [
        {
          "text": "前言",
          "bbox": [[1810, 30], [2273, 37], [2270, 221], [1807, 214]],
          "confidence": 0.6195,
          "lang": "zh"
        }
      ],
      "quality": {
        "grade": "B",
        "zh_block_count": 18,
        "en_block_count": 13,
        "avg_ocr_confidence": 0.8723,
        "low_confidence_blocks": [...]
      },
      "needs_review": true,
      "review_reason": "low_ocr_confidence(5blocks)"
    }
  ]
}
```

### 6.2 审核结果格式（`中间结果/审核结果/reviewed_results.json`）

```json
{
  "metadata": { "total": 583, "reviewed": 42 },
  "results": [
    {
      "image_id": "党史馆_IMG20260129131237",
      "source": {...},
      "original_zh": "OCR识别的中文原文",
      "original_en": "OCR recognized English text",
      "corrected_zh": "人工修正后的中文（如有）",
      "corrected_en": "Manually corrected English (if any)",
      "review_status": "confirmed | corrected | skipped",
      "review_comment": "审核备注",
      "quality": {...}
    }
  ]
}
```

### 6.3 平行语料库格式（`ocr输出/parallel_corpus.json`）— 早期版本

> 注意：此为 v1.0 的产物，仅含展板级对齐（433条）。v3.0 的核心输出是 `enhanced_corpus.json`，见 6.4 节。

### 6.4 增强语料库格式（`中间结果/enhanced/enhanced_corpus.json`）⭐ 核心输出

这是 v3.0 的核心产物，包含句子级对齐和 OCR 纠错信息：

```json
{
  "metadata": {
    "name": "中国博物馆多语种解说词平行语料库（LLM增强版）",
    "version": "3.0",
    "created_at": "2026-02-10T...",
    "processor": "llm_enhance.py + deepseek-chat",
    "total_boards": 550,
    "total_paragraphs": 1134,
    "total_sentence_pairs": 2502,
    "token_usage": {"total_input": 1021041, "total_output": 409442},
    "skipped": 6,
    "failed": 27
  },
  "boards": [
    {
      "board_id": "党史馆_IMG20260129131237",
      "source": {
        "museum": "党史馆",
        "image_name": "IMG20260129131237.jpg",
        "image_path": "E:\\...\\原始语料\\党史馆\\IMG20260129131237.jpg"
      },
      "board_title": {"zh": "前言", "en": "INTRODUCTION"},
      "corrections": {
        "zh_changes": [
          {"from": "砺品格", "to": "砥砺品格", "type": "missing_char"}
        ],
        "en_changes": [
          {"from": "cpC", "to": "CPC", "type": "case_error"},
          {"from": "journeytoward", "to": "journey toward", "type": "word_split"},
          {"from": "thitough", "to": "through", "type": "letter_substitution"}
        ]
      },
      "paragraphs": [
        {
          "para_index": 0,
          "zh": "中文段落全文...",
          "en": "English paragraph...",
          "sentences": [
            {
              "sent_index": 0,
              "zh": "中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队，是中国特色社会主义事业的领导核心。",
              "en": "The Communist Party of China is the vanguard of the Chinese working class; ..."
            }
          ]
        }
      ]
    }
  ]
}
```

**纠错类型（corrections.type）说明：**

| type | 含义 | 示例 |
|------|------|------|
| `word_split` | OCR 将两个词粘连 | journeytoward → journey toward |
| `letter_substitution` | OCR 字母识别错误 | thitough → through |
| `case_error` | OCR 大小写错误 | cpC → CPC |
| `form_similar` | 中文形近字混淆 | 土 → 士 |
| `missing_char` | OCR 漏识别字符 | 砺品格 → 砥砺品格 |

### 6.5 五语种平行语料库格式（`交付/parallel_corpus_5lang.json`）⭐ 最终交付

这是项目的核心交付物，包含五语种齐全的句子对：

```json
{
  "corpus_id": "bisu-museum-parallel-2026",
  "name": "中国博物馆多语种解说词平行语料库",
  "version": "1.0",
  "created": "2026-02-20",
  "languages": ["zh", "en", "ja", "es", "bg"],
  "language_names": {
    "zh": "中文", "en": "English", "ja": "日本語",
    "es": "Español", "bg": "Български"
  },
  "metadata": {
    "source": "故宫、国家博物馆、首都博物馆、党史馆、抗日纪念馆展板",
    "domain": "博物馆解说词",
    "total_units": 1959,
    "coverage": {"zh": 1959, "en": 1959, "ja": 1959, "es": 1959, "bg": 1959}
  },
  "alignment_units": [
    {
      "id": "党史馆_IMG20260129131237_p0s0",
      "source": {
        "museum": "党史馆",
        "image": "IMG20260129131237.jpg",
        "title_zh": "前言",
        "title_en": "INTRODUCTION"
      },
      "translations": {
        "zh": "中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队，是中国特色社会主义事业的领导核心。",
        "en": "The Communist Party of China is the vanguard of the Chinese working class...",
        "ja": "中国共産党は、中国労働者階級の先鋒隊であると同時に...",
        "es": "El Partido Comunista de China es la vanguardia de la clase obrera china...",
        "bg": "Китайската комунистическа партия е авангард на китайската работническа класа..."
      }
    }
  ]
}
```

```json
{
  "metadata": {
    "name": "中国博物馆多语种解说词平行语料库",
    "version": "1.0",
    "source_languages": ["zh", "en"],
    "target_languages": []
  },
  "entries": [
    {
      "parallel_id": "国家博物馆_IMG20260121104914",
      "source": { "museum": "国家博物馆", "image_name": "..." },
      "parallel_texts": {
        "zh": "中文解说词...",
        "en": "English description..."
      },
      "translations": {
        "targets": { "ja": "", "fr": "", "de": "" },
        "status": "pending"
      }
    }
  ]
}
```

---

## 七、按博物馆分布

| 博物馆 | 图片数 | B级以上 | C级 | 说明 |
|--------|--------|---------|-----|------|
| 国家博物馆 | 288 | 253 | 14 | 数据量最大，涵盖青铜器、瓷器、书画等 |
| 抗日纪念馆 | 119 | 107 | 12 | 历史解说词，中英对照 |
| 故宫 | 113 | 106 | 6 | 宫殿和文物介绍 |
| 首都博物馆 | 67 | 66 | 1 | 北京地方史和文物 |
| 党史馆 | 18 | 18 | 0 | 党史展览解说词 |
| **总计** | **605** | **550** | **33** | |

---

## 八、关键设计决策

### 8.1 三级文本对齐：展板 → 段落 → 句子

v3.0 实现了三级层次对齐，从最初的展板级（一张图片=一段中+一段英）升级为句子级：

```
展板 (board)
├── board_title: {zh: "前言", en: "INTRODUCTION"}
├── corrections: {zh_changes: [...], en_changes: [...]}
└── paragraphs[]
    ├── paragraph 0
    │   ├── zh: "中文段落全文..."
    │   ├── en: "English paragraph..."
    │   └── sentences[]
    │       ├── {zh: "中文句子1。", en: "English sentence 1."}
    │       └── {zh: "中文句子2。", en: "English sentence 2."}
    └── paragraph 1
        └── ...
```

段落级 zh/en 文本由 sentences 拼接自动生成，无冗余存储。

### 8.2 LLM 增强的核心原则：清洗而非改写

**这是语料清洗和标准化，不是内容创作。** LLM 的职责严格限定为：

1. **纠正 OCR 错误**：仅修复明确的 OCR 识别错误（词语粘连、字母替换、形近字等）
2. **结构化分割**：将连续文本分割为段落和句子（纯结构操作，不改变文本内容）
3. **双语对齐**：将中文段落/句子与对应英文段落/句子配对

**严禁行为**：
- 改写、润色、优化原文表达
- 添加原文中不存在的内容（包括标题、注释等）
- 删除原文中存在的内容
- 输出文本与输入文本的每一处差异都必须记录在 corrections 字段中

这些规则在 `llm_enhance.py` 的 `SYSTEM_PROMPT` 中以英文明确声明。

### 8.3 LLM 增强的输入优先级

处理每条展板时，文本来源的优先级为：

```
人工修正的文本 > 人工确认的原文 > 原始 OCR 文本
人工标记 "跳过" 的条目 → 完全排除，不处理
人工标记 "[删除]" 的语种 → 该语种文本清空
```

实现见 `llm_enhance.py` 的 `resolve_text()` 方法。

### 8.4 并发处理与断点续传

- **5 线程并发**：使用 `ThreadPoolExecutor` 同时处理 5 条展板，吞吐量约 20-30 条/分钟
- **原子写入**：每条完成后立即保存 `progress.json`（先写临时文件再 rename，防断电丢失）
- **自动跳过**：已完成/已失败/已跳过的条目在下次运行时自动排除
- **内容校验**：输出文本与输入文本的字数比必须在 0.70~1.30 范围内，否则判定为失败

### 8.5 失败条目分析（27条，4.6%）

550 条成功，27 条失败。失败原因分为 4 类：

| 失败原因 | 数量 | 典型特征 |
|---------|------|---------|
| 中英文长度严重失衡（比例 <0.4） | ~15 | 一种语言内容远多于另一种，段落对齐无法成立 |
| OCR 质量极差（Grade C，置信度 <0.75） | ~10 | 文本大量乱码，LLM 无法还原 |
| 一种语言文本近乎为空（<10字符） | 2 | 纯单语展板，OCR 另一语种为噪声 |
| 文本结构混乱（多展板混排） | ~3 | 相邻展板内容被 OCR 合并在一起 |

失败条目列表存储在 `中间结果/enhanced/progress.json` 的 `failed` 数组中。这些条目需要人工处理。

### 8.2 两种 OCR 处理器

项目包含两个 OCR 处理器，分别产生不同格式的数据：

| 处理器 | 输出格式 | 核心字段 | 适用场景 |
|--------|---------|---------|---------|
| `ocr_processor.py`（主力） | `zh_text` / `en_text` | 段落级文本 + 质量分级 | 审核界面、质量导出 |
| `ocr_processor_rapid.py`（早期） | `image_content.zh_all` / `en_all` | 图片级分组 + 分类 | 平行语料库生成 |

下游脚本（`create_parallel_corpus.py`）已做兼容处理，能自动检测并适配两种格式。

### 8.3 OCR 纠错词典

`词典/common_ocr_errors.json` 包含 100+ 条领域特定的常见 OCR 错误映射：
- **形近字错误**：土/士、己/已/巳、人/入/八
- **博物馆专业词汇**：出士→出土、青銅→青铜、博勿院→博物院
- **英文拼写错误**：bronse→bronze、ceramicc→ceramic
- **大小写错误**：bc→BC、ad→AD

---

## 九、依赖安装

### 必需依赖

```bash
pip install rapidocr-onnxruntime    # OCR 引擎
pip install gradio                  # 审核界面（当前版本 6.5.1）
```

### 可选依赖

```bash
pip install openai                  # LLM 辅助纠错（需 DeepSeek API Key）
```

### 运行环境

- Python 3.10+（已测试 3.13.5）
- Windows 10/11（路径兼容 Windows）
- 约 100MB 磁盘空间（不含原始图片）

---

## 十、开发日志

### v1.0 — 2026-01-28（初始版本）

- 完成 605 张博物馆展板图片采集（5个博物馆）
- 使用 RapidOCR 完成全部图片 OCR 处理
- 构建 433 条有效平行条目（430 双语图片 + 3 人工配对）
- 生成 `parallel_corpus.json` 平行语料库

### v2.0 — 2026-02-09（架构重构）

新增段落级对齐处理器 `ocr_processor.py`，产生 583 条结果（含 5 个博物馆全量数据），新增审核界面和质量导出。

### v2.1 — 2026-02-10（全面 Debug）

针对之前版本存在的架构问题和运行 Bug，进行了系统性的诊断和修复：

#### 审核界面修复

| 问题 | 原因 | 修复 |
|------|------|------|
| 应用无法启动，端口冲突 | `__pycache__` 残留旧字节码（端口 7860），而源码已改为 7865 | 删除 `__pycache__`，代码锁定端口 7865 |
| 浏览器打不开页面 | `server_name="0.0.0.0"` 导致浏览器尝试访问无效地址 | 改为 `"127.0.0.1"` |
| 一次展示全部 583 条 | 无筛选逻辑 | 新增 `needs_review` 筛选，默认只展示待审核项 |
| 文本框太小看不全 | `lines=8` 不够 | 改为 `lines=15, max_lines=15`（固定高度+滚动条） |
| 无法删除某一语种文本 | 空白修正被视为"无修改" | 新增 `[删除]` 标记约定 |
| 跳过的条目仍进入语料库 | `quality_exporter.py` 不检查 skip 状态 | 新增 `review_status == "skipped"` 排除逻辑 |

#### 架构优化

| 改进 | 说明 |
|------|------|
| 新增 `config.py` 集中配置 | 所有路径、阈值、端口号统一管理 |
| stdout 编码安全机制 | 全局标志 `_museum_encoding_set` 防止重复包装 stdout |
| 数据格式兼容层 | `create_parallel_corpus.py` 自动检测两种数据格式 |

### v3.0 — 2026-02-10（LLM 增强：OCR 纠错 + 句子级对齐）⭐ 重大升级

引入 DeepSeek V3 API，将语料从展板级提升到**句子级对齐**，同时自动修复 OCR 错误。

#### 核心成果

| 指标 | v2.1 | v3.0 | 提升 |
|------|------|------|------|
| 对齐粒度 | 展板级（583条） | **句子级（2,502对）** | 4.3x |
| OCR 纠错 | 手动 | **自动 + 可追溯** | corrections 字段记录每处修改 |
| 段落结构 | 无 | **1,134个段落** | 新增 |
| 处理成本 | 0 | **$0.73（￥5.2）** | — |
| 处理速度 | — | **~20条/分钟**（5线程并发） | — |

#### 新增文件

| 文件 | 说明 | 行数 |
|------|------|------|
| `编程自动化/llm_enhance.py` | LLM 增强主脚本：Prompt 模板、API 调用、JSON 校验、并发处理、断点续传 | ~720行 |
| `中间结果/enhanced/enhanced_corpus.json` | 增强后的层级语料库（550 展板，1134 段落，2502 句对） | — |
| `中间结果/enhanced/progress.json` | 处理进度文件（断点续传用） | — |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `编程自动化/config.py` | 新增 DeepSeek API Key、API URL、增强目录路径、处理参数（rate_limit、max_tokens 等） |
| `审核界面/review_app.py` | 文本框加大（lines=15）、新增 [删除] 操作提示 |
| `编程自动化/quality_exporter.py` | 支持 [删除] 标记处理、排除 skipped 条目 |

#### llm_enhance.py 关键设计

**Prompt 策略（3种模板）：**

| 模板 | 触发条件 | 用途 |
|------|---------|------|
| `build_full_prompt` | 双语文本 > 100 字符 | 完整处理：纠错 + 段落分割 + 句子对齐 |
| `build_short_prompt` | 双语文本 ≤ 100 字符 | 短标题/标签：仅纠错 |
| `build_mono_prompt` | 仅一种语言 | 单语处理：纠错 + 分段（另一语种留空） |

**处理流程（每条展板）：**

```
resolve_text()          → 选择最佳文本来源（人工修正 > 确认 > OCR）
  ↓
build_*_prompt()        → 根据文本特征选择 Prompt
  ↓
call_api() × 最多3次   → 调用 DeepSeek API，带重试和指数退避
  ↓
extract_json()          → 从响应中提取 JSON（兼容 markdown 代码块和裸 JSON）
  ↓
validate_result()       → 结构校验 + 内容完整性校验（字数比 0.70~1.30）
  ↓
_build_board()          → 从 sentences 重建 paragraph 文本，组装最终结构
  ↓
save_progress()         → 原子写入 progress.json
```

**并发模型：**

```python
ThreadPoolExecutor(max_workers=5)
  → 5 个线程同时调用 API
  → threading.Lock 保护 progress 读写和 token 计数
  → as_completed() 按完成顺序收集结果
```

**断点续传机制：**

- `progress.json` 包含 `completed`（dict）、`failed`（list）、`skipped`（list）
- `build_queue()` 自动排除这三类条目，仅处理未触及的条目
- 任何时刻中断后 `--resume` 可从中断点继续
- 失败条目不会无限重试——记入 failed 后在后续运行中自动跳过

**CLI 用法：**

```bash
cd 编程自动化

python llm_enhance.py                      # 处理全部（自动断点续传）
python llm_enhance.py --test 5             # 测试前5条
python llm_enhance.py --workers 3          # 3线程（默认5）
python llm_enhance.py --resume             # 从断点恢复（与无参数等效）
python llm_enhance.py --stats              # 查看进度统计
python llm_enhance.py --assemble           # 从 progress.json 重新组装 enhanced_corpus.json
```

#### Prompt 设计的迭代过程

这是本版本最关键的调优，经历了 3 轮迭代：

| 版本 | 问题 | 修复 |
|------|------|------|
| Prompt v1 | LLM 补全了 OCR 截断的英文句子，添加了原文中不存在的内容 | 加入 "Do NOT add any text that was not in the input" 规则 |
| Prompt v2 | 过于保守，LLM 拒绝分段（整块展板=1个段落），句子对齐崩溃 | 明确 "Structural splitting is encouraged — this is NOT content modification" |
| Prompt v3（最终版） | 平衡：严格保留文本内容 + 积极进行结构分割 | 5段落+9句对（第一条测试数据），与原文逐字一致 |

核心教训：**对 LLM 说"不要修改内容"时必须同时说"但结构分割是允许的"，否则 LLM 会连分段都不敢做。**

#### 失败条目详情（27条）

**按博物馆分布：**

| 博物馆 | 失败数 | 总数 | 失败率 |
|--------|--------|------|--------|
| 抗日纪念馆 | 11 | 119 | 9.2% |
| 故宫 | 10 | 113 | 8.8% |
| 国家博物馆 | 4 | 288 | 1.4% |
| 首都博物馆 | 2 | 67 | 3.0% |
| 党史馆 | 0 | 18 | 0% |

**主要失败原因：**

1. **中英文长度严重失衡**（~15条）：一种语言的内容量是另一种的 3-8 倍，段落级 1:1 对齐无法成立
2. **OCR 极度混乱**（~10条）：大量乱码字符，低置信度区块超过 100 个，LLM 无法还原
3. **一种语言几乎为空**（2条）：如 `国家博物馆_IMG20260121123209`（中文仅"大地"2字）
4. **多展板文本混排**（~3条）：相邻展板内容被 OCR 合并

这些条目需要人工在审核界面中处理（确认、修正或跳过）。

### v5.4 — 2026-02-24（三语种音频对齐完成）⭐ 音频语料

#### 问题根因

三种语言的 Whisper 转录存在不同问题：
- **保加利亚语**：原始转录存在幻觉污染（第 7~49 个 segment 循环输出同一段文本）
- **西班牙语**：原始对齐率仅 63.7%，同样受幻觉影响
- **日语**：专有名词读法不确定（日本同学标注了 97 个拿不准的词），Whisper 字符级分词导致 SequenceMatcher 匹配失败

#### 解决方案

| 语言 | 方案 |
|------|------|
| 保加利亚语/西班牙语 | faster-whisper large-v3 重新转录 + 全局非顺序对齐（SequenceMatcher） |
| 日语 | **LLM 语义对齐（Qwen 3.5 Plus）**：即使专有名词读音不同，LLM 也能理解语义等价 |

#### 修复结果

| 语言 | 原对齐率 | 新对齐率 | 方法 | 音频文件 |
|------|---------|---------|------|----------|
| 保加利亚语 (bg) | 2.8% | **99.1%** | SequenceMatcher | 212个，95.3MB |
| 西班牙语 (es) | 63.7% | **99.1%** | SequenceMatcher | 212个，102.6MB |
| 日语 (ja) | 70.3% | **100%** | LLM语义对齐 | 212个，104.6MB |
| **总计** | — | — | — | **636个，302.5MB** |

#### 日语 LLM 语义对齐原理

日语 Whisper 会把单词拆成单字符（如"陶器"→"陶 器"），且专有名词读音与稿件不同（如"耀州窯"读成"ようしゅうよう"）。传统字符级相似度无法匹配。

**解决方案**：用 Qwen 3.5 Plus 做**语义级对齐**——把 Whisper 转录分组成词块，让 LLM 判断每个朗读稿句子对应哪个词块。LLM 能理解"耀州窯"和"ようしゅうよう"是同一个词。

#### 新增文件

| 文件 | 说明 |
|------|------|
| `编程自动化/bg_realign.py` | 多语言重转录+全局对齐脚本，支持 `--lang bg/es/ja` |
| `编程自动化/llm_align_ja.py` | 日语 LLM 语义对齐脚本（Qwen 3.5 Plus） |
| `交付/录音/对齐/alignment_bg_v3.json` | 保加利亚语对齐（99.1%） |
| `交付/录音/对齐/alignment_es_v3.json` | 西班牙语对齐（99.1%） |
| `交付/录音/对齐/alignment_ja_llm.json` | 日语对齐（100%） |
| `交付/录音/处理后/bg/sentences_v3/` | 保加利亚语音频（212个） |
| `交付/录音/处理后/es/sentences_v3/` | 西班牙语音频（212个） |
| `交付/录音/处理后/ja/sentences_llm/` | 日语音频（212个） |

---

### v5.3 — 2026-02-20（数据修复：一板两拍合并 + 缺失翻译补全）⭐ 最终交付

针对五语种对齐数据缺失问题进行系统修复，将五语种齐全句子从 1,841 提升至 1,959。

#### 问题诊断

| 问题类型 | 数量 | 原因 |
|---------|------|------|
| 一板两拍 | 4 对 | 展板太大，用户分两次拍摄（中文版+英文版），OCR 识别为两个独立条目 |
| OCR 漏识别 zh | ~371 | 中文 OCR 识别失败（字体、反光、遮挡等） |
| OCR 漏识别 en | ~323 | 英文 OCR 识别失败 |
| 翻译失败 | ~129 | LLM 无法正常输出（边缘情况） |

#### 修复方案

**1. 合并 4 对"一板两拍"展板：**

| 中文板 | 英文板 | 内容 |
|--------|--------|------|
| 国家博物馆_IMG20260121114840 | 国家博物馆_IMG20260121114842 | 宋代其他名窑 |
| 国家博物馆_IMG20260121121509 | 国家博物馆_IMG20260121121512 | 前言农民画 |
| 故宫_IMG20260115144710 | 故宫_IMG20260115144715 | 前言陶瓷 |
| 首都博物馆_IMG20260120153150 | 首都博物馆_IMG20260120153157 | 人类文明黄金 |

**2. 补全缺失的单语种翻译：**

| 缺失语言 | 句子数 | 成功补全 |
|---------|--------|---------|
| 仅缺 ja | 33 | 32 |
| 仅缺 es | 69 | 69 |
| 仅缺 bg | 14 | 14 |

#### 修复结果

| 阶段 | 五语种齐全 | 变化 |
|------|-----------|------|
| 修复前 | 1,841 | - |
| 合并展板后 | 1,844 | +3 |
| 补全翻译后 | **1,959** | **+115** |

#### 新增脚本

| 文件 | 说明 |
|------|------|
| `编程自动化/full_merge_fix.py` | 合并一板两拍展板 |
| `编程自动化/fix_merged_translations.py` | 为合并后展板补全翻译 |
| `编程自动化/fill_missing_translations.py` | 补全缺失的单语种翻译 |

---

### v5.2 — 2026-02-20（项目收尾：目录整理 + 交付归档）

三语种翻译全部完成（ES 100%，JA/BG 99.9%），项目进入收尾阶段。

- 新建 `交付/` 目录，汇聚五个核心交付物（multilingual_corpus、enhanced_corpus、museum_glossary、corpus.tmx、corpus_stats）
- 新建 `没用/` 目录，归档约 35 个自动生成的备份文件（enhanced × 10，multilingual × 25）
- 整理根目录：6 份会议记录移至 `会议记录/`，6 份竞赛官方文件移至 `竞赛文档/`
- 新建 `项目日志.md`，完整记录六个工作阶段历程

---

### v5.1 — 2026-02-20（多语种翻译：空ZH句修复 + 重跑完成）⭐ 重要修复

修复了99条翻译失败段落（JA 16 + ES 24 + BG 59），根因是部分句子的中文为空（OCR仅识别英文）导致 LLM 输出数量不匹配。

#### 修复方案

| 改动 | 说明 |
|------|------|
| 新增 `translate_en_fallback()` | 对空ZH句，用EN→target兜底翻译（而非ZH→target） |
| 修改 `translate_paragraph()` | 预处理分离空ZH句，主翻译只处理有中文的句子，最后按原位置合并 |
| 统一 `_apply_merge()` 内部函数 | 所有返回路径（包括TransA/B单方面失败）均经过合并逻辑 |

#### 重跑结果

| 语言 | 修复前 | 修复后 | 成功率 |
|------|--------|--------|--------|
| 日语（JA） | 818/834 | **833/834** | **99.88%** |
| 西班牙语（ES） | 810/834 | **834/834** | **100%** ✅ |
| 保加利亚语（BG） | 775/834 | **~830/834** | **~99.5%** |

#### 剩余失败（约2-5条）

属于 Type 3 边缘情况：1句段落，两模型均无法输出恰好1条译文。这些段落 ZH 为空、EN 也较短，无法有效翻译。

---

### v5.0 — 2026-02-19（多语种平行语料库：多模型博弈翻译）⭐ 重大升级

引入**多模型博弈翻译流水线**，为每个句对添加日语（ja）、保加利亚语（bg）、西班牙语（es）翻译，构成五语种平行语料库。

#### 核心设计

| 要素 | 说明 |
|------|------|
| 架构 | 每语种 2 个翻译模型并行 → LLM 裁判选最优 / 合成更好译文 |
| 处理单元 | 段落级批量（一次 API 调用翻译段落内所有句子，减少 API 调用次数） |
| 断点续传 | 原子写入 `translation_progress.json`，任何时刻中断可从断点恢复 |
| 候选保留 | `_candidates.{lang}` 字段保留 A/B 候选，供人工审核者对比验证 |

#### 模型分配（基于排行榜 arena.ai 2026-02-16）

| 语言 | 翻译A | 翻译B | 裁判 | 排行榜依据 |
|------|-------|-------|------|-----------|
| 日语（JA） | Gemini 3 Flash | Qwen-MT-Turbo | Gemini 3 Flash | JA榜 #2 |
| 西班牙语（ES） | Kimi K2.5 | Gemini 3 Flash | Gemini 3 Flash | ES榜 #3，已配置 |
| 保加利亚语（BG） | Claude Sonnet | Gemini 3 Flash | Claude Sonnet | WMT24冠军（欧洲语言） |

#### 新增文件

| 文件 | 说明 |
|------|------|
| `编程自动化/multilingual_translate.py` | 多语种翻译主脚本（Multi-Model Ensemble + LLM-as-Judge） |
| `词典/museum_glossary.json` | 五语种博物馆术语表（朝代名、文物名、历史概念等，110+条） |
| `中间结果/enhanced/multilingual_corpus.json` | 五语种平行语料库（556展板，2678句对，自动生成） |
| `中间结果/enhanced/translation_progress.json` | 翻译进度文件（断点续传） |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `编程自动化/config.py` | 新增 302.ai 中转配置、Gemini/Claude/Qwen-MT/Kimi 翻译模型标识符、`TRANSLATION_MODELS` 分配表、各语言特殊指令 |
| `README.md` | 更新工作流程图（阶段七）、技术栈表格、目录结构、待完成工作状态 |

#### 输出格式（sentence 对象新增字段）

```json
{
  "sent_index": 0,
  "zh": "中国共产党是中国工人阶级的先锋队...",
  "en": "The Communist Party of China is the vanguard...",
  "ja": "中国共産党は中国の労働者階級の先鋒...",
  "bg": "Китайската комунистическа партия е авангард...",
  "es": "El Partido Comunista de China es la vanguardia...",
  "_candidates": {
    "ja": {"a": "Gemini译文", "b": "Qwen-MT译文"},
    "bg": {"a": "Claude译文", "b": "Gemini译文"},
    "es": {"a": "Kimi译文",  "b": "Gemini译文"}
  }
}
```

段落级 `_translation_meta` 字段记录各语言实际使用的模型：

```json
"_translation_meta": {
  "ja": {"translator_a": "Gemini3-Flash", "translator_b": "Qwen-MT-Turbo", "judge": "Gemini3-Flash"},
  "bg": {"translator_a": "Claude-Sonnet", "translator_b": "Gemini3-Flash", "judge": "Claude-Sonnet"},
  "es": {"translator_a": "Kimi-K2.5",    "translator_b": "Gemini3-Flash", "judge": "Gemini3-Flash"}
}
```

#### CLI 用法

```bash
cd 编程自动化

# 翻译单种语言
python multilingual_translate.py --lang ja
python multilingual_translate.py --lang es
python multilingual_translate.py --lang bg

# 翻译全部三种语言（顺序执行）
python multilingual_translate.py --lang all

# 测试模式（仅处理前5个段落）
python multilingual_translate.py --lang ja --test 5

# 查看进度统计
python multilingual_translate.py --stats

# 仅从进度文件组装输出，不发起新翻译
python multilingual_translate.py --assemble
```

#### 前置条件

需设置 302.ai API Key（用于接入 Gemini 3 Flash 和 Claude Sonnet）：

```bash
set AI302_API_KEY=your_302ai_api_key    # Windows
export AI302_API_KEY=your_302ai_api_key # Linux/Mac
```

Kimi K2.5 和 Qwen-MT 使用已有 API Key（`KIMI_API_KEY`、`QWEN_API_KEY`），无需额外配置。

---

### v4.0 — 2026-02-18（多模态增强：Qwen 3.5 Plus 看图校对）⭐ 重大升级

引入 **Qwen 3.5 Plus（通义千问 VL）** 多模态大模型，实现"看图校对"——让 VLM 直接读取展板图片，对 OCR 文本进行校验和纠错，同时保留 DeepSeek V3 的句子级对齐能力。

#### 核心改进

| 指标 | v3.0 (DeepSeek) | v4.0 (Qwen VL) | 说明 |
|------|-----------------|----------------|------|
| 输入 | 纯文本 | **图片 + 文本** | VLM 直接"看"展板 |
| 纠错准确率 | ~85% | **~95%** | 能识别 OCR 遗漏的细节 |
| 处理成功率 | 550/577 (95.3%) | **550/553 (99.5%)** | 失败条目大幅减少 |
| 段落数 | 1,134 | **1,087** | 更精准的分段 |
| 句对数 | 2,502 | **1,840** | 更严格的对齐（排除低质量） |
| 费用 | $0.73 | **¥4.90** | Qwen VL 按图片+文本计费 |

#### 新增文件

| 文件 | 说明 |
|------|------|
| `编程自动化/multimodal_enhance.py` | 多模态增强主脚本（~980行） |
| `中间结果/enhanced/multimodal/progress.json` | 多模态处理进度（断点续传） |
| `审核界面/corpus_cleaner.py` | 增强语料库人工清洗界面（Gradio） |

#### 技术架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        多模态增强流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│  展板图片 ────→ Qwen 3.5 Plus (VL) ────→ 结构化输出                   │
│      ↓              ↓                    ↓                          │
│  OCR 文本 ─→ 看图校验 + OCR纠错 ─→ 段落/句子对齐                      │
│                    ↓                                                 │
│              Kimi K2.5 (fallback)                                    │
│              当 Qwen 失败时自动切换                                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Prompt 设计要点

- **图片优先**：VLM 直接读取图片，不依赖 OCR 准确性
- **保守纠错**：只修复明确的 OCR 错误，不添加/补全内容
- **结构分割**：按视觉段落分组，再按句子切分
- **对齐验证**：中英文字数比 0.3~3.0 为有效对齐
- **失败保护**：`paragraphs` 为空或极端失衡时标记失败

#### 处理结果

| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 多模态增强成功 | 550 | 正常处理 |
| ⏭️ 人工清洗跳过 | 30 | 之前已人工处理 |
| ❌ 失败 | 3 | 图片质量问题（单语/无法识别） |

**失败条目（需人工处理）：**
- `抗日纪念馆_IMG20260129110923` — 中文字数比 0.07 极端异常
- `国家博物馆_IMG20260121123401` — paragraphs 为空
- `故宫_IMG20260115152806` — paragraphs 为空

#### CLI 用法

```bash
cd 编程自动化

python multimodal_enhance.py                    # 全量处理（自动断点续传）
python multimodal_enhance.py --provider qwen    # 指定 Qwen（默认）
python multimodal_enhance.py --provider kimi    # 使用 Kimi K2.5
python multimodal_enhance.py --only-failed      # 只处理 LLM 增强失败的条目
python multimodal_enhance.py --workers 3        # 3线程并发
python multimodal_enhance.py --stats            # 查看进度统计
```

#### 语料清洗界面

```bash
cd 审核界面
python corpus_cleaner.py     # 启动清洗界面（端口 7866）
# 或双击 start_cleaner.bat
```

功能：
- 左侧：原始展板图片
- 右侧：展板信息 + LLM 纠错记录
- 中间：可编辑的句对表格
- 操作：确认无误 / 保存修改 / 删除展板 / 跳过

---

## 十一、项目完成状态

### 当前状态（v5.5）

**五语种平行语料库已完成！** 共 1,959 条五语种齐全的句子对。**三语种音频语料全部完成！** 共 636 个句子级音频文件，总计 302.5 MB。

### 核心任务完成情况

| 任务 | 状态 | 说明 |
|------|------|------|
| OCR 识别 | ✅ 已完成 | 583/605 张图片 |
| LLM 增强 | ✅ 已完成 | OCR 纠错 + 句子级对齐 |
| 多模态校验 | ✅ 已完成 | Qwen VL 看图校对 |
| 日语翻译 | ✅ **98.1%** | 2,628/2,678 句完成 |
| 西班牙语翻译 | ✅ **97.8%** | 2,619/2,678 句完成 |
| 保加利亚语翻译 | ✅ **99.3%** | 2,658/2,678 句完成 |
| 一板两拍合并 | ✅ 已完成 | 4 对展板合并，恢复 3 条完整句子 |
| 缺失翻译补全 | ✅ 已完成 | 补全 115 条单语种缺失 |
| **五语种齐全** | ✅ **1,959 对** | 最终交付 |
| 术语表 | ✅ 已完成 | `词典/museum_glossary.json`，五语种对照，78条 |
| **技术报告** | ✅ **已完成** | 约2,500字，`交付/技术报告.md` |
| 展示材料 | ✅ 已完成 | PPT（用户已完成） |
| **BG音频对齐** | ✅ **99.1%** | 212句，95.3MB，`交付/录音/处理后/bg/sentences_v3/` |
| **ES音频对齐** | ✅ **99.1%** | 212句，102.6MB，`交付/录音/处理后/es/sentences_v3/` |
| **JA音频对齐** | ✅ **100%** | 212句，104.6MB，`交付/录音/处理后/ja/sentences_llm/`（LLM语义对齐） |

### 交付文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 五语种平行语料库 | `交付/parallel_corpus_5lang.json` | **核心交付物**，1,959 条五语种对齐单元 |
| 完整五语种语料 | `交付/multilingual_corpus.json` | 含候选译文和元数据 |
| 中英双语语料 | `交付/enhanced_corpus.json` | LLM 增强后的中英对照 |
| 术语表 | `交付/museum_glossary.json` | 五语种博物馆专业术语 |
| TMX 格式 | `交付/corpus.tmx` | 兼容翻译工具 |
| **三语种音频** | `交付/录音/处理后/` | BG/ES/JA 共 636 个句子级 wav 文件（302.5MB） |

### 提交截止日期

**2026年3月1日**

### 需提交材料清单

1. **语料成果包** — ✅ 已完成（`交付/` 目录）
2. **技术报告** — ✅ 已完成（`交付/技术报告.md`，约2,500字）
3. **展示材料** — ✅ 已完成（PPT）

---

## 十二、注意事项

1. **中文连接规则**：中文文本块直接连接，不加空格
2. **英文连接规则**：英文文本块用空格连接
3. **审核建议**：优先审核低置信度（<0.85）的图片
4. **不要删除** `原始语料/` 和 `中间结果/` 目录，它们是核心数据
5. **配置修改**：只需编辑 `编程自动化/config.py`，其他脚本自动继承
6. 如需重新运行 OCR，结果会覆盖 `中间结果/ocr_results.json`

---

## 十三、致谢

本项目为北京第二外国语学院第一届语料库技术大赛参赛作品。

### 语料来源

故宫博物院、中国国家博物馆、首都博物馆、中国共产党历史展览馆、中国人民抗日战争纪念馆展板解说词。

### 技术支持

本项目全程采用**国产 AI 技术栈**完成，感谢以下企业和开源社区的慷慨支持：

#### 大模型与翻译服务

| 企业/项目 | 服务 | 本项目用途 |
|----------|------|-----------|
| **深度求索 DeepSeek** | DeepSeek V3 API | LLM 增强（OCR 纠错 + 句子级对齐），性价比极高 |
| **通义千问 Alibaba Qwen** | Qwen 3.5 Plus (VL) | 多模态增强（看图校对），Qwen-MT-Turbo 翻译 |
| **月之暗面 Moonshot** | Kimi K2.5 | 多模态兜底 + 西班牙语翻译主力 |
| **智谱 AI Zhipu** | GLM 系列模型 | 翻译候选模型（早期测试） |
| **Google Gemini** | Gemini 2.0 Flash | 日语/保加利亚语翻译 + LLM 裁判 |
| **Anthropic Claude** | Claude Sonnet 4 | 保加利亚语翻译主力（WMT24 冠军水准） |
| **302.ai / AIHubMix** | 多模型中转服务 | 统一接入 Gemini、Claude 等国际模型 |

#### 开发工具与基础设施

| 工具 | 用途 |
|------|------|
| **Cherry Studio** | 本地 LLM 调试与 Prompt 设计，大幅提升开发效率 |
| **RapidOCR** | 开源 OCR 引擎，离线运行，中英文识别效果出色 |
| **Gradio** | Python Web UI 框架，快速搭建人工审核界面 |
| **OpenAI SDK** | 统一的 API 调用接口，兼容多厂商 |

#### AI 编程助手

本项目的 **2,000+ 行代码** 全部由 AI 辅助完成，感谢以下智能编程伙伴：

| 模型 | 角色 | 贡献 |
|------|------|------|
| **Claude Opus 4.6** | 架构设计 + 复杂逻辑 | 核心脚本架构、并发处理、断点续传机制 |
| **Claude Sonnet 4.6** | 日常开发 + 调试 | 功能实现、Bug 修复、代码审查 |
| **智谱 GLM-5** | 本地快速迭代 | Prompt 调优、数据格式转换脚本 |
| **Kimi 2.5** | 文档 + 长文本处理 | README 撰写、技术文档整理 |

> 人机协作新模式：人类提供需求与方向，AI 负责实现与优化。本项目是 AI 辅助编程的一次成功实践。

### 特别感谢

- **北京第二外国语学院** — 提供比赛平台，让我们有机会探索语料库技术的前沿应用
- **指导教师 刘斌老师** — 专业指导与耐心答疑
- **所有开源贡献者** — RapidOCR、Gradio 等项目的维护者，让技术民主化成为可能

### 写在最后

这个项目从 2026 年 1 月开始筹备，历时一个月完成。我们见证了国产大模型从"能用"到"好用"的飞跃——DeepSeek V3 的超高性价比、Qwen VL 的多模态能力、Kimi 的超长上下文……每一项技术的进步都在推动着语料库构建的边界。

**"博古通译"**，既是对博物馆文化的传承，也是对 AI 时代的拥抱。感谢这个时代，让我们有机会用技术讲好中国故事。

---

*最后更新：2026-02-20*


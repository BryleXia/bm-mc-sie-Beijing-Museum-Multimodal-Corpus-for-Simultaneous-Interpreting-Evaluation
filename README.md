# 面向AI同声传译评测的北京博物馆多模态平行语料库构建

> 北京第二外国语学院第一届语料库技术大赛 - 语料库构建类（主赛道）- 文旅与社会传播类
---
## 一、项目概述

本项目以中国博物馆展板解说词为语料来源，通过手机实地拍摄、OCR自动识别、人工审核校对、**LLM自动纠错与多级对齐**的技术流程，构建了一个**句子级对齐的五语种平行语料库**（中、英、日、西班牙、保加利亚语）。

### 核心特点

- **金标准中英对照**：中文和英文原文来自博物馆实地展板，由专业翻译团队提供，是绝对可靠的参考译文，可作为AI评测的黄金标准
- **五语种平行**：中、英、日、西班牙、保加利亚语，1,959 条五语种齐全的句子对
- **句子级对齐**：展板 → 段落 → 句子，三级层次结构完整保留从属关系
- **多模型博弈翻译**：每种目标语言使用 2 个翻译模型 + 1 个裁判模型，确保翻译质量
- **LLM 自动纠错**：DeepSeek V3.2 + Qwen 3.5 Plus 自动修复 OCR 错误
- **质量三级分级**：基于 OCR 置信度和人工审核状态，将语料分为 A/B/C 三级
- **全流程自动化**：从原始图片到结构化语料库，提供一键式处理管道
- **人工审核闭环**：内置 Web 审核界面，支持确认、修正、跳过操作
- **多格式导出**：JSON、TMX（翻译记忆库）、纯文本平行语料

### 数据规模

| 指标 | 数值 |
|------|------|
| 原始图片总数 | **605 张** |
| 覆盖博物馆 | 5 个（故宫、国家博物馆、首都博物馆、党史馆、抗日纪念馆） |
| OCR 成功处理 | 583 张 |
| **最终展板数** | **557 条** |
| **段落级对齐** | **834 对** |
| **句子级对齐** | **2,705 对** |
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
原始图片 → OCR提取 → 人工审核 → LLM增强 → 多模态校验 → 多语种翻译 → 数据修复 → 导出语料
   ↓          ↓           ↓          ↓            ↓            ↓           ↓           ↓
 605张    RapidOCR    Gradio   DeepSeek    Qwen 3.5    ja/bg/es   合并+补全    多格式
                                   V3.2       Plus       多模型博弈
                                    ↓
                               OCR纠错+
                              段落/句子分割
```

**阶段说明：**

| 阶段 | 名称 | 输入 | 输出 | 关键技术 |
|------|------|------|------|----------|
| 一 | OCR提取 | 展板图片 | OCR文本 | RapidOCR |
| 二 | 人工审核 | OCR结果 | 审核后文本 | Gradio Web UI |
| 三 | LLM增强 | 审核后文本 | 句子级对齐 | DeepSeek V3.2 |
| 四 | 多模态校验 | 图片+文本 | 看图校验 | Qwen 3.5 Plus |
| 五 | 多语种翻译 | 中英文本 | 日/西/保译文 | 多模型博弈翻译 |
| 六 | 数据修复 | 缺漏数据 | 补全后数据 | 规则+LLM |
| 七 | 导出 | 完整数据 | 多格式语料 | JSON/TMX |

### 2.2 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| OCR 引擎 | RapidOCR (onnxruntime) | 离线运行，无需联网 |
| 语言检测 | Unicode 字符集正则匹配 | CJK 范围检测中文，Latin 检测英文 |
| 审核界面 | Gradio 6.x Web UI | 浏览器内操作，支持图片预览 |
| **LLM 增强** | **DeepSeek V3.2 API** | **OCR 纠错 + 段落/句子级对齐，5线程并发** |
| **多语种翻译** | **Gemini 3 Flash + Kimi K2.5 + Claude Sonnet 4.6（302.ai中转）** | **多模型并行 + LLM裁判，3语种×2翻译模型** |
| 数据格式 | JSON / TMX | 兼容主流翻译工具和 NLP 框架 |

### 2.3 模型选择依据

本项目基于 **Text Arena**（arena.ai）排行榜选择翻译模型，每个目标语言采用 **2个翻译模型 + 1个裁判模型** 的博弈机制：

| 目标语言 | 翻译模型A | 翻译模型B | 裁判模型 | 说明 |
|----------|-----------|-----------|----------|------|
| 日语 | Gemini 3 Flash | Qwen-MT-Turbo | Gemini 3 Flash | 排行榜日语榜前排，通用性能稳定 |
| 西班牙语 | Kimi K2.5 | Gemini 3 Flash | Gemini 3 Flash | 排行榜西班牙语榜前排 |
| 保加利亚语 | Claude Sonnet 4.6 | Gemini 3 Flash | Gemini 3 Flash | WMT翻译冠军模型 |
| OCR 纠错 | DeepSeek V3.2 | - | - | 国产开源模型，性价比极高 |
| 多模态校验 | Qwen 3.5 Plus | - | - | 国产多模态最强，可直接读取展板图片校对 |

> **选择逻辑**：通过 302.ai 中转统一接入多个模型，使用双翻译 + LLM 裁判机制选择最佳译文。

### 2.4 质量分级标准

> 注：以下为OCR阶段的质量分级。最终交付的1,959条五语种齐全句子对均经过人工审核或LLM增强处理。

| 等级 | 条件 | 数量（OCR阶段） | 占比 |
|------|------|-----------------|------|
| **A 级** | 人工已确认或修正 + LLM 增强处理成功 | 0 → 1,959 | 0% → **100%** |
| **B 级** | OCR 置信度 0.8~0.9 | 550 | 94.3% |
| **C 级** | OCR 置信度 < 0.8 | 33 | 5.7% |

> 经过人工审核、LLM增强、多模态校验、数据修复等后续处理后，OCR阶段的B级和C级语料大部分经过人工确认或修正，并成功完成LLM增强，最终形成1,959条五语种齐全的高质量句子对。少量处理失败的条目已被舍弃。

---

## 三、目录结构

```
nlp-museum-corpus/
│
├── README.md                         # 项目主文档
├── .gitignore                       # Git忽略文件
│
├── 编程自动化/                       # 核心代码 (~800KB)
│   ├── ocr_processor.py             # OCR处理
│   ├── llm_enhance.py               # LLM增强（OCR纠错+句子对齐）
│   ├── multilingual_translate.py    # 多语种翻译
│   ├── whisper_align.py             # 音频对齐
│   ├── multimodal_enhance.py        # 多模态增强
│   ├── config.py                    # 配置
│   └── ... (其他40+脚本)
│
└── 黄金标准/                         # 质量分级语料 (~1.2MB)
    ├── grade_A_corpus.json          # A级语料
    ├── grade_B_corpus.json          # B级语料
    └── ...
```

> **注意**：大语料数据（3.4GB）和原始数据（3GB）不在本仓库中，请访问 HuggingFace 或 Google Drive 下载。

---

## 四、快速开始

### 4.1 环境准备

```bash
# Python 3.10+ 推荐
pip install rapidocr-onnxruntime gradio openai

# LLM 辅助纠错（可选）
set DEEPSEEK_API_KEY=your_api_key    # Windows
export DEEPSEEK_API_KEY=your_api_key # Linux/Mac
```

### 4.2 处理流程

| 阶段 | 技术 | 输入 | 输出 |
|------|------|------|------|
| 1. OCR 提取 | RapidOCR | 展板图片 | OCR 文本（JSON） |
| 2. 人工审核 | Gradio Web UI | OCR 结果 | 审核后的文本 |
| 3. LLM 增强 | DeepSeek V3.2 | 审核后文本 | 句子级对齐 |
| 4. 多模态校验 | Qwen 3.5 Plus | 图片+文本 | 看图校验 |
| 5. 多语种翻译 | Gemini/Kimi/Claude | 中英文本 | 日/西/保译文 |
| 6. 数据修复 | 脚本 | 缺漏数据 | 补全后数据 |
| 7. 导出 | 多格式导出 | 完整数据 | JSON/TMX |

### 4.3 核心脚本使用

```python
# OCR 处理
from ocr_processor import OCRProcessor
processor = OCRProcessor()
results = processor.process_image("path/to/image.jpg")

# LLM 增强
from llm_enhance import LLMEnhancer
enhancer = LLMEnhancer()
enhanced = enhancer.enhance(ocr_result)

# 多语种翻译
from multilingual_translate import MultilingualTranslator
translator = MultilingualTranslator()
translations = translator.translate(text, target_langs=["ja", "es", "bg"])
```

---

## 五、数据格式

### 5.1 五语种平行语料库（核心交付物）

```json
{
  "corpus_id": "bisu-museum-parallel-2026",
  "name": "中国博物馆多语种解说词平行语料库",
  "version": "1.0",
  "created": "2026-02-20",
  "languages": ["zh", "en", "ja", "es", "bg"],
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
        "zh": "中国共产党是中国工人阶级的先锋队...",
        "en": "The Communist Party of China is the vanguard...",
        "ja": "中国共産党は、中国労働者階級の先鋒隊...",
        "es": "El Partido Comunista de China es la vanguardia...",
        "bg": "Китайската комунистическа партия е авангард..."
      }
    }
  ]
}
```

### 5.2 博物馆术语表

```json
{
  "glossary": [
    {"zh": "青铜器", "en": "bronze", "ja": "青銅器", "es": "bronce", "bg": "бронз"},
    {"zh": "瓷器", "en": "porcelain", "ja": "陶磁器", "es": "cerámica", "bg": "порцелан"}
  ]
}
```

---

## 六、关键设计决策

### 6.1 三级文本对齐：展板 → 段落 → 句子

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

### 6.2 LLM 增强的核心原则：清洗而非改写

**这是语料清洗和标准化，不是内容创作。** LLM 的职责严格限定为：

1. **纠正 OCR 错误**：仅修复明确的 OCR 识别错误
2. **结构化分割**：将连续文本分割为段落和句子
3. **双语对齐**：将中文与对应英文配对

**严禁行为**：
- 改写、润色、优化原文表达
- 添加原文中不存在的内容
- 删除原文中存在的内容

### 6.3 并发处理与断点续传

- **5 线程并发**：使用 `ThreadPoolExecutor` 同时处理 5 条展板
- **原子写入**：每条完成后立即保存进度
- **自动跳过**：已完成/已失败/已跳过的条目自动排除

---

## 七、项目完成状态

### 核心任务完成情况

| 任务 | 状态 | 说明 |
|------|------|------|
| OCR 识别 | ✅ 已完成 | 583/605 张图片 |
| LLM 增强 | ✅ 已完成 | OCR 纠错 + 句子级对齐 |
| 多模态校验 | ✅ 已完成 | Qwen 3.5 Plus 看图校对 |
| 日语翻译 | ✅ **98.1%** | Gemini + Qwen-MT 双模型翻译 |
| 西班牙语翻译 | ✅ **97.8%** | Kimi + Gemini 双模型翻译 |
| 保加利亚语翻译 | ✅ **99.3%** | Claude + Gemini 双模型翻译 |
| **五语种齐全** | ✅ **1,959 对** | 中英为金标准，日/西/保采用双模型翻译 |
| BG音频对齐 | ✅ **99.1%** | 212句 |
| ES音频对齐 | ✅ **99.1%** | 212句 |
| JA音频对齐 | ✅ **100%** | 212句（LLM语义对齐） |

---

## 八、致谢

本项目为北京第二外国语学院第一届语料库技术大赛参赛作品。

### 技术支持

| 企业/项目 | 服务 | 本项目用途 |
|----------|------|-----------|
| **DeepSeek** | DeepSeek V3.2 API | LLM 增强 |
| **Qwen** | Qwen 3.5 Plus | 多模态增强、翻译 |
| **Kimi** | Kimi K2.5 | 西班牙语翻译 |
| **Google Gemini** | Gemini 3.0 Flash | 日语/保加利亚语翻译 |
| **Anthropic Claude** | Claude Sonnet 4.6 | 保加利亚语翻译 |
| **RapidOCR** | 开源 OCR 引擎 | 离线 OCR 识别 |
| **Gradio** | Web UI 框架 | 人工审核界面 |

### AI 编程助手

本项目的 **2,000+ 行代码** 全部通过 **Claude Code** 和其他 AI 助手完成。

---

## 九、数据分享

**大语料数据不在本仓库中**，如需获取：

- **五语种平行语料库（3.4GB）**：请访问 HuggingFace 或 Google Drive
- **原始展板图片（3GB）**：请访问 Google Drive
- **音频语料（292MB）**：请访问 HuggingFace Audio 数据集

---

*最后更新：2026-02-27*

# Text Embedding Ablation Report

**范围：** Qwen3 / BERT / 投影维度消融

**用途：** 面试准备与实验收口

**口径：** offline full test Recall@50

> 本报告只整理已完成实验，不修改 canonical 配置，不替换 README 或简历主数字。

---

## 1. 实验矩阵

### 1.1 Transformer TT 单路，投影维度 64

| 文本向量来源 | Full test Recall@50 |
| --- | ---: |
| MiniLM | 0.103168 |
| Qwen3 plain | 0.103033 |
| Qwen3 instruct | 0.103394 |
| bert-base | 0.102101 |

### 1.2 Transformer TT 单路，投影维度 128

| Seed | MiniLM | Qwen3 plain | Qwen3 - MiniLM |
| ---: | ---: | ---: | ---: |
| 42 | 0.110416 | 0.111437 | +0.001021 |
| 2024 | 0.115101 | 0.112980 | -0.002121 |
| 2025 | 0.110160 | 0.109302 | -0.000858 |
| Mean | 0.111892 | 0.111240 | -0.000653 |

### 1.3 Text Semantic 单路

三个变体使用完全相同的纯文本召回算法：`history_max_len=100`、`decay=0.8`、逐行 L2 normalize。唯一变量是 item text embedding 来源。

| 文本向量来源 | Full test Recall@50 |
| --- | ---: |
| MiniLM | 0.027859 |
| Qwen3 plain | 0.026696 |
| Qwen3 instruct | 0.027784 |

### 1.4 波动边界

- Transformer TT 训练存在随机性：历史 seed robustness 的 `std=0.0034`。
- Text Semantic 单路是确定性纯检索，不包含训练噪声。

---

## 2. 正交分解

### 2.1 文本向量来源：未观察到可采纳增益

在 Transformer TT `@64` 中，Qwen3 plain、Qwen3 instruct 和 bert-base 与 MiniLM 的差异均处于 TT seed 波动范围内。不能据此声称更换 embedding 来源提升了模型。

在确定性的 Text Semantic 单路中，Qwen3 plain 低于 MiniLM；Qwen3 instruct 能追回大部分差距，但仍未超过 MiniLM。这个结果直接否定了“仅替换 embedding 文件即可增强 Text Semantic 通路”的假设。

### 2.2 投影维度：`64 -> 128` 是更强信号

MiniLM 从 `@64` 到 `@128` 的结果由 `0.103168` 提升到 `0.110416`，约为 `+7%`。这一级别的变化明显大于 TT 的 `std=0.0034`，说明 latent / text projection 宽度比 embedding 来源更值得关注。

Qwen3 plain `@128` 在 seed 42 下略高于 MiniLM，但在 seed 2024 和 2025 下均低于 MiniLM。三个 seed 的均值也略低于 MiniLM。这个结果否定了“Qwen3 在 `@128` 下跨 seed 稳定提升”的假设。

---

## 3. 机理解释：合理假设，不是因果证明

### 3.1 任务对齐比模型规模更重要

`all-MiniLM-L6-v2` 使用 sentence-pair 对比学习目标，并面向 sentence similarity / semantic search。`Qwen3-Embedding-0.6B` 同样是专门面向 embedding 与 ranking 的模型，不应描述为“未经检索优化的通用 LLM embedding”。

本项目的结果只能支持更窄的判断：更大的 embedding 模型不自动带来更好的推荐召回；模型与“电影 / 电视商品短文本 + 用户历史语义聚合”这一分布的匹配程度更重要。

### 3.2 Pooling 归纳偏置可能影响短文本

MiniLM 使用 mean pooling；Qwen3 官方实现使用 last-token pooling。对于标题较短的商品文本，mean pooling 可能更稳健，而 last-token pooling 未必形成优势。

这是解释假设，不是已证明原因。当前实验没有单独控制 pooling 策略，因此不能断言 Qwen3 的差异由 last-token pooling 导致。

### 3.3 Instruction 有效，但没有突破 MiniLM

Qwen3 instruct 相比 Qwen3 plain 更接近 MiniLM，说明任务提示对该模型有帮助。但在确定性的 Text Semantic 单路中，它仍停留在追平附近，没有形成可用于 4ch 重跑的明确增益。

---

## 4. 决策

### 4.1 当前收口

- canonical 保持 MiniLM `@64` 不变。
- 不启动 Qwen3 版 4ch 融合重跑。
- Qwen3、bert-base 和 `@128` 配对实验保留为消融证据，不改 README、简历主数字或 canonical 配置。

### 4.2 Qwen3 是否被严格证伪

已经被否定的是：

- 在当前数据、当前文本拼接方式和当前召回算法下，直接把 MiniLM 替换为 Qwen3 plain / instruct，没有提升 Text Semantic 单路。
- 在 TT `@64` 下，Qwen3 没有超出 seed 波动范围的可采纳增益。

尚未被否定的是：

- 更系统的 query / document 非对称 prompt 设计。
- 针对商品短文本的输入模板设计。
- 领域微调，或针对 Qwen3 单独设计 / 调优 projection 结构。

这些方向不是当前收口所必需。只有在新的面试需求或明确收益假设出现时，才值得单独立项。

---

## 5. 诚实边界

- 所有结果都是 offline evaluation，不是线上 A/B。
- Qwen3 未提升当前主指标，不能写成“升级成功”。
- `@128` 是有价值的诊断信号，但尚未完成 4ch 融合或 Faiss 复验，不能替换 MiniLM `@64` canonical。
- Text Semantic 单路本身较弱；即使单路上升，也需要 valid-selected 4ch 融合复验才能判断系统边际收益。

---

## 面试口述话术

我做了文本 embedding 来源和投影维度的正交消融。直接把 MiniLM 换成 Qwen3 或 bert-base，并没有在当前离线口径下带来稳定增益；Qwen3 instruct 比 plain 更好，但 Text Semantic 单路仍只追平 MiniLM。投影维度从 64 扩到 128 的信号更明显，但 Qwen3 在 `@128` 三个 seed 中只赢一次，均值反而略低于 MiniLM。这个实验让我确认，embedding 不是越大越好，任务分布匹配和系统瓶颈定位比模型规模更重要。我保持 MiniLM `@64` 作为 canonical，只把这组结果作为诚实的消融结论。

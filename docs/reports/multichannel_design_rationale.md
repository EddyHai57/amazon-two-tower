# Multi-Channel Retrieval Design Rationale

**范围：** 四路召回设计理由与评价体系

**用途：** 面试准备与项目叙述收口

**口径：** offline full test evaluation

> 本报告只汇总现有结论，不新增实验，不修改 canonical、README 或简历主数字。

---

## 1. 为什么使用四路召回

多路召回不是为了堆模块，而是因为单一路径无法覆盖所有目标分布。现有 bucket 分析表明，不同通路在不同场景下各有作用。

| 通路 | 主要信号 | 保留理由 |
| --- | --- | --- |
| ItemCF | 物品共现 | 在头部和长尾 bucket 上有优势，是稳定的协同过滤基线 |
| Transformer TT | 用户历史序列表示 | 在中热度 bucket 上更强，补充 ItemCF 的中段不足 |
| Text Semantic | 历史物品文本语义 | 单路较弱，但提供与共现信号不同的候选；用于补充文本可用物品与协同信号较弱场景 |
| Popularity Fallback | 全局热门物品 | 作为兜底信号，避免候选不足；通过较低权重控制热门偏置 |

需要特别说明：当前标准评估会排除 cold target。因此 Text Semantic 的“冷启动价值”是架构层面的覆盖能力，不是本轮 full test 已单独证明的 cold-target 提升。

---

## 2. 融合方式：Weighted RRF

四路候选使用 Weighted Reciprocal Rank Fusion（加权倒数排名融合，Weighted RRF）：

```text
score(item) = sum(weight_channel / (k + rank_channel(item)))
```

选择 RRF 的原因：

- 各通路分数空间不同，直接比较原始 score 不可靠。
- RRF 只依赖通路内排名，便于融合异构召回器。
- 权重允许区分主信号和补充信号：ItemCF / TT 是主通路，Text / Popularity 是补充通路。

---

## 3. 评价体系：不能只看单路 Recall

多路召回需要同时评估三层证据。

### 3.1 单路有效性

每条通路首先要证明自己不是随机噪声。ItemCF 与 Transformer TT 都有独立的 offline full test Recall@50；Text Semantic 虽然较弱，但有非零命中；Popularity 是受控兜底。

### 3.2 互补性

ItemCF 与 Transformer TT 的 `Jaccard@50=0.040`，说明两路候选重合较低。但低 Jaccard 本身不等于高质量互补：如果其中一路质量差，随机噪声同样会导致低重合。

要把“低重合”解释为“有效互补”，必须同时满足：

1. 两路单路 Recall 都有效。
2. 融合结果高于任一单路。
3. bucket 分析可以解释两路为什么命中不同物品子群：ItemCF 强在头部 / 长尾，Transformer TT 强在中热度。

三条证据同时成立，才可以说候选差异具有业务意义，而不是随机分歧。

### 3.3 融合增益

最终四路 valid-selected wRRF 的 offline full test Recall@50 为 `12.52%`，高于任何单路。这个结果证明互补候选经过融合后转化成了真实命中，而不是只增加候选多样性。

---

## 4. 如何避免 test-tuning

融合权重必须在 valid set 上选择，然后冻结配置，在 test set 上只运行一次。

```text
valid sweep
-> 选择 frozen config
-> test run once
```

这个流程避免根据 test label 反复调权重。对外表述应使用“valid-selected weighted RRF”，而不是只说“调参后达到 12.52%”。

---

## 5. 设计边界

- 这是 offline 多路召回实验系统，不是线上部署。
- Weighted RRF 是候选融合层，不是新的模型训练阶段。
- 低 Jaccard 只是互补性的必要证据之一，不能脱离单路质量和融合增益单独使用。
- Text Semantic 的作用是弱补充，不应描述为最强单路。
- Popularity Fallback 必须保持受控权重，不应包装成个性化建模。

---

## 面试口述话术

我没有把多路召回理解成简单堆模型，而是先看 bucket 分布。ItemCF 在头部和长尾更稳，Transformer TT 对中热度物品更强，Text Semantic 提供与共现不同的语义候选，Popularity 只做低权重兜底。评价时我同时看单路有效性、候选互补性和最终融合增益：ICF 与 TT 的 `Jaccard@50=0.040` 很低，但我不会只凭低 Jaccard 下结论，还要确认两路本身有效、bucket 差异可解释，并且 valid-selected wRRF 的 full test Recall@50 最终达到 `12.52%`。权重只在 valid 上选择，test 只跑一次，避免 test-tuning。

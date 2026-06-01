# Transformer LogQ Correction Investigation

## 1. Scope

本文记录 Transformer Two-Tower 中 LogQ correction 的隔离实验、机制负对照、风险边界
和下一步受控验证。所有结果均为 offline temporal leave-one-out evaluation，不是线上 A/B。

当前 canonical 不修改。README、简历和 CLAUDE 主数字不修改。

## 2. Why LogQ

基础 in-batch InfoNCE 将 batch 内其他正样本当作负样本。训练交互存在长尾分布时，热门
item 更容易进入 batch，因此会被更频繁地作为负样本参与 softmax。

本轮使用 train-only item frequency：

```python
q = bincount(train_df["item_idx"], minlength=n_items).clamp_min(1)
q = q / q.sum()
corrected_logits = logits - log(q[item_idx]).unsqueeze(0)
```

推理阶段不减 `log(q)`，不做 popularity rerank，不修改 ANN、RRF、模型结构或评估口径。

## 3. Orthogonal Smoke

固定 Transformer `@64`、`3 epochs`、`50K limited-valid`：

| Variant | LogQ | Duplicate masking | Recall@50 |
|---|---:|---:|---:|
| baseline | off | off | 0.124420 |
| mask-only | off | on | 0.124880 |
| logq-only | on | off | 0.180120 |
| logq-mask | on | on | 0.179900 |

主要增益来自 LogQ。duplicate masking 作为单独消融证据保留，不作为候选主方案。

## 4. Multi-Seed Full Test

| Seed | Historical baseline Recall@50 | LogQ Recall@50 | Delta |
|---:|---:|---:|---:|
| 42 | 0.103168 | 0.149926 | +0.046758 |
| 2024 | 0.103704 | 0.149485 | +0.045781 |
| 2025 | 0.096223 | 0.147757 | +0.051534 |

总 Recall@50 提升跨 seed 稳定。但总指标不能单独证明个性化质量改善。

## 5. Full-Test Effect Audit

### 5.1 Protocol

```text
temporal leave-one-out
test history = concat(train, valid)
test seen mask = merge_seen_items(train_seen, valid)
exclude is_cold_item_for_eval
exact Top50 inner product
```

### 5.2 Target item popularity bucket Recall@50

| Target 热度桶 | Baseline | LogQ alpha=1.0 | Delta |
|---|---:|---:|---:|
| 1-5 | 0.026480 | 0.001741 | -0.024740 |
| 6-20 | 0.060390 | 0.013782 | -0.046608 |
| 21-100 | 0.096687 | 0.068465 | -0.028222 |
| >100 | 0.138252 | 0.292048 | +0.153795 |

### 5.3 Exposure distribution

| Metric | Baseline | LogQ alpha=1.0 |
|---|---:|---:|
| avg_pop | 106.81 | 892.21 |
| median_pop | 27 | 286 |
| P90 pop | 207 | 2701 |
| catalog coverage | 152691 | 57610 |
| Top50 `>100` share | 19.89% | 81.99% |
| Top50 average Jaccard | - | 0.094075 |

### 5.4 Hit transition

```text
baseline_only = 18838
logq_only      = 42052
both_hit       = 32382
neither_hit    = 403198
```

### 5.5 Interpretation boundary

`alpha=1.0` 的 LogQ correction 确实改变了向量空间，并稳定提高总体 Recall@50。但增益
高度集中在 head item，其他三个 target popularity buckets 全部回退，catalog coverage
下降。因此不能将其描述为“已经解决 popularity bias”，也不能直接升级 canonical。

## 6. Mechanism Negative Control

| Variant | limited-valid Recall@50 |
|---|---:|
| baseline | 0.124420 |
| empirical-logq | 0.180120 |
| shuffled-logq | 0.078340 |

真实 `q(item)` 映射优于 shuffled-q，说明收益来自按真实 item frequency 改变训练目标，
而不是任意 logits 扰动。

## 7. Alpha Strength Smoke

### 7.1 Purpose

`logq_alpha` 控制 LogQ correction 强度：

```python
corrected_logits = logits - logq_alpha * log(q[item_idx]).unsqueeze(0)
```

- `alpha=0.0`：等价于基础 InfoNCE。
- `alpha=1.0`：完整 LogQ correction，即已完成的强修正版本。
- `alpha=0.25 / 0.50 / 0.75`：逐步减弱修正，寻找 Recall 与曝光分布的平衡。

### 7.2 Matrix

| Variant | logq_alpha | Scope |
|---|---:|---|
| alpha-000 | 0.00 | 3 epochs + 50K limited-valid |
| alpha-025 | 0.25 | 3 epochs + 50K limited-valid |
| alpha-050 | 0.50 | 3 epochs + 50K limited-valid |
| alpha-075 | 0.75 | 3 epochs + 50K limited-valid |
| alpha-100 | 1.00 | 3 epochs + 50K limited-valid |

每组补充 limited-valid exposure audit：总 Recall@50、item popularity bucket Recall、
`avg_pop`、median、P90、catalog coverage、Top50 热门商品占比。

### 7.3 Manual gate

本轮只做候选筛选，不自动进入 full train。候选至少需要：

```text
Recall@50 明显高于 baseline
中热度桶不能系统性下降
Top50 热门商品占比不能接近 alpha=1.0 的 81.99%
catalog coverage 不能明显塌缩
```

### 7.4 Completed limited-valid results

| Alpha | Recall@50 | avg_pop | median_pop | P90 pop | coverage | Top50 `>100` share |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.124460 | 113.78 | 28 | 219 | 141547 | 20.97% |
| 0.25 | 0.163560 | 270.17 | 68 | 587 | 128908 | 40.13% |
| 0.50 | 0.187340 | 516.40 | 149 | 1263 | 88259 | 60.56% |
| 0.75 | 0.189520 | 735.41 | 221 | 2068 | 57240 | 73.50% |
| 1.00 | 0.179980 | 915.66 | 290 | 2844 | 40621 | 81.83% |

Target item popularity bucket Recall@50：

| Alpha | `1-5` | `6-20` | `21-100` | `>100` |
|---:|---:|---:|---:|---:|
| 0.00 | 0.032455 | 0.072903 | 0.114014 | 0.160671 |
| 0.25 | 0.035128 | 0.069072 | 0.128784 | 0.236509 |
| 0.50 | 0.013364 | 0.048808 | 0.125305 | 0.300559 |
| 0.75 | 0.005346 | 0.034968 | 0.107300 | 0.324005 |
| 1.00 | 0.001527 | 0.020759 | 0.084351 | 0.325052 |

### 7.5 Candidate interpretation

`alpha=0.25` 是当前离散网格中的 Pareto candidate：

- Recall@50 相对 `alpha=0.00` 增加 `+0.039100`。
- `1-5` 和 `21-100` buckets 提升。
- `6-20` bucket 轻微回退。
- coverage 只下降约 `8.9%`，明显小于 `alpha>=0.50`。
- Top50 `>100` share 为 `40.13%`，仍需业务目标和 full-test multi-seed 验收。

这不是 universal optimum。不同数据分布、batch size、`Q(item)` 估计方式和业务目标会改变
最优折中点。

## 8. Mechanism Notes and External References

### 8.1 Correct interpretation

LogQ correction 的初衷是修正 sampled softmax / in-batch negatives 中的 sampling bias，
不是单调降低热门曝光的 rerank knob。训练阶段修改 logits 会改变 embedding learning
dynamics；推理阶段仍使用 raw dot product，因此更强 correction 不保证更均衡的最终曝光。

本项目当前使用简化估计：

```python
q(item) = train_item_frequency / total_train_interactions
```

Uber 的公开工程文章使用 batch appearance probability：

```text
Q = 1 - (1 - w)^B
```

其中 `w` 是 item 数据权重，`B` 是 batch size。两者不能直接写成完全相同。

RecSys 2025 的论文进一步指出，传统 LogQ derivation 没有严格区分 positive item 总是出现
和 negative item 被采样出现的概率，并提出 refined correction。该方向属于后续研究候选，
本轮不继续实现。

### 8.2 Primary sources

- Uber Engineering: [Innovative Recommendation Applications Using Two Tower Embeddings at Uber](https://www.uber.com/en-GB/blog/innovative-recommendation-applications-using-two-tower-embeddings/)
- RecSys 2025: [Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval](https://arxiv.org/abs/2507.09331)
- Official RecSys 2025 code: [NonameUntitled/logq](https://github.com/NonameUntitled/logq)
- Google Research: [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- Survey: [A Survey on Popularity Bias in Recommender Systems](https://arxiv.org/abs/2308.01118)
- IPL regularization: [Popularity Debiasing from Exposure to Interaction in Collaborative Filtering](https://arxiv.org/abs/2305.05204)

## 9. Interview Answer

> 我实现了 train-only frequency 的 LogQ correction，用于分析 in-batch negatives 的
> popularity sampling bias。初始 `alpha=1.0` 在三个 seed 上都显著提高总体 Recall，
> 但 effect audit 发现增益主要来自热门商品，coverage 也明显下降，因此没有直接替换
> 主模型。随后我把 correction 写成 `alpha * log(q)` 的受控消融，并同时检查 Recall、
> item 热度桶和曝光覆盖率。`alpha=0.25` 是当前 smoke 的 Pareto 候选，但仍需 full-test
> multi-seed 验收。这个过程说明 sampled-softmax 风格修正需要结合业务目标验收，不能
> 只看一个总 Recall 数字。

## 10. Evidence Files

```text
outputs/transformer_logq_smoke/
outputs/text_timeaware_transformer_max100_logq*_full_eval/eval_summary.json
outputs/transformer_logq_effect_audit/audit_summary.json
outputs/transformer_logq_alpha_smoke/
outputs/transformer_logq_alpha_smoke_audit/
```

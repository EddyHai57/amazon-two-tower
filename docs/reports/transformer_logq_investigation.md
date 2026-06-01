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

## 8. Interview Answer

> 我实现了 train-only frequency 的 LogQ correction，用于分析 in-batch negatives 的
> popularity sampling bias。初始 `alpha=1.0` 在三个 seed 上都显著提高总体 Recall，
> 但 effect audit 发现增益主要来自热门商品，coverage 也明显下降，因此没有直接替换
> 主模型。随后我把 correction 写成 `alpha * log(q)` 的受控消融，并同时检查 Recall、
> item 热度桶和曝光覆盖率。这个过程说明 sampled-softmax 风格修正需要结合业务目标
> 验收，不能只看一个总 Recall 数字。

## 9. Evidence Files

```text
outputs/transformer_logq_smoke/
outputs/text_timeaware_transformer_max100_logq*_full_eval/eval_summary.json
outputs/transformer_logq_effect_audit/audit_summary.json
```

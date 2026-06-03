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

> 我实现了 LogQ / BatchQ 系列 correction，用于分析 in-batch negatives 的 popularity
> sampling bias。强 LogQ 在三个 seed 上都显著提高总体 Recall，但 effect audit 发现
> 增益主要来自 head item，coverage 明显下降，因此没有直接替换主模型。随后我用
> Uber batch appearance Q 和较低 `alpha=0.10` 做受控 Pareto 验证：seed42 上总 Recall
> 和长尾都提升，但 seed2024 上虽然总 Recall 继续提升，`1-5` 与 `6-20` 低热度目标桶
> 显著回退，long-tail bootstrap CI 也整体为负。因此最终不采用 LogQ / BatchQ，保留
> 基础 InfoNCE Transformer TT + 4ch RRF 作为 canonical。这个过程说明 sampled-softmax
> 修正必须结合长尾、coverage 和多 seed 验收，不能只看一个总 Recall 数字。

## 10. Evidence Files

```text
outputs/transformer_logq_smoke/
outputs/text_timeaware_transformer_max100_logq*_full_eval/eval_summary.json
outputs/transformer_logq_effect_audit/audit_summary.json
outputs/transformer_logq_alpha_smoke/
outputs/transformer_logq_alpha_smoke_audit/
```

## 11. Uber BatchQ Low-Alpha Pareto Smoke

### 11.1 Constraint objective

本轮不再以最大化单一 Recall 为目标，而是使用预先锁定的约束：

```text
max Recall@50
s.t. 长尾桶基本不退化
     Top50 热门占比不过度增加
     catalog coverage 基本保持
     exposure entropy / Gini 不明显恶化
```

所有组均使用相同 `seed=42`、`3 epochs`、`50K limited-valid`、exact Top50 和
Uber batch appearance Q：

```text
Q = 1 - (1 - w) ** batch_size
```

### 11.2 Completed results

| Alpha | Recall@50 | coverage | Top50 `>100` share | entropy | Gini | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.124460 | 141547 | 20.97% | 0.9261 | 0.6460 | baseline |
| 0.05 | 0.130240 | 139931 | 23.97% | 0.9208 | 0.6643 | FAIL: `6-20` bucket |
| 0.10 | 0.137700 | 140411 | 25.84% | 0.9173 | 0.6732 | PASS |
| 0.15 | 0.145220 | 137705 | 29.97% | 0.9089 | 0.6982 | FAIL: Gini |
| 0.25 | 0.157980 | 129609 | 38.23% | 0.8884 | 0.7534 | reference only |

`alpha=0.10` 的 target item popularity bucket Recall@50：

| Bucket | Baseline | Uber BatchQ `alpha=0.10` | Delta |
|---|---:|---:|---:|
| `1-5` | 0.032455 | 0.043528 | +0.011073 |
| `6-20` | 0.072903 | 0.073520 | +0.000618 |
| `21-100` | 0.114014 | 0.124817 | +0.010803 |
| `>100` | 0.160671 | 0.180362 | +0.019691 |

### 11.3 Interpretation boundary

`alpha=0.10` 是 limited-valid 的温和 Pareto candidate，不是新 canonical。它在 coverage
基本保持、热门占比受控的前提下提高了四个 target 热度桶，但仍需要 full-test 与
multi-seed 验收。面试中应说明：LogQ 是 sampling-bias correction，不是单调去热门的
旋钮；强修正会改变 embedding dynamics，因此必须使用带约束的离线审计，而不能只报
总 Recall。

### 11.4 Other sampling methods already tested

| Method | Limited-valid Recall@50 | coverage | Top50 `>100` share | Current decision |
|---|---:|---:|---:|---|
| Refined LogQ `alpha=1.0` | 0.180360 | 40867 | 81.67% | diagnostic-only, reject strong setting |
| MNS `50% uniform` | 0.154040 | 90824 | 41.22% | reject current ratio |
| MNS + Refined LogQ | 0.161880 | 28560 | 85.51% | diagnostic-only, reject |

这些结果不证明 refined LogQ 或 MNS 永久无效，只证明当前强度或比例不适合作为本轮主候选。
在 Uber BatchQ `alpha=0.10` 完成 full 验收前，不继续扩展新的采样 sweep。

## 12. Uber BatchQ Alpha=0.10 Full Validation Queue

本轮预先锁定三个关卡，任何一关失败都自动停止：

1. `alpha=0.00 seed42` full sanity：同一 Uber BatchQ 分支将 correction 归零，
   `|full_test_recall@50 - 0.103168| < 0.001`。
2. `alpha=0.10 seed42` full train + full effect audit：四个 target 热度桶均不低于 canonical，
   coverage 至少为 canonical 的 `95%`，Top50 `>100` share `<30%`，Gini `<0.70`。
3. `alpha=0.10 seed2024 / seed2025` full train + full test：三个 paired delta 全部为正，
   candidate 最差 seed 高于 canonical 最差 seed。

输出写入：

```text
outputs/transformer_sampling_uber_alpha010_full_validation/
outputs/text_timeaware_transformer_sampling_full/uber-batchq-alpha000-sanity-seed42/
outputs/text_timeaware_transformer_sampling_full/uber-batchq-alpha010-seed*/
```

本轮不自动启动 4ch、Faiss 或 canonical replacement。

### 12.1 Full validation hardening

正式启动前补充两项可信度增强：

1. 三个 seed 均使用对应 historical baseline checkpoint 做 paired effect audit，而不是只审计 seed42。
2. 每个 paired audit 增加 deterministic bootstrap CI：

```text
bootstrap_seed = 42
bootstrap_resamples = 10000
overall Recall delta CI95
long-tail Recall delta CI95, where target item train popularity <= 20
```

overall Recall delta 的 `CI95 low > 0` 是每个 seed 的硬 gate。长尾 CI 只报告，不作为硬 gate；
长尾不伤害仍由 `1-5` 和 `6-20` 两个 bucket 分别不回退约束保证。

### 12.2 Interrupted run archive boundary

一次误启动的 gate0 在人工停止前写出 partial checkpoint，但没有 full eval 或 gate JSON。
正式重跑前必须先将 partial outputs 和 server log 移动到：

```text
/workspace/server-logs/archives/transformer_sampling_uber_alpha010_interrupted_<timestamp>/
```

不得续跑、覆盖或删除 partial 现场。

## 13. Final Closeout: Uber BatchQ `alpha=0.10` Rejected

### 13.1 Gate summary

正式队列已完成并按预注册 gate 自动停止。服务器未启动 4ch、Faiss 或 canonical
replacement。

| Gate | Result | Key metric | Decision |
|---|---|---:|---|
| Gate0 `alpha=0.00` sanity | PASS | full test R@50 = 0.103116 | 与 canonical 0.103168 对齐，口径干净 |
| Gate1 `alpha=0.10` seed42 | PASS | full test R@50 = 0.112047 | 四个 target popularity buckets 均提升 |
| Gate2 `alpha=0.10` seed2024 | FAIL | full test R@50 = 0.114982 | 总 Recall 提升，但低热度桶显著回退 |

### 13.2 Seed42 looked healthy

Seed42 上 `alpha=0.10` 的总体提升和长尾提升均通过 gate：

```text
overall Recall delta point estimate = +0.008879
overall CI95 = [0.008311, 0.009461]
long-tail Recall delta point estimate = +0.002055
long-tail CI95 = [0.001171, 0.002964]
Top50 >100 share = 24.80%
Gini = 0.6697
```

如果只看这个 seed，`alpha=0.10` 会显得是健康改进。

### 13.3 Seed2024 exposed instability

Seed2024 上总体 Recall 继续上升，但低热度目标桶退化：

```text
overall Recall delta point estimate = +0.011278
overall CI95 = [0.010895, 0.011660]
1-5 bucket delta = -0.004252
6-20 bucket delta = -0.002779
long-tail Recall delta point estimate = -0.003202
long-tail CI95 = [-0.003751, -0.002670]
Top50 >100 share = 26.92%
Gini = 0.6919
```

长尾 CI95 整体为负，说明回退不是随机噪声。该结果触发预注册 gate：

```text
failed_constraints = ["bucket_1-5", "bucket_6-20"]
status = stopped_after_alpha010_seed2024_full_audit
```

### 13.4 Final decision

本项目最终不采用 LogQ / Uber BatchQ：

```text
Accepted canonical:
  Transformer TT trained with base InfoNCE
  + 4ch valid-selected Weighted RRF

Rejected / diagnostic:
  old empirical LogQ alpha=1.0
  old empirical LogQ alpha=0.25
  Uber BatchQ alpha=0.25
  Uber BatchQ alpha=0.10
```

原因：

- LogQ / BatchQ 能稳定提高总体 Recall。
- 但强 correction 造成明显 head concentration 和 coverage loss。
- 温和 `alpha=0.10` 在 seed42 健康，但 seed2024 显著伤害长尾。
- 继续调 `alpha` 容易变成事后搜索，不符合当前收口目标。

### 13.5 Interview closeout wording

> 我最终没有把 LogQ 放进主模型。它确实能提高总 Recall，但强修正会把收益转移到
> 热门商品。温和的 Uber BatchQ `alpha=0.10` 在 seed42 上看起来很健康，但 multi-seed
> 验证时 seed2024 暴露了显著长尾退化。因为我的预注册目标是“总 Recall 提升且长尾不
> 受伤”，所以按 gate 拒绝它。这个负结果本身是项目亮点：我没有被漂亮的单指标诱导，
> 而是用 bucket、coverage、多 seed 和 bootstrap CI 验证了模型是否真的健康。

# max_len=50 Pooling Paired Smoke 报告

**报告日期：** 2026-05-19  
**脚本：** `scripts/train_maxlen50_pooling_smoke.py`  
**输出目录：** `outputs/attention_pooling_maxlen50_smoke/`  
**状态：** ✅ 完成 — 2026-05-19 16:29 UTC  
**结论：⛔ 两个候选变体均未达到继续训练阈值，立即停止，不进入主线**

---

## 1. 实验动机

前次 attention smoke（`outputs/attention_pooling_smoke/`）存在一个关键限制：`build_history_matrix` 将历史截断至 `max_len=20`，导致 `history_lengths` 数组中不存在长度 > 20 的用户，gt20 bucket 永远为空，无法评估注意力机制对长历史用户的影响——而这正是提出 attention 的原始动机。

本次实验通过两项改进重新验证：

1. **`max_len=50`**：扩大历史窗口，让 33,342 个 >20 交互用户（占 6.7%）能被真实评估
2. **`compute_raw_history_lengths`**：bucket 分配使用截断前的真实历史长度，修复前次 bug

同时引入第三个变体（**gated fusion**），测试是否存在比单一 time_decay 或 attention 更好的混合策略。

**本次结果不改变主线结论，作为 diagnostic finding 记录。**

---

## 2. 实验设置（Paired Smoke）

以下所有设置三个模型完全一致：

| 参数 | 值 |
| --- | --- |
| 数据集 | Movies_and_TV 5-core（同 final model） |
| item tower | 完全相同（item_id_emb + text_proj + has_text_mask，additive fusion） |
| embedding_dim | 64 |
| batch_size | 4096 |
| learning_rate | 0.001 |
| weight_decay | 1e-6 |
| temperature τ | 0.15 |
| use_l2_norm | True |
| seed | 42 |
| **history_max_len** | **50**（前次 smoke 为 20） |
| history_weight | 1.0 |
| epochs | 3 |
| eval_max_users | 50,000（limited valid） |
| eval_batch_size | 256 |
| eval_k_list | [20, 50, 100] |
| seen_mask | train items only（valid eval 口径） |
| **bucket 分配** | **raw history length（截断前）** |

唯一差异：`pooling_type`（time_decay / time_aware_attention / gated）

### 三种 pooling 设计

**time_decay**（基线）：指数衰减加权均值，与 final model 完全相同逻辑，仅 max_len 从 20 改为 50。

**time_aware_attention**：Scaled dot-product attention，加入 log-decay 位置偏置：
```
score_k = (user_emb · hist_emb_k) / sqrt(dim) + (L-1-k) * log(decay_rate)
```
位置偏置 = 0（最新），约 −10.9（最旧，L=50）。当注意力评分均匀时，softmax 退化为 time_decay 权重。无额外参数（与 time_decay 参数量相同：41,715,840）。

**gated**：可学习门控融合 time_decay pool 和 pure attention pool：
```
gate = sigmoid(W_g · user_id_emb)     [W_g: 1×64, 初始化为零 → gate=0.5]
pooled = gate * td_pool + (1-gate) * attn_pool
```
额外参数：64（`gate_proj` 权重），总参数：41,715,904。

---

## 3. 数据集基本统计（max_len=50 视角）

| 统计量 | 值 |
| --- | --- |
| 总用户数 | 497,449 |
| 非空历史用户 | 497,449 |
| 平均原始历史长度 | 8.68 |
| 原始历史 >20 的用户 | 33,342（6.7%）← gt20 bucket 来源 |
| 原始历史 >50 的用户 | 6,964（1.4%）← max_len=50 仍会截断 |

---

## 4. 三模型训练结果

### 4.1 Time-decay max_len=50（基线）

| Epoch | Train Loss | R@50 | NDCG@50 | MRR@50 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.168614 | 0.114500 | 0.048822 | 0.032141 | 49.8s |
| **2** | **6.090448** | **0.119740** | **0.051148** | **0.033708** | 49.0s |
| 3 | 5.831892 | 0.119740 | 0.050704 | 0.033137 | 48.9s |

### 4.2 Time-aware Attention max_len=50

| Epoch | Train Loss | R@50 | NDCG@50 | MRR@50 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.168651 | 0.114500 | 0.048840 | 0.032168 | 50.6s |
| 2 | 6.090326 | 0.119560 | 0.051118 | 0.033712 | 49.3s |
| **3** | **5.831716** | **0.119700** | **0.050705** | **0.033150** | 49.5s |

### 4.3 Gated max_len=50

| Epoch | Train Loss | R@50 | NDCG@50 | MRR@50 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.117812 | 0.112080 | 0.047304 | 0.030908 | 51.0s |
| 2 | 6.071824 | 0.117560 | 0.050033 | 0.032876 | 51.9s |
| **3** | **5.826532** | **0.118620** | **0.050053** | **0.032620** | 51.9s |

---

## 5. Overall Recall@50 对比

| 模型 | best_epoch | Recall@50 | NDCG@50 | MRR@50 | Δ vs td | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Time-decay（基线） | 2 | **0.119740** | **0.051148** | **0.033708** | — | 147.8s |
| Time-aware Attention | 3 | 0.119700 | 0.050705 | 0.033150 | **−0.000040** | 149.4s |
| Gated (TD+Attn) | 3 | 0.118620 | 0.050053 | 0.032620 | **−0.001120** | 154.8s |

**两个候选变体均低于 time_decay baseline。**

---

## 6. History Bucket Recall@50（真实历史长度分桶）

| 桶 | 用户数 | td R@50 | attn R@50 | Δ(attn) | gated R@50 | Δ(gated) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤5 | 27,333 | 0.139538 | 0.139502 | −0.000037 | 0.139319 | −0.000220 |
| 6-20 | 19,349 | 0.102383 | **0.102486** | **+0.000103** | 0.100574 | −0.001809 |
| **>20** | **3,318** | **0.057866** | 0.056962 | **−0.000904** | 0.053345 | **−0.004521** |

**gt20 bucket 现已正确评估（3,318 用户）。**

- time_decay max_len=50 在 gt20 桶 R@50=0.057866（有意义的非零基准）
- time_aware_attention 在 gt20 微弱落后（−0.0009），在 6-20 桶有极小提升（+0.0001）
- gated 在 gt20 落后明显（−0.0045），显示纯注意力分支拉低了长历史用户表现

---

## 7. Sanity Check

### time_aware_attention

| 检查项 | 结果 |
| --- | --- |
| NaN 计数 | **0** ✅ |
| Inf 计数 | **0** ✅ |
| attention weight sum（均值） | **1.0000** ✅ |
| 最大单位置权重 | 0.4128（未退化为 1-hot） |

### gated

| 检查项 | 结果 |
| --- | --- |
| gate 均值 | **0.4949**（≈初始值 0.5，几乎未学到偏好） |
| gate 范围 | [0.3734, 0.6634] |
| pct_td_preferred (gate>0.5) | **39.5%** |
| NaN 计数（attention 分支） | **0** ✅ |
| Inf 计数 | **0** ✅ |
| attention weight sum（均值） | **1.0000** ✅ |
| 最大单位置权重（attention 分支） | 0.3361 |

gated 模型的 gate 均值极接近初始值（0.5），说明在 3 epoch 内未收敛到有意义的偏好。

---

## 8. Unique Hit 分析

| 指标 | Time-decay | Time-aware Attn | Gated |
| --- | ---: | ---: | ---: |
| 总命中用户数 | 5,987 | 5,985 | 5,931 |
| 三者均命中 | 5,179 | — | — |
| **仅此模型命中** | **623** | **92** | **111** |

time_aware_attention 独有命中仅 92，远少于 time_decay 独有的 623，说明注意力分支未能覆盖 time_decay 无法命中的用户。

---

## 9. 是否满足继续训练条件

**否。**

| 条件 | attn | gated |
| --- | --- | --- |
| R@50 ≥ td + 0.001 | ❌ Δ = −0.000040 | ❌ Δ = −0.001120 |
| gt20 bucket 改善 | ❌ Δ = −0.000904 | ❌ Δ = −0.004521 |
| 6-20 bucket 改善 | ✅ Δ = +0.000103（过小） | ❌ Δ = −0.001809 |

**结论：两个变体均未达到继续训练阈值，立即停止，不进行 full training。**

---

## 10. max_len=20 → max_len=50 对 time_decay 的影响

| 实验 | max_len | R@50 |
| --- | --- | ---: |
| 前次 attention smoke td baseline | 20 | 0.119840（epoch 2） |
| 本次 td baseline | 50 | 0.119740（epoch 2） |

两者几乎完全相同（差 −0.0001），说明：
- 多数用户历史较短（avg=8.68），max_len=20 已覆盖绝大多数用户的全部历史
- 扩大到 max_len=50 对 time_decay 提升极小，长尾历史对当前模型无显著增益

---

## 11. 对 time_aware_attention 设计的诊断

time_aware_attention 的 Δ 极小（−0.000040）而不是明显正向或负向，原因：

**log-decay 位置偏置主导注意力评分。**

以 L=50、decay_rate=0.8 为例：
- 最旧 item 的位置偏置：49 × log(0.8) ≈ −10.93
- 典型注意力评分 (q·k)/8 范围：±0.1~0.5

位置偏置的量级（~10）远大于注意力评分（~0.5），导致 softmax 后的权重几乎与 time_decay 权重相同。仅对注意力评分极度突出的 item（差异 >2~3）才会产生可观察的偏差。

**这解释了为什么 time_aware_attention ≈ time_decay。**

要让时间感知注意力真正起作用，需要缩放位置偏置，例如：
- `time_bias = (L-1-k) * log(decay_rate) / scale`（按 sqrt(dim) 归一化）
- 或可学习的偏置权重

但这不是本次 smoke 的范围。

---

## 12. 是否改变 final model 选择

**否。**

当前 final model（Text + Time-decay Mean Pool τ=0.15，max_len=20）的 full test Recall@50 = 0.078315，与本次 limited valid smoke 结果不可直接比较。本次两个候选变体在 50K limited valid 上均低于 time_decay baseline，无推翻 final model 的依据。

**final model 保持不变：Text + Time-decay Mean Pool τ=0.15，max_len=20。**

---

## 13. 是否建议继续 Transformer / 更深 attention

**否（当前阶段）。**

诊断结论：
1. 在当前数据（avg history=8.68，majority ≤ 20）下，time_decay 已充分利用历史信息
2. 扩大 max_len=20→50 对 time_decay 几乎无提升（Δ=−0.0001）
3. attention 机制（无论是否加时间先验）在 3 epoch smoke 中均未超过 time_decay
4. gt20 bucket（3,318 用户，6.7%）的基准 R@50=0.057866，低于总体 0.119740，与前次 CLAUDE.md 记录一致（long-history users recall naturally lower due to diverse interests）

在上述前提下，Transformer 用户塔的额外复杂度在当前数据集上 ROI 不明确。若面试后有反馈认为用户塔建模不够深入，可考虑作为 M12 升级点（参见 decision_log AGENTS.md Section 4）。

---

## 14. 是否建议写入 README / 简历

**README：否。** 本次为诊断性 smoke test，未达阈值，不入主线。

**简历：否。** 负向结果不直接写入简历。

**面试口述用途：** 可用于回答"如何验证用户塔设计"——描述用 max_len=50 paired smoke + gt20 bucket 专项评估长历史用户，验证 time_decay 已充分；并解释为何 time_aware_attention ≈ time_decay（位置偏置主导）。体现系统性实验思维。

---

## 15. 局限性

```text
1. Limited eval（50K users，非 full valid 497K）：
   Delta 较小（<0.002）时不能外推至 full eval。

2. 仅 3 epochs：
   gated model epoch 3 仍未收敛（gate≈0.5），更多 epoch 可能学到更明确偏好，
   但 epoch 3 总体 R@50 仍落后 td，不具备继续依据。

3. time_aware_attention 位置偏置过强：
   log-decay 偏置 ~10× 大于典型注意力评分，实质上退化为 time_decay。
   若要真正测试时间感知注意力，需对偏置幅度加约束或归一化。

4. 无 full test eval：
   所有数字来自 limited valid（50K users，3 epoch），
   不能与 final model full test Recall@50 = 0.078315 直接比较。

5. gt20 bucket 样本量小：
   3,318 评估用户，统计噪声较大。Δ ≈ 0.001 在此桶上不具有显著性。
```

---

## 附录：对比汇总（JSON）

```
outputs/attention_pooling_maxlen50_smoke/unique_hit_comparison.json
```

关键字段：
- `attn_vs_td_delta: -0.000040`（threshold: +0.001 → not met）
- `gated_vs_td_delta: -0.001120`（threshold: +0.001 → not met）
- `bucket_delta_attn_vs_td.gt20: -0.000904`
- `bucket_delta_gated_vs_td.gt20: -0.004521`
- `meets_threshold_attn: false`
- `meets_threshold_gated: false`

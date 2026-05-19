# Attention Pooling 用户塔 Smoke Test 报告

**报告日期：** 2026-05-19  
**脚本：** `scripts/train_attention_pooling_smoke.py`  
**输出目录：** `outputs/attention_pooling_smoke/`  
**状态：** ✅ 完成 — 2026-05-19 15:26 UTC  
**结论：⛔ 未达到继续训练阈值，建议停止，不进入主线**

---

## 1. 为什么做 Attention Smoke

当前 final model（Text + Time-decay Mean Pool Two-Tower）的用户历史池化使用指数时间衰减加权均值，不含注意力机制。

本次 smoke test 的动机是：

- **诊断性问题**：time-decay mean pooling 是否对长历史用户已足够？
- **潜在改进假设**：attention pooling 可以让用户动态关注与自身 profile 最相关的历史 item，理论上对 history > 20 的用户更有益（这些用户历史信息更丰富，但 mean pooling 会平均稀释）

评估重点：
1. 整体 Recall@50 是否提升 ≥ 0.001（继续训练阈值）
2. history > 20 bucket 是否有明显改善

**本次结果不改变主线结论，attention smoke 作为 diagnostic finding 记录。**

---

## 2. Paired Smoke 设置

以下所有设置两模型完全一致，确保公平对比：

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
| history_max_len | 20 |
| history_weight | 1.0 |
| epochs | 3 |
| eval_max_users | 50,000（limited valid） |
| eval_batch_size | 256 |
| eval_k_list | [20, 50, 100] |
| seen_mask | train items only（valid eval 口径） |

唯一差异：`pooling_type`（time_decay vs attention）

### Attention 设计

- 用户历史 item_id_emb 为 Keys/Values
- Query = `user_id_emb`（无新增可学习参数，参数量与 time_decay 版本相同：41,715,840）
- Scaled dot-product：`score_k = (user_emb · hist_emb_k) / sqrt(dim)`
- Softmax over valid positions（padding 位置 -inf mask）
- 空历史用户 pooled = 0（`nan_to_num` 处理）
- 无 Transformer、无多层 attention、无多兴趣建模

---

## 3. Time-decay Smoke Baseline 结果

| 指标 | 值 |
| --- | ---: |
| eval users | 50,000（limited valid） |
| **best epoch** | **2** |
| train_loss (ep1 → ep3) | 7.173 → 6.091 → 5.832 |
| **Recall@50** | **0.119840** |
| **NDCG@50** | **0.051130** |
| **MRR@50** | **0.033675** |
| Recall@20 | — |
| total train time | 138.2s（3 epoch） |

### Bucket Recall@50

| 历史长度桶 | 用户数 | 命中数 | Recall@50 |
| --- | ---: | ---: | ---: |
| ≤5（短历史） | 27,333 | 3,816 | **0.139611** |
| 6-20（中等历史） | 22,667 | 2,176 | **0.095999** |
| >20（长历史） | 0 | 0 | — |

*注：gt20 bucket 为 0 原因见第 12 节局限性说明。*

---

## 4. Attention Smoke 结果

| 指标 | 值 |
| --- | ---: |
| eval users | 50,000（limited valid） |
| **best epoch** | **3** |
| train_loss (ep1 → ep3) | 7.119 → 6.072 → 5.827 |
| **Recall@50** | **0.116040** |
| **NDCG@50** | **0.048945** |
| **MRR@50** | **0.031887** |
| total train time | 141.3s（3 epoch） |

### Bucket Recall@50

| 历史长度桶 | 用户数 | 命中数 | Recall@50 |
| --- | ---: | ---: | ---: |
| ≤5（短历史） | 27,333 | 3,739 | **0.136794** |
| 6-20（中等历史） | 22,667 | 2,063 | **0.091013** |
| >20（长历史） | 0 | 0 | — |

### Attention Sanity Check

| 检查项 | 结果 |
| --- | --- |
| NaN 计数 | **0** ✅ |
| Inf 计数 | **0** ✅ |
| attention weight sum（均值） | **1.0000** ✅（归一化正确） |
| attention weight sum（min/max） | 0.99999988 / 1.00000012 ✅ |
| 最大单位置权重 | 0.3360（未退化为 1-hot） |
| 最小有效位置权重 | 0.0495（权重有合理分布） |

Attention 机制工作正常，权重归一化、无 NaN/Inf、未退化。

---

## 5. Overall Recall@50 对比

| 模型 | best epoch | Recall@50 | NDCG@50 | MRR@50 | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Time-decay（baseline） | 2 | **0.119840** | **0.051130** | **0.033675** | 138s |
| Attention | 3 | 0.116040 | 0.048945 | 0.031887 | 141s |
| **Delta** | — | **−0.003800** | −0.002184 | −0.001787 | +3s |

**Attention 在所有指标上均低于 time-decay baseline。**

- Delta R@50 = −0.0038（不足于继续训练阈值 +0.001，方向相反）
- 训练时间基本相同（两者均约 140s / 3 epochs，attention 无额外开销）
- Attention 在 epoch 1 显著低于 time-decay（0.1075 vs 0.1144），此后追赶但未能超越
- Time-decay 在 epoch 2 到达峰值后下降（epoch 3: 0.11974），Attention 仍在上升（epoch 3: 0.11604）

---

## 6. >20 History Bucket 对比

| 桶 | Time-decay R@50 | Attention R@50 | Delta |
| --- | ---: | ---: | ---: |
| ≤5 | 0.139611 | 0.136794 | **−0.002817** |
| 6-20 | 0.095999 | 0.091013 | **−0.004985** |
| **>20** | **0.000000** | **0.000000** | **0** |

**gt20 bucket 在本次 smoke eval 中 user 数为 0，无法评估。**

原因：eval 使用 `build_history_matrix` 返回的截断后历史长度（最大值 = max_len = 20），因此没有用户的存储历史长度超过 20，gt20 bucket 永远为空（见局限性第 2 条）。

可观察的两个有效桶（≤5 和 6-20）中，**attention 均弱于 time-decay**，差距分别为 −0.0028 和 −0.0050。

---

## 7. Unique Hit 对比

对 50,000 eval 用户中，两个模型各自在 valid Recall@50 中命中的用户集合进行集合分析：

| 指标 | 数值 |
| --- | ---: |
| Time-decay 总命中用户数 | **5,992** |
| Attention 总命中用户数 | 5,802 |
| **两者均命中** | **5,135** |
| **仅 attention 命中**（attention 有、td 无） | **667** |
| **仅 time-decay 命中**（td 有、attn 无） | **857** |

Attention 带来了 667 个 time-decay 未命中的新用户，但同时失去了 857 个 time-decay 能命中的用户。**净差 = −190 用户命中**，不是正向增益。

---

## 8. 是否满足继续训练条件

**否。**

| 停止条件 | 判断 |
| --- | --- |
| attention R@50 ≥ time-decay R@50 + 0.001 | ❌ 实际 delta = **−0.0038**（阈值相反） |
| 训练更稳定 | ❌ attention epoch 1 落后更多（0.1075 vs 0.1144） |
| 长历史用户有提升 | ❌ 无法评估（gt20 bucket 为空）；6-20 bucket 明显更差（−0.005） |

**结论：Attention pooling smoke 未达到继续训练阈值，建议立刻停止，不进行 epoch > 3 的完整训练。**

---

## 9. 是否改变当前 Final Model 结论

**否。**

当前 final model（Text + Time-decay Mean Pool τ=0.15 decay_rate=0.8）在：
- 20-epoch full test Recall@50 = **0.078315**（brute-force）
- valid-selected multichannel Recall@50 = **0.104776**（4-channel RRF）

本次 attention smoke 在 3-epoch limited valid 上的 Recall@50 = 0.116040 < time-decay 的 0.119840，差距 −0.0038，且为 50K 用户的 limited eval，**不具备推翻 20-epoch full test 结论的统计可靠性**。

**当前 final model 保持不变：Text + Time-decay Mean Pool Two-Tower。**

---

## 10. 是否建议进入 README

**否。**

README 记录的是项目主线已完成并验证的结果。本次 attention smoke：
- 未达到继续训练阈值
- 未完成 full test eval
- 未改变任何正式指标
- 属于诊断性探查，不是升级

不建议在 README 中添加 attention pooling 相关内容。

---

## 11. 是否建议进入简历

**否。**

简历应只记录正式完成、完整评估、有明确提升的结果。本次 attention smoke 为负向结果（-0.0038），不满足简历要求。

如果要在面试中提及，可以用作 "我们也探索了 attention pooling，但 3-epoch smoke 显示它在 50K limited valid 上落后 time-decay，因此决定不进入主线" 作为 diagnostic / ablation 结果，体现系统性实验设计。

---

## 12. 局限性

```text
1. Limited eval（50K users，非 full valid 497K）：
   结果为指示性参考，不代表全量评估结论。
   Delta 较小时（<0.005）需谨慎外推至 full eval。

2. gt20 bucket 无法评估（关键限制）：
   build_history_matrix 将历史长度截断至 max_len=20，
   因此 history_lengths 数组中不存在长度 > 20 的用户，
   gt20 bucket 永远为空。
   正确实现应在截断前统计实际历史长度（train_df.groupby("user_idx").size()）。
   当前无法评估注意力机制对 >20 历史用户的影响，
   这正是提出 attention 的原始动机所在。

3. 仅 3 epochs：
   Attention 在 epoch 3 仍未到达收敛（R@50 仍在上升），
   time-decay 在 epoch 2 已收敛后下降。
   可能需要更多 epoch 才能看到 attention 真实上限，
   但考虑到 epoch 3 与 time-decay ep2 差距仍有 -0.0038，
   以及 full training（20 epoch）的时间成本，不建议继续。

4. Attention query 设计局限：
   当前使用 user_id_emb 直接作为 query（无额外投影层）。
   user_id_emb 同时承担两个角色：静态用户表示 + 注意力查询。
   这种 dual-role 设计可能限制 attention 的表达能力。
   更强的设计（如 learnable query projection）未测试。

5. 无 full test eval：
   本报告所有数字来自 limited valid（50K users, 3 epoch）。
   不能与 final model full test Recall@50 = 0.078315 直接比较。
```

---

## 附录：训练日志

### Time-decay baseline train log（3 epoch）

| Epoch | Train Loss | R@50 | NDCG@50 | MRR@50 | 耗时（s） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.172756 | 0.114420 | 0.048829 | 0.032171 | 46.3 |
| **2** | **6.091473** | **0.119840** | **0.051130** | **0.033675** | 45.8 |
| 3 | 5.832237 | 0.119740 | 0.050711 | 0.033152 | 46.1 |

### Attention train log（3 epoch）

| Epoch | Train Loss | R@50 | NDCG@50 | MRR@50 | 耗时（s） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.118933 | 0.107480 | 0.044915 | 0.029122 | 47.8 |
| 2 | 6.071516 | 0.114520 | 0.048457 | 0.031681 | 46.3 |
| **3** | **5.827014** | **0.116040** | **0.048945** | **0.031887** | 47.3 |

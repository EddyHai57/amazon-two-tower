# Multi-channel Retrieval 结果审计报告

**审计日期：** 2026-05-19  
**审计对象：** `scripts/run_multichannel_retrieval.py` + `outputs/multichannel_itemcf_twotower_v1/`  
**审计人：** Claude Code（自动审计 + 交叉验证）  
**审计结论：** ✅ **PASSED — 无数据泄漏，无 eval 口径违规，核心数字复核一致**

---

## 1. 数据口径检查

| 检查项 | 预期 | 实测 | 状态 |
| --- | --- | --- | --- |
| data_dir | `data/processed/movies_tv_5core` | `data/processed/movies_tv_5core` | ✅ |
| n_users | 497,449 | 497,449 | ✅ |
| n_items | 153,977 | 153,977 | ✅ |
| train interactions | 4,319,438 | 4,319,438 | ✅ |
| eval split | test | test | ✅ |
| eval users（非 cold） | 496,470 | 496,470 | ✅ |
| cold target skipped | 979 | 979 | ✅ |

**结论：** 数据口径与 canonical preprocess 完全对齐。

---

## 2. seen-item mask 检查

### ItemCF 通道

| 检查项 | 实现 | 正确性 |
| --- | --- | --- |
| test eval seen-item mask | `add_valid_to_seen(train_seen, valid_df)` → train + valid items | ✅ |
| 与 standalone ItemCF 对比 | 两者均使用 `eval_seen_filter=train_valid` | ✅ 一致 |
| target item 处理 | `if candidate_item in seen_items and candidate_item != target_item` | ✅ |

**注：** 条件中的 `candidate_item != target_item` 是冗余判断（test target 不在 train/valid seen 中，永远不会触发），但与 standalone `run_itemcf.py` 中 `recommend_for_user` 逻辑完全一致，不影响结果。

### Two-Tower 通道

| 检查项 | 实现 | 正确性 |
| --- | --- | --- |
| test eval seen-item mask | `merge_seen_items(train_seen, valid_df)` → train + valid | ✅ |
| test history matrix | `concat(train_df, valid_df)` → train + valid history | ✅ |
| 与 standalone eval_only 对比 | 逻辑完全相同（源码比对确认） | ✅ 一致 |

### Target item 泄漏验证

随机抽取 1,000 名 test 用户，验证其 test target item 是否出现在 train/valid seen 中：

```
Target-in-seen leaks (1000 users sample): 0  (expected 0)
```

**结论：** test target 未泄漏进 seen-item mask。

---

## 3. candidate generation 检查

### ItemCF 候选

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| similarity 构建数据源 | `build_train_sets(train_df)` — 仅使用 train history | ✅ |
| sim_topk | 100（与 canonical ItemCF 一致） | ✅ |
| max_user_history | 100（与 canonical ItemCF 一致） | ✅ |
| candidates_per_channel | 200（足够填满 quota 50 + RRF 50） | ✅ |

### Two-Tower 候选

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| checkpoint | `outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt` epoch=17 | ✅ |
| model config | pooling_type=time_decay, decay_rate=0.8, temperature=0.15, embedding_dim=64 | ✅ |
| item embedding | encode_all_items → shape=(153977, 64)，3 批完整编码 | ✅ |
| user history | test_history_matrix = train + valid（与 standalone eval_only 相同） | ✅ |
| seen-item filter | `s[seen_arr] = -np.inf` 屏蔽 train+valid items | ✅ |

### quota merge 检查

- 所有 quota 组合总和均为 50（检查了全部 7 组）✅
- dedup 逻辑正确（set 去重，保留先序） ✅
- 两路候选集 Jaccard overlap 仅 7.6%，实际 top-20/30 中重叠概率极低，quota 基本都能填满 50 个

### RRF 检查

- score = sum of `1/(k + rank)` over channels ✅
- 只使用 rank 信息，不使用任何 test label ✅
- 输出 top-`rrf_top_n`=50 个候选 ✅

---

## 4. metric calculation 检查

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| denominator | `n_eval_users = 496,470` | ✅ |
| cold target exclusion | `eval_targets = test_df[~cold_mask]` | ✅ |
| Recall@K | target 在 top-K 中则 +1，否则 0；sum / n | ✅ |
| NDCG@K | `1/log2(rank+1)`（binary relevance，single target）| ✅ |
| MRR@K | `1/rank` if rank <= K | ✅ |
| overlap 定义 | **Jaccard index**：`|A∩B| / |A∪B|`（不是 `|A∩B|/K`）| ✅（需注意） |
| unique hit 定义 | 某通路 top-50 命中 target 但另一路 top-50 未命中 | ✅ |
| bucket pop 定义 | `Counter(train_df["item_idx"])` — 与 canonical 一致 | ✅ |
| bucket 用户数之和 | 496,470（等于 n_eval_users）| ✅ |

**注：Recall@100 = Recall@50**（见已知限制第 1 条）。

---

## 5. 关键数字交叉验证

### 单路 sanity check（最重要的验证）

| 指标 | 本实验 | standalone 结果 | 是否一致 |
| ---: | ---: | ---: | --- |
| quota_icf50_tt0 R@50 | **0.083570** | ItemCF clean full eval: **0.083570** | ✅ 精确一致 |
| quota_icf0_tt50 R@50 | **0.078315** | TwoTower full eval: **0.078315** | ✅ 精确一致 |

> 两路候选集分别退化为单路时，完全复现 standalone 结果，证明候选生成和 eval 逻辑均正确。

### Hit count 交叉验证

通过 overlap_stats 反推单路 Recall，与 standalone 逐一比对：

```
TT Recall@50 = (overlap_hits + tt_unique_hits) / n
            = (23083 + 15798) / 496470
            = 38881 / 496470
            = 0.078315  ✓ 精确等于 standalone TwoTower R@50

ICF Recall@50 = (overlap_hits + icf_unique_hits) / n
             = (23083 + 18407) / 496470
             = 41490 / 496470
             = 0.083570  ✓ 精确等于 standalone ItemCF R@50
```

> 此验证从 hit 计数角度独立复现了两路单路指标，与 eval 计算路径完全独立，是最强的一致性证明。

### RRF 边界检查

```
下界（单路最大）:    0.083570  （ItemCF）
RRF k=60 实测:      0.096727
上界（union@50）:   0.115391  （两路 top-50 覆盖的理论最大 Recall）
在界内:             ✓
RRF 捕获了 41.3% 的理论增量
```

### 核心指标确认

| 指标 | 精确值 | 四舍五入 |
| --- | ---: | ---: |
| best quota（icf20+tt30）Recall@50 | 0.089552 | 8.96% |
| best RRF（k=60）Recall@50 | 0.096727 | 9.67% |
| avg_candidate_overlap@50（Jaccard） | 0.076189 | 7.62% |
| TwoTower unique hits@50 | 15,798 | — |
| ItemCF unique hits@50 | 18,407 | — |
| overlap hits@50 | 23,083 | — |

---

## 6. 已知限制（非 BLOCKER）

### 限制 1：Recall@100 ≡ Recall@50（文档问题）

**原因：** `rrf_top_n=50` 且 quota 总和=50，merged list 最多 50 个候选。任何 rank≤50 的 target 对 K=100 也成立，故 Recall@100 = Recall@50。

**影响：** 结果报告中 Recall@100 列无独立信息，容易误导。

**修复建议：** 在报告和文档中明确标注"R@100 = R@50（候选集上限为 50）"。若需真 R@100，需重跑 `rrf_top_n=100`。

**严重等级：** ⚠️ DOCUMENTATION（不影响 R@50 结论）

### 限制 2：overlap 指标是 Jaccard，非 |A∩B|/K

**定义：** `avg_candidate_overlap@50` = avg of `|A∩B|/|A∪B|` over users（Jaccard index）。

**不是：** `|A∩B|/50`。

**数量含义：** Jaccard=0.076 ⟹ 平均每用户交集约 7 个 item（从 `|A∩B| ≈ 0.076×|A∪B| ≈ 0.076×93 ≈ 7`）。

**修复建议：** 报告中明确写"Jaccard overlap = 0.076"，并补充"avg absolute intersection ≈ 7 items per user out of 50"。

**严重等级：** ⚠️ DOCUMENTATION

### 限制 3：encode_all_items 中一行死代码

**位置：** `scripts/run_multichannel_retrieval.py` 第 122 行。

```python
end = min(start + num_items, num_items)  # 错误（start+num_items 不是批大小）
end = min(start + 65536, num_items)      # 正确，覆盖了上一行
```

**影响：** 零。第 123 行覆盖了第 122 行，所有 153,977 个 item 均正确编码。

**严重等级：** 🔵 COSMETIC（可在后续 cleanup 时修复）

### 限制 4：ItemCF 与 Two-Tower 使用不同 history 口径（预期行为）

- ItemCF history：train-only（用于构建 co-occurrence similarity，加入 valid 会改变图结构）
- TwoTower history：train+valid（用于计算 user embedding，反映 test 时刻的完整历史）

**影响：** 两路口径不同，但各自与 standalone 方法一致，且 sanity check 完全通过。这是合理的设计选择，不是 bug。

---

## 7. 审计结论

### BLOCKER：无

### 正式结论

| 维度 | 结论 |
| --- | --- |
| 数据泄漏风险 | **无** — target 未进入 history 或 seen mask |
| eval 口径 | **正确** — 与 canonical preprocess 和 standalone eval 完全对齐 |
| 单路 sanity check | **通过** — quota 退化结果精确等于 standalone 基线 |
| hit count 交叉验证 | **通过** — 从计数角度独立复现两路 Recall |
| RRF 实现 | **正确** — 只用 rank，不用 label，结果在理论边界内 |
| 关键指标 | **可信** — best quota R@50=0.089552，best RRF R@50=0.096727 |

### 对 README 的建议

审计通过，但在进入 README 前，需要解决以下文档问题：

1. **明确指出 Recall@100 = Recall@50** 是候选集上限造成的，不是评估代码问题
2. **明确指出 overlap 是 Jaccard**，补充绝对 intersection 估算
3. **明确说明 history 口径差异**（ItemCF: train-only，TwoTower: train+valid）
4. **标注所有指标为 offline evaluation**，不是线上 A/B

在以上文档收口后，数字和方法论均经审计，可进入 README。建议 Eddy 人工确认后再操作。

---

*本报告基于代码审查、数据验证和数字交叉验证生成。审计时间：2026-05-19。*

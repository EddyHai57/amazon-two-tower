# Multi-channel Retrieval v3 Audit

**审计日期：** 2026-05-19  
**脚本：** `scripts/run_multichannel_retrieval_v3.py`  
**配置：** `configs/multichannel_v3_balanced.yaml`  
**结论：** ✅ PASSED，无 BLOCKER

---

## 1. Weighted RRF 实现审计

### 1.1 核心函数（v3.py 第 93–110 行）

```python
def weighted_rrf_merge_n(channel_cands, weights, k, top_n):
    scores = defaultdict(float)
    for cands, w in zip(channel_cands, weights):
        if w <= 0:
            continue
        for rank, item in enumerate(cands, start=1):
            scores[item] += w / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [item for item, _ in ranked[:top_n]]
```

| 检查项 | 结果 |
| --- | --- |
| `score(item) = Σ w / (k + rank)` 实现正确 | ✅ |
| rank 从 1 开始（非 0） | ✅ |
| k=60（与报告一致） | ✅ 从 config 读取 |
| ICF weight=1.0，TT weight=1.0 | ✅ 常量 `wrrf_icf_w=1.0`, `wrrf_tt_w=1.0` |
| text_w=0.0 时 text 通路被完全跳过（`w <= 0` check） | ✅ |
| 只使用 rank，不使用 label 或 target | ✅ |
| top_n=50（rrf_top_n=50，同 v1） | ✅ |

### 1.2 权重传递（v3.py 第 392–421 行）

```python
wrrf_k = int(config.get("wrrf_k", 60))        # ← 60
wrrf_icf_w = float(config.get("wrrf_icf_w", 1.0))  # ← 1.0
wrrf_tt_w = float(config.get("wrrf_tt_w", 1.0))    # ← 1.0
...
weights = [wrrf_icf_w, wrrf_tt_w, text_w, pop_w]  # [1.0, 1.0, text_w, pop_w]
lambda u, w=weights: weighted_rrf_merge_n([
    icf_cands, tt_cands, text_cands, pop_cands
], w, wrrf_k, rrf_top_n)
```

| 检查项 | 结果 |
| --- | --- |
| weights 列表顺序与 channel_cands 顺序一致 | ✅ [icf, tt, text, pop] |
| `wrrf_pop0.5_text0.3` → weights=[1.0, 1.0, 0.3, 0.5] | ✅ JSON 验证 |
| Lambda 使用默认参数 `w=weights` 避免闭包引用 bug | ✅ |
| Quota 也使用默认参数 `iq=icf_q, tq=tt_q, pq=pop_q` | ✅ |

**JSON 验证（all_results_full.json，wrrf_pop0.5_text0.3）：**
```
icf_w=1.0  tt_w=1.0  text_w=0.3  pop_w=0.5  wrrf_k=60  ✅
```

---

## 2. 数据泄漏审计

### 2.1 Popularity 通路

```python
# v3.py 第 211 行
item_pop_counter: Counter[int] = Counter(bundle.train_df["item_idx"].tolist())
```

| 检查项 | 结果 |
| --- | --- |
| 只使用 `train_df`，不使用 valid 或 test | ✅ |
| `Counter` 仅统计 train 交互次数 | ✅ |
| pop_sorted_items = train 降序排列 | ✅ |
| 候选生成时过滤 `test_seen`（train+valid） | ✅ |

### 2.2 Text Semantic 通路

| 检查项 | 结果 |
| --- | --- |
| `item_text_norm_gpu` = 预计算文本 embedding，非 test label | ✅ |
| 用户 query = 历史 item 文本 embedding 时间衰减均值 | ✅ |
| 历史矩阵 = train + valid（无 test label） | ✅ |
| `has_text_q` 基于 query norm，不看 target | ✅ |
| seen-item mask 在 cosine 排序后应用（`s[seen_arr] = -np.inf`） | ✅ |
| `n_zero_text_query_users = 0`（全部 496,470 用户有有效文本 query） | ✅ JSON 验证 |

### 2.3 Seen-item mask

```python
# v3.py 第 192–193 行
train_seen = build_seen_items(bundle.train_df)
test_seen = merge_seen_items(train_seen, bundle.valid_df)  # train + valid
```

| 检查项 | 结果 |
| --- | --- |
| `test_seen` = train + valid（correct for test eval） | ✅ |
| ICF `icf_eval_seen = add_valid_to_seen(icf_full_seen, bundle.valid_df)` | ✅ |
| TwoTower `test_seen` 传入 `generate_twotower_candidates` | ✅ |
| Text Semantic `test_seen` 传入 `generate_text_semantic_candidates` | ✅ |
| Popularity `test_seen` 传入 `generate_popularity_candidates` | ✅ |

### 2.4 Test target 未进入 seen history

- Test target = 用户在 test split 的交互 item（temporal leave-one-out 最后一个）
- Test history matrix 包含 train + valid 交互（max 20 items）
- `test_seen = train + valid`：test target 不在 train/valid，不会被 mask
- ✅ test target 不在 seen mask，也不在历史 matrix 中

### 2.5 Cold-target 用户排除

```python
cold_mask = bundle.test_df["is_cold_item_for_eval"].astype(bool)
eval_targets_df = bundle.test_df[~cold_mask].copy()
```

| 检查项 | 结果 |
| --- | --- |
| Cold target items 排除（979 users） | ✅ |
| eval_targets 再次过滤（冗余但无害） | ✅ |
| 非冷用户 = 496,470 | ✅ JSON 确认 |

---

## 3. 指标一致性审计

从 `outputs/multichannel_v3_balanced/all_results_full.json` 直接读取：

| 指标 | 期望值 | 实测值 | 状态 |
| --- | ---: | ---: | --- |
| `run_type` | "full" | "full" | ✅ |
| `n_eval_users` | 496,470 | 496,470 | ✅ |
| v1 baseline Recall@50 | 0.096727 | **0.096727** | ✅ delta=0.000000 |
| v2 baseline Recall@50 | 0.108766 | **0.108766** | ✅ delta=0.000000 |
| v3 winner Recall@50 | 0.103384 | **0.103384** | ✅ |
| v3 winner avg_pop | 443 | **442.9** | ✅ |
| v3 winner icf_w | 1.0 | 1.0 | ✅ |
| v3 winner tt_w | 1.0 | 1.0 | ✅ |
| v3 winner text_w | 0.3 | 0.3 | ✅ |
| v3 winner pop_w | 0.5 | 0.5 | ✅ |
| v3 winner wrrf_k | 60 | 60 | ✅ |

---

## 4. Trade-off 复核

### 4.1 avg_pop 数据

| 组合 | avg_pop | avg_pop / v1 |
| --- | ---: | ---: |
| v1 ref | 264.5 | 1.0× |
| wrrf_pop0.1_text0.3 | 297.6 | 1.1× |
| wrrf_pop0.2_text0.3 | 330.3 | 1.2× |
| wrrf_pop0.3_text0.3 | 363.4 | 1.4× |
| wrrf_pop0.5_text0.3 | 442.9 | **1.7×** |
| **wrrf_pop1.0_text0.3** | **1946.3** | **7.4×** ← 相变 |
| v2 ref | 1642.0 | 6.2× |

**pop_w=0.5 → pop_w=1.0 时 avg_pop 从 443 跳至 1946（×4.4 跳变）** ✅ 确认

### 4.2 Bucket 分解（v3 winner vs v1 vs v2）

| 桶 | v1 | v3 winner | v2 | v3 vs v1 | v3 vs v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤5 | 0.044029 | **0.045342** | 0.044600 | +3.0% | +1.7% |
| 6-20 | 0.064008 | **0.065639** | 0.064732 | +2.5% | +1.4% |
| 21-100 | 0.085018 | **0.086057** | 0.082440 | +1.2% | +4.4% |
| >100 | 0.127714 | 0.141582 | **0.157393** | +10.8% | -10.0% |

- v3 winner **优于 v1，所有桶** ✅
- v3 winner **优于 v2，≤5 / 6-20 / 21-100 三个非头部桶** ✅
- v3 winner 在 >100 头部桶弱于 v2（预期：pop 通路全部命中 >100 桶，pop_w 减小必然降低头部 recall）

---

## 5. 非 BLOCKER 注意事项

| 编号 | 描述 | 影响 |
| --- | --- | --- |
| ⚠️ 1 | v2 ref 日志行打印 `delta vs V2_BEST_RECALL50`，但存储字段为 `delta_vs_v1`（vs v1） | 仅日志格式问题，JSON 数据正确 |
| ⚠️ 2 | `eval_targets` 对 cold-item 做了双重过滤（eval_df 已过滤一次，eval_targets 再过滤一次） | 冗余但结果无差异 |
| ⚠️ 3 | `item_coverage` 在全量评估下无区分度（153,928–153,936 / 153,977，约 99.97%） | 指标不具判别力，已在报告 Notes 中标注 |
| ⚠️ 4 | `Recall@100 = Recall@50`（rrf_top_n=50，同 v1） | 报告中已标注，不影响结论 |

---

## 6. 审计结论

| 检查分类 | 结果 |
| --- | --- |
| Weighted RRF 实现（权重、k、rank） | ✅ PASS |
| ICF/TT 权重固定为 1.0 | ✅ PASS |
| Text w≤0 时通路跳过 | ✅ PASS |
| Lambda 闭包正确捕获权重 | ✅ PASS |
| Popularity 只用 train split | ✅ PASS |
| Text semantic 无 test 泄漏 | ✅ PASS |
| Seen-item mask = train+valid | ✅ PASS |
| Test target 不在 seen mask | ✅ PASS |
| eval users = 496,470 | ✅ PASS |
| v1 baseline 精确复现 0.096727 | ✅ PASS |
| v2 baseline 精确复现 0.108766 | ✅ PASS |
| v3 winner 0.103384，avg_pop=443 | ✅ PASS |
| pop 相变点 pop_w=0.5→1.0 确认 | ✅ PASS |
| 非头部桶不低于 v1 | ✅ PASS |
| 非头部桶不低于 v2 | ✅ PASS |

**总结：✅ AUDIT PASSED。无 BLOCKER。V3 实验数据可信，可作为项目主结论。**

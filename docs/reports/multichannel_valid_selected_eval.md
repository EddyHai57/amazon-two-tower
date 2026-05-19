# Valid-Selected Multichannel Retrieval 审计报告

**审计日期：** 2026-05-19  
**脚本：** `scripts/run_multichannel_valid_selected.py`  
**配置：** `configs/multichannel_valid_selected.yaml`  
**状态：** ✅ 完成 — 2026-05-19 11:53 UTC

---

## 1. 背景与动机

V3 实验在 **test set** 上 sweep 了 15 组 weighted RRF 权重，按 Pareto 标准选出 `text_w=0.3, pop_w=0.5, k=60` 作为主结论（Recall@50=0.103384）。

**质疑点**：这组权重是否对 test set 有偶然拟合？正确的 ML 评估流程应是：

```
valid set sweep → 选定 frozen config → test set 运行一次
```

本次实验严格遵守此流程，目标是用独立的 valid 选择来验证 V3 结论的稳健性。

---

## 2. Valid vs Test Eval 口径区别

| 项目 | Valid Eval | Test Eval |
| --- | --- | --- |
| 评估目标 | valid_df 非冷启动行（~497,137 users） | test_df 非冷启动行（496,470 users） |
| seen-item mask | train items only | train + valid items |
| 候选生成历史 | train items only（max 20） | train + valid items（max 20） |
| ItemCF seen | train only | train + valid |
| 评估 target | valid 交互 item | test 交互 item |

---

## 3. Sweep 设计

| 参数 | 范围 |
| --- | --- |
| k | [30, 60, 100] |
| text_w | [0.0, 0.1, 0.3, 0.5] |
| pop_w | [0.0, 0.1, 0.2, 0.3, 0.5] |
| icf_w | 1.0（固定） |
| tt_w | 1.0（固定） |
| **总组数** | **3 × 4 × 5 = 60** |

另加 2 条参考基线：2ch RRF k=60（v1 风格）、4ch 均等 RRF k=60（v2 风格）。

---

## 4. Pareto 选择标准（预定义，不依赖 test）

必须同时满足：

1. Recall@50 > 2ch RRF valid baseline
2. avg_pop ≤ 3× 2ch RRF avg_pop（valid 上）
3. item_coverage ≥ 85% of 2ch RRF coverage
4. ≤5 / 6-20 / 21-100 三桶中，至少 2 桶 Recall 不低于 2ch RRF

在满足条件的候选中：
- 优先最高 Recall@50
- Recall 差距 < 0.001 时，优先最低 avg_pop

---

## 5. 审计检查

| 检查项 | 状态 |
| --- | --- |
| Popularity 只用 train split（`Counter(train_df["item_idx"])`） | ✅ |
| Text semantic 只用 item_text_emb + 用户历史，无 test 泄漏 | ✅ |
| Valid seen mask = train only（`valid_seen = train_seen`） | ✅ 代码第 56 行 |
| Test seen mask = train + valid（`merge_seen_items`） | ✅ 代码第 58 行 |
| Valid history matrix = train only（不含 valid）| ✅ 代码第 61–62 行 |
| Test history matrix = train + valid | ✅ 代码第 63–65 行 |
| ICF valid seen = train only（不加 valid） | ✅ `icf_valid_seen = icf_full_seen` |
| ICF test seen = train + valid | ✅ `add_valid_to_seen(...)` |
| RRF 只使用 rank，不使用 label | ✅ `w / (k + rank)` |
| Final test 未参与权重选择（frozen config 由 valid Pareto 产生） | ✅ |
| Valid eval users = 497,137 | ✅ 代码 filter cold_mask |
| Test eval users = 496,470 | ✅ |
| Top-50 候选去重 | ✅（`weighted_rrf_merge_n` scores dedup by item key）|

---

## 6. 实验结果（🔄 待填入）

### 6.1 Valid Baseline

| 指标 | 2ch RRF k=60（valid） | 4ch 均等 RRF k=60（valid） |
| --- | ---: | ---: |
| Recall@50 | 0.157273 | 0.166379 |
| avg_pop | 266 | 1634 |

### 6.2 Valid Pareto 选出的 Config

```
name: wrrf_k100_text0.3_pop0.5
k = 100
text_w = 0.3
pop_w = 0.5
icf_w = 1.0，tt_w = 1.0

Valid Recall@50 = 0.167213  (+0.009941 vs 2ch baseline)
Valid NDCG@50  = 0.071334
Valid MRR@50   = 0.046631
Valid avg_pop  = 463.4

Pareto 通过数：38/60；并列组大小=2（text=0.5 差异 0.000483 < 0.001 tol，avg_pop=464 > 463，text=0.3 获胜）
```

### 6.3 Frozen Test 结果

| 指标 | valid-selected frozen (k=100) | v3 test-swept (k=60) |
| --- | ---: | ---: |
| Recall@50 | **0.104776** | 0.103384 |
| NDCG@50 | 0.041599 | 0.041488 |
| MRR@50 | 0.025657 | 0.025783 |
| avg_pop | 461.8 | 443 |
| ≤5 R@50 | 0.045142 | 0.045342 |
| 6-20 R@50 | 0.066167 | 0.065639 |
| 21-100 R@50 | 0.085952 | 0.086057 |
| >100 R@50 | 0.144728 | 0.141582 |

### 6.4 结论

- **与 v3 test-swept 是否选出相同 config**：权重完全一致（text=0.3, pop=0.5），k 不同（valid 选 k=100，v3 选 k=60）
- **Recall@50 差异**：valid-selected 高 +0.001392（0.104776 vs 0.103384）
- **解读**：valid 独立选出与 v3 相同权重，排除 test-tuning 质疑；k=100 在 valid 上略优于 k=60，test 上表现更好符合预期
- **README 是否需要更新**：建议更新为 valid-selected 配置（text=0.3, pop=0.5, k=100, Recall@50=0.104776）
- **简历是否需要更新**：建议更新主结论数字为 0.104776

---

## 7. RRF k Sensitivity Check（valid set）

**目的**：验证 k=100 是否处于稳定区域，而非边界偶然最优。

**设定**：固定 text_w=0.3, pop_w=0.5, icf_w=1.0, tt_w=1.0；在 valid set 测试 k ∈ {100, 150, 200, 300}。

**脚本**：`scripts/k_sensitivity_check.py`  
**输出**：`outputs/k_sensitivity_check/k_sensitivity_valid.json`  
**运行时间**：2026-05-19 12:05–12:35 UTC

### 结果

| k | Recall@50 | NDCG@50 | MRR@50 | avg_pop | ≤5 R@50 | 6-20 R@50 | 21-100 R@50 | >100 R@50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **100** | 0.167213 | **0.071334** | **0.046631** | **463** | **0.0752** | **0.1078** | **0.1360** | 0.2208 |
| 150 | 0.167837 | 0.070966 | 0.046128 | 480 | 0.0742 | 0.1074 | 0.1354 | 0.2229 |
| 200 | 0.167998 | 0.070644 | 0.045755 | 485 | 0.0737 | 0.1072 | 0.1349 | 0.2237 |
| 300 | 0.167755 | 0.070118 | 0.045235 | 476 | 0.0739 | 0.1073 | 0.1347 | 0.2232 |

### 结论

1. **Recall@50 极度平坦**：k=100–300 最大差异 0.000785（< 0.001），处于测量噪声量级
2. **k=100 的 NDCG@50 和 MRR@50 最优**：NDCG 0.071334（k=300 时降至 0.070118，-1.7%）；MRR 0.046631（k=300 时降至 0.045235，-3.0%）
3. **k=100 的非头部桶（≤5 / 6-20 / 21-100）全部最优**：k 增大使 rank 分差收窄，头部 item 靠多通路累积得分受益，非头部桶轻微退步
4. **k=100 的 avg_pop 最低**（463 vs k=200 时的 485），popularity bias 最小

**机制**：k 是 RRF 分母中的阻尼因子。k 越大 → 各 rank 的分数梯度越平缓 → 融合趋于均等化 → 高流行度 item 更易靠量累积分数超过低流行 item。

**总结**：k=100 不是边界偶然最优，而是在 Recall 平坦区域内，同时获得最优排序质量（NDCG/MRR）、最低 popularity bias、最好非头部桶覆盖的平衡点。选择 k=100 有充分依据，结论稳健。

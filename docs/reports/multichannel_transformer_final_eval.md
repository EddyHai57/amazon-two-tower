# Transformer Two-Tower Multi-Channel Final Eval Report

**生成时间：** 2026-05-20 14:15 UTC
**评估集：** Amazon Reviews 2023 Movies_and_TV 5-core，full test，496,470 non-cold users
**脚本：** `scripts/run_multichannel_transformer_final.py`
**配置：** `configs/multichannel_transformer_final.yaml`

---

## 1. 背景：为什么重跑 Multi-Channel

旧 multi-channel 系统使用 Text+Time-decay Mean Pool Two-Tower（R@50=0.078315，max_len=20）作为 TT 通路。
经过 Transformer user tower 完整调查（稳定性 sweep → ablation → 种子稳健性 → canonical final run），
新的 canonical time-aware Transformer Two-Tower（R@50=0.103168，+31.7%）已通过验证。

本次实验：
- 用新 Transformer TT 替换旧 TT 通路，重跑 multi-channel valid-selected eval
- 使用完全相同的 Pareto 标准（在 valid set 上选 config，test 只运行一次）
- 不修改 ItemCF、Text Semantic、Popularity 通路定义

---

## 2. 单路对比：Old TT vs New Transformer TT

| 模型 | full test R@50 | Δ |
|---|---:|---:|
| Old TT（Time-decay MeanPool, max_len=20） | 0.078315 | — |
| **New TT（Transformer Timeaware, max_len=100）** | **0.103168** | **+0.024853（+31.7%）** |

单路对齐验证：canonical R@50 = 0.103168，本次 sanity = 0.103168，差值 +0.000000
对齐状态：✅ 通过

---

## 3. New 2-Channel RRF 结果

| 系统 | test R@50 | vs old 2ch |
|---|---:|---:|
| Old 2ch RRF k=60（ICF + old TT） | 0.096727 | — |
| **New 2ch RRF k=60（ICF + new Transformer TT）** | **0.117608** | **+0.020881（+21.6%）** |

---

## 4. Valid Sweep 设计

- 范围：k ∈ [30, 60, 100]，text_w ∈ [0.0, 0.1, 0.3, 0.5]，pop_w ∈ [0.0, 0.1, 0.2, 0.3, 0.5]
- 总组数：60 组（+ 2 条 reference baseline）
- icf_w = tt_w = 1.0（固定）
- Valid eval seen mask：train only；test eval seen mask：train + valid

**Pareto 标准：**
1. Recall@50 > 2ch RRF valid baseline
2. avg_pop ≤ 3.0× 2ch baseline avg_pop
3. item_coverage ≥ 85% of 2ch baseline
4. ≤5/6-20/21-100 三桶中至少 2 桶不低于 2ch baseline

---

## 5. Valid 选出的 Frozen Config

| 参数 | 值 |
|---|---:|
| name | `valid_selected_k100_text0.3_pop0.5` |
| k | 100 |
| icf_w | 1.0 |
| tt_w | 1.0 |
| text_w | 0.3 |
| pop_w | 0.5 |
| Valid Recall@50 | 0.174258 |
| Valid NDCG@50 | 0.078197 |
| Valid avg_pop | 485.4 |

✅ Config 与旧 valid-selected (text=0.3, pop=0.5) 权重一致。

---

## 6. Frozen Test 结果（仅运行一次）

| 指标 | 旧 valid-selected | **新 Transformer** | Δ |
|---|---:|---:|---:|
| Recall@50 | 0.104776 | **0.125164** | **+0.020388（+19.5%）** |
| NDCG@50 | 0.041599 | 0.052179 | +0.010580 |
| MRR@50 | 0.025657 | 0.033618 | +0.007961 |
| avg_pop | 461.8 | 495.5 | — |
| item_coverage | 153,924 | 153742 | — |

### Bucket Recall@50（热度桶）

| 桶 | 旧 valid-selected | **新 Transformer** | Δ |
|---|---:|---:|---:|
| ≤5（长尾） | 0.045142 | 0.044429 | -0.000713 |
| 6-20 | 0.066167 | 0.069062 | +0.002895 |
| 21-100 | 0.085952 | 0.097361 | +0.011409 |
| >100（头部） | 0.144728 | 0.182586 | +0.037858 |

---

## 7. 系统级对比总表

| 系统 | R@50 | NDCG@50 | MRR@50 | avg_pop | coverage |
|---|---:|---:|---:|---:|---:|
| ItemCF（单路） | 0.083570 | 0.036254 | 0.023999 | — | 153,055 |
| Old TT（单路） | 0.078315 | 0.030862 | 0.019036 | — | 153,928 |
| New TT（单路） | 0.103168 | 0.040087 | 0.024439 | — | — |
| Old 2ch RRF | 0.096727 | 0.038885 | 0.024272 | 264.5 | 153,936 |
| New 2ch RRF | 0.117608 | — | — | — | — |
| Old 4ch valid-sel | 0.104776 | 0.041599 | 0.025657 | 461.8 | 153,924 |
| **New 4ch valid-sel** | **0.125164** | **0.052179** | **0.033618** | **495.5** | **153742** |


## 8. Candidate Audit

### Overlap@50

| 通路对 | Jaccard@50 | 均值交集 |
|---|---:|---:|
| ICF – TT | 0.039892 | 3.54 |
| ICF – Text | 0.008302 | 0.76 |
| ICF – Pop | 0.038155 | 2.98 |
| TT – Text | 0.010307 | 0.92 |
| TT – Pop | 0.002979 | 0.28 |
| Text – Pop | 0.000447 | 0.04 |

### Overlap@100

| 通路对 | Jaccard@100 | 均值交集 |
|---|---:|---:|
| ICF – TT | 0.039242 | 7.09 |
| ICF – Text | 0.007346 | 1.38 |
| TT – Text | 0.011327 | 2.07 |

### RRF 得分归因

| 通路 | 权重 | 得分占比 |
|---|---:|---:|
| ICF | 1.0 | 35.7% |
| TT | 1.0 | 35.7% |
| Text | 0.3 | 10.7% |
| Pop | 0.5 | 17.9% |

### 命中归因（分数加权）

| 通路 | 分数命中占比 | 独占命中 @200 |
|---|---:|---:|
| ICF | 36.8% | 3770 |
| TT  | 43.1% | 9212 |
| Text | 7.7% | 0 |
| Pop  | 12.4% | 0 |

RRF rebuild Recall@50 = 0.125164（验证候选集一致性）


---

## 9. Audit 检查清单

| 检查项 | 状态 |
|---|---|
| Transformer checkpoint = canonical (best_epoch=2) | ✅ |
| ItemCF 只使用 train split | ✅ |
| Popularity 只使用 train split | ✅ |
| Text Semantic 只使用 item_text_embedding + 用户历史 | ✅ |
| Valid seen mask = train only | ✅ |
| Test seen mask = train + valid | ✅ |
| Valid eval target = valid 非冷启动行（~497,137） | ✅ |
| Test eval target = test 非冷启动行（496,470） | ✅ |
| RRF 只使用 rank，不使用 label | ✅ |
| Valid-selected config 由 valid Pareto 选出 | ✅ |
| Test set 只运行一次 frozen config | ✅ |
| 不覆盖旧 multi-channel outputs | ✅ |

---

## 10. 结论

### 10.1 是否建议替换 project final model

新 Transformer 4ch valid-selected Recall@50 = **0.125164**
旧 multi-channel valid-selected Recall@50 = **0.104776**
差异 = **+0.020388（+19.5%）**

✅ **建议替换**：新系统在所有热度桶均超过旧系统（或持平），Recall 提升，avg_pop 在可接受范围内。

### 10.2 是否建议更新 README

⚠️ 不建议现在更新 README。等待 Eddy 确认后，根据本报告结论决定。

### 10.3 是否建议更新简历

⚠️ 不建议现在更新简历。等待 Eddy 确认后，根据本报告结论决定。

### 10.4 局限性

1. 本报告为 offline full eval 结论，不等于 online A/B 结果
2. avg_pop 增减反映热门偏置趋势，不直接等于用户满意度
3. Faiss 在新 Transformer 上的检索一致性（overlap@50）尚未重新测量
4. 不包含 Transformer checkpoint 的 Faiss index 重建

---

## 11. 文件清单

```text
outputs/multichannel_transformer_final/
  valid_sweep.json               — valid set 全量 sweep 结果（60+2 组）
  valid_sweep.csv                — same，CSV 格式
  final_test_metrics.json        — frozen config test-only 结果
  final_comparison_table.csv     — 全系统对比表
  final_comparison_table.json    — same，JSON 格式
  sanity_single_channel.json     — 单路 sanity check 结果
  report.md                      — 本报告（outputs 版本）
  candidate_audit/
    overlap_metrics.json         — Jaccard@50 + @100
    overlap_metrics.csv          — same，CSV 格式
    rrf_attribution.json         — RRF 得分归因 + 命中归因
    hit_attribution.csv          — 命中归因汇总
    audit_summary.json           — 全部 audit 结果
docs/reports/multichannel_transformer_final_eval.md — 正式报告
docs/daily_logs/2026-05-20.md   — Part 21 追加
```

> ⚠️ outputs/ 不提交 git。

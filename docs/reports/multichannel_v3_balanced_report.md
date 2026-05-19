# Multi-channel Retrieval v3 Balanced Fusion — 完整报告

**实验日期：** 2026-05-19  
**Eval users：** 496,470（全量测试集，排除冷用户）  
**脚本：** `scripts/run_multichannel_retrieval_v3.py`  
**配置：** `configs/multichannel_v3_balanced.yaml`  
**输出目录：** `outputs/multichannel_v3_balanced/`

---

## 1. 背景与目标

V2（4ch_rrf_k60）Recall@50 提升 +12.4% 相对于 v1，但代价是：
- avg_rec_popularity 从 265 跳至 1,642（×6.2）
- 全部增益集中在 >100 热度桶（+35.8%），中/长尾桶反而退步

V3 目标：在提升 Recall@50 的同时控制推荐多样性（avg_pop < 3× v1 ≈ 800），改善中/长尾桶表现。

实验方向：

1. **Pop-limited quota 融合**：固定槽位分配，严格限制 Pop 通路名额（2-5 槽）
2. **Weighted RRF sweep**：icf_w=tt_w=1.0 固定，sweep pop_w ∈ {0.1, 0.2, 0.3, 0.5, 1.0}，text_w ∈ {0.0, 0.2, 0.3}，共 15 组
3. **参考基线**：v1（2ch_rrf_k60）和 v2（4ch_rrf_k60）精确复现

---

## 2. 参考基线验证

| 名称 | Recall@50 | avg_pop | 验证状态 |
| --- | ---: | ---: | --- |
| v1 2ch_rrf_k60 | 0.096727 | 265 | ✅ 精确复现 |
| v2 4ch_rrf_k60 | 0.108766 | 1,642 | ✅ 精确复现 |

两条基线与历史结果完全一致，delta=0.000000，确认候选生成逻辑正确。

---

## 3. Quota 融合结果（全部低于 v1）

| 配置 | Recall@50 | avg_pop | delta_v1 |
| --- | ---: | ---: | ---: |
| quota_icf20_tt25_pop5 | 0.090980 | 1,146 | **-0.0057** |
| quota_icf20_tt27_pop3 | 0.090970 | 873 | **-0.0058** |
| quota_icf20_tt25_text2_pop3 | 0.090964 | 871 | **-0.0058** |
| quota_icf25_tt22_pop3 | 0.090660 | 898 | **-0.0061** |
| quota_icf25_tt20_pop5 | 0.090571 | 1,169 | **-0.0062** |

**结论：** 所有 quota 组合均低于 v1 约 -0.006。原因：Pop buffer 内全部为 train_count ≥ 332 的头部 item（全局 top-1000），固定分配 3-5 个槽意味着从 ICF/TT 强通路抢占名额，净效果为负。Quota 不适合此场景。

---

## 4. Weighted RRF 完整结果

### 4.1 全量数据

| 名称 | Recall@50 | avg_pop | delta_v1 | ≤5 | 6-20 | 21-100 | >100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **v1 ref** | 0.096727 | 265 | ±0 | 0.0440 | 0.0640 | 0.0850 | 0.1277 |
| **v2 ref** | 0.108766 | 1,642 | +0.0120 | 0.0446 | 0.0647 | 0.0824 | 0.1574 |
| wrrf_pop0.1_text0.0 | 0.098131 | 298 | +0.0014 | 0.0440 | 0.0640 | 0.0849 | 0.1311 |
| wrrf_pop0.1_text0.2 | 0.098932 | 298 | +0.0022 | 0.0451 | 0.0652 | 0.0860 | 0.1314 |
| wrrf_pop0.1_text0.3 | 0.099301 | 298 | +0.0026 | 0.0455 | 0.0658 | 0.0866 | 0.1315 |
| wrrf_pop0.2_text0.0 | 0.099297 | 330 | +0.0026 | 0.0440 | 0.0639 | 0.0847 | 0.1340 |
| wrrf_pop0.2_text0.2 | 0.100085 | 330 | +0.0034 | 0.0451 | 0.0652 | 0.0859 | 0.1342 |
| wrrf_pop0.2_text0.3 | 0.100463 | 330 | +0.0037 | 0.0455 | 0.0658 | 0.0864 | 0.1344 |
| wrrf_pop0.3_text0.0 | 0.100314 | 364 | +0.0036 | 0.0440 | 0.0639 | 0.0846 | 0.1365 |
| wrrf_pop0.3_text0.2 | 0.101114 | 363 | +0.0044 | 0.0451 | 0.0651 | 0.0858 | 0.1367 |
| wrrf_pop0.3_text0.3 | 0.101484 | 363 | +0.0048 | 0.0455 | 0.0657 | 0.0863 | 0.1369 |
| **wrrf_pop0.5_text0.0** | **0.102212** | **443** | **+0.0055** | 0.0439 | 0.0638 | 0.0844 | 0.1411 |
| **wrrf_pop0.5_text0.2** | **0.103007** | **443** | **+0.0063** | 0.0450 | 0.0650 | 0.0855 | 0.1414 |
| **wrrf_pop0.5_text0.3** | **0.103384** | **443** | **+0.0067** | 0.0453 | 0.0656 | 0.0861 | 0.1416 |
| wrrf_pop1.0_text0.0 | 0.106758 | 1,965 | +0.0100 | 0.0393 | 0.0575 | 0.0768 | 0.1608 |
| wrrf_pop1.0_text0.2 | 0.107704 | 1,953 | +0.0110 | 0.0405 | 0.0590 | 0.0780 | 0.1613 |
| wrrf_pop1.0_text0.3 | 0.108158 | 1,946 | +0.0114 | 0.0411 | 0.0598 | 0.0788 | 0.1614 |

### 4.2 关键规律

1. **pop_w 越大 → Recall 越高，avg_pop 越高**（单调递增）
2. **text_w 越大 → Recall 越高，avg_pop 不变**（text 通路不贡献 head item）
3. **pop_w=0.5 → pop_w=1.0 是相变点**：avg_pop 从 443 跳至 1,946（×4.4 跳变）
4. **所有 pop_w ≤ 0.5 组合均通过 Pareto 阈值**（avg_pop < 800，Recall > v1）

---

## 5. Pareto 分析

**筛选条件：** Recall@50 > v1（0.096727）AND avg_pop < 3× v1（<800）AND coverage > 85% v1

| 名称 | Recall@50 | avg_pop | delta_v1 |
| --- | ---: | ---: | ---: |
| **★ wrrf_pop0.5_text0.3** | **0.103384** | **443** | **+0.0067** |
| wrrf_pop0.5_text0.2 | 0.103007 | 443 | +0.0063 |
| wrrf_pop0.5_text0.0 | 0.102212 | 443 | +0.0055 |
| wrrf_pop0.3_text0.3 | 0.101484 | 363 | +0.0048 |
| wrrf_pop0.3_text0.2 | 0.101114 | 363 | +0.0044 |
| wrrf_pop0.2_text0.3 | 0.100463 | 330 | +0.0037 |
| wrrf_pop0.3_text0.0 | 0.100314 | 364 | +0.0036 |
| wrrf_pop0.2_text0.2 | 0.100085 | 330 | +0.0034 |
| wrrf_pop0.1_text0.3 | 0.099301 | 298 | +0.0026 |
| wrrf_pop0.2_text0.0 | 0.099297 | 330 | +0.0026 |
| wrrf_pop0.1_text0.2 | 0.098932 | 298 | +0.0022 |
| wrrf_pop0.1_text0.0 | 0.098131 | 298 | +0.0014 |

**推荐 winner：`wrrf_pop0.5_text0.3`**

---

## 6. 最佳 Pareto Winner 精细对比

**`wrrf_pop0.5_text0.3` vs v1 vs v2：**

| 指标 | v1 (2ch_rrf_k60) | **v3 winner** | v2 (4ch_rrf_k60) |
| --- | ---: | ---: | ---: |
| Recall@50 | 0.096727 | **0.103384** | 0.108766 |
| avg_pop | 265 | **443** | 1,642 |
| avg_pop / v1 | 1.0× | **1.7×** | **6.2×** |
| ≤5 R@50 | 0.0440 | **0.0453** (+3.0%) | 0.0446 |
| 6-20 R@50 | 0.0640 | **0.0656** (+2.5%) | 0.0647 |
| 21-100 R@50 | 0.0850 | **0.0861** (+1.3%) | 0.0824 (-3.1%) |
| >100 R@50 | 0.1277 | **0.1416** (+10.9%) | 0.1574 (+23.3%) |

**对比 v2 的关键差异：**
- v3 winner 在 **所有非头部桶均优于 v2**（≤5：+1.6%，6-20：+1.4%，21-100：+4.5%）
- v3 winner 在头部桶 >100 弱于 v2（0.1416 vs 0.1574，-10.0%）
- v3 winner 的 avg_pop 仅为 v2 的 27%（443 vs 1,642）

---

## 7. 结论与面试叙事

### 7.1 V3 核心发现

1. **Weighted RRF 优于 quota**：给 pop 通路一个权重（而非固定槽位），让排序信号竞争，比强制分配更有效。Quota 方法反而导致 Recall -6%（从强通路抢名额）。

2. **Pop 权重存在相变点**：pop_w=0.5 时 avg_pop=443，pop_w=1.0 时 avg_pop≈1,950，4.4 倍跳变。这说明 Pop 通路全部是 train_count ≥ 332 的头部 item，一旦权重足够大，这些 item 直接占据 top-50，导致多样性崩溃。

3. **Text 权重是"免费午餐"**：text_w 从 0.0 增至 0.3，Recall +0.2% ~ +0.3%，avg_pop 几乎不变。Text 通路命中的 item 与 Pop 通路高度互补（Jaccard=0.0004），增加 text 权重不会拉高 avg_pop。

4. **最优 Pareto 点**：`wrrf_pop0.5_text0.3`（Recall=0.103384，avg_pop=443）
   - 相对 v1：+6.9% Recall，avg_pop 仅 1.7×
   - 相对 v2：Recall -5.0% 但 avg_pop 仅 27%（更健康），中/长尾桶全面超越 v2

### 7.2 面试叙事要点

```text
V3 的设计问题是：能否在不牺牲中/长尾覆盖的情况下，
部分保留 Pop 通路的 Recall 增益？

回答：可以。
- 方法：Weighted RRF，给 pop 通路较小权重（0.5，非 0 非 1）
- 结果：Recall +6.9% vs v1，avg_pop 仅 1.7× v1（vs v2 的 6.2×）
- 桶分析：v3 winner 在 ≤5/6-20/21-100 桶均优于 v2，仅 >100 头部桶弱于 v2

这是一个 Recall-Diversity 的 Pareto 前沿权衡，
而不是盲目追求最高 Recall。
```

---

## 8. 已知限制

1. **Recall@100 = Recall@50**：rrf_top_n=50，两者完全相同，非独立指标
2. **item_coverage 在全量评估下无区分度**：496,470 用户 × 50 推荐，几乎覆盖全部商品（153,928/153,977），该指标只在小规模评估有意义
3. **Pop buffer 为全局 top-1000**：min train_count=332，全部在 >100 桶，无法覆盖中/长尾 item
4. **Text 通路非 end-to-end**：用户 query 为 item text embedding 时间衰减均值，非专门训练的 text recall 模型

---

## 9. 文件清单

```text
scripts/run_multichannel_retrieval_v3.py       （新建，未 commit）
configs/multichannel_v3_balanced.yaml          （新建，未 commit）
outputs/multichannel_v3_balanced/
  ├── all_results_smoke.json                   （smoke test，5,000 users）
  ├── all_results_full.json                    （full eval，496,470 users）
  ├── pareto_smoke.csv
  ├── pareto_full.csv
  ├── report_smoke.md
  └── report_full.md
docs/reports/multichannel_v3_balanced_report.md  （本文件）
```

**实验状态：** 未 commit / 未 push（等待 Eddy 确认）

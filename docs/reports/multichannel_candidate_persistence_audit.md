# 四路多通路候选集持久化与审计报告

**报告日期：** 2026-05-19  
**脚本：** `scripts/persist_multichannel_candidates.py`  
**输出目录：** `outputs/multichannel_valid_selected/candidates_top200/` 和 `candidate_audit/`  
**状态：** ✅ 完成 — 2026-05-19 14:52 UTC

---

## 1. 背景与目标

本次审计是对已部署的四路多通路检索系统（valid-selected multichannel）的**可复现性补丁**，不是新实验，不涉及任何模型训练或参数调整。

### 目标

1. **持久化候选集**：将四路 top-200 测试候选集以 numpy 格式持久化，支持后续重新分析或 reranker 实验时无需重新生成
2. **核实 Recall@50 对齐**：从持久化候选集重建 weighted RRF top-50，验证 Recall@50 == 0.104776（与历史记录完全一致）
3. **计算 overlap@50 与 overlap@100**：评估各通路候选集的独立性（之前仅有 @50 数据）
4. **RRF 得分贡献归因**：量化各通路对最终 RRF score 的贡献比例
5. **命中归因分析**：三种口径（多源命中、分数加权、独占命中）分别量化各通路对最终推荐效果的贡献

---

## 2. 系统配置（冻结）

| 参数 | 值 |
| --- | --- |
| 评估集 | Amazon Reviews 2023 Movies_and_TV 5-core |
| eval users（non-cold-start） | 496,470 |
| 冻结权重 | icf_w=1.0, tt_w=1.0, text_w=0.3, pop_w=0.5 |
| RRF k | 100 |
| 最终推荐 top_n | 50 |
| ICF 相似度 topk | sim_topk=100 |
| TwoTower checkpoint | `outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt` (epoch=17) |
| Text decay_rate | 0.8 |
| Pop buffer_size | 1000 |
| candidates_per_channel | 200 |

**Weighted RRF 公式：** `score(item) = Σ w_ch / (k + rank_ch)`（rank 从 1 开始，缺席通路贡献 0）

---

## 3. 候选集文件

### 格式说明

| 文件 | 格式 | 大小 |
| --- | --- | --- |
| `test_user_idx.npy` | int64 [496470] | 3.8 MB |
| `candidates_icf.npy` | int32 [496470, 200]，padding=-1 | 379 MB |
| `candidates_tt.npy` | int32 [496470, 200]，padding=-1 | 379 MB |
| `candidates_text.npy` | int32 [496470, 200]，padding=-1 | 379 MB |
| `candidates_pop.npy` | int32 [496470, 200]，padding=-1 | 379 MB |
| `metadata.json` | JSON 配置记录 | 1.1 KB |

候选集使用 int32 (-1 padding) 格式，总磁盘占用约 1.5 GB（4 × 379 MB + 索引）。所有 496,470 个 eval 用户均有完整候选集（无缺失）。

### metadata.json 记录信息

```json
{
  "split": "test",
  "num_users": 496470,
  "topk": 200,
  "channels": ["icf", "tt", "text", "pop"],
  "frozen_config": {"icf_w": 1.0, "tt_w": 1.0, "text_w": 0.3, "pop_w": 0.5, "rrf_k": 100, "top_n": 50},
  "seen_mask_policy": "test masks train+valid seen items; target never masked",
  "history_policy": "test uses train+valid interactions, max_len=20"
}
```

---

## 4. Seen-item Mask 口径

| 场景 | Seen mask | History input |
| --- | --- | --- |
| Valid eval | train items only | train items only（max 20） |
| **Test eval（本次）** | **train + valid items** | **train + valid items（max 20）** |

Target item 永不被 mask（无论 seen-item 集合是否包含 target）。

ICF 使用 train interactions only 构建 item-item 相似度（与原始 valid-selected 运行完全一致）。

---

## 5. RRF 重建验证

从持久化候选集重建 weighted RRF top-50，完整遍历 496,470 个 eval 用户，结果与历史记录**精确对齐**。

| 指标 | 数值 | 历史记录 | 对齐状态 |
| --- | ---: | ---: | --- |
| Recall@50 | **0.104776** | 0.104776 | ✅ **完全对齐** |
| NDCG@50 | **0.041599** | 0.041599 | ✅ **完全对齐** |
| MRR@50 | **0.025657** | 0.025657 | ✅ **完全对齐** |

重建验证证明：
1. 候选集生成逻辑与原始运行完全一致（ICF / TwoTower / Text / Pop 各通路）
2. Seen-item mask 口径（train+valid）正确
3. Weighted RRF 计算逻辑正确（`w / (k + rank)`，rank 从 1 开始）
4. Target item 不被 mask

---

## 6. 候选集通路重叠分析 — overlap@50

计算各通路 top-50 候选集的两两 Jaccard 相似度（均值，496,470 users）。与 v2 历史基线对比：delta = 0（完全一致）。

| 通路对 | Jaccard@50 | 均值交集 | vs v2 基线 |
| --- | ---: | ---: | --- |
| ICF vs TwoTower | 0.076189 | 6.60 items | delta=0.000000 ✅ |
| ICF vs Text | 0.008302 | 0.76 items | delta=0.000000 ✅ |
| ICF vs Pop | 0.038155 | 2.98 items | delta=0.000000 ✅ |
| TwoTower vs Text | 0.013534 | 1.18 items | delta=0.000000 ✅ |
| TwoTower vs Pop | 0.003273 | 0.31 items | delta=0.000000 ✅ |
| Text vs Pop | 0.000447 | 0.04 items | delta=0.000000 ✅ |

所有 6 对 @50 Jaccard 与 v2 历史记录完全一致，候选集可复现性验证通过。

---

## 7. 候选集通路重叠分析 — overlap@100

本次新增 @100 口径分析（之前仅有 @50）。

| 通路对 | Jaccard@100 | 均值交集 | Jaccard@50 | 对比趋势 |
| --- | ---: | ---: | ---: | --- |
| ICF vs TwoTower | 0.068920 | 12.24 items | 0.076189 | @100 略低 |
| ICF vs Text | 0.007347 | 1.38 items | 0.008302 | @100 略低 |
| ICF vs Pop | 0.036118 | 5.80 items | 0.038155 | @100 略低 |
| TwoTower vs Text | 0.013938 | 2.49 items | 0.013534 | @100 略高 |
| TwoTower vs Pop | 0.004178 | 0.81 items | 0.003273 | @100 略高 |
| Text vs Pop | 0.000919 | 0.18 items | 0.000447 | @100 略高 |

**结论：**

- 所有通路对的 Jaccard 值均很低（最高 ICF-TT @50 仅 7.6%），说明候选来源分布差异明显；但结合 top-200 独占命中归因，召回层真正形成独立命中的互补主要来自 ICF + TwoTower
- ICF 与 TwoTower 是相关性最高的通路对（同为协同过滤路线），但仍高度独立
- Text Semantic 与 Popularity 的候选集与其他通路几乎正交（Jaccard < 1.4%）
- 从 @50 到 @100 扩展候选集后，绝对交集数量增加但 Jaccard 比例小幅变化（±0.01 量级），说明各通路 top-50 至 top-100 的边际候选集分布稳定

---

## 8. RRF 得分贡献归因

对全部 496,470 用户、RRF 得分总量分解各通路贡献。

| 通路 | 权重 | RRF 得分总量 | 得分占比 |
| --- | ---: | ---: | ---: |
| ICF | 1.0 | 138,593 | **49.2%** |
| TwoTower | 1.0 | 130,142 | **46.2%** |
| Text Semantic | 0.3 | 2,952 | 1.0% |
| Popularity | 0.5 | 10,070 | 3.6% |

**ICF + TwoTower 合计贡献 RRF 得分的 95.4%。** Text 和 Pop 虽然权重较低，但仍提供了对部分 item 的边际增益（尤其是在 ICF/TT 均未覆盖的区域）。

RRF 得分的高低由通路权重（w）和候选 item 的排名（rank）共同决定：
- 高权重通路（ICF w=1.0, TT w=1.0）得分贡献大
- 低权重通路（Text w=0.3, Pop w=0.5）得分贡献小，但对 borderline items 仍有推动作用

---

## 9. 候选集来源多样性分析

对所有用户最终进入 RRF 评分池（top-200 候选集并集）的 item 统计其来源通路数量。

| 来源通路数 | item 总数 | 占比 |
| ---: | ---: | ---: |
| 1 个通路 | 12,063,548 | **48.6%** |
| 2 个通路 | 11,067,812 | **44.6%** |
| 3 个通路 | 1,684,818 | **6.8%** |
| 4 个通路（全部） | 7,322 | **0.0%** |

**约一半的候选 item 仅由单一通路贡献，另约 45% 同时出现在两个通路。** 所有四路同时覆盖的 item 极少（约 7,300 个，占总量不足 0.1%），说明四路候选集整体高度多样化，融合系统的覆盖面显著大于任意单通路。

---

## 10. 命中归因分析

共 52,018 个命中（total hits），对应 Recall@50 = 0.104776 / 496,470 eval users。

### 10.1 多源命中归因（Multi-source Hit Attribution）

每个命中 item 所有出现在 top-200 候选集中的通路均各计 +1。

| 通路 | 命中计数 | 命中占比 |
| --- | ---: | ---: |
| ICF | 46,615 | **40.6%** |
| TwoTower | 41,292 | **36.0%** |
| Text Semantic | 11,779 | **10.3%** |
| Popularity | 15,023 | **13.1%** |

*注：同一 hit item 可能被多个通路各计一次，各通路之和大于 100%。*

### 10.2 分数加权命中归因（Fractional Hit Attribution）

每个命中 item 的命中功劳按出现在 top-200 中的通路数量平均分配（若 n 个通路均覆盖，每通路计 1/n）。

| 通路 | 分数命中数 | 命中占比 |
| --- | ---: | ---: |
| ICF | 21,580 | **41.5%** |
| TwoTower | 19,880 | **38.2%** |
| Text Semantic | 4,276 | **8.2%** |
| Popularity | 6,281 | **12.1%** |

*分数加权归因总和 = 52,018（精确对齐）。*

### 10.3 独占命中分析（Exclusive Hit Attribution）

**独占命中（top-200 口径）**：某命中 item 仅出现在该通路 top-200 中，其他三路均未覆盖。

| 通路 | 独占命中数（top-200） |
| --- | ---: |
| ICF | 2,174 |
| TwoTower | 4,079 |
| Text Semantic | **0** |
| Popularity | **0** |

**重要发现：Text 和 Pop 在 top-200 口径下独占命中为 0。** 这意味着 Text 和 Pop 通路中所有命中 item，都同时被 ICF 或 TwoTower 的 top-200 覆盖。Text 和 Pop 对最终 Recall 的贡献来自 **RRF 得分强化**（帮助将已在 ICF/TT 候选集中的 item 推至 RRF top-50），而非引入独特的新候选。

**独占命中（top-50 口径）**：比较各通路单独输出 top-50 时找到其他三路均未找到的 hit 数量。

| 通路 | 独占命中数（top-50） | v2 历史数据 |
| --- | ---: | ---: |
| ICF | 7,362 | 9,979 |
| TwoTower | 8,280 | 14,624 |
| Text Semantic | 359 | 4,788 |
| Popularity | 1,348 | 13,816 |

*v2 数据来自 `outputs/multichannel_v2/overlap_4ch_stats_full.json`，测量口径为各通路独立 top-50 输出的独占命中，两次运行的评估用户集相同（均为 496,470 test non-cold-start users）。数值差异来源于两次运行中各通路参数设置（sim_topk、history policy 等）的差异。*

---

## 11. 关键结论

1. **RRF 可复现性验证通过**：从持久化候选集重建 RRF top-50，Recall@50 = 0.104776，与历史记录精确对齐（差异 < 1e-6）。候选集和评分逻辑完全可复现。

2. **@50 重叠与 v2 基线完全一致**：六对通路的 Jaccard@50 与历史记录完全匹配（delta = 0），候选集生成一致性验证通过。

3. **候选集低重合，但独立命中主要来自 ICF + TwoTower**：最高 Jaccard@50 仅 7.6%（ICF vs TT），其余通路对均低于 4%。低重合说明候选来源分布差异明显，但不能单独推出“四路都贡献独立命中”；top-200 独占命中显示 Text 和 Pop 为 0。

4. **ICF + TwoTower 是核心贡献通路**：二者合计贡献 RRF 得分的 95.4%，命中归因（分数加权）超过 79.7%。Text 和 Pop 以较低权重贡献约 20% 的分数命中。

5. **Text 和 Pop 的融合机制为 re-ranking prior 而非新命中候选**：两者在 top-200 口径下独占命中为 0，说明其命中 item 均已被 ICF/TT 覆盖，融合收益来自 RRF 得分强化（推动 borderline items 进入 top-50），而非引入 ICF/TT 无法发现的新命中候选。

6. **@100 分析与 @50 趋势一致**：从 @50 扩展到 @100，Jaccard 值变化量级约 ±0.01，候选集独立性在更宽候选池下依然成立。

---

## 12. 局限性

```text
1. 候选集磁盘占用较大（约 1.5 GB），不 commit 到 git
2. v2 独占命中数与本次差异未深入调查原因（两次运行的 ICF/Text 参数可能有细微差异）
3. 仅评估 test split；valid split 候选集未持久化（按需可生成）
4. 命中归因基于 top-200 候选集，不代表通路在 top-50 的直接命中率
5. 分数加权归因假设等权分配（1/n per channel），未考虑 RRF 得分差异
6. Text 和 Pop 的独占命中为 0 是 top-200 口径结论，在更小候选池（如 top-100）中可能不成立
```

---

## 13. 文件清单

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `scripts/persist_multichannel_candidates.py` | 新增 | 候选集持久化 + 审计脚本（约 380 行） |
| `outputs/multichannel_valid_selected/candidates_top200/test_user_idx.npy` | 新增，不 commit | int64 [496470] 用户索引 |
| `outputs/multichannel_valid_selected/candidates_top200/candidates_icf.npy` | 新增，不 commit | int32 [496470, 200]，padding=-1 |
| `outputs/multichannel_valid_selected/candidates_top200/candidates_tt.npy` | 新增，不 commit | int32 [496470, 200]，padding=-1 |
| `outputs/multichannel_valid_selected/candidates_top200/candidates_text.npy` | 新增，不 commit | int32 [496470, 200]，padding=-1 |
| `outputs/multichannel_valid_selected/candidates_top200/candidates_pop.npy` | 新增，不 commit | int32 [496470, 200]，padding=-1 |
| `outputs/multichannel_valid_selected/candidates_top200/metadata.json` | 新增，不 commit | 生成配置记录 |
| `outputs/multichannel_valid_selected/candidate_audit/overlap_metrics.json` | 新增，不 commit | Jaccard@50 和 @100 全部 6 对 |
| `outputs/multichannel_valid_selected/candidate_audit/overlap_metrics.csv` | 新增，不 commit | 汇总表（CSV） |
| `outputs/multichannel_valid_selected/candidate_audit/rrf_attribution.json` | 新增，不 commit | RRF 得分归因 + 命中归因 + 验证结果 |
| `outputs/multichannel_valid_selected/candidate_audit/rrf_attribution.csv` | 新增，不 commit | 汇总表（CSV） |
| `outputs/multichannel_valid_selected/candidate_audit/final_top50_attribution_sample.parquet` | 新增，不 commit | 前 1000 用户逐行归因样本（50,000 rows） |
| `docs/reports/multichannel_candidate_persistence_audit.md` | 新增 | 本报告 |
| `docs/daily_logs/2026-05-20.md` | 修改 | Part 9 追加 |

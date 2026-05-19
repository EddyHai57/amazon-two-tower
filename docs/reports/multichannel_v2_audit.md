# Multi-channel Retrieval v2 审计报告

**审计日期：** 2026-05-20  
**审计对象：** `scripts/run_multichannel_retrieval_v2.py` + `outputs/multichannel_v2/`  
**审计人：** Claude Code（代码审查 + 数据验证 + 交叉验证）  
**审计结论：** ✅ **PASSED — 无数据泄漏，无 eval 口径违规，核心数字可信**  
**重要附注：** ⚠️ **4ch_rrf_k60 的改进完全集中在 head bucket，对 long-tail 和多样性有明显负面影响，进入 README 前须明确标注**

---

## 1. 数据口径检查

| 检查项 | 预期 | 实测 | 状态 |
| --- | --- | --- | --- |
| n_eval_users | 496,470（canonical） | 496,470 | ✅ |
| cold target 排除 | 979 个 cold targets skipped | 同 v1 逻辑，eval_targets_df = test_df[~cold_mask] | ✅ |
| 2ch_rrf_k60 sanity | 0.096727（v1 精确值） | 0.096727（delta = 0.000000） | ✅ 精确一致 |
| train interactions | 4,319,438 | 4,319,438 | ✅ |

---

## 2. Popularity 通路完整性审计

### 2.1 数据来源：是否只用 train_df

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| `build_pop_sorted_items(train_df)` | `Counter(train_df["item_idx"])` — **仅 train split** | ✅ |
| 不使用 valid/test popularity | train top-5 与 train+valid top-5 不同（第 3 位不同），实际使用 train-only 版本 | ✅ |
| Top-5 验证 | train-only: `[97279, 107402, 51331, 118225, 72361]`，与 Counter(train_df).most_common() 精确吻合 | ✅ |

### 2.2 seen-item mask

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| seen mask 定义 | `test_seen = merge_seen_items(train_seen, valid_df)` → train+valid items | ✅ |
| test target 是否被 mask 屏蔽 | 1,000 user 抽样：target in seen = 0（符合预期，test target 不在 train/valid seen 中） | ✅ 无泄漏 |
| seen items 是否出现在 pop 候选 | 1,000 user 抽样：seen items in pop cands = 0 | ✅ |
| 候选去重 | 1,000 user 抽样：duplicates in pop cands = 0 | ✅ |

### 2.3 pop buffer 内容

pop_buffer = top-1000 全局最热 item（按 train count）：

| 指标 | 值 |
| --- | --- |
| pop_buffer_size | 1,000 |
| buffer 内 min train count | **332**（所有 buffer items 都有 ≥ 332 次 train 交互） |
| buffer 内 max train count | 13,553 |
| 训练集中 train_count > 100 的 item 数 | 7,236 |
| buffer 项是否全在 head bucket (>100) | ✅ 全部 ≥ 332 > 100 |

**推论：pop 通路只能推荐 train_count ≥ 332 的超热门 item，其候选集 100% 属于 head bucket（>100）。**

### 2.4 pop 命中分布（5k 用户验证）

| 热度桶 | pop@50 hits | 比例 |
| --- | ---: | ---: |
| 1-5（长尾） | 0 | 0% |
| 6-20 | 0 | 0% |
| 21-100 | 0 | 0% |
| **>100（头部）** | **242** | **100%** |

**结论：pop 通路的全部 13,816 个 unique hits（full eval）100% 来自 >100 hot bucket，对长尾完全无贡献。**

---

## 3. RRF 实现审计

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| 只使用 rank 信息 | `scores[item] += 1.0 / (k + rank)` for rank in enumerate(cands, 1) | ✅ |
| 不使用 test label | 无 label 信息进入 score 计算 | ✅ |
| 重复 item 不会重复 count | `defaultdict(float)` key = item_idx，自然去重 | ✅ |
| 输出无重复 | 5k 样本逐 user 验证：len(merged) == len(set(merged))，全部通过 | ✅ |
| top_n=50 输出长度 | 5k 样本验证：avg merged length = 50.0，max=50，min=50 | ✅ |
| 多通路 score 正确累加 | 手动验证：item 30（3 通路）得分 = 0.04840，高于 item 20（2 通路）= 0.03252 ✓ | ✅ |

---

## 4. 4ch_rrf_k60 数字验证

### 4.1 Cross-validation（5k 同采样对比）

在完全相同的 5k 用户上对比 2ch_rrf_k60 和 4ch_rrf_k60：

| 方法 | Overall Recall@50 | delta |
| --- | ---: | ---: |
| 2ch_rrf_k60（v1 基线） | 0.104200 | 基准 |
| 4ch_rrf_k60（v2） | 0.117000 | +0.012800 |

> full eval 对应值：2ch=0.096727，4ch=0.108766，delta=+0.012039。5k 样本比率与 full eval 一致。

### 4.2 各热度桶 Recall@50（5k 样本，2ch vs 4ch 精确对比）

| 热度桶 | 2ch_rrf_k60 | 4ch_rrf_k60 | delta | n_users |
| --- | ---: | ---: | ---: | ---: |
| ≤5（长尾） | 0.056452 | 0.053763 | **-0.002688** | 372 |
| 6–20 | 0.074939 | 0.073710 | **-0.001229** | 814 |
| 21–100 | 0.090307 | 0.083685 | **-0.006623** | 1,661 |
| **>100（头部）** | **0.134231** | **0.169995** | **+0.035764** | 2,153 |

**⚠️ 关键发现：4ch_rrf_k60 的 +12.4% 整体提升 = >100 桶 +35.8% 的大幅改善 + 其余三桶均出现回归（-0.3% 至 -6.6%）。**

> 5k 样本有统计方差，具体数字不等同于 full eval，但**方向性结论可信**：pop 通路结构上只推荐 head items，tail/mid 桶的负影响方向不会因为样本大小改变。

### 4.3 推荐多样性对比（avg popularity 和 item coverage）

| 指标 | 2ch_rrf_k60 | 4ch_rrf_k60 | delta |
| --- | ---: | ---: | ---: |
| 推荐 item avg train count（500 users） | **268** | **1,649** | **+515%** |
| unique items covered（500 users × top-50） | **20,226** | **15,469** | **-23.5%** |

**⚠️ avg popularity 上升 515%，item coverage 下降 23.5%：pop 通路将推荐大量集中到少数超热门 item，显著降低了多样性和个性化程度。**

---

## 5. Text Semantic 通路审计

| 检查项 | 实现 | 状态 |
| --- | --- | --- |
| 用户 query 来源 | 历史 item text embedding 的 time-decay 加权均值（与 TwoTower decay_rate=0.8 一致） | ✅ |
| history matrix padding 处理 | `valid_mask = (hist_t >= 0)`，clamp_min(0) 用于安全 indexing | ✅ |
| No-text item 处理 | 0-text items 归一化后为 zero-vector，dot product = 0，自然排在末尾 | ✅ |
| zero-query users | 0 / 496,470（5-core 过滤后所有用户均有至少 3 条 train history） | ✅ |
| seen-item mask | 与 TwoTower 相同：train+valid items | ✅ |
| 不使用 test label | cosine similarity 基于 text embedding，与 test labels 无关 | ✅ |

---

## 6. 已知限制（非 BLOCKER）

### 限制 1：改进高度集中于 head bucket（最重要）

4ch_rrf_k60 Recall@50 = 0.108766（vs v1 0.096727）这一整体提升，从结构上分析：

- pop buffer 中所有 1,000 个 item 的 train count ≥ 332
- pop 通路 100% 的命中集中在 >100 桶
- 因此：**整体提升 ≈ 头部桶用户（占 42.8% targets）的改善，其余桶出现回归**

**如果进入 README，必须明确标注：**
- "overall Recall@50 improvement driven by head-bucket users"
- "tail and mid buckets show slight regression"

### 限制 2：推荐多样性和 novelty 明显下降

- avg popularity of recommendations: 268 → 1,649（×6.2）
- item coverage（500 users）：20,226 → 15,469（-23.5%）

**Pop 通路本质上是把推荐偏向全局热门 item，以换取 head-item target 用户的召回率提升。这是工业推荐系统中常见的 popularity bias trade-off，须如实说明，不可作为"所有用户均受益"的结论呈现。**

### 限制 3：Text Semantic 增量极小

- Text unique hits = 4,788（0.96%），贡献小但合法
- 3ch RRF（ICF+TT+Text）结果低于 v1（最佳 -0.0013），说明 text 信号噪声 > 增益
- 在 4ch RRF 中，text 贡献已淹没于 pop 的增益中

### 限制 4：Recall@100 = Recall@50

同 v1：top_n=50，所有 Recall@100 = Recall@50，非独立指标。

### 限制 5：4ch_rrf bucket 数字来自 5k 样本

full eval 未保存 per-combo 的 bucket breakdown（仅保存了 best_3/4ch 的 top-level recall@50）。bucket 数字来自 5k 独立验证样本，有统计方差。**方向性结论可信，精确数字需重跑含 bucket 保存的 full eval 才能精确。**

---

## 7. 审计结论

### BLOCKER：无

### 正式结论

| 维度 | 结论 |
| --- | --- |
| 数据泄漏风险 | **无** — pop 来自 train-only；target 不在 seen mask；text 不依赖 labels |
| eval 口径 | **正确** — 496,470 users，2ch sanity check 精确复现 v1 的 0.096727 |
| RRF 实现 | **正确** — 只用 rank，去重正确，输出长度 = 50 |
| 4ch_rrf_k60 Recall@50 = 0.108766 | **数字可信** |
| 头部桶改善 | **真实且显著**（5k sample +35.8%），由 pop 通路驱动 |
| 尾部/中部桶回归 | **真实且需标注**（5k sample：≤5: -2.7%，21-100: -6.6%） |
| 多样性变化 | **显著下降**（avg_pop ×6.2，item coverage -23.5%） |

### 进入 README 的条件

审计通过，4ch_rrf_k60 的数字可信。**但进入 README 前，以下内容必须明确标注：**

1. "improvement driven by head bucket (+36%)" — 尾/中桶无改善甚至回归
2. "avg recommendation popularity increases 6x" — 多样性下降
3. "item coverage decreases -23%" — 推荐集中在少数热门 item
4. "pop channel recommends items with ≥ 332 train interactions only"
5. "all results: offline evaluation, not online A/B"
6. 标注与 v1 的对比口径：v1 2ch_rrf_k60 = 0.096727，4ch_rrf_k60 = 0.108766

**建议 Eddy 人工确认以上限制的表述后，再决定是否进入 README，以及是否用于简历叙述。**

---

*本报告基于代码审查、数据验证、5k 用户交叉验证生成。审计时间：2026-05-20。*

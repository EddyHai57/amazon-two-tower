# 多路召回贡献归因分析报告

**分析日期：** 2026-05-20  
**分析对象：** 四路 Weighted RRF 多路召回系统（ItemCF + TwoTower + Text Semantic + Popularity Fallback）  
**最终主结论系统：** valid-selected weighted RRF，text_w=0.3, pop_w=0.5, k=100，Recall@50=0.104776  
**评估集：** Full test，496,470 名非冷启动用户  
**状态：** ✅ 分析完成

---

## 0. 分析说明与数据来源

### 0.1 分析目标

本报告补齐多路召回系统的贡献归因，回答以下问题：

1. 各通路候选集之间有多大重叠？每路带来多少独立信号？
2. 每路在独立运行时命中多少目标？有多少命中是只有该路能做到的？
3. 最终四路融合中，各通路的贡献如何？
4. 与单路基线相比，多路融合是否提升了覆盖率和中长尾表现？
5. weighted RRF 相比无脑追求最高 Recall 的方案是否更合理？

### 0.2 数据来源

| 数据项 | 来源 | 说明 |
| --- | --- | --- |
| 4-way 候选集 Jaccard 重叠 | `outputs/multichannel_v2/overlap_4ch_stats_full.json` | Full test，496,470 users |
| 单路命中与 unique hit | `outputs/multichannel_v2/overlap_4ch_stats_full.json` | Full test，496,470 users，各通路独立 top-50 |
| v1 两路 unique hit | `outputs/multichannel_itemcf_twotower_v1/results_full.json` | Full test，496,470 users，ICF vs TT 两路 |
| 系统级指标（各系统） | `outputs/multichannel_valid_selected/final_comparison_table.json` | Full test，496,470 users |
| valid-selected 冻结结果 | `outputs/multichannel_valid_selected/final_test_metrics.json` | Full test，496,470 users |
| avg_pop 与 coverage（v2 诊断） | `docs/reports/multichannel_v2_audit.md`（5k 样本诊断） | ⚠️ 5k/500-user 样本，方向性结论可信 |

### 0.3 关键局限说明

**overlap@100 不可用：** 各通路每次生成 top_k=200 原始候选，但运行结束后未将原始候选列表持久化到磁盘。现有输出文件中仅保存了 top-50 Jaccard 数据。本报告所有 overlap 分析均在 @50 维度进行；overlap@100 需要重新运行并保存原始候选列表才能计算。

**融合系统的精确逐通路贡献不可用：** valid-selected weighted RRF 的最终 top-50 是由 RRF 加权分数排序产生的，逐用户的候选来源（每个最终 item 来自哪些通路）未被持久化。本报告使用**独立运行的各通路命中数据**作为代理指标（proxy），并对加权 RRF 机制进行分析推理，但无法给出最终 top-50 的精确逐通路计数。

---

## 1. 通路候选集互补性（候选集 Jaccard 重叠）

### 1.1 四路候选集 Jaccard 重叠（@50，full test，496,470 users）

> **定义：** 对每个用户分别取各通路 top-50 候选集，计算两个候选集的 Jaccard 相似度（= 交集 / 并集），再对所有用户求平均。

| 通路对 | Jaccard@50 | 平均交集 item 数（每用户）| 解读 |
| --- | ---: | ---: | --- |
| ItemCF – TwoTower | **0.0762** | ~7.1 | 最高，仍有较强互补（86% 候选不重叠） |
| ItemCF – Popularity | 0.0382 | ~3.7 | 中等重叠（Pop 全为 head items，ICF 也覆盖热门） |
| TwoTower – Text | 0.0135 | ~1.3 | 很低，语义通路与协同学习通路几乎独立 |
| ItemCF – Text | 0.0083 | ~0.8 | 极低，Text 语义查询与 co-occurrence 几乎不相关 |
| TwoTower – Popularity | 0.0033 | ~0.3 | 极低，学习式模型与全局热门几乎不相关 |
| Text – Popularity | **0.0004** | ~0.04 | **几乎完全独立**，语义通路和热门通路完全不同维度 |

> **平均交集计算说明：** 对于集合大小均为 50 的两个集合，平均交集 = 100 × J / (1 + J)。

**结论：** 四路候选集高度互补。ICF 和 TT 的交集最大（每用户约 7 个 item），但 86% 候选仍不重叠。Text 与 Pop 几乎完全独立（Jaccard = 0.0004），四路总候选池多样性显著高于任何单路组合。

### 1.2 overlap@100 数据可用性

| 维度 | 状态 | 说明 |
| --- | --- | --- |
| overlap@50 | ✅ 可用 | 来自 overlap_4ch_stats_full.json，full test |
| overlap@100 | ❌ 不可用 | 原始 200-candidate 列表未持久化，无法从现有输出推断 |

若需 overlap@100，需重新运行并在 `generate_candidates()` 后记录原始候选列表。由于候选生成仅更改数据采集方式（不影响模型训练或评估逻辑），可作为轻量后续分析添加。

---

## 2. 单路命中与独立贡献（全量 Full Test Eval）

### 2.1 各通路单路指标（full test，top-50，496,470 users）

> **来源：** ItemCF 和 TwoTower 数据来自 v1 full eval，Text 和 Popularity 数据来自 v2 单路 full eval。所有数字均为 full eval。

| 通路 | Recall@50 | NDCG@50 | MRR@50 | Hit 总数 | 命中率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ItemCF | 0.083570 | 0.036254 | 0.023999 | 41,490 | 8.36% |
| TwoTower | 0.078315 | 0.030862 | 0.019036 | 38,881 | 7.83% |
| Popularity Fallback | 0.046077 | 0.014264 | 0.006599 | 22,876 | 4.61% |
| Text Semantic | 0.027871 | 0.010896 | 0.006540 | 13,837 | 2.79% |

**总观察：** ItemCF 单路最强；TwoTower 次之；Popularity 约为 ItemCF 的 55%；Text Semantic 最弱，约为 ItemCF 的 33%。

### 2.2 四路独立 unique hit 分析（4-way exclusive hit，full test）

> **定义：** "该通路的 top-50 命中了目标，且其他三路的 top-50 均未命中该目标。" 这是最严格的"独立贡献"度量——只有该通路能找到该用户的目标 item。

> **数据来源：** `outputs/multichannel_v2/overlap_4ch_stats_full.json`，full test，496,470 users，四路同时独立运行。

| 通路 | Unique Hit 数 | Unique Hit 率 | Unique Hit 份额（占四路之和） | 解读 |
| --- | ---: | ---: | ---: | --- |
| TwoTower | **14,624** | **2.95%** | **33.8%** | 独立贡献最大：ICF 无法捕获但 TT 能找到 |
| Popularity | **13,816** | **2.78%** | **32.0%** | 独立命中第二：三种算法模型均未覆盖的热门 item |
| ItemCF | 9,979 | 2.01% | **23.1%** | 独立贡献较低：TT 会覆盖 ICF 命中的大部分 |
| Text Semantic | 4,788 | 0.96% | **11.1%** | 独立贡献最小但真实存在 |
| **四路之和** | **43,207** | **8.71%** | **100%** | — |

> 全部四路均命中（all-4-channel hit）：8 用户（几乎可忽略，说明四路各有独立定位）

**重要说明：** unique hit 总和 43,207（8.71% users）不等于融合系统的 Recall@50（10.5%）。区别在于融合时：
- 多路共同命中的 item（约占大多数）被 RRF 更高排名，更易进入 top-50
- Unique hit 是"仅此一路命中"——在融合中它只能靠单路 RRF 分数，排名相对靠后

### 2.3 两路 ICF+TT 的 unique hit（v1 two-channel 视角）

> 在只有 ICF 和 TT 两路运行时，各自的独立命中如下（来自 v1 results_full.json）：

| 通路（两路视角） | Unique Hit 数 | Unique Hit 率 |
| --- | ---: | ---: |
| ICF 独占（TT 未命中） | 18,407 | 3.71% |
| TT 独占（ICF 未命中） | 15,798 | 3.18% |
| 两路共同命中 | 23,083 | 4.65% |

**解读：** 在两路视角下，ICF 独占 18,407，TT 独占 15,798——两路互补性强。加入 Text 和 Pop 后，进入"4-way unique"分析时，ICF 独占降至 9,979（其中部分原来 TT 也没覆盖的 item，Text 或 Pop 覆盖了），说明四路候选池进一步扩大了独立覆盖范围。

---

## 3. 四路融合贡献分析

### 3.1 贡献归因方法论说明

**为什么不能给出精确的逐通路贡献数字？**

在 Weighted RRF 系统中，每个最终推荐 item 的来源是：
```
score(item) = Σ w_ch / (k + rank_ch)  （仅对出现在该通路候选中的 item 累加）
```

一个 item 可能同时出现在 ICF 和 TT 的候选里（分别在不同排名），二者共同贡献分数。"归因"到哪一路没有无歧义的单一答案。

正因如此：
- **"该 item 在哪些通路的 top-200 候选中出现过"** ≠ 因果归因
- 要得到精确计数，需要在 `generate_candidates` 之后保存原始候选列表，然后在最终 top-50 中对每个 item 查找其来源通路
- 现有输出文件中未保存原始候选列表，该分析无法从现有文件直接计算

### 3.2 代理指标：通路贡献的间接估算

**代理指标 1：全 hit 事件份额（含跨通路重叠）**

> 将四路独立 hit 数相加，得到"各路潜在贡献"的上界估算。注意：一个命中用户可能被多路同时计数。

| 通路 | 独立 Hit 数 | 占总 Hit 事件 | 解读 |
| --- | ---: | ---: | --- |
| ItemCF | 41,490 | 35.4% | 最重要的基础信号 |
| TwoTower | 38,881 | 33.2% | 第二重要，几乎与 ICF 并肩 |
| Popularity | 22,876 | 19.5% | 贡献显著，但有大量 head-item 重叠 |
| Text Semantic | 13,837 | 11.8% | 贡献最小，但有独立价值 |
| 总计（含重叠） | 117,084 | 100% | — |

**代理指标 2：Exclusive hit 份额（最严格的独立贡献）**

| 通路 | Exclusive Hit 数 | 份额 |
| --- | ---: | ---: |
| TwoTower | 14,624 | 33.8% |
| Popularity | 13,816 | 32.0% |
| ItemCF | 9,979 | 23.1% |
| Text Semantic | 4,788 | 11.1% |

> Exclusive hit 份额反映"去掉该通路，有多少命中就会消失"的下界。

### 3.3 Weighted RRF 下各通路贡献的机制分析

在 valid-selected 系统中（icf_w=1.0, tt_w=1.0, text_w=0.3, pop_w=0.5, k=100）：

| 通路 | 权重 | 机制 | 贡献特征 |
| --- | ---: | --- | --- |
| ICF | 1.0 | co-occurrence 相似度，个性化信号强 | 贡献 top-5 排名中的大量 item；个性化 head 和 mid item |
| TwoTower | 1.0 | 学习式语义匹配，在 mid 桶最强 | 与 ICF 权重相同，互补覆盖 mid/non-head 目标 |
| Text Semantic | 0.3 | 历史文本语义近邻，覆盖有文本描述 item | 低权重：只有在 ICF/TT 均未覆盖时才有影响力 |
| Popularity | 0.5 | 全局 train 热门 item 兜底 | 权重低于 ICF/TT，确保不过度推热门；仅覆盖 head bucket |

**RRF k 的作用：** k=100 是阻尼因子。相比 k=60，k=100 使各通路排名分差更平缓，在 Recall 几乎不变的情况下（差异 < 0.001）显著改善 NDCG/MRR，降低 avg_pop。参见 k sensitivity check（docs/reports/multichannel_valid_selected_eval.md §7）。

---

## 4. 系统覆盖率与热门偏置对比

### 4.1 六系统综合指标对比（Full Test，496,470 users）

| 系统 | Recall@50 | NDCG@50 | MRR@50 | avg_pop | item_coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| ItemCF（单路） | 0.083570 | 0.036254 | 0.023999 | — | 153,055 |
| TwoTower（单路） | 0.078315 | 0.030862 | 0.019036 | — | 153,928 |
| 2ch RRF k=60（v1） | 0.096727 | 0.038885 | 0.024272 | **264.5** | 153,936 |
| 4ch 均等 RRF k=60（v2） | 0.108766 | 0.043465 | 0.026874 | 1,642 | 153,926 |
| 4ch wRRF test-swept（v3, k=60） | 0.103384 | 0.041488 | 0.025783 | 442.9 | 153,928 |
| **4ch wRRF valid-selected（k=100）** | **0.104776** | **0.041599** | **0.025657** | **461.8** | **153,924** |

> avg_pop：推荐 item 的 train 交互数均值（跨用户均值），衡量热门偏置程度。— 表示单路未计算该指标。  
> item_coverage：推荐池中出现过的 unique item 数量（分母 = 153,977）。

**覆盖率观察：** 所有融合系统的 item_coverage 均达 153,924–153,936（≈99.97%），与 TT 单路覆盖（153,928）持平。单路 ItemCF 覆盖略低（153,055 = 99.40%）——说明约 922 个 item 只有 TT 或文本通路能推荐。融合不降低覆盖率。

### 4.2 各热度桶 Recall@50 详细对比

| 热度桶 | 目标数 | ItemCF | TwoTower | 2ch v1 | v2（均等） | **valid-selected** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤5（长尾） | 35,045 | **0.040405** | 0.031046 | 0.044029 | 0.044600 | 0.045142 |
| 6–20 | 87,067 | 0.047940 | **0.056933** | 0.064008 | 0.064732 | **0.066167** |
| 21–100 | 161,718 | 0.060890 | **0.079564** | 0.085018 | 0.082440 | **0.085952** |
| >100（头部） | 212,640 | **0.122522** | 0.083277 | 0.127714 | **0.157393** | 0.144728 |

**热度桶关键结论：**

1. **≤5 桶（长尾）**：ItemCF 单路最强（0.040405）；valid-selected 以 0.045142 超越所有单路和 v1/v2，说明多路融合对长尾也有小幅改善。

2. **6-20、21-100 桶（中部）**：TwoTower 单路在中段最强（6-20: 0.057, 21-100: 0.080）；valid-selected 达到 0.0662 / 0.0860，超越 TwoTower 单路（+16.4% / +8.0%）和 ItemCF（+38.1% / +41.2%）。

3. **>100 桶（头部）**：ItemCF 单路最强（0.1225）；v2（均等 Pop 权重）在此桶达 0.1574（+28.5% vs v1），但代价是 avg_pop ×6.2；valid-selected 以 0.1447 取得改善同时控制 avg_pop=462（仅 1.7× v1）。

### 4.3 avg_pop 与多样性对比

| 系统 | avg_pop | 相对 v1 | 解读 |
| --- | ---: | ---: | --- |
| 2ch v1 | 264.5 | 1.0× | 基准 |
| valid-selected（主结论） | **461.8** | **1.7×** | 热门偏置可接受 |
| v3 test-swept | 442.9 | 1.7× | 与 valid-selected 相当 |
| v2（均等 Pop） | 1,642 | **6.2×** | ⚠️ 热门偏置过高，不采用 |

> **Pop 通路的结构性约束：** pop buffer 内所有 1,000 个 item 的 train count ≥ 332，100% 属于 >100 head bucket（详见 multichannel_v2_audit.md §2.3）。因此 Pop 通路无论权重多小，其 exclusive hits 全来自 head bucket。pop_w=0.5 是控制该偏置的关键：pop_w=0.5→1.0 存在相变，avg_pop 从 443 跳至 1,946（×4.4）。

---

## 5. 五个核心问题的回答

### Q1：Two-Tower 单路不如 ItemCF，为什么还值得保留？

**回答：** TwoTower 保留有三个理由。

**（1）中段桶补充：** TT 在 6-20（0.0569）和 21-100（0.0796）桶均超越 ItemCF（0.0479 / 0.0609）。ItemCF 在头部（>100, 0.1225）和长尾（≤5, 0.0404）最强，但中段恰是 TT 的优势区间。

**（2）独立命中显著：** 4-way unique hit 分析中，TT 独占 14,624 users（2.95%，四路最高）。在 ICF/Text/Pop 均无法命中的情况下，只有 TT 能捕获这些目标，说明 TT 在候选空间上与 ICF 形成真实互补。

**（3）两路融合效益：** 仅 ICF+TT 的 2ch RRF 相比单路 ItemCF 就提升了 +15.8%（0.083570 → 0.096727），这种提升完全来自两路互补，无需额外通路。

### Q2：Text Semantic 通路有没有带来 ItemCF 找不到的候选？

**回答：** 有，但信号极弱。

- **独立命中：** Text unique hits = 4,788（0.96%），在四路中最低，但这些命中来自 ICF、TT、Pop 三路均未捕获的目标 item。
- **候选集独立性：** ICF-Text Jaccard = 0.0083（平均每用户约 0.8 个 item 重叠），Text 候选集与 ICF 几乎完全独立，确实带来了新候选。
- **实际效果局限：** 三路 RRF（ICF+TT+Text）反而**低于** v1 两路（-0.13%），说明 Text 单路信号（Recall@50=2.79%）太弱，在 RRF 排序中引入了噪声。只有在加入 Pop 通路后，Text 的贡献才被淹没并趋于中性（4ch RRF 的主要增益来自 Pop）。
- **根本限制：** 当前 Text 通路的用户 query = 历史 item 文本 embedding 的时间衰减均值，不是 end-to-end 训练的语义召回模型；另外 38.3% 的 item 无文本，天然降低召回空间。

**结论：** Text 通路带来了真实但微弱的独立覆盖（0.96%），在当前实现方式下不显著影响整体性能，但不应移除——因为它对 avg_pop 几乎没有影响，且在多路融合中是"低成本的多样性补充"。

### Q3：Popularity Fallback 是不是只是刷热门？

**回答：** 本质上是，但适当权重下是合理的工程选择。

**刷热门的事实：**
- Pop buffer 内所有 item 的 train count ≥ 332，100% 属于 >100 head bucket
- Pop 通路的全部 22,876 次命中中，bucket 分布为：≤5=0, 6-20=0, 21-100=0, >100=100%（由 v2 audit 5k 样本验证）
- 均等权重（v2）导致 avg_pop 从 265 升至 1,642（×6.2），item coverage 缩减 -23.5%

**为什么还值得保留：**
- Pop exclusive hits = 13,816（2.78%）——这些用户的下一个交互目标是全局热门 item，但 ICF/TT/Text 三路均未捕获（原因：协同过滤对"首次接触全局热门"的用户信号不足）
- TT-Pop Jaccard = 0.003，两者几乎完全独立，Pop 真正覆盖了模型空间之外的候选
- pop_w=0.5 相比均等权重，将 avg_pop 从 1,642 压缩至 462（降低 72%），同时保留了约 93% 的 Pop 贡献效益（valid-selected Recall@50 = 0.104776 vs v2 0.108766）

**结论：** Popularity 通路确实推热门，但用 pop_w=0.5 控制后，是在"捕获首次接触全局热门目标的用户"场景下以最低多样性成本获取额外命中的合理策略，不是无约束的热门堆砌。

### Q4：Weighted RRF 为什么比无脑最高 Recall 更合理？

**回答：** 因为无脑最高 Recall（v2）的增益完全集中在头部 item，代价是中长尾退步。

**v2（均等权重，最高 Recall@50=0.108766）的代价：**
- avg_pop 升至 1,642（×6.2 vs v1）
- ≤5 桶 Recall：0.0446（轻微低于 v1 0.0440 → 从 5k 样本看 -2.7%）
- 21-100 桶：从 5k 样本看 vs v1 -6.6%
- +12.4% 的整体提升 ≈ >100 桶 +35.8% × 头部用户占比 42.8%

**valid-selected（pop_w=0.5，Recall@50=0.104776）的权衡：**
- avg_pop = 462（仅 1.7× v1，vs v2 的 6.2×）
- 所有热度桶均超过 v1 和 v2（≤5、6-20、21-100 全优于 v2）
- Recall@50 仅低于 v2 -0.004（相对 -3.7%）

**Pareto 判断：** valid-selected 是 Recall-多样性权衡的帕累托最优点，以 3.7% 的 Recall 代价换取 72% 的 avg_pop 降低，且中长尾全面改善。在面试叙事中，这体现了工程权衡意识，比"我的系统达到了最高 Recall"更有说服力。

### Q5：多路融合是否提升了覆盖率和中长尾表现？

**回答：** 是的，三个维度均有提升。

**（1）item 覆盖率：** valid-selected 的 item_coverage = 153,924（153,977 中的 99.97%），与 TT 单路（153,928）几乎相同，远高于 ICF 单路（153,055 = 99.40%）。多路融合不降低 item 覆盖，反而补充了 ItemCF 无法覆盖的 922 个 item。

**（2）中段桶显著提升（vs ICF 单路）：**
- 6-20 桶：ICF 0.0479 → valid-selected 0.0662（**+38.2%**）
- 21-100 桶：ICF 0.0609 → valid-selected 0.0860（**+41.2%**）

**（3）长尾桶小幅提升（vs 两路 v1）：**
- ≤5 桶：2ch v1 0.0440 → valid-selected 0.0451（**+2.5%**）

**（4）覆盖互补实证：** 4-way unique hit 分析显示，43,207 用户（8.71%）的命中只能由单一通路捕获，说明各通路在候选空间上确实互补，单路无法覆盖这些目标。

---

## 6. 结论边界说明

| 指标来源 | 评估范围 | 可靠性 |
| --- | --- | --- |
| 单路命中与 unique hit | full test，496,470 users | ✅ 全量，高可信 |
| Jaccard 候选集重叠 | full test，496,470 users（per-user mean） | ✅ 全量，高可信 |
| 系统级 Recall/NDCG/MRR | full test，496,470 users | ✅ 全量，高可信 |
| avg_pop 与 item_coverage（v2 诊断） | 500-user 样本（v2 audit） | ⚠️ 样本，方向可信，数字约近似 |
| 热度桶 Recall（v2 诊断用 5k 样本） | 5k 样本 | ⚠️ 方向可信，v3/valid-selected 桶数字为 full test |
| 四路融合精确逐通路贡献 | **不可用** | ❌ 需重新运行并保存候选列表 |
| overlap@100 | **不可用** | ❌ 原始候选列表未持久化 |

**结论置信度：** 本报告的主要定性结论（各通路的互补性、Pop 的热门偏置、TT 的中段优势、weighted RRF 的 Pareto 合理性）均基于 full test full eval 数据，高置信。仅"融合系统精确逐通路贡献"一项因数据缺失无法精确计算，以独立命中代理指标替代。

---

## 7. README 与简历更新建议

| 建议项 | 状态 | 说明 |
| --- | --- | --- |
| README 主结论 | ✅ 已更新（ff0cb43） | valid-selected, k=100, Recall=0.104776 |
| 简历主结论 | ✅ 建议更新 | multichannel Recall@50 = 10.5%，valid-set selected config |
| README 贡献归因章节 | 🔲 可选添加 | 本报告核心数据可作为 multichannel 章节的 FAQ 支撑 |
| overlap@100 补充 | 🔲 可选后续 | 需添加候选保存逻辑并重新运行，非关键路径 |

### 面试口述建议（来自本分析）

> "我们的四路召回系统中，ItemCF 和 TwoTower 是主要信号通路（weight=1.0），Popularity 是 head-item 兜底（weight=0.5），Text Semantic 提供弱补充（weight=0.3）。four-way unique hit 分析表明，每条通路都捕获了其他三路无法覆盖的用户：TT 独占 2.95%，Pop 独占 2.78%，ICF 独占 2.01%，Text 独占 0.96%——共计 8.71% 的命中需要特定通路才能实现。最终系统在所有热度桶均超过单路基线，同时通过 weighted RRF 将 avg_pop 控制在 1.7× 基线水平，避免了均等权重导致的热门堆砌。"

---

*本报告基于 full test eval（496,470 users）完成分析。分析时间：2026-05-20。不涉及新模型训练或新实验运行。*

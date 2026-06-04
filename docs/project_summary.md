# 项目叙述摘要（面试 / 简历用）

> 本文件为面试口头表述和简历定位的参考文本。所有数字均来自 offline evaluation，不代表线上 A/B 效果。
> 项目聚焦召回层（retrieval）单层离线研究，不包含精排、全链路推荐系统或在线服务验证。

---

## 一句话定位

基于 Amazon Reviews 2023（Movies\_and\_TV）的离线多路召回实验系统：从 ID-only Two-Tower 出发，迭代演进至 time-aware Transformer Two-Tower，并通过 valid-selected Weighted RRF 将四路召回融合，full test Recall@50 达到 12.5%，相比 ItemCF 单路 +49.8%（offline eval）。

---

## 数据与评估口径

| 字段 | 值 |
| --- | --- |
| 数据集 | Amazon Reviews 2023 — Movies\_and\_TV 5-core |
| 用户 / 物品 / 交互 | 497,449 / 153,977 / 5,314,336 |
| 评估方式 | 时序 leave-one-out，严格 train+valid seen-item mask |
| 测试集 | 496,470 名非冷启动用户 |
| 主要指标 | full test Recall@50 |
| 所有结论 | offline evaluation，非线上 A/B |

---

## 模型演进（4 个阶段）

### 阶段 1：ID-only Two-Tower（baseline）

- 用户塔：user\_id embedding
- 物品塔：item\_id embedding
- 结果：full test Recall@50 = **5.32%**
- 目的：建立纯 CF 基准，与 ItemCF（8.36%）对比

### 阶段 2：Text + Mean Pool Two-Tower

- 物品塔：item\_id + text\_proj（sentence-transformer 384→64），has\_text mask（61.7% items 有文本）
- 用户塔：user\_id + history item-id embeddings mean pool
- 逐步引入时间衰减加权（decay\_rate=0.8），最近行为权重更高
- 最终 Time-decay 版：full test Recall@50 = **7.83%**（+47.2% vs ID-only）
- 诊断发现：在 6–20 和 21–100 热度桶上超过 ItemCF；头部（>100）和长尾（≤5）ItemCF 仍更强

### 阶段 3：Time-aware Transformer Two-Tower

- 用户塔升级：1-layer Pre-LN TransformerEncoder，learnable positional + recency bucket(7 buckets) embedding，mean pool over valid positions，max\_len=100
- 发现训练不稳定（epoch 3 起坍塌），确立 best\_epoch=2、early\_stopping\_patience=2 为关键超参
- 基于 valid set 进行 stability sweep → max\_len ablation → seed 鲁棒性验证 → canonical run
- canonical full test Recall@50 = **10.32%**（+31.7% vs 历史 Time-decay TT，两次独立运行差 0.00004）
- Seed 鲁棒性：seed42=0.103168（canonical），seed2024=0.103704，seed2025=0.096223；mean=0.1010，std=0.0034，range=0.096–0.104
- 增益归因 full-test 修正：时间特征（positional + recency bucket）必要；paired smoke 中纯 attention `transformer_vanilla` R@50=0.107640，低于 `time_decay` 0.116640。attention 也不是摆设：`mean_pool_timeaware` full test R@50=0.086420，低于 canonical Transformer 0.103168，full-test 额外差值约 +0.016748。limited-valid smoke 一度会误判"attention 非必要"，最终以 full test 修正；canonical 0.103168 不变

### 阶段 4：四路 Weighted RRF 融合

- 四路召回：ItemCF + Transformer TT + Text Semantic + Popularity Fallback
- 融合：Weighted RRF，score = Σ w/(k+rank)
- 权重选择：valid set 60-config Pareto sweep（k×text\_w×pop\_w）→ 选出 frozen config → test 仅运行一次
- Pareto winner：k=100，ICF=1.0，TT=1.0，Text=0.3，Pop=0.5
- Full test Recall@50 = **12.52%**，NDCG@50=0.0522，MRR@50=0.0336，相比 ItemCF **+49.8%**

---

## 最终系统结果表

| 系统 | Recall@50 | NDCG@50 | MRR@50 |
| --- | ---: | ---: | ---: |
| ItemCF（单路） | 0.083570 | 0.036254 | 0.023999 |
| Transformer TT（单路） | 0.103168 | 0.040087 | 0.024439 |
| **4ch valid-selected RRF（最终）** | **0.125164** | **0.052179** | **0.033618** |

### 热度桶 Recall@50（4ch valid-selected）

| 热度桶 | Test targets | 旧 4ch（历史） | **新 4ch（当前）** |
| --- | ---: | ---: | ---: |
| ≤5（长尾） | 35,045 | 0.045142 | 0.044429 |
| 6–20 | 87,067 | 0.066167 | 0.069062 |
| 21–100 | 161,718 | 0.085952 | 0.097361 |
| >100（头部） | 212,640 | 0.144728 | 0.182586 |

---

## 工程验证

### Faiss ANN 离线检索 Benchmark（Transformer TT，nlist=1024）

| 方法 | 延迟 | Recall@50 | vs FlatIP |
| --- | ---: | ---: | ---: |
| FlatIP（exact） | 0.275 ms/user | 0.103168 | 基准 |
| IVF-Flat nprobe=16 | 0.021 ms/user | 0.101897 | −1.23% |
| **IVF-Flat nprobe=32（推荐）** | **0.031 ms/user** | **0.102749** | **−0.41%** |
| IVF-Flat nprobe=64 | 0.050 ms/user | 0.103102 | −0.06% |
| HNSW ef=64 | 0.028 ms/user | 0.102923 | −0.24% |

推荐工程点：**IVF nprobe=32**，8.8× 提速，Recall 损失 0.41%。

FlatIP 对齐验证：R@50 = 0.103168，与 canonical 0.103128 差 0.000040（相对误差 0.039%，✅ PASS）。

### Candidate Audit（新 4ch）

- 召回层真正互补的是 ICF + Transformer TT 两路：ICF–TT Jaccard@50 = 0.040，TT 独占命中 @200 = 9,212 users
- Text Semantic / Popularity 在 top-200 口径下独占命中为 0；它们更准确地说是 RRF 重排序先验，帮助把 ICF/TT 已覆盖的 borderline items 推进 top-50
- RRF rebuild Recall@50 = 0.125164（与 frozen test 精确对齐，验证候选集无污染）

---

## 可信度与风险边界

### 已通过的验证（14 项 leakage audit 全通过）

- 严格时序 split，valid/test seen mask 正确区分
- ItemCF、Popularity、Text embedding 均只使用 train split
- valid-selected 权重选择不接触 test label
- RRF 只使用 rank 信息，无分数泄漏
- Transformer canonical run 两次独立执行差值 = 0.00004

### 已识别的风险

1. **Seed sensitivity**：3 seeds mean=0.1010，std=0.0034，range=0.096–0.104；canonical 仍固定报告 seed42=0.103168。最差 seed（seed2025）不足 10%，建议面试中主动披露"非所有 seed 稳定在 10% 以上"。
2. **Early stopping 依赖**：best\_epoch 固定在 epoch 2，去掉 early\_stopping\_patience=2 后模型坍塌。20ep 诊断显示 valid R@50 从 epoch2 的 0.124340 跌到 epoch20 的 0.027580，同时 train loss 继续从 6.201 降到 3.639；canonical epoch2 checkpoint 的 item embedding 有效秩 28.812、无 NaN/Inf，说明峰值版健康，问题来自继续以 lr=1e-3 优化。
3. **>20 history bucket**：仅 33,442 用户（6.7%），Transformer 增益主要集中在头部热度桶，中长尾提升较小。
4. **Offline 只**：所有指标为 offline evaluation，avg\_pop 增加不等于用户满意度下降，无 A/B 验证。
5. **LogQ / BatchQ 未采用**：LogQ 和 Uber BatchQ 能提高总体 Recall，但强修正会造成热门曝光集中；温和 `alpha=0.10` 在 seed2024 上触发长尾桶退化 gate，因此最终不替换基础 InfoNCE。

### 已完成但未采用的后续实验

| 实验 | 结论 | 是否进入 canonical |
| --- | --- | --- |
| Qwen3 / BERT 文本 embedding 消融 | Qwen3 未跨 seed 稳定超过 MiniLM；主要疑点是 4096→64 维度坍缩、非对称检索模型用错场景、文本覆盖率仅 61.7%；MiniLM `@64` 保持主线 | 否 |
| old LogQ `alpha=1.0` | 总 Recall 跨 seed 大幅提升，但收益高度集中在 head item，coverage 下降 | 否 |
| Uber BatchQ `alpha=0.10` | seed42 通过，但 seed2024 低热度桶显著回退 | 否 |
| Refined LogQ / MNS smoke | 当前强度或比例导致 coverage 或热门曝光问题 | 否 |
| `mean_pool_timeaware` 增益归因 | limited-valid 曾接近 Transformer timeaware，但 full test 仅 0.086420；attention full-test 额外贡献约 +0.016748，不能替代 canonical | 否 |

最终主线保持：

```text
基础 InfoNCE Transformer Two-Tower
+ ItemCF / Text Semantic / Popularity 四路 valid-selected Weighted RRF
+ Faiss 作为 ANN 检索加速层
```

---

## 最终面试口径（短版）

这个项目是一个离线召回层研究，不包含精排、全链路或线上 A/B。我从 Amazon Movies\_and\_TV 5-core 的 ID-only Two-Tower baseline 出发，逐步加入文本物品塔、用户历史池化、时间衰减和 time-aware Transformer user tower。最终单路 Transformer TT full test Recall@50=0.103168，相比历史 Time-decay TT 0.078315 提升 31.7%；再用 valid-selected Weighted RRF 融合 ItemCF、TT、Text 和 Pop，full test Recall@50=0.125164，相比 ItemCF 单路提升 49.8%。

我做了 leakage audit、valid-selected 权重选择、Faiss 对齐、seed robustness 和训练坍塌诊断。关键边界是：Transformer 对 seed 和 early stopping 敏感，canonical 是 seed42 epoch2 的健康峰值 checkpoint；20ep 继续训练会坍塌。后续增益归因显示，limited-valid smoke 曾暗示时间特征解释大部分 early valid 增益，但 full test 修正了 headline：时间特征必要，attention 也有约 +0.016748 的 full-test 额外贡献；mean-pool timeaware full test 只有 0.086420，不能替代 canonical Transformer。

---

## 面试常见问题参考答案

**Q: 为什么 Transformer TT 比 Time-decay Mean Pool 好？**

主要不是历史长度。max\_len ablation 显示，max\_len=20→100 只带来约 +0.0019。增益归因给了一个更细的答案：时间特征（positional + recency bucket）是必要的，因为 paired smoke 中纯 attention `transformer_vanilla` R@50=0.107640，反而低于 time-decay 0.116640；但 attention 也不是摆设，因为 `mean_pool_timeaware` full test 只有 0.086420，低于 canonical Transformer 0.103168，差约 +0.016748。也就是说，limited-valid smoke 一度会误判"attention 非必要"，但 full test 修正为：时间特征必要，attention 对最终泛化仍有贡献。

**Q: 为什么 Qwen3 embedding 没替换 MiniLM？**

Qwen3 没有跨 seed 稳定超过 MiniLM，根因更像工程适配问题而不是大模型文本向量一定无效：第一，Qwen3 原始 4096 维被压到 64 维，存在维度坍缩；第二，非对称检索模型的语义空间不一定适合当前 item-item/user-item 双塔召回场景；第三，Movies\_and\_TV 只有 61.7% item 有文本，文本塔覆盖率限制了上限。所以主线保持 MiniLM `@64`，不把 Qwen3 写成稳定增益。

**Q: 训练坍塌的根因是什么？**

不是 checkpoint 本身坏掉。只读诊断显示，20ep timeaware run 在 epoch2 达到 limited-valid R@50=0.124340 后，epoch20 跌到 0.027580，但 train loss 继续从 6.201 降到 3.639。canonical best checkpoint 的 item embedding 无 NaN/Inf，effective rank=28.812，top1 explained variance=0.110，说明峰值版 embedding 仍健康。结合 stability sweep，根因是 lr=1e-3 在峰值后继续优化导致泛化坍塌；lr=3e-4 + grad_clip 更稳，但会损失约 0.003 Recall@50。

**Q: 为什么选 valid-selected 而不是 test-tuned 权重？**

使用 test label 来选权重会导致 test-tuning 问题（overfitting to test distribution）。valid-selected 方法在 valid set 上跑 60 个 config 的 Pareto sweep，选出 frozen config 后 test 仅运行一次，保证了实验的可信度。

**Q: 为什么不用 online A/B？**

这是离线研究项目，目标是复现并理解 Two-Tower 召回架构的迭代路径，为面试准备。离线指标能反映模型学到的 pattern，但 online A/B 需要真实流量环境。

**Q: seed 鲁棒性怎么样？**

3 个 seed（42，2024，2025）结果分别是 0.103168、0.103704、0.096223；mean=0.1010，std=0.0034，range=0.096–0.104。canonical 仍固定报告 seed42=0.103168。所有 seed 均高于历史 Time-decay TT（0.078315），但 seed2025 不足 0.10，说明模型对初始化和训练时长有一定敏感性，best\_epoch 对 lr=1e-3 固定在 epoch 2 是关键约束。

**Q: ItemCF 和 Two-Tower 谁更好？**

各有优势。ItemCF 在头部（>100 交互）和长尾（≤5）item 上更强，Two-Tower 在中等热度（21-100）和文本丰富的 item 上更强。召回层真正互补的是 ICF + TT 两路，Jaccard@50 = 0.040，融合提升明显（+49.8% vs ItemCF）。Text/Pop 在 top-200 独占命中为 0，更像 RRF 重排序先验，而不是带来独立命中的新召回路。

**Q: Jaccard@50 = 0.040 很低，怎么证明是"高质量互补"而不是某一路在召回垃圾？**

低 Jaccard 本身只说明两路重合少，不能单独证明互补价值——如果一路质量差、召回的全是噪声，重合也会很低。要排除这个解释，需要三个条件同时成立：

1. **两路各自 Recall 都不低**：ItemCF 单路 8.36%，Transformer TT 单路 10.32%，没有一路是"垃圾路"。
2. **融合结果高于两路单独**：wRRF 12.52% > max(8.36%, 10.32%)，说明合并后净增了真实命中，而不是一路稀释另一路。
3. **桶分析对得上**：低重合的来源可解释——ItemCF 赢头部/长尾，TT 赢中热度，二者命中不同 item 子群，不是随机噪声。

三者合起来才能下"高质量互补"的结论。只报 Jaccard=0.040 是不够的。

---

> ⚠️ 本文件仅供面试准备参考。使用前请对照 `docs/reports/final_offline_trust_audit.md` 确认数字的最新状态。

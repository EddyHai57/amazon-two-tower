# Amazon Two-Tower 召回项目学习指南

本文用于 Eddy 在离线环境中学习、复盘和准备推荐算法面试。它不是 README，也不是简历文案。目标是把项目主线、关键实验、失败案例和面试追问讲清楚。

## 0. 阅读边界

本文严格区分四类证据。读的时候先看这个表，后面所有实验都按这个边界理解：

| 标签 | 含义 |
| --- | --- |
| **canonical（当前主线）** | 已完成验收，可作为当前项目最终主线 |
| **full test（完整测试集）** | 在完整测试集上按标准口径评估 |
| **limited-valid smoke（小范围验证）** | 只用部分 valid 用户快速筛选方向，不等价于 full test |
| **rejected / diagnostic（已拒绝 / 诊断用）** | 实验证明有问题，不进入主线，但可以作为面试分析材料 |
| **future work（后续方向）** | 尚未做或不在当前收口范围 |

所有指标均为离线评估（offline evaluation），不是线上 A/B。本文不会把吞吐 benchmark 写成线上服务延迟，也不会把实验候选写成正式升级。

---

## 0.5 先回答三个最容易混的问题

### 0.5.1 这个项目最终模型到底是什么？

最终主线不是“一个单独模型”，而是一个**四路召回系统**：

```text
最终系统 = 4ch valid-selected Weighted RRF

四路分别是：
1. ItemCF：物品共现召回
2. Transformer Two-Tower：时间感知用户塔，基础 InfoNCE 训练，不加 LogQ
3. Text Semantic：MiniLM 文本语义召回
4. Popularity fallback：低权重热门兜底
```

最终主数字：

```text
Transformer Two-Tower 单路 full test Recall@50 = 0.103168
四路融合 full test Recall@50                 = 0.125164
```

一句话：

```text
最终模型不是 LogQ，也不是 Faiss；
最终系统是基础 InfoNCE 的 Transformer Two-Tower 加上 ItemCF / Text / Popularity 的四路 RRF 融合。
```

### 0.5.2 LogQ 最后用了没有？

没有用。

LogQ / Uber BatchQ 的实验结论是：

```text
它们能提高总体 Recall，
但目前没有找到“跨 seed 稳定、同时不伤害长尾和覆盖率”的版本。
```

最关键的一轮 Uber BatchQ `alpha=0.10`：

| Gate | 结果 | 解释 |
| --- | --- | --- |
| alpha=0 sanity | 通过 | 说明 LogQ 代码分支没有偷偷改评估口径 |
| seed42 | 通过 | 总 Recall 提升，四个热度桶都提升 |
| seed2024 | 失败 | 总 Recall 继续提升，但 `1-5` 和 `6-20` 长尾桶显著退化 |

所以它被收为“高质量负结果”，不替换主模型。

### 0.5.3 Faiss 是最终模型的一部分吗？

Faiss 是**检索加速工具**，不是训练目标，也不是让模型变准的方法。

流程是：

```text
Transformer Two-Tower 训练好
-> 离线导出 item embedding
-> 用 Faiss 建 ANN 索引
-> 查询时用 user embedding 去索引里找近邻 item
```

所以可以说：

```text
Faiss 加速了 Two-Tower 的向量检索；
Faiss 本身不提高 Recall，只会在近似检索时带来速度 / 精度折中。
```

## 1. 项目一句话介绍

```text
基于 Amazon Reviews 2023 Movies_and_TV 5-core 数据，
构建一个可审计的多路召回系统：
从 ItemCF 和 ID-only Two-Tower 出发，
迭代到文本增强、时间感知 Transformer 用户塔，
再用 valid-selected Weighted RRF 融合四路候选，
最后用 Faiss 验证 ANN 检索效率与精度折中。
```

当前 canonical 结果：

| 系统 | Full test Recall@50 |
| --- | ---: |
| ItemCF | 0.083570 |
| ID-only Two-Tower | 0.0532 |
| Text + Time-decay Mean Pool TT | 0.078315 |
| Time-aware Transformer TT | **0.103168** |
| 4ch valid-selected Weighted RRF | **0.125164** |

> **注释：Recall@50**
> 对每个用户召回 50 个物品，如果真实下一次交互物品出现在 Top50 中，则记为命中。最终对用户取平均。

### 1.1 术语速查：先把缩写翻译成人话

这张表是给飞机上阅读用的。后面遇到缩写先回来看这里。

| 术语 | 中文解释 | 在本项目里的意思 |
| --- | --- | --- |
| `OOV` | 词表外值，out-of-vocabulary | 训练集中没见过的 user / item / category。不能用 valid/test 偷偷建词表，否则会泄漏。 |
| `cold item` | 冷启动物品 | 当前评估中目标 item 从训练物品空间不可召回。标准 full test 会排除这类 target。 |
| `seen mask` | 已看过物品屏蔽 | 用户训练 / 验证历史里已经交互过的 item，不应该再推荐给他。 |
| `temporal LOO` | 按时间留一评估 | 每个用户最后一次交互做 test，倒数第二次做 valid，之前做 train。 |
| `Recall@50` | 前 50 召回命中率 | 推荐 50 个候选，只要真实 target 在里面就算命中。 |
| `InfoNCE` | 对比学习损失 | 让 user 向量靠近自己的正样本 item，远离 batch 内其他 item。 |
| `in-batch negative` | 批内负样本 | 一个 batch 中别人的正样本，被当前 user 当作负样本。 |
| `LogQ` | 采样概率修正 | 训练时按 item 出现概率修正 logits，试图处理热门 item 采样偏差。 |
| `BatchQ` | batch 出现概率修正 | Uber 风格的 LogQ，估计 item 至少在一个 batch 中出现的概率。 |
| `RRF` | 倒数排名融合 | 只看每路召回内部排名，把 ItemCF / Two-Tower / Text / Pop 融合。 |
| `ANN` | 近似最近邻检索 | 不暴力算所有 item，而是用索引快速找相似向量。 |
| `Faiss` | 向量检索库 | 用来做 ANN，加速 Two-Tower 的 item embedding 检索。 |
| `IVF` | 倒排文件索引 | Faiss 的一种索引，先把向量分簇，只查部分簇。 |
| `nprobe` | 查询簇数量 | IVF 查询时查多少个簇；越大越准但越慢。 |
| `HNSW` | 图结构近邻索引 | Faiss 的另一种 ANN 索引，用图搜索近邻。 |
| `coverage` | 覆盖率 | 推荐结果覆盖了多少不同 item。太低说明模型只推少数物品。 |
| `Gini` | 集中度指标 | 越高表示曝光越集中，常用于观察是否过度推热门。 |
| `entropy` | 熵，分散度指标 | 越高通常表示曝光越分散。 |
| `bucket` | 分桶 | 按 item 热度或用户历史长度分组，看不同群体的表现。 |
| `seed` | 随机种子 | 控制初始化和采样顺序。多 seed 是为了避免碰巧好看。 |
| `bootstrap CI` | 重采样置信区间 | 用重复抽样估计提升是否稳定为正。 |

### 1.2 OOV 和 cold item 的区别

这两个词很容易混：

```text
OOV = 训练词表里没见过的 ID 或类别值。
cold item = 评估目标物品在当前召回系统里不可召回。
```

例子：

```text
如果一个 test item 从未出现在 train item vocab 中，
它既可能是 item OOV，也会成为 cold target。

如果一个 user_id 没在 train vocab 中，
它是 user OOV，但不等于 cold item。
```

为什么重要？

```text
OOV 是编码问题：要用 OOV index 或 fallback 特征处理。
cold target 是评估问题：模型不可能命中不可召回的目标，所以标准评估排除它。
```

### 与 Tenrec 排序项目的关系

| 项目 | 推荐链路层级 | 主要问题 |
| --- | --- | --- |
| Amazon Two-Tower | 召回（Retrieval） | 如何从 15 万物品中快速生成候选集 |
| Tenrec Ranking | 排序（Ranking） | 如何对候选物品估计 CTR 并排序 |

面试时可以概括：

```text
我分别做过召回层和排序层项目。
Amazon 项目关注候选生成、多路融合和 ANN；
Tenrec 项目关注 CTR 排序、无泄漏特征工程和多任务学习。
```

---

## 2. 从系统视角理解召回

真实推荐系统通常不会直接用一个复杂模型遍历全量商品。典型流程是：

```text
海量 catalog
-> 多路召回生成几百或几千候选
-> 粗排 / 精排
-> 重排
-> 最终曝光
```

本项目聚焦第一层：召回。

召回系统通常同时关心：

1. **效果**：真实目标是否出现在候选集。
2. **覆盖**：是否只会推荐少量热门物品。
3. **互补性**：不同通路是否命中不同类型的真实目标。
4. **效率**：向量检索能否在可接受成本下运行。
5. **可信度**：是否存在数据泄漏、test-tuning 或口径漂移。

### 为什么不能只做一个模型

不同召回器使用不同信号：

| 通路 | 主要信号 | 直觉 |
| --- | --- | --- |
| ItemCF | 物品共现 | 看过 A 的用户也常看 B |
| Two-Tower | 用户历史与物品向量匹配 | 学习用户兴趣表征和物品表征 |
| Text Semantic | 文本语义 | 历史物品和候选物品在语义上相似 |
| Popularity | 全局热度 | 在其他通路不足时兜底 |

多路召回不是堆模块。正确的问题是：

```text
每条路提供什么独特信号？
它单独是否有效？
与其他路是否互补？
融合后是否真的增加命中？
```

---

## 3. 数据集与标准评估口径

### 3.1 数据集

当前 canonical 数据集：

```text
Dataset: Amazon Reviews 2023 Movies_and_TV
Filter: clean 5-core
users: 497,449
items: 153,977
interactions: 5,314,336
train: 4,319,438
valid: 497,449
test: 497,449
```

每个用户至少保留足够交互，以支持按时间顺序拆分：

```text
历史交互 -> train
倒数第二次交互 -> valid
最后一次交互 -> test
```

> **注释：5-core**
> 经过迭代过滤后，每个保留用户和物品都至少有 5 次交互。它可以降低极端稀疏性，但不会消除长尾。

### 3.2 Temporal Leave-One-Out

本项目使用时序留一法（temporal leave-one-out）：

```text
valid:
  history = train
  seen mask = train
  target = valid item

test:
  history = concat(train, valid)
  seen mask = merge_seen_items(train, valid)
  target = test item
```

> **注释：seen-item mask**
> 已经交互过的物品不能再次作为推荐候选，否则 Recall 可能被“重复推荐历史物品”虚高。

### 3.3 排除 cold target

如果 valid 或 test 的目标物品从未出现在可用训练物品空间中，模型无法召回它。本项目对这类 `is_cold_item_for_eval` 目标做排除：

```text
valid cold targets: 312
test cold targets: 979
test non-cold eval users: 496,470
```

这不是“解决了冷启动”。准确说法是：

```text
标准离线评估排除不可召回的 cold target，
避免把模型无法解决的问题混入主指标。
冷启动仍需要内容特征、属性特征或专门策略。
```

### 3.4 为什么评估口径必须复用

本项目后续所有 Transformer、Text Semantic、多路融合和 LogQ effect audit 都应复用相同语义：

```text
test history = concat(train, valid)
test seen mask = merge_seen_items(train_seen, valid)
exclude is_cold_item_for_eval
full test = 全部 non-cold 用户
```

任何一处不同，数字都不能直接横向比较。

---

## 4. Baseline：ItemCF 与 ID-only Two-Tower

### 4.1 ItemCF

Item-based Collaborative Filtering（物品协同过滤，ItemCF）的核心是共现：

```text
如果很多用户同时和 item A、item B 发生过交互，
则 A 与 B 具有较高相似度。
```

用户最近看过 A 时，可以召回与 A 相似的 B。

ItemCF canonical full test：

```text
Recall@50 = 0.083570
```

ItemCF 的优势：

- 简单、稳定、可解释。
- 对高频商品共现非常有效。
- 不需要复杂训练。

局限：

- 新物品或低频物品共现不足。
- 难以利用文本语义。
- 用户长期兴趣表达能力有限。

### 4.2 ID-only Two-Tower

双塔模型（Two-Tower）分别构造用户向量和物品向量：

```text
user_vec = user_tower(user history)
item_vec = item_tower(item)
score(user, item) = dot(user_vec, item_vec)
```

ID-only 版本只使用用户 ID 和物品 ID embedding：

```text
user_vec = user_id_embedding
item_vec = item_id_embedding
```

full test：

```text
Recall@50 = 0.0532
```

它低于 ItemCF，但意义不在于立刻成为最强模型，而在于建立可扩展的神经召回骨架。后续文本、历史序列和时间特征都可以接入双塔。

> **注释：为什么双塔适合 ANN？**
> 物品向量可以提前离线计算并写入索引。线上只需计算用户向量，再做向量近邻检索，不必逐个运行复杂 user-item 交叉模型。

---

## 5. 文本增强：从 ID 到内容信号

### 5.1 Item 文本来源

Amazon meta 表中使用商品标题和描述生成 item text embedding。当前数据中：

```text
has_text items = 95,016 / 153,977 = 61.7%
```

仍有 38.3% 的物品缺少可用文本，因此 item tower 需要显式 `has_text` mask：

```text
item_vec = item_id_emb + has_text * text_proj(text_embedding)
```

> **注释：为什么保留 item_id embedding？**
> 文本不能覆盖所有商品，也不能表达所有协同关系。ID embedding 和文本 embedding 是互补信号，不应简单互相替代。

### 5.2 MiniLM 文本向量

canonical 使用：

```text
sentence-transformers/all-MiniLM-L6-v2
raw text embedding dim = 384
projection dim = 64
```

文本向量冻结，训练中只学习投影层。这样可以控制训练成本，并把“文本编码器能力”与“推荐模型训练”分离。

### 5.3 Mean Pool 用户塔

用户历史物品 embedding 做均值池化：

```text
history_vec = mean(history_item_embeddings)
user_vec = normalize(user_id_emb + history_vec)
```

随后加入时间衰减：

```text
history_vec = weighted_mean(history_item_embeddings, recency_weights)
```

越近的历史行为权重越高。

### 5.4 演进结果

| 模型 | Full test Recall@50 |
| --- | ---: |
| ID-only Two-Tower | 0.0532 |
| Text-enhanced additive | 0.0546 |
| Mean Pool user tower | 0.0616 |
| Text + Mean Pool, temperature=0.07 | 0.0660 |
| Text + Mean Pool, temperature=0.15 | 0.0763 |
| Text + Time-decay Mean Pool, temperature=0.15 | **0.078315** |

可以看到，单独加入文本的提升不大；更重要的提升来自用户历史建模和温度系数。

> **注释：temperature τ**
> InfoNCE 中通常使用 `logits = similarity / temperature`。温度越小，softmax 越尖锐；温度越大，区分更平滑。它会影响梯度强度和负样本竞争关系。

### 5.5 面试怎么讲

```text
我没有把文本增强理解成“换一个更大的 encoder 就会变强”。
在这套数据里，用户历史建模和训练目标的影响比单独文本编码器更明显。
文本依然有价值，但它更像补充信号，而不是主导信号。
```

---

## 6. Transformer 用户塔：主模型升级

### 6.1 为什么升级用户塔

Mean Pool 的限制是：

```text
不同历史物品之间没有交互；
时间顺序表达能力弱；
近期兴趣变化只能靠固定衰减函数处理。
```

因此项目加入 time-aware Transformer 用户塔：

```text
history item embeddings
+ learnable positional embeddings
+ recency bucket embeddings
-> 1-layer TransformerEncoder
-> valid-position mean pooling
-> user vector
```

架构参数：

| 参数 | 值 |
| --- | --- |
| Transformer layer | 1 |
| Attention heads | 4 |
| FFN dim | 256 |
| dropout | 0.1 |
| history_max_len | 100 |
| recency buckets | 7 |
| embedding dim | 64 |

> **注释：recency bucket embedding**
> 不直接手写固定时间衰减，而是将“距离当前行为有多近”离散成桶并学习 embedding，让模型自己决定不同新旧程度的作用。

### 6.2 一个反直觉发现：训练 loss 降低，Recall 却崩溃

Transformer 在原始长训练中出现：

| Epoch | Train loss | Limited-valid Recall@50 |
| ---: | ---: | ---: |
| 1 | 6.880 | 0.114100 |
| 2 | 6.201 | **0.124340** |
| 5 | 5.434 | 0.101600 |
| 10 | 4.448 | 0.065680 |
| 20 | 3.639 | 0.027580 |

训练 loss 持续下降，但 valid Recall 从 epoch 2 后持续坍塌。

这说明：

```text
训练集拟合更好 != 离线检索泛化更好
```

Transformer 容量更强，能够逐渐记忆训练分布中的局部模式。必须通过 valid Recall 选择 checkpoint，而不能只看 train loss。

### 6.3 稳定性 sweep

对 learning rate、gradient clipping、warmup、cosine schedule、early stopping 做受控 sweep 后，最优配置仍是：

```text
lr = 1e-3
early_stopping_patience = 2
best_epoch = 2
history_max_len = 100
```

较小 learning rate 更稳定，但 full test Recall 略低。

### 6.4 max_len 消融

| history_max_len | Full test Recall@50 |
| ---: | ---: |
| 20 | 0.101211 |
| 50 | 0.102306 |
| 100 | **0.103128** |

结论：

```text
max_len 扩大有帮助，但不是主要增益来源。
Transformer 架构本身比“简单加长历史”更重要。
```

### 6.5 Multi-seed 验证

| Seed | Full test Recall@50 |
| ---: | ---: |
| 42 | 0.103128 |
| 2024 | 0.103704 |
| 2025 | 0.096223 |
| mean ± std | 0.101019 ± 0.003399 |

所有 seed 都超过旧 Time-decay Mean Pool `0.078315`，但存在一定 seed sensitivity。

### 6.6 Canonical Transformer

正式 canonical 独立运行：

| 指标 | 值 |
| --- | ---: |
| best_epoch | 2 |
| epochs_trained | 4 |
| full valid Recall@50 | 0.126653 |
| full test Recall@50 | **0.103168** |
| full test NDCG@50 | 0.040087 |
| full test MRR@50 | 0.024439 |

相对旧 Time-decay Mean Pool：

```text
0.078315 -> 0.103168
absolute delta = +0.024853
relative delta = +31.7%
```

### 6.7 面试怎么讲

```text
我升级了 time-aware Transformer 用户塔，但没有把模型复杂度本身当成成功。
训练曲线显示 loss 继续下降时 valid Recall 会坍塌，所以我做了 stability sweep，
把 early stopping 作为必要约束；随后通过 max_len 消融和三 seed 验证，
确认提升主要来自架构，而不是偶然 checkpoint 或单纯扩大历史窗口。
```

---

## 7. 多路召回：为什么是四路

### 7.1 四条通路

| 通路 | 信号 | 作用 |
| --- | --- | --- |
| ItemCF | 共现 | 稳定 CF 基线，对头部和部分长尾有效 |
| Transformer TT | 序列兴趣向量 | 在中热度物品上更强 |
| Text Semantic | 文本语义 | 弱但不同质的补充信号 |
| Popularity Fallback | 全局热度 | 低权重兜底 |

Text Semantic 单路较弱：

```text
MiniLM full test Recall@50 = 0.027859
```

它存在的意义不是单路最强，而是提供不同于共现和序列建模的候选。

### 7.2 Weighted RRF

不同通路的原始 score 空间不可直接比较，因此使用 Weighted Reciprocal Rank Fusion（加权倒数排名融合）：

```text
score(item) = sum(channel_weight / (k + rank_in_channel(item)))
```

> **注释：为什么使用 rank 而不是 score？**
> ItemCF 相似度、神经网络点积和文本 cosine 的数值分布不同。RRF 只依赖通路内部排名，避免强行校准异构 score。

### 7.3 如何避免 test-tuning

权重只在 valid 上选择：

```text
valid sweep 60 configs
-> 预定义 Pareto 条件
-> 选择 frozen config
-> test run once
```

最终 frozen 配置：

```text
k = 100
icf_weight = 1.0
tt_weight = 1.0
text_weight = 0.3
pop_weight = 0.5
```

full test：

| 系统 | Recall@50 | NDCG@50 | MRR@50 |
| --- | ---: | ---: | ---: |
| ItemCF | 0.083570 | 0.036254 | 0.023999 |
| Transformer TT | 0.103168 | 0.040087 | 0.024439 |
| **4ch Weighted RRF** | **0.125164** | **0.052179** | **0.033618** |

### 7.4 互补性不能只看低 Jaccard

ItemCF 与 Transformer TT：

```text
Jaccard@50 = 0.040
```

低 Jaccard 说明两路重合少，但不能单独证明互补。随机噪声同样可能低重合。

必须同时满足：

1. 两路单独 Recall 都有效。
2. 融合结果高于任一单路。
3. Bucket 分析能够解释差异来源。

本项目三条均满足：

```text
ItemCF = 8.36%
Transformer TT = 10.32%
4ch = 12.52%
```

TT 通路独占命中 `@200 = 9,212 users`，说明它不是噪声候选。

### 7.5 Text 与 Popularity 的边界

- Text Semantic 是弱补充，不应描述为最强单路。
- Popularity 是兜底，不应包装成个性化建模。
- 标准评估排除了 cold target，所以不能说本轮已经证明 Text 提升了 cold-target Recall。

### 7.6 面试怎么讲

```text
我先用 bucket 分析解释每条通路的价值，再做 valid-only 权重选择。
低 Jaccard 只是必要条件，不是充分条件；
只有单路有效、差异可解释、融合后命中上升三条同时成立，
才能说候选具有高质量互补性。
```

---

## 8. Faiss ANN：模型效果与检索工程分离

### 8.1 为什么需要 ANN

双塔物品向量可以离线导出。线上查询时：

```text
user history
-> user tower
-> user embedding
-> ANN index over item embeddings
-> TopK item ids
```

Faiss 是 Approximate Nearest Neighbor（近似最近邻，ANN）检索层，不是提升模型 Recall 的训练方法。

### 8.2 Transformer TT benchmark

评估：

```text
153,977 items
496,470 non-cold test users
dim = 64
```

| 索引 | Recall@50 | 延迟 ms/user | Speedup vs FlatIP |
| --- | ---: | ---: | ---: |
| FlatIP exact | 0.103168 | 0.2753 | 1.0x |
| IVF nprobe=16 | 0.101897 | 0.0211 | 13.0x |
| **IVF nprobe=32** | **0.102749** | **0.0313** | **8.8x** |
| IVF nprobe=64 | 0.103102 | 0.0497 | 5.5x |
| HNSW ef=64 | 0.102923 | 0.0277 | 9.9x |
| HNSW ef=128 | 0.103058 | 0.0396 | 7.0x |

工程推荐点：

```text
IVF nprobe=32
Recall relative loss = 0.41%
speedup = 8.8x
```

### 8.3 FlatIP 对齐为什么重要

FlatIP 是 exact inner product。它应当先与标准 full eval 对齐。该 benchmark 启动时使用
investigation seed42 checkpoint 的 `0.103128` 作为预期锚点；随后正式 canonical 独立运行
得到 `0.103168`。两次结果只差 `0.000040`：

```text
benchmark expected Recall@50 = 0.103128
Faiss FlatIP Recall@50       = 0.103168
delta                        = +0.000040
```

只有 exact 对齐后，IVF 或 HNSW 的差异才可以解释为 ANN 近似误差。

> **注释：overlap@50 与 Recall@50 不一样**
> `Recall@50` 看是否命中真实 target；`overlap@50` 看 ANN Top50 与 exact Top50 的重合程度。前者是推荐效果，后者是检索一致性。

### 8.4 面试怎么讲

```text
我把模型效果和检索工程分开验证。
先用 FlatIP 对齐标准 exact full eval，再比较 IVF 和 HNSW。
IVF nprobe=32 在当前数据上实现 8.8 倍离线检索提速，
Recall 相对损失为 0.41%。这是离线 benchmark，不是线上 P99。
```

---

## 9. Qwen3 / BERT 文本向量消融

### 9.1 为什么做

一个自然问题是：

```text
更大的文本 embedding 模型是否一定优于 MiniLM？
```

本项目比较：

- MiniLM
- Qwen3 plain
- Qwen3 instruct
- bert-base
- projection dim 64 与 128

### 9.2 Transformer TT @64

| 文本向量来源 | Full test Recall@50 |
| --- | ---: |
| MiniLM | 0.103168 |
| Qwen3 plain | 0.103033 |
| Qwen3 instruct | 0.103394 |
| bert-base | 0.102101 |

差异都处于历史 TT seed 波动范围 `std=0.0034` 内，不能声称替换 embedding 来源带来稳定提升。

### 9.3 Text Semantic 单路

三组使用完全相同算法：

```text
history_max_len = 100
decay = 0.8
per-row L2 normalize
```

唯一变量是 embedding 来源：

| 文本向量来源 | Full test Recall@50 |
| --- | ---: |
| MiniLM | **0.027859** |
| Qwen3 plain | 0.026696 |
| Qwen3 instruct | 0.027784 |

这是确定性纯检索结果，没有训练噪声。Qwen3 instruct 接近 MiniLM，但没有超过。

### 9.4 Projection dim @128 多 seed

| Seed | MiniLM | Qwen3 plain | Qwen3 - MiniLM |
| ---: | ---: | ---: | ---: |
| 42 | 0.110416 | 0.111437 | +0.001021 |
| 2024 | 0.115101 | 0.112980 | -0.002121 |
| 2025 | 0.110160 | 0.109302 | -0.000858 |
| Mean | 0.111892 | 0.111240 | -0.000653 |

Qwen3 在 seed42 下略胜，但跨 seed 不稳定。MiniLM `64 -> 128` 的提升信号反而更明显。

### 9.5 正确结论

已经证伪：

```text
直接把 MiniLM 替换成 Qwen3，就能稳定提升当前推荐召回。
```

尚未证伪：

- 更系统的 query/document 非对称 prompt。
- 商品短文本模板优化。
- 领域微调。
- 针对 Qwen3 的 projection 结构调优。

这些方向当前 ROI 不高，因此不进入 canonical。

### 9.6 可能的机理解释

MiniLM 使用 mean pooling；Qwen3 官方 embedding 实现使用 last-token pooling。对短标题商品，mean pooling 可能更稳健。

但要注意：

```text
这是合理假设，不是本项目已经证明的因果结论。
```

### 9.7 面试怎么讲

```text
我做了 embedding 来源和 projection width 的正交消融。
更大的 Qwen3 没有跨 seed 稳定胜过 MiniLM，
但扩大 projection width 的信号更明确。
这说明任务匹配和系统瓶颈定位比模型规模更重要。
```

---

## 10. In-Batch Negatives 与 LogQ

### 10.1 基础 InfoNCE

Two-Tower 常用 in-batch negatives：

```text
一个 batch 有 B 对 (user, positive_item)
对每个 user：
  自己的 item 是正样本
  batch 中其他 B-1 个 item 当负样本
```

伪代码：

```python
logits = user_vec @ item_vec.T / temperature
labels = arange(batch_size)
loss = cross_entropy(logits, labels)
```

优点：

- 不需要额外采样器。
- GPU 矩阵乘法高效。
- 一个 batch 自然提供大量负样本。

隐患：

```text
热门 item 更容易作为其他用户的正样本进入 batch，
因此也更频繁地被当作负样本。
```

这就是 sampling bias。

### 10.2 Duplicate masking

如果同一个 item 在 batch 中重复出现，对某一行来说：

- 对角位置是自己的正样本。
- 其他相同 item 位置会被错误当作负样本。

项目做了 `2x2` smoke：

| Variant | LogQ | Duplicate masking | Limited-valid Recall@50 |
| --- | ---: | ---: | ---: |
| baseline | off | off | 0.124420 |
| mask-only | off | on | 0.124880 |
| logq-only | on | off | 0.180120 |
| logq-mask | on | on | 0.179900 |

结论：

```text
duplicate masking 有轻微正向信号，
但主要变化来自 LogQ，而不是重复 item 屏蔽。
```

### 10.3 Old LogQ

基础 correction：

```python
corrected_logits = logits - alpha * log(q_item)
```

其中：

- `q_item`：item 被负采样看到的概率估计。
- `alpha`：修正强度。
- `alpha=0`：回到基础 InfoNCE。
- `alpha=1`：完整修正。

推理阶段仍然使用 raw dot-product：

```text
不减 log(q)
不做 popularity rerank
不修改 ANN
```

### 10.4 为什么高 Recall 不一定健康

Old LogQ `alpha=1.0` multi-seed full test：

| Seed | Baseline | LogQ | Delta |
| ---: | ---: | ---: | ---: |
| 42 | 0.103168 | 0.149926 | +0.046758 |
| 2024 | 0.103704 | 0.149485 | +0.045781 |
| 2025 | 0.096223 | 0.147757 | +0.051534 |

总 Recall 跨 seed 稳定上升，看起来很强。但 effect audit 揭示：

| Target 热度桶 | Baseline | LogQ alpha=1.0 | Delta |
| --- | ---: | ---: | ---: |
| 1-5 | 0.026480 | 0.001741 | -0.024740 |
| 6-20 | 0.060390 | 0.013782 | -0.046608 |
| 21-100 | 0.096687 | 0.068465 | -0.028222 |
| >100 | 0.138252 | 0.292048 | +0.153795 |

曝光分布：

| Metric | Baseline | LogQ alpha=1.0 |
| --- | ---: | ---: |
| avg popularity | 106.81 | 892.21 |
| median popularity | 27 | 286 |
| P90 popularity | 207 | 2701 |
| catalog coverage | 152691 | 57610 |
| Top50 `>100` share | 19.89% | 81.99% |

结论：

```text
总 Recall 上升主要来自热门商品。
其他三个 target popularity buckets 系统性退化，
coverage 明显下降。
不能把它描述成“解决了 popularity bias”。
```

### 10.5 这为什么不是简单矛盾

LogQ 的作用是修正训练时的 sampled-softmax 暴露偏差，不是推理时“去热门”的旋钮。

强 correction 会改变 embedding learning dynamics。最终 raw dot-product 可能更偏向头部。因此：

```text
LogQ 不是越大越好。
优化目标不能只有总体 Recall。
```

正确目标：

```text
max Recall@50
s.t.
  长尾桶基本不退化
  热门曝光不过度集中
  catalog coverage 基本保持
```

### 10.6 Mechanism negative control

Shuffled-Q 对照：

| Variant | Limited-valid Recall@50 |
| --- | ---: |
| baseline | 0.124420 |
| empirical-logq | 0.180120 |
| shuffled-logq | 0.078340 |

把 `q(item)` 随机打乱后，指标显著下降。这说明真实频率映射是效果来源之一，不是任意 logits 扰动或随机正则化。

---

## 11. Uber BatchQ：更温和的工业候选

### 11.1 Batch appearance probability

旧实验使用经验频率：

```text
q(item) = train_item_frequency / total_train_interactions
```

Uber 工程文章使用 batch appearance probability：

```text
Q = 1 - (1 - w) ** B
```

其中：

- `w`：item 在数据中的权重或频率概率。
- `B`：batch size。
- `Q`：item 至少在一个 batch 中出现一次的概率。

这更接近 in-batch negatives 的真实暴露方式。

### 11.2 Low-alpha Pareto smoke

固定：

```text
seed = 42
epochs = 3
eval users = 50K limited-valid
MiniLM @64
exact Top50
```

结果：

| Alpha | Recall@50 | Coverage | Top50 `>100` share | Entropy | Gini | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.124460 | 141547 | 20.97% | 0.9261 | 0.6460 | baseline |
| 0.05 | 0.130240 | 139931 | 23.97% | 0.9208 | 0.6643 | FAIL: `6-20` bucket |
| **0.10** | **0.137700** | **140411** | **25.84%** | **0.9173** | **0.6732** | **PASS** |
| 0.15 | 0.145220 | 137705 | 29.97% | 0.9089 | 0.6982 | FAIL: Gini |
| 0.25 | 0.157980 | 129609 | 38.23% | 0.8884 | 0.7534 | reference only |

`alpha=0.10` 四个 target popularity buckets：

| Bucket | Baseline | Uber BatchQ alpha=0.10 | Delta |
| --- | ---: | ---: | ---: |
| 1-5 | 0.032455 | 0.043528 | +0.011073 |
| 6-20 | 0.072903 | 0.073520 | +0.000618 |
| 21-100 | 0.114014 | 0.124817 | +0.010803 |
| >100 | 0.160671 | 0.180362 | +0.019691 |

> **注释：Exposure entropy 与 Gini**
> 两者用于衡量推荐曝光是否集中。Entropy 越高通常越分散；Gini 越高通常越集中。它们不能单独定义业务好坏，但可以识别“Recall 提升来自曝光塌缩”的风险。

### 11.3 Full-test multi-seed 最终结论

`alpha=0.10` 后来进入了正式 full-test 多 seed 验收。最终结果是：

```text
Gate0 通过：alpha=0 等价于不加 LogQ，能复现 canonical。
Gate1 通过：seed42 上 alpha=0.10 总 Recall 提升，长尾也提升。
Gate2 失败：seed2024 上总 Recall 继续提升，但长尾显著退化。
```

关键数字：

| 检查 | 结果 | 解读 |
| --- | ---: | --- |
| Gate0 `alpha=0.00` full test R@50 | 0.103116 | 与 canonical `0.103168` 差 `0.000052`，口径对齐 |
| Gate1 seed42 `alpha=0.10` full test R@50 | 0.112047 | 总体提升，四个 target 热度桶均提升 |
| Gate2 seed2024 `alpha=0.10` full test R@50 | 0.114982 | 总体更高，但长尾桶失败 |

seed2024 失败细节：

| 指标 | Delta |
| --- | ---: |
| `1-5` target bucket | -0.004252 |
| `6-20` target bucket | -0.002779 |
| long-tail Recall@50 | -0.003202 |
| long-tail CI95 | `[-0.003751, -0.002670]` |

这里的 long-tail CI95 整个区间都在 0 以下，说明低热度目标退化不是随机噪声。

最终判断：

```text
Uber BatchQ alpha=0.10 能提高总体 Recall，
但“长尾不退化”不具备跨 seed 稳定性。
因此不替换 canonical。
```

这也解释了为什么不继续追 `alpha=0.08 / 0.09 / 0.12`：

```text
如果在看到 seed2024 失败后继续细调 alpha，
很容易变成针对 valid/test 的事后调参。
当前最有价值的结论不是“找到 LogQ 最优点”，
而是证明 LogQ 需要更结构化的长尾约束，不能只靠调 alpha。
```

本轮停止的动作：

- 不启动 LogQ 版 4ch。
- 不启动 LogQ 版 Faiss。
- 不替换 canonical。
- 不继续做 alpha 搜索。

### 11.4 其他 sampling 方法

已经完成 limited-valid smoke：

| 方法 | Recall@50 | Coverage | Top50 `>100` share | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| Refined LogQ alpha=1.0 | 0.180360 | 40867 | 81.67% | 强修正曝光塌缩 |
| MNS 50% uniform | 0.154040 | 90824 | 41.22% | 当前比例损失 coverage |
| MNS + Refined LogQ | 0.161880 | 28560 | 85.51% | 当前组合曝光塌缩 |

这些结果不证明 Refined LogQ 或 Mixed Negative Sampling（混合负采样，MNS）永远无效，只说明当前参数 ROI 低于 Uber BatchQ `alpha=0.10`。

保留为 future work：

- MNS uniform fraction `10% / 25%`
- Refined LogQ 小 alpha
- PDA
- DICE
- 服务层 calibration rerank

### 11.5 面试怎么讲

```text
我没有采用 Recall 最高的强 LogQ。
Effect audit 显示它把 Top50 热门物品占比推到 82%，coverage 明显下降。
所以我把目标改成带约束的 Pareto 优化：
在 Recall 上升的同时约束长尾桶、coverage、热门占比、entropy 和 Gini。
Uber BatchQ alpha=0.10 在 limited-valid 和 seed42 full test 上都很漂亮，
但 multi-seed 验证时 seed2024 暴露了显著长尾退化。
因此我没有把它替换成主模型，而是把它作为“总 Recall 指标陷阱”的负结果。
```

---

## 12. 如何理解项目中的“严谨”

严谨不是多跑几个模型，而是每个结论都知道证据边界。

### 12.1 Smoke 与 full test 分开

```text
smoke:
  快速筛选方向
  可以淘汰明显差的方案
  不能直接替换 canonical

full test:
  完整 non-cold 用户
  标准 seen mask
  标准 test history
  用于正式验收
```

### 12.2 Multi-seed

单一 seed 的提升可能来自初始化偶然性。Transformer canonical 与 Qwen3 `@128` 都通过 multi-seed 显示了 seed sensitivity。

因此需要报告：

```text
mean
std
min
paired delta
```

而不是挑最好 seed。

### 12.3 Paired audit

对同一个用户比较 baseline 和 candidate：

```text
baseline-only hit
candidate-only hit
both-hit
neither-hit
```

这比只看总体 Recall 更能说明候选是否新增了真实命中。

### 12.4 Bootstrap CI

Uber BatchQ full 验收加入 deterministic paired bootstrap：

```text
bootstrap_seed = 42
bootstrap_resamples = 10000
```

基于每个用户的命中差值：

```text
candidate_hit - baseline_hit in {-1, 0, +1}
```

输出总体 Recall delta 和长尾 Recall delta 的 95% 置信区间。

> **注释：为什么 paired bootstrap？**
> Baseline 和 candidate 面对的是同一批用户。直接对用户级命中差值重采样，可以更准确地估计改进的不确定性。

### 12.5 预注册 gate

在结果出来前锁定：

```text
什么算通过
什么算失败
失败后是否自动停止
哪些动作明确禁止自动执行
```

这样避免看到结果后临时放宽标准。

---

## 13. 常见面试追问与回答

### Q1：为什么 Transformer TT 比 Mean Pool 好？

```text
Mean Pool 只做固定聚合，表达不了历史 item 之间的关系。
Time-aware Transformer 加入 positional embedding 和 recency bucket embedding，
能够根据序列位置与新旧程度学习更灵活的用户表示。
max_len 消融显示 20 -> 100 只带来约 0.0019，
所以主要提升来自架构，而不是单纯加长历史。
```

### Q2：Transformer 训练为什么需要 early stopping？

```text
它在 epoch 2 达到 valid Recall 峰值，之后 train loss 继续下降，
但 valid Recall 持续坍塌。这说明模型开始拟合训练分布中的局部模式，
而不是提升检索泛化。因此 checkpoint 必须按 valid Recall 选择。
```

### Q3：为什么不用 test 选择 RRF 权重？

```text
用 test label 调权重会产生 test-tuning。
我在 valid 上 sweep 60 组配置，按预定义 Pareto 条件选择 frozen config，
然后 test 只运行一次。
```

### Q4：低 Jaccard 为什么能支持融合？

```text
低 Jaccard 本身不能证明互补，噪声也可能低重合。
我同时检查：两路单独有效、bucket 差异可解释、融合后 Recall 高于任一单路。
三条一起成立，才能说明是高质量互补。
```

### Q5：Text Semantic 单路这么弱，为什么保留？

```text
它不是主路，而是与共现和序列建模不同质的补充信号。
多路召回看的是融合后的边际命中，不要求每条辅助路单独最强。
```

### Q6：为什么 Qwen3 没有替换 MiniLM？

```text
Qwen3 plain / instruct 在 Text Semantic 单路没有超过 MiniLM；
TT @64 的差异处于 seed 噪声范围；
@128 三 seed 中 Qwen3 只赢一次，均值略低。
因此不能声称更大 encoder 自动更优。
```

### Q7：Faiss 是否提高了模型 Recall？

```text
不是。Faiss 是检索工程层。
FlatIP exact 先与标准 full eval 对齐，再比较 IVF/HNSW 的速度和近似损失。
```

### Q8：In-batch negatives 有什么偏差？

```text
热门 item 更容易作为正样本进入 batch，
也更频繁地成为其他用户的负样本。
这是 sampling bias。LogQ 尝试按 item 暴露概率修正训练 logits。
```

### Q9：为什么强 LogQ 总 Recall 上升仍然不能采用？

```text
因为 bucket audit 显示增益几乎集中在 head item：
Top50 热门占比从约 20% 升到约 82%，coverage 大幅下降，
其他三个 target popularity buckets 回退。
总 Recall 上升不等于个性化质量改善。
```

### Q10：Uber BatchQ alpha=0.10 为什么最后不用？

```text
因为它只在 seed42 上同时满足“总 Recall 上升 + 长尾不伤害”。
换到 seed2024 后，总 Recall 仍然上升，但 1-5 和 6-20 低热度桶显著回退，
long-tail bootstrap CI 也整体为负。
这说明它的健康性依赖随机种子，不够稳健。
所以它是很好的负结果，不是最终模型。
```

### Q11：为什么不继续无限调 alpha？

```text
Alpha 越大时 Recall 上升，但热门曝光集中、Gini 上升、coverage 下降。
目标是满足业务约束下的 Pareto 最优，不是刷单一 Recall。
继续扩大 sweep 的 ROI 低，还会增加 valid 过拟合风险。
```

### Q12：离线结果有什么边界？

```text
离线 temporal LOO 只能说明在当前数据和协议下的候选生成能力。
它不能证明线上 CTR、用户满意度、留存或服务 P99。
这些需要线上流量和 A/B 实验。
```

---

## 14. 项目演进路线图

```text
Phase 0  数据加载与数据契约
  Amazon Reviews 2023 Movies_and_TV clean 5-core
  temporal LOO / seen mask / cold exclusion

Phase 1  基线
  ItemCF
  ID-only Two-Tower

Phase 2  内容与历史增强
  MiniLM item text embeddings
  has_text mask
  Mean Pool user tower
  temperature tuning
  time-decay weighting

Phase 3  Transformer 用户塔
  time-aware Transformer
  collapse diagnosis
  stability sweep
  max_len ablation
  multi-seed
  canonical TT = 0.103168

Phase 4  多路召回
  ItemCF + TT + Text Semantic + Popularity
  valid-selected Weighted RRF
  canonical 4ch = 0.125164

Phase 5  ANN 工程验证
  FlatIP exact alignment
  IVF / HNSW benchmark
  IVF nprobe=32: 8.8x speedup, 0.41% relative Recall loss

Phase 6  文本 embedding 消融
  MiniLM vs Qwen3 plain / instruct vs bert-base
  projection dim 64 / 128
  结论：Qwen3 未稳定提升，MiniLM canonical 保持不变

Phase 7  Sampling-bias 调查
  duplicate masking
  old LogQ + shuffled-Q negative control
  exposure audit
  Uber BatchQ low-alpha Pareto smoke
  alpha=0.10 full-test multi-seed validation: rejected by seed2024 long-tail gate
```

---

## 15. 核心句子速记

飞行途中可以反复复述以下句子：

```text
1. 召回系统不只看总体 Recall，还要看 coverage、互补性、bucket 和 ANN 成本。

2. temporal LOO 的 test 口径是：
   history=train+valid，seen mask=train+valid，排除 cold target。

3. ItemCF 是稳定共现基线；Two-Tower 的价值是可扩展，并天然适配 ANN。

4. 文本增强有价值，但本项目更大的增益来自用户历史建模和训练目标。

5. Transformer 的关键不是“更复杂”，而是经过 collapse diagnosis、
   stability sweep、max_len ablation 和 multi-seed 后仍然成立。

6. 多路融合不能只报低 Jaccard；
   单路有效、差异可解释、融合增益三条必须同时成立。

7. RRF 用 rank 融合异构召回器，权重只在 valid 选择，test 只运行一次。

8. Faiss 是检索工程层，不是提升模型效果的模型；FlatIP exact 必须先对齐。

9. Qwen3 消融说明 embedding 不是越大越好，任务匹配比模型规模更重要。

10. In-batch negatives 会产生 popularity sampling bias，
    热门 item 更频繁进入 batch，也更频繁被当成负样本。

11. LogQ 不是单调去热门旋钮；强修正可能让 raw dot-product 更偏头部。

12. 高 Recall 可能是指标幻觉：
    必须同时审计长尾桶、热门占比、coverage、entropy 和 Gini。

13. Uber BatchQ alpha=0.10 不是最终模型：
    seed42 很好，但 seed2024 长尾显著退化，所以按预注册 gate 拒绝。

14. 所有结果都是 offline evaluation，不是线上 A/B。
```

---

## 16. 当前可以讲与不能讲

### 可以讲

- 完成 Amazon Reviews 2023 Movies_and_TV 5-core 的 temporal LOO 召回系统。
- Time-aware Transformer TT full test Recall@50 = `0.103168`。
- 4ch valid-selected Weighted RRF full test Recall@50 = `0.125164`。
- RRF 权重只在 valid 上选择，test 只运行一次。
- ItemCF 与 TT 的低 Jaccard 通过单路 Recall、bucket 和融合增益共同解释。
- Faiss IVF `nprobe=32` 离线 benchmark：`8.8x` 提速，Recall 相对损失 `0.41%`。
- Qwen3 未稳定超过 MiniLM，因此保留为诚实消融。
- 强 LogQ 出现热门曝光集中，因此没有直接替换 canonical。
- Uber BatchQ `alpha=0.10` 通过 seed42 但失败于 seed2024 长尾约束，因此最终未采用。

### 不能讲

- “线上 A/B 提升 12.5%”。
- “Transformer 已部署生产”。
- “Faiss 提升了模型 Recall”。
- “Qwen3 升级成功”。
- “LogQ 已经解决 popularity bias”。
- “Uber BatchQ alpha=0.10 已替换主模型”。
- “最终模型用了 InfoNCE + LogQ”。
- “Text Semantic 已证明解决 cold-start”。
- “离线平均延迟就是线上 P99”。

---

## 17. 建议复习顺序

如果只有 2 小时：

```text
§3 标准评估口径
-> §6 Transformer 用户塔
-> §7 多路召回
-> §10 LogQ
-> §13 面试追问
-> §15 核心句子
```

如果有 4 小时：

```text
全文顺序阅读
-> 把 §13 的回答遮住后自己复述
-> 对照 §15 核心句子做第二轮
```

如果准备深挖面试：

```text
重点复习：
seen mask 为什么不同
Transformer collapse 为什么不能只看 loss
低 Jaccard 为什么不是充分条件
Faiss FlatIP 为什么必须先对齐
LogQ 为什么会造成热门曝光集中
Uber BatchQ 为什么 seed42 通过但 seed2024 失败后必须拒绝
```

---

## 18. 参考报告

继续深挖时读取：

```text
docs/project_summary.md
docs/reports/transformer_user_tower_investigation.md
docs/reports/multichannel_design_rationale.md
docs/reports/multichannel_valid_selected_eval.md
docs/reports/faiss_transformer_two_tower_benchmark.md
docs/reports/embedding_ablation_report.md
docs/reports/transformer_logq_investigation.md
docs/decision_log.md
docs/issue_log.md
```

外部阅读：

- Uber Engineering: [Innovative Recommendation Applications Using Two Tower Embeddings at Uber](https://www.uber.com/en-GB/blog/innovative-recommendation-applications-using-two-tower-embeddings/)
- Google Research: [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- RecSys 2025: [Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval](https://arxiv.org/abs/2507.09331)
- PDA: [Causal Intervention for Leveraging Popularity Bias](https://arxiv.org/abs/2105.06067)
- DICE: [Disentangling Interest and Conformity for Recommendation](https://arxiv.org/abs/2006.11011)

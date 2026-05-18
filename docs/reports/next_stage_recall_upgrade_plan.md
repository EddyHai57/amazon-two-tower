# 下一阶段召回升级计划

**状态：计划（Plan）**  
**创建时间：** 2026-05-19  
**作者：** Eddy（Claude Code 辅助整理）  
**适用阶段：** Post-5/15 投递期后续升级（需 Eddy 明确 go-ahead 后逐项实施）

> **重要说明：** 本文档记录的三个方向均为计划（Plan/Backlog），未开始实现。文中数字仅为历史已有事实，不包含任何预测或模拟结果。

---

## 当前项目基线事实

| 模型 | Full test Recall@50 | 备注 |
| --- | ---: | --- |
| ItemCF | 0.083570（8.36%） | 正式传统 baseline |
| ID-only Two-Tower（20ep） | 0.053198（5.32%） | ID baseline |
| Text + Time-decay Mean Pool τ=0.15 | 0.078315（7.83%） | 最终主模型，+47.2% vs ID-only |

**关键诊断结论（已有事实）：**
- ItemCF 在头部 item（>100 train 交互）胜出（Recall@50=0.122522 vs Two-Tower 0.083277）
- Two-Tower 在中等热度（21–100 桶，32.6% targets）超过 ItemCF（0.079564 vs 0.060890）
- Two-Tower 整体 Recall@50 不超过 ItemCF（7.83% < 8.36%）
- 两路模型在候选集上的重叠比例尚未量化（此为计划项之一）

---

## 计划一：多路召回融合（Multi-channel Retrieval）

**状态：计划，未开始**  
**所需前置条件：** Eddy 明确确认（AGENTS.md §4 中 Hybrid retrieval 需显式确认）

### 目标

将当前"单路 Two-Tower 实验"升级为"离线多路召回系统"，验证 neural recall 能否补充 ItemCF 未覆盖的候选。

### 候选召回通路

| 通路 | 描述 | 当前状态 |
| --- | --- | --- |
| ItemCF recall | 基于 user-item 共现的传统协同过滤召回 | ✅ 已完成（Recall@50 = 0.083570） |
| Two-Tower recall | Text + Time-decay Mean Pool τ=0.15 | ✅ 已完成（Recall@50 = 0.078315） |
| Text semantic recall | 基于 item text embedding 的纯语义检索 | ⚠️ 已有脚本（eval_pure_text_retrieval.py），full eval 未复现 |
| Popularity fallback recall | 全局热门 item 补充兜底 | ❌ 未实现 |

### 候选融合方式（均为计划）

1. **Union merge**：各路 TopK 取并集，按 score 或排名重排
2. **Quota merge**：固定各路配额，例如 ItemCF 30 + Two-Tower 20
3. **RRF（Reciprocal Rank Fusion）**：基于排名倒数融合，不依赖 score 可比性

### 评估指标（计划中需要收集）

| 指标 | 含义 | 注意事项 |
| --- | --- | --- |
| Recall@50 / Recall@100 | 整体命中率 | 主要对比指标 |
| NDCG@50 | 排序质量 | 融合排序相关 |
| 与 ItemCF 的 overlap | 两路共同命中比例 | 不能将 overlap 写成 Recall |
| unique hit contribution | 某通路独占命中（另一路未命中）数量 | 关键价值指标 |
| item coverage | 候选集中 unique item 数量 | 覆盖多样性 |
| avg item popularity | 候选集平均热度（train 交互数） | 是否覆盖长尾 |
| 按 popularity bucket 的 recall | head/mid/tail 分桶 Recall@50 | 结构性诊断 |

### 项目意义

单路 Two-Tower 当前没有整体超过 ItemCF，但如果 Two-Tower 能召回 ItemCF 召不到的候选（尤其是中等热度 item），则多路融合的 Recall@100 可能超过任何单路。这比硬凹单模型指标更接近工业召回系统的真实设计思路。

### 实施前提（需确认）

1. Eddy 明确 go-ahead（hybrid retrieval 需显式确认，见 AGENTS.md §4）
2. 最终模型 checkpoint 可用（新服务器当前无 checkpoint）
3. ItemCF topK candidates 已保存或可重新生成
4. 确定融合方式（quota / RRF）和评估口径

### 风险

- 实现复杂度高于单路 benchmark，需要多路 candidate 存储和对齐
- 如果融合结果提升不明显，面试叙事需要解释"为什么加了还不如 ItemCF"

---

## 计划二：召回贡献分析（Overlap / Unique Contribution Analysis）

**状态：计划，未开始**  
**所需前置条件：** 最终模型 checkpoint 可用，ItemCF topK 候选集可用

### 目标

回答"某一路召回是否有独立价值"，量化各路召回在 test set 上的独立贡献。

### 分析维度

| 分析项 | 含义 |
| --- | --- |
| ItemCF 和 Two-Tower topK 的重叠比例 | overlap@K：两路共同命中 / 任一路命中 |
| Two-Tower 独占命中数 | Two-Tower 命中但 ItemCF 未命中的 test target 数量 |
| ItemCF 独占命中数 | ItemCF 命中但 Two-Tower 未命中的 test target 数量 |
| text recall 独占命中 | 仅 text semantic recall 命中的数量 |
| 多路融合新增命中 | 融合后命中但单路均未命中的数量（理论值） |
| 按 popularity bucket 的独占贡献 | 各 bucket 下哪路召回更有独立价值 |
| long-tail item coverage | ≤5 train 交互 item 是否被更多覆盖 |
| avg item popularity 变化 | 融合后候选集热度是否下降（更多长尾） |

### 关键概念区分（面试核心解释点）

```text
overall Recall@K    : 某路 TopK 里命中 test target 的比例
overlap             : 两路 TopK 共同命中的比例（不是 Recall）
unique hit          : 某路独占命中，另一路未命中（衡量独立价值）
coverage            : 候选集覆盖的 unique item 数量（多样性指标）
novelty             : 候选集中 popularity 较低的 item 占比（长尾指标）
```

**注意：不要把 overlap 当 Recall 写。**

### 项目意义

即使 Two-Tower 单路 Recall@50 低于 ItemCF，只要 Two-Tower 能召回 ItemCF 召不到的样本，或覆盖更多中长尾 item，就能作为多路召回中的有效通路。这是面试中解释"Two-Tower 价值"的关键逻辑。

### 预期面试叙事结构

```text
1. 单路对比：Two-Tower 7.83% < ItemCF 8.36%（整体不如，但赢在中等热度桶）
2. 重叠分析：Two-Tower 和 ItemCF 候选集有 XX% overlap，各有 YY 个独占命中
3. 价值结论：Two-Tower 能补充 ItemCF 未覆盖的中等热度 item，在多路召回中有独立价值
4. 延伸：若 Two-Tower 独占命中率高于 text semantic recall，则作为多路通路的 ROI 更高
```

### 实施前提（需确认）

1. 最终模型 checkpoint 可用（新服务器当前无 checkpoint）
2. ItemCF topK（K=50 或 K=100）候选集可以重新生成或从输出目录恢复
3. 确定分析脚本方案：复用现有 eval 脚本还是新建独立分析脚本

---

## 计划三：Faiss Benchmark 完整化

**状态：部分已完成（历史数字可信），部分待修复**

### 当前已有事实（来自 2026-05-14 issue_log）

**最终模型 Faiss FlatIP Benchmark（全量 test，496,470 用户）：**

| 指标 | FlatIP（exact） |
| --- | ---: |
| Recall@50 | 0.078315 |
| Recall@20 | 0.052724 |
| Recall@100 | 0.104792 |
| NDCG@50 | 0.030862 |
| MRR@50 | 0.019036 |
| 吞吐量 | 1,165 users/s |
| 平均延迟 | 0.858 ms/user |

> 注：FlatIP Recall@50 = full eval Recall@50（delta=0），一致性已验证。

**最终模型 Faiss IVF Benchmark（nlist=4096，nprobe=32/64）：**

| 指标 | FlatIP（exact） | IVF nprobe=32 | IVF nprobe=64 |
| --- | ---: | ---: | ---: |
| Recall@50 | 0.078315 | 0.078172（-0.18%） | 0.078365（+0.06%） |
| 吞吐量 | 1,165 users/s | 29,114 users/s | — |
| 平均延迟 | 0.858 ms/user | 0.034 ms/user | — |
| Speedup | 1× | **25.0×** | — |

> 注：以上均为 offline retrieval benchmark latency，不是线上 A/B 实测延迟。

### 当前仓库问题

| 问题 | 状态 | 严重等级 |
| --- | --- | --- |
| `scripts/benchmark_faiss_time_decay_text_mean_pool.py` **不存在** | ISSUE-20260519-002（Open） | High |
| `benchmark_faiss_ivf_time_decay_text_mean_pool.py` import 缺失模块，无法运行 | 依赖上一条解决 | High |
| `configs/faiss_id_two_tower_clean_20epoch.yaml` 仅有 ID-only 版本，无最终模型 Faiss 配置 | 低优先级 | Low |

### Benchmark 完整化 Backlog

**P0（修复复现路径）：**
1. 重建 `scripts/benchmark_faiss_time_decay_text_mean_pool.py`（参考 `benchmark_faiss_id_two_tower.py` 适配最终模型）
2. 验证 IVF 脚本 import 正常
3. 提交两个脚本到 git

**P1（扩展 benchmark，需 Eddy 确认）：**

| 扩展项 | 参数 | 面试价值 | AGENTS.md 约束 |
| --- | --- | --- | --- |
| IVF nprobe sweep（8/16/32/64/128） | nlist=4096 | 展示 recall/speed trade-off 曲线 | 需明确确认（Faiss 深度参数 sweep） |
| HNSW benchmark | M=32/64，efSearch=50/100 | 对比 IVF 图索引替代方案 | **永久禁用**（GPU Faiss / HNSW 在 AGENTS.md §4 中禁止） |
| PQ benchmark | m=8/16 | 量化压缩 index size | **永久禁用**（PQ 在 AGENTS.md §4 中禁止） |

> **注意**：HNSW 和 PQ 在 AGENTS.md §4 中明确标注为"永久禁用（out of scope）"，除非面试官特别询问，否则不做。IVF nprobe sweep 在 AGENTS.md §4 中需要 Eddy 显式确认才能启动。

**P1 中唯一合理的扩展是 IVF nprobe sweep**（如果 Eddy 确认）：

```text
当前已有：nprobe=32（Recall@50 损失 0.18%，25× speedup）
计划补充：nprobe=8/16/64/128，形成完整 recall/speed 曲线
简历价值：可以说"量化了 ANN 检索的精度-速度 trade-off 曲线"
```

### Benchmark 所需指标模板

每次 Faiss benchmark 必须记录：

| 字段 | 内容 |
| --- | --- |
| index_type | FlatIP / IVF / HNSW / PQ |
| nlist | IVF 的分桶数 |
| nprobe | IVF 查询桶数 |
| M / efSearch | HNSW 参数（如适用） |
| Recall@50 loss | 相对 FlatIP 的损失 |
| avg latency | ms/user（offline benchmark） |
| throughput | users/s |
| search time（total） | 全量 test 搜索总时间 |
| index size | 索引文件大小（MB） |
| build time | 建立索引耗时 |

### 项目意义

Faiss 不提升模型 Recall，它验证召回工程链路：embedding 导出、ANN 检索、召回一致性、速度和精度 trade-off。IVF nprobe=32 已实现 25× speedup 同时仅损失 0.18% Recall，这是项目中的关键工程结论，也是面试中的 ANN 知识考点。

---

## 优先级排序

| 优先级 | 任务 | 所需条件 | 预估工作量 |
| --- | --- | --- | --- |
| **P0** | 重建 `benchmark_faiss_time_decay_text_mean_pool.py` | 只需代码，无需 checkpoint | 小（参考现有脚本） |
| **P0** | 训练最终模型（新服务器复现） | `.venv` + 数据已就绪 | 中（约 20 epoch GPU 训练） |
| **P1** | 重跑 Faiss FlatIP + IVF benchmark | checkpoint 可用后 | 小 |
| **P1** | 召回贡献分析（overlap/unique hit） | checkpoint + ItemCF candidates | 中 |
| **P2** | IVF nprobe sweep（Eddy 确认后） | checkpoint 可用后 | 小 |
| **P2** | 多路召回融合实现（Eddy 确认后） | 多路 candidates 准备 | 大 |

---

## 明确不做的事（AGENTS.md §4 约束）

- HNSW benchmark（permanently disabled）
- PQ benchmark（permanently disabled）
- GPU Faiss（permanently disabled）
- 声称 Two-Tower 整体超过 ItemCF（事实不支持）
- 声称线上 A/B 或工业级部署
- Hard Negative Mining 系列（已完成 4 个实验，均失败，marked as future work）

---

*本文档内容为计划级别，任何实施需经 Eddy 确认。*

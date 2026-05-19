# Faiss ANN 检索工程验证报告 — 最终 Two-Tower 模型

**报告日期：** 2026-05-20  
**脚本：** `scripts/benchmark_faiss_two_tower.py`  
**输出目录：** `outputs/faiss_two_tower_benchmark/`  
**状态：** ✅ 完成 — 2026-05-20 14:02 UTC

---

## 1. 背景与目标

本次 Faiss benchmark 的目的是**工程验证**：验证最终 Two-Tower 向量召回通路能否从 brute-force 全量 scoring 替换为 Faiss ANN 检索，并量化 Recall 损失、延迟、吞吐和索引大小的工程 trade-off。

**不是**提升 Recall 的建模实验。Recall 提升来自多路融合（见 multichannel_valid_selected 系列）。

### 对齐目标

```text
FlatIP 对齐目标：standalone final Two-Tower full test Recall@50 = 0.078315
不是四路融合 Recall@50 = 0.104776
```

这两个数字代表不同的评估系统，不能混用：

| 数字 | 含义 |
| --- | --- |
| 0.078315 | 单独 final Two-Tower 的全测试集 Recall@50 |
| 0.104776 | 四路加权 RRF 融合（ICF+TT+Text+Pop）的全测试集 Recall@50 |

---

## 2. 使用的模型与数据

| 项目 | 详情 |
| --- | --- |
| 模型 | Text + Time-decay Mean Pool Two-Tower |
| 超参 | τ=0.15, decay_rate=0.8, dim=64 |
| Checkpoint | `outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt` |
| 最佳 epoch | 17（limited valid Recall@50 = 0.121140） |
| 数据集 | Amazon Reviews 2023 Movies_and_TV 5-core |
| 候选 item 数 | 153,977 |
| 评估 user 数 | 496,470（non-cold-start test users） |
| 冷启动跳过数 | 979 |
| **无重新训练** | checkpoint 来自之前已完成的 20-epoch 训练 |

---

## 3. Seen-item Mask 口径

| 场景 | Seen mask | History input |
| --- | --- | --- |
| Valid eval | train items only | train items only（max 20） |
| **Test eval（本次）** | **train + valid items** | **train + valid items（max 20）** |

Target item 永不被 mask（不论 seen-item 集合是否包含 target）。

K_SEARCH=300：Faiss 每个 query 检索 300 个 item，再对每个 user 过滤掉 seen items，取 top-50。平均 seen items 约 10 个（train+valid），300 的 over-fetch 余量充足。

---

## 4. FlatIP 正确性校验

**结论：完全对齐，通过校验。**

| 指标 | 数值 |
| --- | --- |
| 对齐目标 Recall@50 | 0.078315 |
| FlatIP 实测 Recall@50 | **0.078315** |
| 相对误差 | **0.0000%** ✅ |

FlatIP 与 brute-force full scoring 结果完全一致，验证：

1. 向量编码（`encode_items` / `encode_users`）与 full eval 使用同一 checkpoint
2. L2 归一化后 inner product = cosine similarity，与 full eval 的 dot product 评分口径一致
3. Seen-item mask 逻辑正确（train+valid，target 不被 mask）
4. Test set 用户、target item、seen items 与 full eval 严格一致

---

## 5. 索引参数选择理由

### IVFFlat（倒排文件索引）

- **nlist = 1024**：Faiss 建议每个 centroid 至少 39 个训练点。153,977 ÷ 1024 = 150 ≥ 39 ✓。不产生训练数据不足警告。
- **nprobe 候选值：16 / 32 / 64**：覆盖从激进省时（1.6% cells）到高精度（6.3% cells）的工程区间。
- 索引构建：train=3.33s，add=0.36s，快速且可重复。

### HNSW（分层可导航小世界图）

- **M = 32，efConstruction = 40**：标准配置，构建时间约 2.3s。
- **efSearch = 64 / 128**：控制搜索精度 vs 速度。
- 向量已 L2 归一化，METRIC_INNER_PRODUCT 在 faiss-cpu 1.13.2 中实测可用。

---

## 6. 结果总览

### 6.1 完整结果表（full test，496,470 users）

| 索引 | Recall@50 | NDCG@50 | MRR@50 | Recall@20 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlatIP（精确基线） | **0.078315** | **0.030862** | **0.019036** | 0.052724 | 0.104792 |
| IVF nprobe=16 | 0.076752 | 0.030344 | 0.018772 | 0.051828 | 0.102818 |
| IVF nprobe=32 | 0.077721 | 0.030675 | 0.018946 | 0.052438 | 0.104087 |
| IVF nprobe=64 | 0.078091 | 0.030788 | 0.019000 | 0.052547 | 0.104504 |
| HNSW efSearch=64 | 0.076909 | 0.030278 | 0.018662 | 0.051681 | 0.103225 |
| HNSW efSearch=128 | 0.077574 | 0.030539 | 0.018821 | 0.052164 | 0.104000 |

### 6.2 召回损失 vs FlatIP

| 索引 | R@50 delta | 相对损失 |
| --- | ---: | ---: |
| FlatIP | 0.000000 | 0.000% |
| IVF nprobe=16 | −0.001563 | −**1.996%** |
| IVF nprobe=32 | −0.000594 | −**0.759%** |
| **IVF nprobe=64** | **−0.000224** | **−0.285%** ← 推荐 |
| HNSW efSearch=64 | −0.001406 | −1.795% |
| HNSW efSearch=128 | −0.000741 | −0.946% |

### 6.3 延迟与吞吐（offline batch benchmark）

| 索引 | 搜索总时间 | 吞吐（users/s） | 均摊延迟（ms/user） | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| FlatIP | 125.40 s | 3,959 | 0.2526 | 1.0× |
| IVF nprobe=16 | 10.5 s | 47,265 | 0.0212 | **11.9×** |
| IVF nprobe=32 | 15.7 s | 31,497 | 0.0317 | 8.0× |
| **IVF nprobe=64** | 25.3 s | 19,615 | **0.0510** | **5.0×** |
| HNSW efSearch=64 | 14.7 s | 33,763 | 0.0296 | 8.5× |
| HNSW efSearch=128 | 21.6 s | 23,028 | 0.0434 | 5.8× |

> **注意：上述延迟是 offline batch 模式的均摊延迟**，即对 496,470 个 query 批量执行、总时间均摊至每个 user。单次在线 query 延迟因索引加载、冷启动等开销会更高。本报告不代表生产环境延迟。

---

## 7. 索引大小

两种索引均存储相同的向量数据（主要开销）：

```text
向量存储（仅 float32 向量）：153,977 × 64 × 4 bytes ≈ 39.4 MB
IVFFlat 额外开销：1,024 个 centroid 向量 ≈ 0.26 MB
HNSW 额外开销：图结构（M=32 每节点约 32 × 4 bytes）≈ 19.7 MB
```

实际内存占用（含 faiss 内部结构）约 40–60 MB，适合单节点内存部署。

---

## 8. 推荐工程折中点

**推荐：IVF nprobe=64**

| 理由 | 数据支持 |
| --- | --- |
| Recall 损失可忽略 | −0.285% 相对损失，绝对值仅 −0.000224 |
| 显著加速 | 5.0× 搜索速度，吞吐从 3,959 → 19,615 users/s |
| 中等 nprobe 稳定性 | nprobe=16 损失达 2%，nprobe=64 曲线趋于平缓 |
| 索引构建极快 | train+add < 4s，可频繁重建 |

如果追求更高吞吐（牺牲约 0.8% recall）：IVF nprobe=32（8× 加速）。

如果追求极速（可接受 2% recall 损失）：IVF nprobe=16（12× 加速）。

HNSW 在相近 recall 损失下略逊于 IVF（efSearch=128 比 nprobe=32 加速低，但召回更差）。IVFFlat 在此 embedding 规模（n=153,977）下是更优选择。

---

## 9. 关键结论

1. **FlatIP 完全对齐**：Recall@50=0.078315，与 full eval 精确一致（0.000% 误差），向量编码和 mask 逻辑正确。

2. **IVF nprobe=64 是稳健折中点**：在 5× 加速下仅损失 0.285% Recall，满足工程可用性条件。

3. **IVF nprobe 收益递减**：nprobe 从 16→32→64 时，recall loss 从 2.0%→0.76%→0.29% 快速改善；64→128 的改善已接近饱和（未测，但趋势可预期）。

4. **HNSW 可用但非最优**：在此规模（153,977 items, dim=64）下，HNSW 的 speedup 和 recall 不如相近参数的 IVF。

5. **向量编码极快（GPU）**：item 编码 0.25s，496K user 编码 0.35s，不是系统瓶颈。

---

## 10. 局限性

```text
1. 离线 batch benchmark：延迟是均摊值，不代表线上单次 query 延迟
2. 单机 CPU 检索：faiss-cpu，未测试 GPU faiss / 分布式
3. 无真实服务部署：无 HTTP 服务器开销、负载均衡、序列化开销
4. 无 P99 延迟测量：未测尾部延迟
5. nlist 选择未 sweep：nlist=1024 是合理默认值，未穷举其他选项
6. 评估指标为 Recall@K（推荐效果）：与 overlap@K（检索一致性）是不同口径，结果不可直接对比
```

---

## 11. 是否建议写入 README

**建议补充，但非立即必须。**

README 目前仅记录了 ID-only 模型的 Faiss benchmark（IVF1024 nprobe=32，overlap@50=0.768）。本次新增了最终 Two-Tower 的 Faiss Recall@50 评估。

建议在 README 的 Faiss 章节中补充：

```text
最终模型（Text+Time-decay τ=0.15）Faiss ANN 验证：
- FlatIP 精确检索 Recall@50 = 0.078315（与 brute-force 完全对齐）
- IVF nlist=1024, nprobe=64：Recall@50 = 0.078091（−0.285%），5× 搜索加速
```

推迟到第一轮面试反馈后再更新，目前项目叙事已完整。

---

## 12. 是否建议更新简历

**需谨慎，涉及已有简历数字。**

当前简历记录的是 ID-only 模型的 Faiss benchmark：

```text
简历现有：Faiss IVF speedup: 25× over brute-force, 0.18% recall loss
（来源：ID-only two-tower + overlap@50 口径，见 CLAUDE.md §6）
```

本次新数字为最终 Two-Tower + Recall@50 口径：

```text
新数字：IVF nprobe=64 → 5× speedup, 0.285% Recall@50 loss（batch benchmark）
```

两套数字的测量口径不同（overlap@50 vs Recall@50，不同模型，不同测量方式），**不建议直接替换简历原有数字**。

建议方案：保留原有 ID-only 数字（历史事实），另加一行：

```text
final Two-Tower（Text+Time-decay）Faiss ANN 验证：IVF nprobe=64，Recall@50 损失 <0.3%，5× 加速
```

具体措辞待 Eddy 确认后再修改简历。

---

## 13. 文件清单

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `scripts/benchmark_faiss_two_tower.py` | 新增 | FlatIP + IVF + HNSW 统一 benchmark 脚本 |
| `outputs/faiss_two_tower_benchmark/faiss_benchmark_results.json` | 新增 | 完整 benchmark 数字（JSON） |
| `outputs/faiss_two_tower_benchmark/faiss_benchmark_results.csv` | 新增 | 汇总表（CSV） |
| `outputs/faiss_two_tower_benchmark/faiss_benchmark_report.md` | 新增 | 原始结果报告（英文） |
| `outputs/faiss_two_tower_benchmark/item_embeddings.npy` | 新增 | 153,977 × 64 item 向量（不 commit） |
| `outputs/faiss_two_tower_benchmark/test_user_embeddings.npy` | 新增 | 496,470 × 64 user 向量（不 commit） |
| `outputs/faiss_two_tower_benchmark/test_user_idx.npy` | 新增 | user 索引（不 commit） |
| `docs/reports/faiss_two_tower_benchmark.md` | 新增 | 本报告 |
| `docs/daily_logs/2026-05-20.md` | 修改 | Part 8 追加 |

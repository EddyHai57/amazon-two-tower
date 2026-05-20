# Transformer Two-Tower Faiss Offline Benchmark

**生成时间：** 2026-05-20 17:17 UTC  
**脚本：** `scripts/benchmark_faiss_transformer_two_tower.py`  
**模型：** Text + Time-aware Transformer Two-Tower，τ=0.15，max\_len=100  
**Checkpoint：** `outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt`（best\_epoch=2）  
**评估集：** Amazon Reviews 2023 Movies\_and\_TV 5-core，full test，496,470 non-cold users

---

## 1. FlatIP 对齐验证

| 项目 | 值 |
|---|---:|
| 预期 Recall@50（canonical full test） | 0.103128 |
| 实际 FlatIP Recall@50 | **0.103168** |
| 相对误差 | 0.0391% |
| 对齐状态 | ✅ PASS |

FlatIP 精确检索与 canonical full eval 对齐，差值 +0.000040（在误差范围内，与两次独立运行差值一致）。

---

## 2. 完整 Benchmark 结果

| 索引 | Recall@50 | NDCG@50 | MRR@50 | Speedup vs FlatIP | 延迟 (ms/user) | R@50 delta |
|---|---:|---:|---:|---:|---:|---:|
| FlatIP（精确） | 0.103168 | 0.040087 | 0.024439 | 1.0× | 0.2753 | — |
| IVF nprobe=16 | 0.101897 | — | — | **13.0×** | **0.0211** | −0.001271（−1.23%） |
| IVF nprobe=32 | 0.102749 | — | — | **8.8×** | **0.0313** | −0.000419（−0.41%） |
| IVF nprobe=64 | 0.103102 | — | — | 5.5× | 0.0497 | −0.000066（−0.06%） |
| HNSW ef=64 | 0.102923 | — | — | 9.9× | 0.0277 | −0.000246（−0.24%） |
| HNSW ef=128 | 0.103058 | — | — | 7.0× | 0.0396 | −0.000111（−0.11%） |

---

## 3. 延迟 / 吞吐量

| 索引 | 总检索耗时（496,470 users） | 吞吐量 (users/s) | 平均延迟 (ms/user) |
|---|---:|---:|---:|
| FlatIP | 136.69 s | 3,632 | 0.2753 |
| IVF nprobe=16 | 10.50 s | 47,280 | **0.0211** |
| IVF nprobe=32 | 15.56 s | 31,907 | 0.0313 |
| IVF nprobe=64 | 24.94 s | 19,904 | 0.0497 |
| HNSW ef=64 | 13.81 s | 35,954 | 0.0277 |
| HNSW ef=128 | 19.57 s | 25,371 | 0.0396 |

---

## 4. 工程折中分析

### 推荐工程点：IVF nprobe=32

| 指标 | 值 |
|---|---:|
| Recall@50 损失 | **−0.41%**（0.103168 → 0.102749） |
| 提速倍数 | **8.8×** |
| 平均延迟 | **0.031 ms/user** |

- nprobe=16：13.0× 提速，但 Recall 损失 1.23%，略高
- nprobe=32：8.8× 提速，Recall 损失仅 0.41%，折中最优
- nprobe=64：5.5× 提速，Recall 损失极小（0.06%），适合对精度要求极高场景
- HNSW ef=64/128：速度与 nprobe=32/16 相近，Recall 损失介于两者之间；build 时间较快（2.2s vs IVF 3.7s）

### 与旧 Time-decay TT Faiss benchmark 对比

| 指标 | 旧 Time-decay TT | **新 Transformer TT** |
|---|---:|---:|
| FlatIP Recall@50 | 0.078315 | **0.103168** |
| IVF nprobe=32 Recall@50 | 0.077675（−0.82%） | **0.102749（−0.41%）** |
| IVF nprobe=32 Speedup | 25.0× | **8.8×** |
| IVF nprobe=32 Latency | 0.034 ms/user | **0.031 ms/user** |

> 注：旧 benchmark 使用 nlist=4096；新 benchmark 使用 nlist=1024（与 items=153,977 匹配，避免 Faiss 训练警告）。两套 benchmark 的绝对延迟不可直接比较（机器环境不同）。

---

## 5. 索引参数说明

- `nlist = 1024`（coarse clusters；153,977 / 1024 ≈ 150 items/cluster，合理范围）
- 向量维度：64（float32），索引大小下限：153,977 × 64 × 4 = **39.4 MB**
- K\_SEARCH = 300（over-fetch，seen-item filtering 后取 top-50）
- Seen-item mask：test 用户 = train + valid 所有交互，target item 永不 mask

---

## 6. Audit 检查

| 检查项 | 状态 |
|---|---|
| Checkpoint = canonical best\_epoch=2 | ✅ |
| FlatIP Recall@50 对齐 canonical（相对误差 <0.1%） | ✅ |
| test seen mask = train + valid | ✅ |
| target item 永不 mask | ✅ |
| 无新训练，无数据修改 | ✅ |
| 不重跑 multi-channel | ✅ |

---

## 7. 结论

1. **FlatIP 对齐通过**：Recall@50 = 0.103168，与 canonical 0.103128 差 0.000040，验证了 Transformer TT 向量空间一致性。
2. **推荐工程点：IVF nprobe=32**：8.8× 提速，Recall 损失 −0.41%（0.102749），适合离线批量召回场景。
3. **HNSW ef=64** 也是可行备选：9.9× 提速，Recall 损失 −0.24%，build time 更短（2.2s）。
4. 新 Transformer TT 的 IVF Recall 损失（−0.41%）比旧 Time-decay TT 的 IVF 损失（−0.82%）更小，说明新模型向量分布对 IVF 聚类更友好。

---

## 8. 文件清单

```text
outputs/faiss_transformer_two_tower_benchmark/
  faiss_benchmark_results.json     — 完整 benchmark 结果
  faiss_benchmark_results.csv      — CSV 汇总
  faiss_benchmark_report.md        — 内联报告（outputs 版本）
  item_embeddings.npy              — 153,977 × 64 item 向量（不提交 git）
  test_user_embeddings.npy         — 496,470 × 64 user 向量（不提交 git）
  test_user_idx.npy                — user index 映射（不提交 git）
docs/reports/faiss_transformer_two_tower_benchmark.md  — 正式报告
```

> ⚠️ outputs/ 不提交 git。

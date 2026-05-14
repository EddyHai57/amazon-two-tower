# Amazon Two-Tower Retrieval

面向工业推荐召回流程的 Two-Tower 离线实验系统，基于 Amazon Reviews 2023（Movies\_and\_TV）构建，覆盖数据预处理、ID-only 基线、用户塔 Mean Pooling 升级、物品侧文本增强、温度超参扫描，以及离线 Faiss 检索 benchmark 的完整实验链路。

---

## 项目背景

工业推荐系统召回阶段的核心挑战是：如何在百万级 item 库中，快速且准确地为每个用户筛选出 top-K 候选。本项目以 Amazon Reviews 2023 Movies\_and\_TV 数据集（49.7 万用户 / 15.4 万 item / 530 万交互）为基础，系统性地复现并诊断 Two-Tower 召回架构的迭代路径：

- 从 ID-only baseline 出发，与传统 ItemCF 正面对比
- 逐步引入 item 文本特征（sentence-transformer frozen embeddings）和用户历史 mean pooling
- 通过温度系数调优、Faiss 索引加速、流行度分桶诊断，揭示不同方法在头部 / 中段 / 长尾 item 上的差异化优势
- 最终架构相比 ID-only baseline 在 full test Recall@50 上提升 **43.5%**

**核心约束**：
- 5-core clean split，严格 leave-one-out 时序划分，无时间泄漏
- 全 offline evaluation，不涉及线上 A/B
- 每次实验独立输出目录，不覆盖已有 checkpoint 或 baseline 结果

---

## 数据集

| 字段 | 值 |
| --- | --- |
| 数据集 | Amazon Reviews 2023 — Movies\_and\_TV |
| 过滤规则 | 5-core（用户、物品各 ≥5 条交互） |
| 用户数 | 497,449 |
| 物品数 | 153,977 |
| 训练集 | 4,319,438 条 |
| 验证集 / 测试集 | 各 497,449 条（leave-one-out） |
| Full valid evaluated users | 497,137（skipped cold users: 312） |
| Full test evaluated users | 496,470（skipped cold users: 979） |

---

## 模型演化路线

```
ID-only Two-Tower
├── Text-enhanced item tower（additive fusion，frozen text embedding）
├── Mean Pooling user tower（user history mean pooling）
└── Text + Mean Pooling（item text + user history mean pooling）
    └── Temperature Sweep（τ = 0.05 / 0.07 / 0.10 / 0.15 / 0.20 / 0.30）
        └── τ = 0.15 → 20epoch → 当前最终主模型
```

---

## 最终主模型架构

**Text + Mean Pooling Two-Tower，τ = 0.15**

### 用户塔

```
user_vec = user_id_emb(u) + history_weight × mean( item_id_emb(h) for h in history )
```

- `user_id embedding`：dim=64，随机初始化
- `history mean pooling`：train split 最近 20 条历史 item-id embedding 均值；训练时排除当前正样本 item，避免 target leakage
- `history_weight = 1.0`

### 物品塔

```
item_vec = item_id_emb(i) + has_text_mask(i) × text_proj( text_emb(i) )
```

- `item_id embedding`：dim=64，随机初始化
- `text_proj`：Linear(384 → 64)，从 sentence-transformer 384-dim frozen embeddings 投影
- `has_text mask`：153,977 个物品中 95,016 个（61.7%）有可用文本（title / description）；无文本物品的 text path 通过 mask 屏蔽，仅保留 item-id embedding
- `item_fusion`：additive（text 信号与 id 信号直接相加）

### 训练目标

```
loss = softmax cross entropy（in-batch negatives） / temperature τ
```

- L2 normalization，temperature τ = 0.15
- AdamW，lr = 0.001，weight\_decay = 1e-6，batch\_size = 4096
- best checkpoint by `valid_recall@50`（eval\_max\_users = 50,000）

---

## 核心实验结果

### Full Offline Evaluation（所有非冷启动用户）

| 模型 | Full Valid Recall@50 | Full Test Recall@50 | Test NDCG@50 | Test MRR@50 |
| --- | ---: | ---: | ---: | ---: |
| ItemCF | 0.140698 | 0.083570 | — | — |
| ID-only Two-Tower（20ep） | 0.092144 | 0.053198 | 0.021494 | 0.013542 |
| Text-enhanced（additive，20ep） | 0.093940 | 0.054561 | — | — |
| Mean Pooling Two-Tower（20ep） | 0.096309 | 0.061601 | 0.025176 | 0.016047 |
| Text + Mean Pooling τ=0.07（20ep） | 0.099628 | 0.066042 | 0.026751 | 0.016922 |
| **Text + Mean Pooling τ=0.15（20ep）** | **0.122606** | **0.076337** | **0.029987** | **0.018414** |

相比 ID-only baseline：
- Full test Recall@50：**+43.5%**（0.053198 → 0.076337）
- Full valid Recall@50：**+33.1%**（0.092144 → 0.122606）

> 注：ItemCF full test Recall@50 = 0.083570，高于当前最终主模型 0.076337，差距主要集中在头部物品（train 交互数 >100）。详见下方 Bucket Evaluation。

### Temperature Ablation（5epoch limited valid，eval\_max\_users=50K）

| τ | Valid Recall@50 | Best Epoch |
| ---: | ---: | ---: |
| 0.05 | 0.088840 | 2 |
| 0.07 | 0.099020 | 3 |
| 0.10 | 0.111880 | 5 |
| **0.15** | **0.117240** | **4** |
| 0.20 | 0.113700 | 4 |
| 0.30 | 0.102440 | 2 |

---

## 诊断实验

### Item Popularity Bucket Evaluation（Full Test Recall@50）

| Train 交互数桶 | Test targets | ItemCF | ID-only | Mean Pool | Text+MP τ=0.15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤5（长尾） | 35,045 | **0.040405** | 0.023284 | 0.024083 | 0.031046 |
| 6–20 | 87,067 | 0.047940 | 0.043748 | 0.046458 | **0.056933** |
| 21–100 | 161,718 | 0.060890 | 0.062918 | 0.063703 | **0.079564** |
| >100（头部） | 212,640 | **0.122522** | 0.054604 | 0.060233 | 0.083277 |

结论：
- ItemCF 在 ≤5 和 >100 桶上更强，头部物品（>100，占 42.8% targets）领先幅度最大。
- Text+MP τ=0.15 在 6–20 和 21–100 桶上超过 ItemCF。
- Text+MP τ=0.15 在所有桶上均超过 ID-only 和 Mean Pooling。

### User History Length Bucket Diagnostic（Full Test，Text+MP τ=0.15）

| Train history 长度 | Test users | Recall@50 |
| --- | ---: | ---: |
| 0 / 1–2 | 0 | n/a（5-core 过滤后无此桶） |
| 3–5 | 271,578 | **0.086826** |
| 6–20 | 191,450 | 0.067401 |
| >20 | 33,442 | 0.042312 |

结论：Recall@50 不随历史长度单调增加；长历史用户兴趣更多元，简单 mean pooling 将多兴趣平均为模糊向量，召回反而下降。

---

## 离线检索 Benchmark（Faiss）

基于 ID-only checkpoint（153,977 items，dim=64），单次 Top-50 检索延迟：

> 注：benchmark 使用 ID-only checkpoint 进行工程化仿真。最终主模型（Text + Mean Pooling）的 item 向量维度相同（dim=64），embedding 结构一致，延迟特性可直接迁移。

| 方法 | P50 latency | overlap@50 vs brute-force |
| --- | ---: | ---: |
| Brute-force exact | ~100 ms | 1.000 |
| Faiss FlatIP | ~6–10 ms | **1.000** |
| Faiss IVF-Flat（nlist=1024，nprobe=32） | **~0.2 ms** | 0.768 |

---

## 探索性实验（不纳入主模型结论）

### Text-based Hard Negative Mining Smoke

- HN 来源：frozen item text embeddings（sentence-transformer cosine similarity），top-5 per item
- 1epoch limited valid Recall@50：0.107840（vs baseline 0.107460，**+0.000380 absolute，约 +0.35% relative**，几乎持平）
- 链路通过；信号过弱，不继续

### Model-based Hard Negative Mining Smoke

- HN 来源：最终主模型导出的 item embeddings，Faiss IndexFlatIP top-50
- 1epoch limited valid Recall@50：0.105200（vs baseline 0.107460，**-0.002260 absolute，约 -2.1% relative**）
- 链路通过；epoch1 结果低于 baseline（推断：λ=0.1 在初始化阶段梯度方向冲突），不继续

---

## 环境与依赖

```bash
# 项目使用独立 .venv，复用系统 PyTorch/CUDA
/venv/main/bin/python -m venv .venv --system-site-packages

# 安装依赖
.venv/bin/python -m pip install \
  'datasets==2.17.0' 'huggingface_hub==0.36.2' \
  pyyaml pandas pyarrow sentence-transformers faiss-cpu
```

| 依赖 | 版本 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 3090 |
| faiss-cpu | 1.13.2 |

---

## 快速运行

### 1. 数据预处理

```bash
HF_HOME=/workspace/.hf_home HF_DATASETS_CACHE=/workspace/.hf_home/datasets \
  .venv/bin/python scripts/preprocess_amazon.py \
  --config configs/preprocess_movies_tv_5core.yaml
```

### 2. 生成 item text embeddings

```bash
.venv/bin/python scripts/build_item_text_embeddings.py \
  --config configs/preprocess_movies_tv_5core.yaml
```

### 3. 训练最终主模型（Text + Mean Pooling τ=0.15）

```bash
.venv/bin/python scripts/train_text_mean_pool_two_tower.py \
  --config configs/two_tower_movies_tv_5core_text_mean_pool_tau015_20epoch.yaml \
  2>&1 | tee logs/text_mean_pool_tau015_20ep.log
```

### 4. Full valid/test offline evaluation

```bash
.venv/bin/python scripts/train_text_mean_pool_two_tower.py \
  --config configs/two_tower_movies_tv_5core_text_mean_pool_tau015_20epoch.yaml \
  --eval_only --full_eval \
  --checkpoint outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt \
  --eval_output_dir outputs/text_mean_pool_tau015_20ep_full_eval
```

### 5. Faiss offline retrieval benchmark

```bash
.venv/bin/python scripts/benchmark_faiss_id_two_tower.py \
  --config configs/faiss_id_two_tower_clean_20epoch.yaml \
  2>&1 | tee logs/faiss_id_two_tower_clean_20epoch.log
```

---

## 目录结构

```text
amazon-two-tower/
├── configs/                          # 所有实验配置（YAML）
│   ├── preprocess_movies_tv_5core.yaml
│   ├── two_tower_movies_tv_5core_clean_20epoch.yaml          # ID-only baseline
│   ├── two_tower_movies_tv_5core_mean_pool_20epoch.yaml      # Mean Pooling
│   ├── two_tower_movies_tv_5core_text_mean_pool_tau015_20epoch.yaml  # 最终主模型
│   ├── two_tower_movies_tv_5core_text_mean_pool_hnm_smoke.yaml       # Text-based HNM
│   ├── two_tower_movies_tv_5core_text_mean_pool_model_hnm_smoke.yaml # Model-based HNM
│   └── faiss_id_two_tower_clean_20epoch.yaml
├── scripts/
│   ├── preprocess_amazon.py                              # 数据预处理
│   ├── build_item_text_embeddings.py                     # 生成 item text embeddings
│   ├── train_two_tower.py                                # ID-only Two-Tower
│   ├── train_mean_pool_two_tower.py                      # Mean Pooling user tower
│   ├── train_text_mean_pool_two_tower.py                 # 最终主模型（含 eval-only）
│   ├── train_text_mean_pool_hard_negative_smoke.py       # Text-based HNM smoke
│   ├── train_text_mean_pool_model_hard_negative_smoke.py # Model-based HNM smoke
│   ├── benchmark_faiss_id_two_tower.py                   # Faiss retrieval benchmark
│   ├── eval_cold_start_buckets.py                        # Item popularity bucket eval
│   └── eval_user_history_buckets.py                      # User history length diagnostic
├── docs/
│   ├── daily_logs/                   # 按日期记录实验过程
│   └── issue_log.md                  # 问题记录与诊断结论
├── data/                             # gitignore（processed data 不提交）
├── outputs/                          # gitignore（checkpoints / embeddings 不提交）
└── logs/                             # gitignore
```

---

## Git 注意事项

以下目录和文件不提交 Git：

```text
data/processed/
outputs/
logs/
.venv/
*.pt  *.npy
~/.ssh/*  私钥  token  credentials
```

SSH push 使用项目专用密钥：

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_amazon_two_tower -o IdentitiesOnly=yes' git push
```

# Runbook：Qwen3 文本 embedding 替换 MiniLM（P0 实验）

目标：把 item 文本向量来源从 `all-MiniLM-L6-v2`（384维）换成 `Qwen3-Embedding-0.6B`（1024维），
在**完全相同**的 time-aware Transformer 双塔配置下（seed=42, max_len=100, τ=0.15, patience=2），
对比 full test Recall@50 是否超过 MiniLM 基线 **0.103168**。

> 设计要点：`text_proj = nn.Linear(text_emb.shape[1], 64)` 自动适配输入维度，
> Text Semantic 召回路的 `load_text_embeddings` 只 assert `shape[0]`，也维度无关。
> **模型与召回代码零改动**，唯一变量是文本 embedding 来源。这是干净的单变量 ablation。

---

## 0. 依赖检查（Vast 服务器）

Qwen3-Embedding 需要较新依赖：

```bash
python -c "import sentence_transformers, transformers; print('st', sentence_transformers.__version__); print('tf', transformers.__version__)"
# 需要 sentence-transformers >= 2.7.0 且 transformers >= 4.51.0
pip install -U "sentence-transformers>=2.7.0" "transformers>=4.51.0"
```

显存：0.6B 模型 fp32 约 2.4GB + 激活。编码 153,977 条文本。
若 OOM，降 `--batch_size` 到 32 或更低。

---

## 1. 构建 Qwen3 embedding（两个变体）

### 1a. plain（无 instruction，文档侧标准做法，主对照）

```bash
python scripts/build_item_text_embeddings.py \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --output_dir outputs/item_text_embeddings_qwen3/movies_tv_5core \
  --batch_size 64 \
  --run_full
```

### 1b. instruct（带检索指令，prompt-design ablation，次对照）

```bash
python scripts/build_item_text_embeddings.py \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --output_dir outputs/item_text_embeddings_qwen3_instruct/movies_tv_5core \
  --batch_size 64 \
  --prompt "Instruct: Represent this movie/TV product for recommendation retrieval.
Product: " \
  --run_full
```

产物（每个变体目录下）：
- `item_text_embedding.npy`  shape=(153977, 1024) float32
- `item_has_text.npy`        与 MiniLM 完全一致（文本规则没变，has_text mask 不变）
- `item_text_meta.json`      含 model_name / prompt / embedding_dim=1024

> 验收：`item_has_text.npy` 的 true 数量应仍是 95,016（61.7%）。若不一致，停下报告。

---

## 2. 训练双塔（两个变体，各自独立 output_dir）

用 stability_sweep 入口：训练 + 自动 full eval，干净，不污染 canonical 文档。

### 2a. plain

```bash
python scripts/train_transformer_stability_sweep.py \
  --config configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_qwen3.yaml
```

### 2b. instruct

```bash
python scripts/train_transformer_stability_sweep.py \
  --config configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_qwen3_instruct.yaml
```

每个约 3-5 epoch 触发 early stopping（patience=2）。

---

## 3. 验收（明早看这几个文件）

full test Recall@50 在：

```text
outputs/text_timeaware_transformer_max100_qwen3_full_eval/eval_summary.json          → "full_test_recall@50"
outputs/text_timeaware_transformer_max100_qwen3_instruct_full_eval/eval_summary.json → "full_test_recall@50"
```

也汇总在：
```text
outputs/transformer_user_tower_investigation/stability_sweep/text_timeaware_transformer_max100_qwen3_result.json
outputs/transformer_user_tower_investigation/stability_sweep/text_timeaware_transformer_max100_qwen3_instruct_result.json
```

**判断基准**：

| 结果 | 结论 |
|---|---|
| 任一变体 full_test R@50 > 0.103168 | ✅ Qwen3 有正向增益，进入 Phase 2（4ch 重跑）+ 写简历 |
| 两变体都 ≈ 0.103168（±0.001） | ⚠️ embedding 升级在此瓶颈下不敏感，仍是可讲的负结果（text_proj 64维瓶颈吃掉了增益）|
| 明显低于 0.103168 | ❌ 检查 embedding 是否未归一化导致 scale 问题，报告 |

> 注意：所有数字是 offline full test，不是线上 A/B。不要自动改 README / 简历 / CLAUDE.md 第6节。
> 结果出来后等 Eddy 确认再决定是否替换 canonical。

---

## 4. Phase 2（可选，仅当 Phase 1 有正向增益时再做）

把胜出变体的 embedding 接进 4 路融合，看 12.52% 能否进一步提升：
- Text Semantic 召回路：改 `run_multichannel_valid_selected.py` 用的 text embedding 路径指向 qwen3 npy
- Transformer TT 召回路：用 qwen3 训练出的新 checkpoint
- 其余三路（ItemCF / Pop）不变
- valid 重新选权重 → test 跑一次

此步骤涉及多文件路径改动，留到 Phase 1 结果确认后单独开 runbook，今晚不做。

---

## 变更文件清单（本地已改，待 push 到服务器）

```text
scripts/build_item_text_embeddings.py    (+--prompt 支持, embedding_dim 从编码结果稳健读取, meta 记录 prompt)
configs/...max100_qwen3.yaml             (新, plain)
configs/...max100_qwen3_instruct.yaml    (新, instruct)
docs/runbooks/qwen3_text_embedding_runbook.md  (本文件)
```

无 outputs/ / *.npy / *.pt 改动。可安全 push 代码与配置。

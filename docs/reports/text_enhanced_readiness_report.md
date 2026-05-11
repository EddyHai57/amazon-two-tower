# Text-enhanced Item Tower Readiness Report

日期：2026-05-11

## 结论

当前 `text-enhanced item tower` 尚未实现。现有 clean Two-Tower pipeline 是 ID-only：

- dataloader 只读取 `user_idx` / `item_idx`。
- model 只包含 user embedding 和 item embedding。
- train / eval 逻辑没有 item title、description、category、meta embedding 或 feature fusion。
- processed clean data 中没有 item text/meta feature 文件。

因此当前不能直接启动 text-enhanced full training。下一步需要先实现 item feature loading、text/item encoder 或预计算 embedding 的接入方式，并保持 clean split evaluation protocol 不变。

## 已检查文件

- `configs/two_tower_movies_tv_5core.yaml`
- `configs/two_tower_movies_tv_5core_clean.yaml`
- `configs/two_tower_movies_tv_5core_clean_overnight.yaml`
- `configs/two_tower_movies_tv_5core_clean_20epoch.yaml`
- `scripts/train_two_tower.py`
- `scripts/preprocess_amazon.py`
- `data/processed/movies_tv_5core/train.parquet`
- `data/processed/movies_tv_5core/valid.parquet`
- `data/processed/movies_tv_5core/test.parquet`
- `data/processed/movies_tv_5core/stats.json`
- `data/processed/movies_tv_5core/item2id.json`
- `docs/data_notes.md`
- `outputs/inspection_movies_and_tv.md`

## 现有实现状态

`scripts/train_two_tower.py` 当前是 ID-only 实现：

- `TRAIN_COLUMNS = ["user_idx", "item_idx"]`
- `EVAL_COLUMNS = ["user_idx", "item_idx", "is_cold_item_for_eval"]`
- `InteractionDataset` 只返回 user/item ID。
- `IDOnlyTwoTower` 只使用 `nn.Embedding(num_users, embedding_dim)` 和 `nn.Embedding(num_items, embedding_dim)`。
- `encode_items` 只编码 item ID，没有 text/meta 输入。

现有 configs 也只覆盖 ID-only 参数，例如：

- `embedding_dim`
- `batch_size`
- `learning_rate`
- `temperature`
- `use_l2_norm`
- `eval_max_users`
- `output_dir`

未发现 text-enhanced 专用 config、text encoder 参数、item feature path、feature fusion 参数或预计算 text embedding path。

## 可用 item text/meta features

raw metadata 中有可用 item-side 字段，记录在 `docs/data_notes.md` 和 `outputs/inspection_movies_and_tv.md`：

- `title`
- `description`
- `features`
- `categories`
- `main_category`
- `store`
- `details`
- `parent_asin`

其中更适合作为第一版 item tower 文本/类别特征的是：

- `title`
- `description`
- `features`
- `categories`
- `main_category`

需要注意：

- `parent_asin` 是与 processed `item2id.json` 对齐的关键 join key。
- review 表中的 `title` / `text` 是用户评论内容，不应直接当作 item tower 的 item text 主来源。
- `average_rating` / `rating_number` 可能引入聚合统计泄漏风险，第一版不建议使用。

## processed clean data 状态

当前 processed clean data 不包含 item text/meta features：

- `train.parquet` columns：`user_id`, `parent_asin`, `user_idx`, `item_idx`, `rating`, `timestamp`
- `valid.parquet` / `test.parquet` columns：以上列加 `is_cold_item_for_eval`
- `stats.json` 不包含 item text/meta feature 路径或 schema。

因此 text-enhanced item tower 需要新增一个轻量 item feature artifact，例如：

- `data/processed/movies_tv_5core/item_features.parquet`

该文件应至少包含：

- `parent_asin`
- `item_idx`
- `title`
- `description`
- `features`
- `categories`
- `main_category`

是否生成该 artifact、如何清洗文本、是否预计算 embedding，属于下一步需要确认的设计/实现范围。

## clean evaluation protocol

现有 `scripts/train_two_tower.py` 的 eval-only protocol 与 clean split 要求一致：

- valid mask：exclude train history
- test mask：exclude train + valid history
- exclude `is_cold_item_for_eval=True`
- candidate item universe 使用 `n_items`
- diagnostics 包含 `target_item_in_candidate_range`

text-enhanced model 实现后应复用这套 eval 逻辑，只替换 item/user encoding，不改变 valid/test mask 口径。

## 缺失项与风险

- 缺少 item text/meta feature artifact 与加载逻辑。
- 缺少 text-enhanced config。
- 缺少 item text encoder / projection / fusion model。
- 缺少训练脚本或现有训练脚本的 text-enhanced 分支。
- 需要处理 metadata 缺失、重复 `parent_asin`、list/dict 字段序列化、文本长度截断。
- 需要避免使用可能泄漏未来整体流行度的 aggregate meta 字段。
- 需要确认 text representation 路线：简单 bag/category embedding、预计算 sentence embedding，或端到端 text encoder。
- 需要确认 GPU/memory 成本，先做 smoke test 再 full training。

## 建议 smoke-test command

当前尚无可运行的 text-enhanced smoke test。实现后建议先运行极小 smoke test，例如：

```bash
source .venv/bin/activate
python scripts/train_two_tower_text.py \
  --config configs/two_tower_movies_tv_5core_text_smoke.yaml
```

smoke test 目标只验证：

- item feature artifact 能加载并与 `item_idx` 对齐。
- dataloader 能返回 text/meta features。
- model forward / loss / valid eval 能跑通。
- clean valid mask 口径不变。

## 建议 full-training command

实现并通过 smoke test 后，再考虑 full training：

```bash
source .venv/bin/activate
python scripts/train_two_tower_text.py \
  --config configs/two_tower_movies_tv_5core_text.yaml \
  2>&1 | tee logs/two_tower_text.log
```

本次没有运行 smoke test，也没有启动 full training。

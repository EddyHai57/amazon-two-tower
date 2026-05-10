# Movies_and_TV 5-core Phase 1 训练数据审查报告

## 1. 本次分析目的

- Movies_and_TV 已被确定为 Phase 1 主实验数据集。
- 本报告用于确认 5-core preprocess 后的数据是否可以进入 ItemCF baseline。
- 本报告不训练模型，只检查训练数据、切分结果和评估可行性。

## 2. 数据来源与预处理规则

- dataset_name：`McAuley-Lab/Amazon-Reviews-2023`
- review_config：`raw_review_Movies_and_TV`
- 正样本定义：`rating >= 4`
- `rating < 4` 暂时不作为显式负样本。
- 不使用 `verified_purchase` 过滤。
- k-core：`user>=5,item>=5`
- 排序规则：`user_id`, `timestamp`, `parent_asin`, `original_row_idx`，并使用 stable sort。
- 切分规则：leave-one-out。
  - 每个用户最后一条交互作为 test。
  - 每个用户倒数第二条交互作为 valid。
  - 其余交互作为 train。
- cold item 处理策略：切分阶段保留并标记，评估时 `exclude_from_metric`。

## 3. preprocess 输出文件

输出目录：`data/processed/movies_tv_5core/`

- `train.parquet`：训练交互数据。
- `valid.parquet`：验证集 target，每行包含 `is_cold_item_for_eval`。
- `test.parquet`：测试集 target，每行包含 `is_cold_item_for_eval`。
- `user2id.json`：原始 `user_id` 到连续 `user_idx` 的映射。
- `item2id.json`：原始 `parent_asin` 到连续 `item_idx` 的映射。
- `id2user.json`：连续 `user_idx` 到原始 `user_id` 的反向映射。
- `id2item.json`：连续 `item_idx` 到原始 `parent_asin` 的反向映射。
- `stats.json`：预处理规模、cold item 统计、seed 和生成时间。
- `README.md`：该输出目录的规则说明。

## 4. 核心数据规模

基于 `data/processed/movies_tv_5core/stats.json`：

- k-core 后总 interaction：5413083
- user 数：505425
- item 数：155957
- train interaction：4402233
- valid interaction：505425
- test interaction：505425
- preprocess_seed：42
- preprocess_timestamp：`2026-05-10T06:47:38.950931+00:00`

## 5. cold item 检查

- valid cold item 数量：337
- valid cold item 比例：0.0667%
- test cold item 数量：955
- test cold item 比例：0.1889%

cold item 不在切分阶段删除，是为了完整保留原始 leave-one-out 结果，避免预处理阶段悄悄改变数据分布。

评估时需要 `exclude_from_metric`，因为 train 中从未出现过的 target item 对 ItemCF 不可召回。如果把这些样本纳入 Recall@K、NDCG@K、MRR 等指标，会让 ItemCF 指标出现假性偏低。

ItemCF 和 Two-Tower 后续都应使用同一套 non-cold eval 样本，保证可比性。

## 6. 与 All_Beauty 的对比

- All_Beauty 在 `user>=5,item>=5` 后只剩 293 interactions。
- Movies_and_TV 在 `user>=5,item>=5` 后有 5413083 interactions。
- 因此 All_Beauty 只适合 Phase 0 工程验证，Movies_and_TV 可以作为 Phase 1 主实验数据集。

## 7. Phase 1 训练可行性判断

- 可以进入 ItemCF baseline。
- 可以进入 ID-only Two-Tower baseline。
- cold item 比例很低，当前不需要在预处理阶段删除 cold item。
- 后续评估时应对 valid/test 中 `is_cold_item_for_eval=True` 的样本执行 `exclude_from_metric`。
- 建议先跑 5-core ItemCF，获得第一组 Recall@50 / NDCG@50 / MRR@50 基线数字。

## 8. 下一步建议

- 进入 ItemCF baseline。
- 不要直接开始复杂双塔。
- 先获得第一个 Recall@50 / NDCG@50 / MRR@50 基线数字。

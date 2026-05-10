# 决策日志

- Amazon 项目必须与 tianchi-two-tower 分开，放在独立的同级项目中。
- Codex 负责实现；Eddy 负责做设计决策。
- 业务参数必须放在配置文件中，不能硬编码在脚本里。
- 第一阶段必须先完成 ItemCF 和仅使用 ID 的双塔模型，再进入文本嵌入或消融实验。
- 数据加载兼容性相关变更必须经过 Eddy 确认。
- 项目文档默认使用简体中文。

## Decision 编号：DECISION-20260509-001

### 决策时间

2026-05-09

### 决策主题

All_Beauty 是否作为 Phase 1 主实验数据集。

### 可选方案

- A. 继续使用 All_Beauty 完整跑 Phase 1。
- B. All_Beauty 只作为 Phase 0 工程验证，Phase 1 切换到更合适的大类目。

### 最终选择

B。

### 选择原因

All_Beauty 在最宽松的 `user>=3,item>=3` k-core 后仅剩 8657 条 interaction，interaction 保留率只有 1.73%。93.28% 的用户只有 1 条正向交互，用户兴趣信号极弱，不适合作为简历主实验数据集。

### 对实验可比性的影响

- All_Beauty 不再作为 Phase 1 主实验数据集，因此后续简历数字不能与 All_Beauty 的模拟结果混用。
- Phase 1 主数据集需要在候选品类对比后确定，后续 ItemCF 和 ID-only Two-Tower 必须在同一数据切分上对比。

### 对后续开发的影响

- 不再继续在 All_Beauty 上投入完整 preprocess、ItemCF、Two-Tower 训练链路。
- All_Beauty 保留为 Phase 0 工程验证和数据分析流程验证样例。
- 下一步扩展 `analyze_interactions.py` 支持多品类配置。
- 明天优先分析候选品类：Electronics、Video_Games、Movies_and_TV。
- 生成 `category_comparison.md` 后，再决定 Phase 1 主数据集。

## Decision 编号：DECISION-20260509-002

### 决策时间

2026-05-09

### 决策主题

Amazon baseline deadline 调整。

### 原计划

5月10日24:00前跑通 Amazon 数据上的 baseline。

### 调整后计划

- 5月10日上午：扩展多品类分析能力，跑候选品类对比。
- 5月10日下午：基于对比表选择 Phase 1 主数据集。
- 5月11日24:00：跑通新品类 ItemCF + 最简 ID-only Two-Tower baseline。
- 5月12日：文本 embedding 对比实验。
- 5月13日：温度扫描 + 简历数字替换。
- 5月14日：简历定稿。
- 5月15日：投递节点不再延后。

### 选择原因

All_Beauty 数据稀疏问题导致需要切换数据集，5月10日24:00 baseline deadline 不再现实。为了保留最终投递节点，需要把数据集选择和 baseline 跑通拆成两个连续步骤。

### 对实验可比性的影响

候选品类必须先通过同一套 interaction analysis 指标比较，再确定 Phase 1 主数据集；确定后再固定 preprocessing、ItemCF 和 ID-only Two-Tower 的共同 test set。

### 对后续开发的影响

明天第一步是生成候选品类对比报告，而不是直接下载 Electronics 并开始训练。

## Decision 编号：DECISION-20260510-001

### 决策时间

2026-05-10

### 决策主题

确定 Movies_and_TV 作为 Phase 1 主实验数据集。

### 背景

All_Beauty 已验证不可用于 Phase 1 主实验数据集。它在最宽松的 `user>=3,item>=3` k-core 后只剩 8657 条 interaction，只适合作为 Phase 0 工程验证数据集。因此需要从更合适的大类目中选择 Phase 1 主实验数据集。

### 候选方案

- Video_Games
- Movies_and_TV
- Electronics

### 最终选择

选择 Movies_and_TV 作为 Phase 1 主实验数据集。

### 选择原因

- Movies_and_TV 在 `k-core(3,3)` 后仍有 8025936 条 interaction，远高于 Video_Games 和 All_Beauty。
- Movies_and_TV 在 `k-core(5,5)` 后仍有 5413083 条 interaction，足够支撑 ItemCF 和 ID-only Two-Tower baseline。
- Movies_and_TV 的 leave-one-out 可用 user 达到 1190601。
- Movies_and_TV 文本内容天然丰富，适合 5月12日做 `title` / `description` embedding 对比。
- Movies_and_TV 的 `analyze_interactions.py` 已 full_load 成功，工程风险可控。

### 备选说明

- Video_Games 可作为 fallback 数据集。它的 `k-core(3,3)` 后剩余 1165395 条 interaction，工程风险低，但规模小于 Movies_and_TV。
- Electronics 简历叙事接近电商，但 inspection 阶段 full_load 下载到 14.2G / 22.6G 时发生 `HTTPSConnectionPool Read timed out`，已切换 `streaming_fallback`，暂不继续投入。

### 对实验可比性的影响

Phase 1 后续 preprocess、ItemCF、ID-only Two-Tower、text embedding 对比均基于 Movies_and_TV。所有 baseline 必须在同一套 Movies_and_TV train/valid/test 切分上比较。

### 对后续开发的影响

- All_Beauty 仅保留为 Phase 0 工程验证数据集。
- Video_Games 保留为 fallback 数据集。
- Electronics 暂缓，不作为 Phase 1 主数据集。
- 下一步进入 Movies_and_TV preprocess 准备，但具体预处理规则仍需 Eddy 确认。

## Decision 编号：DECISION-20260510-002

### 决策时间

2026-05-10

### 决策主题

Movies_and_TV Phase 1 preprocess 规则。

### 已确认规则

- 正样本定义为 `rating >= 4`。
- `rating < 4` 暂时不作为显式负样本。
- `verified_purchase` 暂时不参与过滤。
- 同时支持两个 k-core 版本：
  - 主实验版本：`user>=5,item>=5`，输出目录为 `data/processed/movies_tv_5core/`
  - 对照版本：`user>=3,item>=3`，输出目录为 `data/processed/movies_tv_3core/`
- 每个用户按 `user_id`, `timestamp`, `parent_asin`, `original_row_idx` 稳定排序后做 leave-one-out 切分。
- 最后一条交互作为 test，倒数第二条作为 valid，其余作为 train。
- 切分阶段不删除 cold target item。
- valid/test 中额外标记 `is_cold_item_for_eval`。
- 评估策略为 `exclude_from_test_metric`：cold target item 不参与 Recall@K、NDCG@K、MRR 等指标计算，但在 `stats.json` 中记录数量和比例。
- `user2id`、`item2id`、`id2user`、`id2item` 只基于 k-core 后的全量交互生成，并在 train/valid/test 中共用。
- `seed` 固定为 42。

### 对实验可比性的影响

5-core 是 Phase 1 主实验版本，后续 ItemCF、ID-only Two-Tower 和文本 embedding 对比都应优先使用 `movies_tv_5core`。3-core 只作为 k-core 阈值影响的对照版本。cold item 评估策略在 ItemCF 和 Two-Tower 中必须保持一致。

### 对后续开发的影响

下一步可以先运行 5-core 预处理，检查输出规模、cold item 比例和文件完整性；确认无误后再运行 3-core 对照版本。

## Decision 编号：DECISION-20260510-003

### 决策时间

2026-05-10

### 决策主题

确认 Movies_and_TV 5-core 作为 Phase 1 主实验版本，并完成 ItemCF baseline。

### 背景

Movies_and_TV 已被确定为 Phase 1 主实验数据集。5-core preprocess 已完成，并生成统一的 train/valid/test 切分，可用于 ItemCF、ID-only Two-Tower 和后续 Text-enhanced Two-Tower 的公平比较。

### 核心数据规模

- users：505425
- items：155957
- interactions：5413083
- test cold item ratio：0.1889%，可接受

### ItemCF baseline 结果

- ItemCF baseline 已完成。
- `Recall@50 = 0.083559`
- `NDCG@50 = 0.034553`
- `MRR@50 = 0.021777`

### 最终确认

确认 `Movies_and_TV 5-core` 作为 Phase 1 主实验版本。后续 ID-only Two-Tower 需要至少超过 ItemCF `Recall@50=0.083559`，才说明深度召回相对传统共现 baseline 有增益。

### 对实验可比性的影响

ItemCF、ID-only Two-Tower 和 Text-enhanced Two-Tower 必须使用同一套 `data/processed/movies_tv_5core/` 数据切分，并保持一致的 cold item 排除策略和 seen item 过滤口径。

后续 Two-Tower 实验必须记录 loss curve 与 valid metrics，用于判断模型是否真实学习，而不是只看最终 test 指标。

### 对后续开发的影响

明日优先任务是 ID-only Two-Tower baseline。暂时不进入负采样实验、LogQ、温度扫描或文本 embedding。

## Decision 编号：DECISION-20260510-004

### 决策时间

2026-05-10

### 决策主题

Evaluation comparison must be aligned on split and eval scope。

### 背景

Movies_and_TV 5-core ItemCF baseline 的评估口径已通过 `outputs/itemcf_movies_tv_5core/` 下的结果文件确认。当前 ID-only Two-Tower overnight 结果也已完成，但二者的 split 和 eval scope 不一致。

### 已确认事实

- ItemCF 5-core baseline `Recall@50=0.083559` 已确认为 test split full non-cold eval。
- ItemCF test 总数为 505425，cold test targets 为 955，eval users 为 504470。
- Two-Tower overnight 当前结果是 valid split 的 50000 users subset。
- Two-Tower valid subset `Recall@50=0.08716` 只能说明趋势，不能作为最终 test 结论。

### 决策内容

后续所有 baseline 对比必须明确记录：

- split：valid / test
- eval users 数量
- 是否 full eval 或 subset eval
- cold item exclude 口径
- seen item mask 口径
- 当前指标是否可直接与其他 baseline 对齐比较

### 对实验可比性的影响

ItemCF 和 Two-Tower 的最终可比数字必须在同一 split 和同一 eval scope 下比较。由于 ItemCF 已经是 test full non-cold eval，Two-Tower 下一步必须跑 full valid / full test eval，尤其是 full test eval，才能和 ItemCF test full eval 对齐。

### 对后续开发的影响

下一步先检查 `scripts/train_two_tower.py` 是否支持 eval-only；若支持，则加载 `outputs/two_tower_movies_tv_5core_overnight/checkpoints/best_model.pt` 跑 full valid / full test evaluation。暂时不进入 text embedding、LogQ、temperature sweep、negative sampling ablation 或 30 epoch training。

## Decision 编号：DECISION-20260510-005

### 决策时间

2026-05-10

### 决策主题

Movies_and_TV preprocess 在正样本过滤前按 `(user_id, parent_asin)` 去重。

### 背景

Two-Tower full valid / full test evaluation 出现异常 gap，进一步诊断发现旧 5-core 数据中存在同一 `(user_idx, item_idx)` 跨 train / valid / test split：

- `test_in_train=2637`
- `test_in_valid=10733`
- `valid_in_train=2641`
- `user_item_pairs_appearing_in_multiple_splits=10737`

### 可选方案

A. 保持旧 preprocess，不处理重复 user-item。

B. 在 `rating >= threshold` 之后去重。

C. 在 `rating >= threshold` 之前去重，对同一 `(user_id, parent_asin)` 保留最新一条 interaction。

### 最终选择

C。

### 选择原因

同一用户对同一 item 的 rating 可能随时间变化。先保留最新 interaction，再判断是否满足 `rating >= threshold`，可以让正样本标签反映该用户对该 item 的最新反馈。

### 对实验可比性的影响

旧的 5-core preprocess、ItemCF 和 Two-Tower 结果不再作为正式可比 baseline。后续正式 baseline 应基于 clean 5-core 数据重新生成。

### 对后续开发的影响

- `scripts/preprocess_amazon.py` 已加入去重逻辑。
- `data/processed/movies_tv_5core/` 已重跑生成 clean split。
- 后续需要先重跑 clean ItemCF baseline，再重新训练 ID-only Two-Tower。
- 3-core 对照版本后续也应使用相同去重策略重新生成。

## Decision 编号：DECISION-20260510-006

### 决策时间

2026-05-10

### 决策主题

Clean test evaluation 使用 `train + valid seen` mask 口径。

### 背景

旧 ItemCF test baseline 只记录为过滤 train seen items；Two-Tower eval-only test 口径已经使用 train + valid seen items。为了让 clean ItemCF 和后续 clean Two-Tower baseline 对齐，需要统一 test seen mask。

### 可选方案

A. test eval 只过滤 train seen items。

B. test eval 过滤 train + valid seen items，并允许当前 test target 作为候选。

### 最终选择

B。

### 选择原因

leave-one-out 流程中 valid 是 test 之前已发生的 held-out interaction。做 test evaluation 时，将 train + valid 作为用户历史更符合时间顺序，也能避免推荐已经在 valid 中出现过的 item。

### 对实验可比性的影响

后续 clean ItemCF 和 clean Two-Tower 的 test 指标必须统一使用 `train + valid seen` mask，并记录 `eval_seen_filter=train_valid`。旧 ItemCF / Two-Tower 指标不再作为正式 baseline。

### 对后续开发的影响

- `scripts/run_itemcf.py` 已支持 `eval_seen_filter`。
- `configs/itemcf_movies_tv_5core_clean.yaml` 使用 `eval_seen_filter: train_valid`。
- clean ItemCF baseline 已生成到 `outputs/itemcf_movies_tv_5core_clean/`。
- 下一步应重跑 clean ID-only Two-Tower 5 epoch baseline。

## Decision 编号：DECISION-20260510-007

### 决策时间

2026-05-10

### 决策主题

以 clean 5-core 数据重建后的结果作为后续正式 baseline 口径。

### 背景

旧 Movies_and_TV 5-core 数据存在重复 `(user_idx, item_idx)` 跨 train / valid / test split 的问题。该问题已通过 preprocess 去重修复，clean split 已验证通过。

### 已确认事实

- clean 5-core 数据：
  - users：497449
  - items：153977
  - interactions：5314336
  - train interactions：4319438
  - valid interactions：497449
  - test interactions：497449
- clean split 验证结果：
  - `test_in_train=0`
  - `test_in_valid=0`
  - `valid_in_train=0`
  - `valid_test_same_target_users=0`
  - `user_item_pairs_appearing_in_multiple_splits=0`
- clean ItemCF test full non-cold eval：
  - `eval_seen_filter=train_valid`
  - `Recall@50=0.083570`
  - `NDCG@50=0.036254`
  - `MRR@50=0.023999`
- clean ID-only Two-Tower 5 epoch valid subset：
  - best_epoch：5
  - `Recall@50=0.081220`
  - `NDCG@50=0.036484`
  - `MRR@50=0.024987`

### 决策内容

旧 ItemCF / Two-Tower 指标不再作为正式 baseline。后续正式实验、报告和简历数字必须基于 clean 5-core 数据和统一的 `train_valid` test seen mask 口径。

### 对实验可比性的影响

clean ItemCF `Recall@50=0.083570` 是当前正式传统 baseline。clean Two-Tower 目前只有 50000 valid users 子集结果，不能直接与 clean ItemCF test full eval 做最终比较。

### 对后续开发的影响

下一步只做 clean Two-Tower eval-only full valid / full test evaluation，加载：

```text
outputs/two_tower_movies_tv_5core_clean_overnight/checkpoints/best_model.pt
```

建议输出目录：

```text
outputs/two_tower_movies_tv_5core_clean_full_eval/
```

在得到 full test 指标前，不进入 text embedding、LogQ、temperature sweep、negative sampling ablation 或 20/25/30 epoch full training。

## Decision 编号：DECISION-20260510-008

### 决策时间

2026-05-10

### 决策主题

Clean Two-Tower full eval 后先做 valid-test gap diagnosis，不直接长训或加复杂模块。

### 背景

clean ID-only Two-Tower 5 epoch checkpoint 已完成 full valid / full test eval。训练链路健康，但 full test 指标明显低于 full valid 和 clean ItemCF baseline。

### 已确认事实

- clean Two-Tower full valid：
  - `Recall@50=0.081591`
  - `NDCG@50=0.036014`
  - `MRR@50=0.024325`
- clean Two-Tower full test：
  - `Recall@50=0.046746`
  - `NDCG@50=0.019344`
  - `MRR@50=0.012461`
- clean ItemCF test full non-cold eval：
  - `Recall@50=0.083570`
  - `NDCG@50=0.036254`
  - `MRR@50=0.023999`

### 决策内容

clean ID-only Two-Tower 已经完成 5 epoch baseline 和 full eval，但当前不能写成超过 ItemCF。下一步先做 clean valid-test gap diagnosis，而不是直接启动 20/25/30 epoch full training，也不进入 text embedding、LogQ、temperature sweep 或 negative sampling ablation。

### 对实验可比性的影响

clean ItemCF `Recall@50=0.083570` 仍是当前正式传统 baseline。clean Two-Tower full test `Recall@50=0.046746` 是当前 ID-only baseline 的诊断结果，但需要进一步解释 valid-test gap。

### 对后续开发的影响

明天优先做：

- valid/test target popularity 分布。
- 按 item popularity 分桶的 `Recall@50`。
- 按 user history length 分桶的 `Recall@50`。
- ItemCF hit 但 Two-Tower miss 的样本类型。
- Two-Tower hit 但 ItemCF miss 的样本类型。

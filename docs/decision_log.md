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

## Decision 编号：DECISION-20260511-001

### 决策时间

2026-05-11

### 决策主题

clean ItemCF valid/test gap 确认后，下一步先做 clean ID-only Two-Tower 20 epoch baseline。

### 背景

clean ID-only Two-Tower 5 epoch full eval 存在明显 valid-test gap，并且 full test 指标低于 clean ItemCF test baseline。随后补充运行 clean ItemCF full valid eval，用于判断 valid-test gap 是否是 Two-Tower-specific evaluation bug。

### 已确认事实

- clean ItemCF：
  - valid `Recall@50=0.140698`
  - test `Recall@50=0.083570`
  - relative drop：约 40.60%
- clean ID-only Two-Tower 5 epoch：
  - full valid `Recall@50=0.081591`
  - full test `Recall@50=0.046746`
  - relative drop：约 42.71%
- ItemCF 和 Two-Tower 在 clean split 上都有同量级 valid-test gap。

### 决策内容

- valid-test gap 不再作为 Two-Tower-specific evaluation bug 继续追。
- 当前 open issue 调整为：`ID-only Two-Tower test baseline underperforms ItemCF`。
- 下一步先做 clean ID-only Two-Tower 20 epoch baseline，用于验证 5 epoch 欠拟合假设。
- 暂不做 text-enhanced、LogQ、temperature sweep 或 negative sampling。

### 对实验可比性的影响

20 epoch baseline 仍应使用 clean Movies_and_TV 5-core 数据、同一 train/valid/test split，以及与 clean Two-Tower full eval 一致的 seen item mask 和 cold item 排除口径。该运行用于判断训练轮数是否是 ID-only Two-Tower underperform 的主要原因，不应与 text-enhanced 或其他增强实验混在同一结论中。

### 对后续开发的影响

下一步只准备并启动 clean ID-only Two-Tower 20 epoch baseline。20 epoch 结果出来前，不进入 text-enhanced item tower、LogQ、temperature sweep 或 negative sampling。

## Decision 编号：DECISION-20260511-002

### 决策时间

2026-05-11

### 决策主题

M6 text-enhanced item tower 的 fusion 策略：ID backbone + projected text + has_text mask，而非 pure text tower。

### 背景

M5.5 pure text retrieval 显示 pure text 检索效果（valid Recall@50=0.028160）明显弱于 ItemCF（valid Recall@50=0.140698）和 ID-only Two-Tower（valid Recall@50=0.092144）。根因是 38.3% item 缺少真实文本，fallback embedding 无语义；且 54.5% 的 test target item 属于 has_text=0。

### 可选方案

- A. 纯 text tower 替代 ID embedding。
- B. ID embedding 主干 + projected text embedding + has_text mask 屏蔽无文本 item。

### 最终选择

B。

### 选择原因

- Pure text（方案 A）对 has_text=0 item 几乎无召回能力（Recall@50≈0.0016-0.0019），在本数据集上不可行。
- ID embedding 是经过训练的协同过滤信号，是主干。
- Text embedding 作为辅助 side feature，对 has_text=1 item 提供语义信号。
- has_text mask 保证 has_text=0 item 的 text path 输出为 0，不引入噪声。

### 对实验可比性的影响

M6 的 item tower 与 ID-only Two-Tower（ID 主干，无 text）相比，text fusion 只在 has_text=1 item 上起作用，对比是受控的。

### 对后续开发的影响

M6.1 已按此方案实现 smoke test，M6.2 正式训练沿用同一架构，不再讨论 pure text 替代方案。

## Decision 编号：DECISION-20260511-003

### 决策时间

2026-05-11

### 决策主题

Item text embedding 使用 all-MiniLM-L6-v2，dim=384，作为 frozen item-side feature。

### 背景

M5.2 全量 embedding 已使用 sentence-transformers/all-MiniLM-L6-v2 生成，shape=(153977, 384)，所有 item 均为 unit-norm，无零行（fallback 为 parent_asin）。

### 最终选择

冻结 all-MiniLM-L6-v2 输出的 embedding，不做 fine-tune。用 Linear(384→64) projection 层将其对齐到 ID embedding 维度。Projection 层是可训练的，embedding 本身为 frozen buffer（persistent=False）。

### 选择原因

- all-MiniLM-L6-v2 是轻量、高质量的 sentence-level encoder，适合 title + description 文本。
- 全量 embedding 已预先生成（236 MB npy），作为 frozen buffer 加载，无需训练时在线 encode。
- persistent=False 避免 checkpoint 体积膨胀（text buffer 不写入 state_dict，从文件重加载）。
- Projection 层可训练，允许模型在 InfoNCE 目标下学习如何利用 text signal。

### 对实验可比性的影响

M6.2 full eval 可在同一候选集上与 ID-only Two-Tower 和 pure text retrieval 做三方对比：

| 方法 | text 使用方式 |
| --- | --- |
| pure text retrieval | 直接用 text embedding 做检索 |
| ID-only Two-Tower | 无 text |
| text-enhanced Two-Tower (M6) | ID + frozen text proj + has_text mask |

### 对后续开发的影响

text embedding 文件路径已写入配置：`outputs/item_text_embeddings/movies_tv_5core/item_text_embedding.npy`。若后续切换更大的 encoder，需重新生成 npy 并更新 config。

## Decision 编号：DECISION-20260511-004

### 决策时间

2026-05-11

### 决策主题

M6.2 正式训练先只看 valid，暂不启动 test eval。

### 背景

M6.1 smoke test 通过。正式 20 epoch 训练配置已就绪。

### 决策内容

- M6.2 正式训练（20 epoch）期间，每 epoch 只在 valid set 上评估（`eval_max_users=50000`）。
- 训练完成后先基于 valid 指标确认最优 checkpoint。
- test eval 不在训练过程中自动运行，避免过拟合 test split 的早停判断。
- test eval 单独作为 M7 阶段执行，并与 ItemCF 和 ID-only Two-Tower 做三方对比。

### 选择原因

clean Two-Tower 20 epoch 训练过程中，valid Recall@50 在 epoch 18 最优，epoch 19/20 略有回落。text-enhanced 模型也可能出现类似平台期，应通过 valid 监控，而非提前用 test 锚定 checkpoint。

### 对实验可比性的影响

M6.2 对应 `save_best_by: valid_recall@50`（已写入 20epoch config）。最终 test eval 将在 M7 阶段统一用 eval-only 模式跑全量用户，口径与 clean ItemCF 和 ID-only Two-Tower 对齐。

### 对后续开发的影响

M7 三方对比命令（待 M6.2 完成后执行，不在本次安排范围）：

```bash
# Text-enhanced Two-Tower full eval（M7，待执行）
.venv/bin/python scripts/train_text_two_tower.py \
  --config configs/text_two_tower_movies_tv_5core_20epoch.yaml \
  --eval_only \
  --checkpoint outputs/text_two_tower_movies_tv_5core_20epoch/checkpoints/best_model.pt \
  --eval_split both
```

## Decision 编号：DECISION-20260511-005

### 决策时间

2026-05-11

### 决策主题

M6 additive residual text-enhanced Two-Tower v1 作为第一个干净 text-enhanced baseline 保留，暂不继续调参。

### 背景

M6.2 additive residual text-enhanced Two-Tower 20 epoch 训练和 full eval（M6.3/M6.4）已完成。

已确认指标：

```text
Full valid Recall@50 = 0.093940
Full test Recall@50  = 0.054561（borderline γ，差距 0.000439 至 β 阈值）
text 相比 ID-only：valid +0.001796（+1.95%），test +0.001363（+2.56%）
has_text=1 target test Recall@50 = 0.070464
has_text=0 target test Recall@50 = 0.041407
```

三方对比（Recall@50）：

| 方法 | valid | test |
| --- | ---: | ---: |
| clean ItemCF | 0.140698 | 0.083570 |
| ID-only Two-Tower 20ep | 0.092144 | 0.053198 |
| Additive text-enhanced 20ep | 0.093940 | 0.054561 |

### 可选方案

- A. 继续调参（更大 projection dim、更多 epoch、unfreezing text encoder），争取突破 β 阈值。
- B. 接受当前结果作为 v1 基线，记录 text 有效但覆盖率限制了增益，后续有需要再迭代。

### 最终选择

B。

### 选择原因

- additive v1 架构干净，text path 和 ID path 对称，ablation 可解释。
- test 未超过 β 阈值（差距仅 0.000439）主要是因为 54.7% 的 test target 无真实文本（has_text=0），这是数据限制，不是模型架构错误。
- 当前 has_text=1 group Recall@50=0.070464，说明 text signal 对有文本的 item 确实有效。
- 继续调参存在过拟合 valid 或拟合 test 的风险，且预期收益有限。
- 保持 additive v1 作为干净基线，更有利于后续增量实验（扩充 text 覆盖率、LogQ、popularity correction）与 v1 形成可比对照。

### 对实验可比性的影响

- additive v1 checkpoint 和输出目录固定为：
  - `outputs/text_two_tower_additive_movies_tv_5core_20epoch/`
- 后续任何新变体必须使用新的 `output_dir`，不覆盖 v1 输出。
- 三方对比表（ItemCF / ID-only / additive text-enhanced）已记录为 M6 阶段正式结果，不因后续实验回溯修改。

### 对后续开发的影响

- 当前不进入 text encoder fine-tune、更大 projection dim、LogQ、temperature sweep 或 popularity correction。
- 后续实验方向留待 Eddy 确认优先级。
- additive v1 的 has_text split 和 popularity bucket 输出（6 个 JSON/md 文件）作为后续增强实验的对照参考。

## Decision 编号：DECISION-20260511-006

### 决策时间

2026-05-11

### 决策主题

D1+D2 诊断完成后停止 5/15 前模型调参，当前诊断证据已足够支撑简历叙事。

### 背景

D1 ID-only has_text split 和 D2 三模型 popularity bucket 矩阵已完成。

D1 关键证据：

```text
has_text=1: ID-only R@50=0.068173, text-enhanced R@50=0.070464, delta=+0.002291
has_text=0: ID-only R@50=0.040811, text-enhanced R@50=0.041407, delta=+0.000596
```

D2 关键证据：

```text
>100 bucket (42.8% targets): ItemCF R@50=0.122522 vs Two-Tower ~0.055（ItemCF 约 2.2×）
21–100 bucket (32.6% targets): ID-only R@50=0.062918, text-enh R@50=0.064433（略优于 ItemCF 0.060890）
text-enhanced 在所有 bucket 均小幅优于 ID-only
```

### 可选方案

- A. 继续调参：text_proj_dim=128、concat+MLP v2、hybrid retrieval，目标突破 β 阈值。
- B. 接受当前 D1+D2 诊断结果，停止模型实验，进入日志整理和简历叙事提炼阶段。

### 最终选择

B。

### 选择原因

- D1+D2 提供了足够的证据用于简历叙事：text 对有元数据 item 有效（+3.4%），ItemCF 对热门 item 优势显著，Two-Tower 对中等热度 item 有泛化优势。
- 继续 text_proj_dim / concat+MLP / hybrid 调参在 5/15 前 ROI 低，且存在过拟合风险。
- 当前叙事已完整：ItemCF baseline → ID-only Two-Tower → additive text-enhanced → 结构性 D1+D2 诊断分析。
- 进一步调参留待 5/15 后根据反馈决定。

### 对实验可比性的影响

- 已有的三组 overall Recall@50 + has_text split + popularity bucket 形成完整对比体系，不需要再引入新 baseline。
- 5/15 后新实验可在此基础上增量叠加，不影响当前记录的 baseline 数字。

### 对后续开发的影响

- 5/15 前：整理 docs、代码 push、提炼简历叙事。
- 5/15 后：视反馈决定是否继续 popularity correction / hybrid retrieval / text 覆盖率扩充。
- 不做 text_proj_dim=128、concat+MLP v2、LogQ、temperature sweep 或 negative sampling 实验。

---

## Decision 编号：DECISION-20260513-001

### 决策时间

2026-05-13

### 决策主题

Hard Negative Mining 全系列（4 实验）关闭，标记为 future work，不进入主线。

### 背景

5/13–5/14 完成 4 次 HNM smoke test（1 epoch limited valid）：

```text
Text-based HNM（λ=1.0）:   0.107840  +0.35%  vs baseline 0.107460
Model-based HNM（λ=0.1）:  0.105200  -2.10%
Semi-hard λ=0.03:           0.108840  +1.28%  ← 系列最优
Semi-hard λ=0.01:           0.107840  +0.35%
```

### 可选方案

- A. 继续 semi-hard λ=0.03，全量训练（约 4.3× 单 epoch 成本）
- B. 关闭 HNM 系列，标记 future work，不进入主线

### 最终选择

B。

### 选择原因

- 最优结果（semi-hard λ=0.03 +1.28%）信号过弱，1 epoch smoke 难以外推至 full training。
- 4.3× 训练成本（每 batch 需 Faiss nearest neighbor 采样）在当前阶段 ROI 不合理。
- Bottleneck 是采样策略（每 batch 只更新少量 HN），而非 loss 权重调整，进一步调 λ 预期收益有限。
- 时间优先：进入多路召回融合和文档整理阶段。

### 对实验可比性的影响

HNM 系列作为"探索性实验"记录，不影响 final model（time-decay mean pool τ=0.15）的 canonical 数字。

### 对后续开发的影响

HNM 标记 future work；若未来有充分训练预算，semi-hard λ=0.03 + full 20 epoch 可作为候选实验。

---

## Decision 编号：DECISION-20260515-001

### 决策时间

2026-05-15

### 决策主题

多路召回最终方案确立为 valid-selected 四通路加权 RRF（Recall@50 = 0.104776）。

### 背景

完成 v1/v2/v3/valid-selected 系列实验：

```text
v1 2ch RRF（ICF+TT）:                     0.096727
v2 4ch equal RRF（+Text+Pop，均等权重）:  0.108766  但 avg_pop 暴涨 ×6.2，头部偏移严重
v3 4ch wRRF（text=0.3, pop=0.5）full test: 0.103384
valid-selected 4ch wRRF（k=100）:          0.104776  ← 最终采用
```

valid-selected 方案：权重在 valid 60-config grid search 选出（Pareto 规则），test set 仅运行一次，无 test-tuning。

### 可选方案

- A. 继续调权重，尝试更多 k / 通路组合
- B. 锁定 valid-selected 结果（0.104776）为最终 pipeline，不再变更

### 最终选择

B。权重冻结：ICF=1.0, TT=1.0, Text=0.3, Pop=0.5, k=100。

### 选择原因

- valid-selected 方案权重选择无 test 污染，方法论严格。
- 进一步调参会增加 test-tuning 风险，已无实质收益空间。
- 当前结果（+25.4% vs ItemCF 单路）已足够支撑简历叙事。

### 对实验可比性的影响

v3 full test（0.103384）和 valid-selected（0.104776）均保留，前者仅作诊断参考，后者为正式主结论。

### 对后续开发的影响

multichannel pipeline 不再变更；后续若扩展通路，需重新做 valid grid search。

---

## Decision 编号：DECISION-20260519-001

### 决策时间

2026-05-19

### 决策主题

Attention pooling smoke 未达阈值，立即停止，final model（time-decay mean pool）保持不变。

### 背景

对 time-decay 和 attention pooling 进行 3 epoch paired smoke（50K limited valid），唯一差异为 pooling_type：

```text
Time-decay  best_epoch=2  R@50=0.119840
Attention   best_epoch=3  R@50=0.116040
Delta                        -0.003800  （阈值 +0.001，方向相反）
```

按桶分析：
```text
≤5 bucket:  time-decay 0.13961 vs attention 0.13679  delta=-0.002817
6-20 bucket: time-decay 0.09600 vs attention 0.09101  delta=-0.004985
>20 bucket:  无法评估（history 截断为 max_len=20，gt20 bucket 永远为空）
```

Unique hit 分析：attention 带来 667 个新命中，但失去 857 个，净差 -190 用户。

### 可选方案

- A. 继续 full training（20 epoch），验证 attention 是否在更多 epoch 后超越
- B. 立即停止，time-decay 保持为 final model

### 最终选择

B。

### 选择原因

- delta = -0.0038，与继续训练阈值（+0.001）相反，gap 较大。
- attention epoch 3 仍在上升（未收敛），但与 time-decay epoch 2 峰值差距 -0.0038，即使继续也不确定能追平。
- gt20 bucket 因实现局限无法评估，但可观察的两个桶（≤5 和 6-20）attention 均弱于 time-decay。
- 20 epoch full training 时间成本高，ROI 不合理。

### 对实验可比性的影响

time-decay mean pool τ=0.15 full test Recall@50=0.078315 保持不变，为项目 canonical 数字。

### 对后续开发的影响

若未来实验 attention，需修复 gt20 bucket 评估 bug（应在截断前统计实际历史长度），并考虑 learnable query projection。

---

## Decision 20：采用 Transformer Two-Tower 作为 Multi-Channel 新 TT 通路

### 决策时间

2026-05-20

### 决策主题

用 canonical time-aware Transformer Two-Tower（R@50=0.103168）替换 multi-channel 系统中的旧 Time-decay Mean Pool Two-Tower（R@50=0.078315），并将新系统作为项目推荐输出的正式 multi-channel 配置。

### 背景

canonical Transformer 模型经过稳定性 sweep（4 配置）、max_len ablation（20/50/100）、seed 鲁棒性验证（seed=42/2024/2025）后，以 max_len=100、seed=42 确认为 final run，full test R@50=0.103168（+31.7% vs old TT）。

重跑 multi-channel valid-selected eval 结果如下：

```text
单路 sanity：R@50=0.103168（对齐=✅）
New 2ch RRF：R@50=0.117608（vs old 2ch 0.096727，+21.6%）
Valid-selected config：k=100, text_w=0.3, pop_w=0.5（与旧系统一致）
New 4ch valid-selected test：R@50=0.125164（vs old 4ch 0.104776，+19.5%）
NDCG@50：0.052179（+25.4%）  MRR@50：0.033618（+31.0%）
```

Candidate audit 验证：RRF rebuild Recall@50 = 0.125164（frozen test 一致性 ✅）。

### 可选方案

- A. 替换 TT 通路，更新 multi-channel 为新系统
- B. 保持旧系统不变（仅 Transformer 作为单独展示）

### 最终选择

A。

### 选择原因

1. 新系统在所有热度桶全面超越旧系统（≤5 桶持平，6-20/21-100/>100 全面提升）
2. valid-selected 权重与旧系统一致（text=0.3, pop=0.5），无 test-tuning 风险
3. Transformer TT 已通过完整稳定性 / ablation / seed 验证，不是 ad-hoc 实验
4. +19.5% Recall@50 提升幅度超过换新系统的工程成本

### 对实验可比性的影响

旧系统数字（0.104776）作为历史 baseline 保留在日志和报告中，不删除。
新系统为项目当前 multi-channel canonical 配置。

### 对后续开发的影响

- README / 简历需 Eddy 确认后单独更新，本决策不触发自动更新
- Faiss index 需基于 Transformer 模型重建（overlap@50 待重测）
- 若未来引入新通路，需重跑 valid grid search

### 参考文件

- `docs/reports/multichannel_transformer_final_eval.md`
- `outputs/multichannel_transformer_final/final_test_metrics.json`
- `outputs/multichannel_transformer_final/candidate_audit/audit_summary.json`

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

### 对后续开发的影响

明日优先任务是 ID-only Two-Tower baseline。暂时不进入负采样实验、LogQ、温度扫描或文本 embedding。

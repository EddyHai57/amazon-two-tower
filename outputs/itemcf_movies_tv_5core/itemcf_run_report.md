# Movies_and_TV 5-core ItemCF Baseline 运行报告

## 1. 本次实验目的

本次实验是 Phase 1 的第一个 baseline，目标是用传统 ItemCF 建立一个可解释、可复现的推荐召回基线。后续 ID-only Two-Tower 需要在同一套 `Movies_and_TV 5-core` 数据切分上与它进行对比。

## 2. 数据输入

本次使用的数据文件：

- `data/processed/movies_tv_5core/train.parquet`
- `data/processed/movies_tv_5core/test.parquet`
- `data/processed/movies_tv_5core/stats.json`

核心数据规模：

- dataset：`Movies_and_TV`
- users：505425
- items：155957
- train interactions：4402233
- test interactions：505425
- cold test targets：955
- non-cold eval users：504470

## 3. 模型方法

- 方法：Item-based Collaborative Filtering
- 共现统计：基于用户 train 历史中的 item-item 共现
- 相似度：`co_count(i, j) / sqrt(count(i) * count(j))`
- `sim_topk`：100
- `max_user_history`：100
- 推荐时过滤用户在 train 中已经交互过的 item
- `is_cold_item_for_eval=True` 的 test target 不参与指标计算

## 4. 评估指标

本次评估使用：

- `Recall@K`
- `NDCG@K`
- `MRR@K`

其中 `K = 20, 50, 100`。

## 5. 实验结果

| K | Recall | NDCG | MRR |
| --- | --- | --- | --- |
| 20 | 0.063260 | 0.030526 | 0.021131 |
| 50 | 0.083559 | 0.034553 | 0.021777 |
| 100 | 0.101283 | 0.037425 | 0.022029 |

其他运行信息：

- eval split：`test`
- skipped cold users：955
- no recommendation users：49
- seed：42
- run timestamp：`2026-05-10T07:23:32.458798+00:00`
- 运行用时：约 7 分 51 秒

## 6. 初步解读

- `Recall@50 = 0.083559` 是后续 ID-only Two-Tower 的直接对照基线。
- ItemCF 只能利用 item 共现关系，不能泛化到语义相似但共现不足的 item。
- 如果后续双塔或文本 embedding 表现更好，可以解释为模型利用了更强的用户/item 表示能力。

## 7. 下一步建议

如果 Eddy 确认本次 ItemCF 结果可作为 Phase 1 baseline，下一步进入 ID-only Two-Tower baseline 准备。不要直接进入负采样实验、LogQ 或温度扫描。

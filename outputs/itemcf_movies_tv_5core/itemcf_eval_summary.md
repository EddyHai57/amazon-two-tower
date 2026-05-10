# Movies_and_TV 5-core ItemCF baseline 评估摘要

## 1. 输入数据

- 数据目录：`data/processed/movies_tv_5core`
- dataset：Movies_and_TV
- train interactions：4402233
- eval split：`test`
- eval users：504470
- cold eval users skipped：955

## 2. ItemCF 设置

- 相似度：`co_count(i, j) / sqrt(count(i) * count(j))`
- 共现统计使用每个用户最近 `max_user_history` 个去重 train item。
- 推荐时过滤用户完整 train 历史中已经交互过的 item。
- `sim_topk`：100
- `max_user_history`：100

## 3. 指标

| K | Recall | NDCG | MRR |
| --- | --- | --- | --- |
| 20 | 0.063260 | 0.030526 | 0.021131 |
| 50 | 0.083559 | 0.034553 | 0.021777 |
| 100 | 0.101283 | 0.037425 | 0.022029 |

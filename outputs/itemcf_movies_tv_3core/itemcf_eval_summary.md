# Movies_and_TV 5-core ItemCF baseline 评估摘要

## 1. 输入数据

- 数据目录：`data/processed/movies_tv_3core`
- dataset：Movies_and_TV
- train interactions：5644734
- eval split：`test`
- eval users：1183084
- cold eval users skipped：7517

## 2. ItemCF 设置

- 相似度：`co_count(i, j) / sqrt(count(i) * count(j))`
- 共现统计使用每个用户最近 `max_user_history` 个去重 train item。
- 推荐时过滤用户完整 train 历史中已经交互过的 item。
- `sim_topk`：100
- `max_user_history`：100

## 3. 指标

| K | Recall | NDCG | MRR |
| --- | --- | --- | --- |
| 20 | 0.054087 | 0.027962 | 0.020327 |
| 50 | 0.068905 | 0.030897 | 0.020797 |
| 100 | 0.082373 | 0.033079 | 0.020988 |

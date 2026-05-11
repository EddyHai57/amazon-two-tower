# Clean Two-Tower valid/test gap diagnosis

## 运行边界

- 本脚本只读取 clean preprocess 数据、clean Two-Tower checkpoint 和 clean ItemCF 配置/metrics。
- 本脚本没有训练模型，没有调参，没有覆盖 clean baseline 输出目录。
- ItemCF top50 只在内存中重新计算 hit 标记，用于 overlap diagnosis，不写入 `outputs/itemcf_movies_tv_5core_clean/`。

## 输入

- data_dir：`data/processed/movies_tv_5core`
- checkpoint：`outputs/two_tower_movies_tv_5core_clean_overnight/checkpoints/best_model.pt`
- itemcf_metrics_path：`outputs/itemcf_movies_tv_5core_clean/metrics.json`

## Two-Tower item popularity bucket Recall@50

| bucket | valid users | valid hit50 | valid Recall@50 | test users | test hit50 | test Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <=5 | 27011 | 824 | 0.030506 | 35045 | 603 | 0.017206 |
| 6-10 | 33199 | 1578 | 0.047532 | 37012 | 1038 | 0.028045 |
| 11-20 | 46476 | 2967 | 0.063839 | 50055 | 1970 | 0.039357 |
| 21-100 | 161631 | 13486 | 0.083437 | 161718 | 8237 | 0.050934 |
| 101-500 | 142872 | 12799 | 0.089584 | 137364 | 7097 | 0.051666 |
| >500 | 85948 | 8908 | 0.103644 | 75276 | 4263 | 0.056632 |

## Two-Tower user history length bucket Recall@50

| bucket | valid users | valid hit50 | valid Recall@50 | test users | test hit50 | test Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-5 | 271910 | 26390 | 0.097054 | 271578 | 14528 | 0.053495 |
| 6-10 | 129755 | 9621 | 0.074147 | 129593 | 5656 | 0.043644 |
| 11-20 | 61970 | 3315 | 0.053494 | 61857 | 2105 | 0.034030 |
| 21-50 | 26746 | 1062 | 0.039707 | 26696 | 766 | 0.028693 |
| >50 | 6756 | 174 | 0.025755 | 6746 | 153 | 0.022680 |

## Two-Tower target rank distribution

| split | users | mean rank | median rank | p75 | p90 | p95 | p99 | Recall@20 | Recall@50 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 497137 | 26556.889010 | 10024.000000 | 39281.000000 | 82481.400000 | 108284.000000 | 140531.920000 | 0.059959 | 0.081591 | 0.102734 |
| test | 496470 | 33401.653987 | 16730.500000 | 52841.000000 | 96866.000000 | 119408.000000 | 144547.310000 | 0.032375 | 0.046746 | 0.061782 |

## ItemCF vs Two-Tower hit overlap on test

- ItemCF diagnostic Recall@50：0.083570
- Two-Tower diagnostic Recall@50：0.046746

| group | users | item popularity mean | item popularity median | user history mean | user history median | popularity<=20 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| both_hit | 17272 | 571.603520 | 110.000000 | 6.150359 | 4.000000 | 0.138780 |
| itemcf_hit_only | 24218 | 1905.913783 | 517.000000 | 8.277603 | 5.000000 | 0.131844 |
| two_tower_hit_only | 5936 | 169.920991 | 67.000000 | 8.967992 | 5.000000 | 0.204515 |
| both_miss | 449044 | 319.494317 | 67.000000 | 8.791945 | 5.000000 | 0.256786 |

## 结论判断

- valid-test gap 在所有 item popularity bucket 中都存在；`<=5` bucket 从 0.030506 降到 0.017206，`>500` bucket 从 0.103644 降到 0.056632。
- test target 比 valid target 更偏长尾，且长尾 bucket 的绝对 Recall@50 最低；这说明长尾 item 是重要因素，但不是唯一因素。
- user history length 分桶中也普遍存在 valid-test gap，`3-5` bucket test Recall@50=0.053495，`>50` bucket test Recall@50=0.022680。
- ItemCF hit only 样本数 24218，其 target item popularity median=517.000000；ItemCF 的优势主要来自能够利用用户历史 item 的局部共现关系。
- Two-Tower hit only 样本数 5936，说明 Two-Tower 仍有 ItemCF miss 但自己 hit 的样本，该组 target item popularity median=67.000000。
- rank sanity check 与已有 full eval 指标一致：valid Recall@50=0.081591，test Recall@50=0.046746；本次未发现新的 evaluation bug 迹象。
- 当前更像是 ID-only 表达能力不足叠加 test target 更长尾、用户兴趣随时间漂移，而不是单纯训练轮数不足。
- 在解释清楚 gap 前，仍不建议直接启动 20/25/30 epoch 长训。

# 报告索引

Amazon Two-Tower / Multi-channel Retrieval 项目离线实验报告。

---

## 当前最终系统

| 报告 | 内容 |
| --- | --- |
| [final_offline_trust_audit.md](final_offline_trust_audit.md) | **首要参考**：14 项 leakage 审计、决策追溯、test-tuning 风险分析、seed 鲁棒性、三套数字（历史 TT / Transformer TT / 4ch）的最终信任判断 |
| [multichannel_transformer_final_eval.md](multichannel_transformer_final_eval.md) | 当前最终系统完整报告：Transformer TT 接入、valid Pareto sweep（60 configs）、frozen test（R@50=0.125164）、candidate overlap audit、RRF hit attribution |

---

## Transformer User Tower 调研链路

| 报告 | 内容 |
| --- | --- |
| [transformer_user_tower_investigation.md](transformer_user_tower_investigation.md) | 完整调研链路：稳定性 sweep（lr/wd/grad_clip/patience）→ max_len ablation → seed 鲁棒性 → canonical final run（0.103168）；包含 epoch 崩塌发现和 early_stopping_patience=2 决策 |

---

## 历史多路召回报告

| 报告 | 内容 |
| --- | --- |
| [multichannel_valid_selected_eval.md](multichannel_valid_selected_eval.md) | 旧 4ch valid-selected 报告：基于 Time-decay TT（0.078315），frozen test Recall@50=0.104776；现已被 Transformer 4ch（0.125164）替代，保留作历史基线 |
| [multichannel_contribution_analysis.md](multichannel_contribution_analysis.md) | 各通路命中归因：分数加权命中占比、Jaccard@50/100、独占命中数；显示 TT 通路独占命中最多（9212），Text/Pop 通路无独占命中 |
| [multichannel_candidate_persistence_audit.md](multichannel_candidate_persistence_audit.md) | 候选集持久化与 RRF rebuild 一致性审计：验证 rebuild Recall 与 frozen test 精确对齐（保证无候选集变更风险） |

---

## 工程验证

| 报告 | 内容 |
| --- | --- |
| [faiss_transformer_two_tower_benchmark.md](faiss_transformer_two_tower_benchmark.md) | **当前**：Transformer TT Faiss ANN benchmark；IVF nprobe=32 = 8.8× 提速，−0.41% Recall；FlatIP 对齐 0.103168 ✅ |
| [faiss_two_tower_benchmark.md](faiss_two_tower_benchmark.md) | 历史：旧 Time-decay TT Faiss benchmark（nlist=4096，不同机器环境；IVF nprobe=32 = 25× 提速，−0.18% Recall） |

---

## 数字速查

| 指标 | 数字 |
| --- | ---: |
| ItemCF full test Recall@50 | 0.083570 |
| ID-only TT full test Recall@50 | 0.053198 |
| Time-decay TT full test Recall@50（历史主模型） | 0.078315 |
| Transformer TT full test Recall@50（当前神经通路） | 0.103168 |
| Transformer 4ch valid-selected Recall@50（当前最终） | 0.125164 |
| Transformer 4ch NDCG@50 | 0.052179 |
| Transformer 4ch MRR@50 | 0.033618 |
| Faiss IVF speedup（旧 TT） | 25× |
| Faiss Recall 损失（旧 TT）| −0.18% |

---

> ⚠️ outputs/ 不提交 git。报告中引用的所有 JSON 结果文件均在 `outputs/` 目录中，不随 repo 分发。

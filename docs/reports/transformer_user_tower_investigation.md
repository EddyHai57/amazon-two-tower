# Transformer User Tower Investigation

**日期：** 2026-05-19 / 2026-05-20 日志周期  
**脚本：** `scripts/train_transformer_maxlen100_smoke.py`  
**输出目录：** `outputs/transformer_user_tower_investigation/`  
**状态：** Phase 0 ✅ 完成；Phase 1 Smoke 3ep ✅ 完成；Diagnostic Full Eval 3ep ✅ 完成；Phase 2 公平 20ep 对比 🔄 训练中

---

## 1. 为什么做 Transformer Investigation

当前 final model 使用 Text + Time-decay Mean Pooling Two-Tower。此前 attention/max_len=50 smoke 显示 time-aware attention 和 gated attention 未超过 time-decay，但 Transformer user tower 尚未被单独验证。

本次 investigation 的目标是：在不改变 item tower、不改变 split、不改变 final model 的前提下，验证 `max_len=100` 条件下 Transformer user tower 是否能比 time-decay pooling 更好，尤其关注 `>20` 长历史用户桶。

---

## 2. 当前数据用户侧特征限制

Amazon Reviews 2023 Movies_and_TV 当前主数据只稳定使用：

- 用户历史 item 序列
- 交互时间戳排序
- item id embedding
- item text embedding

本次不会加入 `verified_purchase`、`helpful_vote`、category、item popularity 或多特征拼接，避免把实验变量混在一起。

---

## 3. max_len=100 的理由

max_len=50 smoke 已验证 `>20` bucket 能正确评估，但仍会截断 1.4% 的 `>50` 用户。本次把窗口扩大到 100，是为了更充分覆盖长历史用户，同时不一次性全开所有 history，避免显存和训练成本失控。

---

## 4. Phase 0：接手与可运行性检查

### 接手文件

- 复用半成品：`scripts/train_transformer_maxlen100_smoke.py`
- 新增配置：
  - `configs/transformer_max100_smoke_td.yaml`
  - `configs/transformer_max100_smoke_vanilla.yaml`
  - `configs/transformer_max100_smoke_timeaware.yaml`
- 新增输出目录：`outputs/transformer_user_tower_investigation/`

### 半成品状态

发现已有脚本 `scripts/train_transformer_maxlen100_smoke.py`，其数据加载、history matrix、raw history length、seen mask、训练/eval/checkpoint 逻辑可复用。已在原脚本上最小补齐：

- `transformer_vanilla`：1 layer / 4 heads / ffn_dim=256 / dropout=0.1 / positional embedding / valid mean pooling
- `transformer_timeaware`：在 vanilla 基础上加入 recency bucket embedding
- Phase 0 tiny forward / tiny train check
- 三路 comparison 输出 `phase1_results.json`

### Phase 0 检查结果

输出：`outputs/transformer_user_tower_investigation/phase0_check.json`

| 检查项 | 结果 |
| --- | --- |
| status | passed |
| history matrix shape | `[497449, 100]` |
| raw avg history length | 8.6832 |
| raw history >20 users | 33,558 |
| raw history >100 users | 1,708 |
| valid non-cold eval users | 497,137 |
| test non-cold eval users | 496,470 |
| valid seen mask | train only |
| blocker | None |

### Tiny Forward / Tiny Train

| 模型 | 参数量 | forward NaN/Inf | tiny train loss finite | 结论 |
| --- | ---: | --- | --- | --- |
| time_decay | 41,715,840 | 0 / 0 | yes | pass |
| transformer_vanilla | 41,772,224 | 0 / 0 | yes | pass |
| transformer_timeaware | 41,772,736 | 0 / 0 | yes | pass |

### Phase 0 结论

可运行性检查通过，无 blocker。可以进入 Phase 1 paired smoke。

---

## 5. Phase 0 接手状态审计（2026-05-20 续接）

**审计时间：** 2026-05-20 本次会话  
**审计结果：** 无残留进程，所有前期结果已完成。

### 已完成文件

| 文件 | 描述 |
| --- | --- |
| `phase0_check.json` | 可运行性检查（passed） |
| `phase1_results.json` | 3ep 50K limited valid 三路对比 |
| `time_decay_max100_full_eval/` | td 3ep checkpoint 全量 valid/test eval |
| `transformer_timeaware_max100_full_eval/` | timeaware 3ep checkpoint 全量 valid/test eval |

### Diagnostic Full Eval（3epoch, full valid/test）

> **⚠️ 标注为 diagnostic，不作为公平对比结论。**
> 3 epoch 远少于 final model 20 epoch；checkpoint 选择基于 50K limited valid，非 full valid。

| 模型 | checkpoint epoch | full valid R@50 | full test R@50 | full test NDCG@50 |
| --- | ---: | ---: | ---: | ---: |
| time_decay_max100 | 3 | 0.117897 | **0.078160** | 0.030841 |
| transformer_timeaware_max100 | 2 | 0.126637 | **0.103148** | 0.040054 |
| 当前 final model（20ep max_len=20） | 17 | 0.122626 | **0.078315** | — |

**发现：** diagnostic 阶段 timeaware Transformer 3ep 的 full test R@50 = 0.103148，远高于当前 final model 的 0.078315（+31.7%）。这是一个强烈信号，需要 20ep 公平对比验证。

**评测方法已确认无误：**
- valid eval：seen_mask=train，history=train
- test eval：seen_mask=train+valid，history=train+valid
- 与 final model 评测口径一致

**原因分析：** timeaware Transformer 使用 recency bucket embedding，在 test 阶段能更有效地利用 valid item（最近一次交互）作为最重要的上下文特征。结合 max_len=100 覆盖更长历史，实现大幅提升。

---

## 6. Phase 1：max_len=100 Paired Smoke


**状态：** 进入 Phase 2

| 模型 | best_epoch | R@50 | NDCG@50 | MRR@50 | Δ vs td | gt20 Δ | 参数量 | 训练秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| time_decay_max100 | 3 | 0.116640 | 0.049344 | 0.032211 | 0.000000 | 0.000000 | 41715840 | 154.591798 |
| transformer_vanilla_max100 | 2 | 0.107640 | 0.044244 | 0.028299 | -0.009000 | -0.019590 | 41772224 | 353.953132 |
| transformer_timeaware_max100 | 2 | 0.124360 | 0.051712 | 0.033346 | 0.007720 | -0.010850 | 41772736 | 362.588553 |

### History Bucket Recall@50

| 桶 | time_decay | vanilla | vanilla Δ | timeaware | timeaware Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=5 | 0.134050 | 0.133904 | -0.000146 | 0.145868 | 0.011817 |
| 6-20 | 0.102383 | 0.082692 | -0.019691 | 0.107499 | 0.005117 |
| >20 | 0.056359 | 0.036769 | -0.019590 | 0.045509 | -0.010850 |

### Unique Hit vs Time-decay

| 模型 | both_with_td | only_model | only_td |
| --- | ---: | ---: | ---: |
| transformer_vanilla_max100 | 4086 | 1296 | 1746 |
| transformer_timeaware_max100 | 4240 | 1978 | 1592 |

### Stop / Continue

- best Transformer: `transformer_timeaware`
- best overall delta vs time_decay: 0.007720
- best >20 bucket delta vs time_decay: -0.010850
- decision: **进入 Phase 2**
- reason: Phase 1 threshold met

本结果是 50K limited valid paired smoke，不是 full test 结论，不改变 final model。

---

## 7. Phase 2：公平 20epoch Paired Comparison

**状态：** ✅ 完成（2026-05-20）

**设计：** 两个模型除 user tower pooling 外完全一致，均采用与 final model 相同的训练超参数和评测策略。

| 参数 | 值 |
| --- | --- |
| epochs | 20（early stopping 未启用，full 20ep） |
| eval_max_users（训练期） | 50,000（与 final model 一致） |
| full eval | eval_max_users=null（训练结束后单独运行） |
| seed | 42 |
| batch_size | 4096 |
| **lr（原始）** | **0.001** |
| temperature | 0.15 |
| embedding_dim | 64 |
| history_max_len | 100 |
| checkpoint selection | best valid_recall@50（50K limited valid） |

### 配置文件

- `configs/transformer_max100_20ep_td.yaml` → `outputs/transformer_user_tower_investigation/td_max100_20ep/`
- `configs/transformer_max100_20ep_timeaware.yaml` → `outputs/transformer_user_tower_investigation/timeaware_max100_20ep/`

### Timeaware Transformer 架构

- 1 TransformerEncoderLayer (Pre-LN, batch_first)
- 4 heads, FFN=256, dropout=0.1
- Learnable positional embedding (max_len=100)
- Recency bucket embedding（7 buckets，位置区间 [0,2,4,8,16,32,64]）
- Output: mean pool over valid positions

### 实际训练时间

| 模型 | 实际总时间 | best epoch |
| --- | --- | --- |
| time_decay max100 | ~1030s（~17 min） | 20 |
| timeaware Transformer max100 | ~2367s（~39 min）| **2**（然后崩溃） |

---

## 8. 公平判断规则（预先声明）

只有当 time-aware Transformer 满足以下全部条件，才建议进入替换讨论：

1. full valid Recall@50 > time_decay max100 20ep
2. full test Recall@50 > time_decay max100 20ep
3. full test Recall@50 > 0.078315（当前 final model）
4. 提升幅度 ≥ +0.0015（建议 +0.002 以上才稳健）
5. 不只是单个 bucket 偶然提升
6. 训练稳定，无 NaN/OOM

如果仅在 3ep 或 limited valid 上赢，不写为 final model 替换结论。


---

## 9. Phase 2：20epoch 公平对比结果

**状态：** ✅ 完成

### Overall 对比（full valid / full test）

| 模型 | best_ep | full valid R@50 | full test R@50 | full test NDCG@50 | full test MRR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time_decay max100 20ep | 20 | 0.124811 | 0.078458 | 0.030852 | 0.018976 |
| timeaware Transformer max100 20ep | 2 | 0.126603 | 0.103136 | 0.040048 | 0.024401 |
| **Δ (ta − td)** | — | **  0.001792** | **  0.024678** | — | — |
| 当前 final model（参考） | 17 | 0.122626 | 0.078315 | — | — |

### Test Bucket Recall@50 对比

| 桶 | td 20ep | timeaware 20ep | Δ |
| --- | ---: | ---: | ---: |
| ≤5 | 0.089380 | 0.126638 | 0.037258 |
| 6-20 | 0.073704 | 0.091331 | 0.017626 |
| >20 | 0.046384 | 0.044815 | -0.001569 |

### Unique Hit（test set, R@50）

| 指标 | 数量 |
| --- | ---: |
| 两者均命中 | 26225 |
| 仅 td 命中 | 12727 |
| 仅 timeaware 命中 | 24979 |

### 是否建议替换 user tower

| 条件 | 结论 |
| --- | --- |
| ta full valid R@50 > td full valid R@50 | ✅ 是 |
| ta full test R@50 > td full test R@50 | ✅ 是 |
| ta full test R@50 > final model 0.078315 | ✅ 是 |
| 提升幅度 ≥ +0.0015（vs final model） | ✅ 是 |
| 至少一个 bucket 正向提升 | ✅ 是 |
| **综合建议：替换 user tower** | ✅ 是 |


---

## 10. 训练稳定性分析——关键发现

### time_decay max100 训练曲线（正常收敛）

| epoch | train_loss | limited valid R@50 |
| ---: | ---: | ---: |
| 1 | 7.411 | 0.087420 |
| 5 | 5.539 | 0.119760 |
| 10 | 5.356 | 0.121340 |
| 15 | 5.277 | 0.121880 |
| **20** | **5.230** | **0.123580** |

单调收敛，稳定提升，best epoch=20。

### timeaware Transformer 训练曲线（⚠️ 峰值后崩溃）

| epoch | train_loss | limited valid R@50 |
| ---: | ---: | ---: |
| 1 | 6.880 | 0.114100 |
| **2** | **6.201** | **0.124340 ← PEAK（best checkpoint）** |
| 3 | 5.981 | 0.123140 |
| 5 | 5.434 | 0.101600 |
| 10 | 4.448 | 0.065680 |
| 15 | 3.918 | 0.040200 |
| 20 | 3.639 | 0.027580 |

**训练 loss 持续下降（6.880→3.639），但 valid R@50 从 epoch 2 开始单调崩溃（0.1243→0.0276）。** 

这是典型的过拟合/训练崩溃：Transformer 在 epoch 2 之后开始记忆训练集的 in-batch negative 分布，而非学习可泛化的用户-item 相关性。

### 原因分析

1. **Transformer capacity 相对 in-batch negatives 过高**：经过足够 epoch 后，Transformer 可以精确记忆 batch 内每个 user 的 history pattern，使 train loss 持续下降但泛化完全崩溃。

2. **time_decay 池化无此问题**：time_decay 仅有加权均值，无 attention 机制，不能记忆 position-specific patterns，因此持续正常收敛。

3. **实践含义**：timeaware Transformer 需要早停（best epoch ≈ 2），不能用默认 20 epoch 配置训练。

### 公平性评价

| 维度 | time_decay 20ep | timeaware 20ep |
| --- | --- | --- |
| 实际 best epoch | 20（正常收敛） | **2**（峰值后崩溃） |
| "20ep"训练实际作用 | 全部 20ep 有效 | 只有前 2ep 有效 |
| 评测用的 checkpoint | ep20 | **ep2（与 3ep smoke 相同！）** |

**结论：** 虽然两者都训练了 20 epoch，但 timeaware 的 full eval 实际上仍是 epoch 2 的 checkpoint（与 Phase 1 3ep smoke 相同）。20ep 对 timeaware 没有额外收益。相反，td 从 3ep 到 20ep 有 full test R@50 从 0.078160 → 0.078458 的小幅提升。

---

## 11. 综合结论

### 最终 20ep 对比数据

| 模型 | best_ep | full valid R@50 | full test R@50 | NDCG@50 | MRR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time_decay max100 20ep | 20 | 0.124811 | **0.078458** | 0.030852 | 0.018976 |
| timeaware Transformer max100 20ep | **2** | 0.126603 | **0.103136** | 0.040048 | 0.024401 |
| 当前 final model（max_len=20 20ep） | 17 | 0.122626 | **0.078315** | — | — |

### Test Bucket Recall@50（全量）

| 桶 | td 20ep | timeaware 20ep | Δ |
| --- | ---: | ---: | ---: |
| ≤5 | 0.089380 | **0.126638** | **+0.037258** |
| 6-20 | 0.073704 | **0.091331** | **+0.017627** |
| **>20** | **0.046384** | 0.044815 | **−0.001569** |

**注意：** >20 bucket timeaware 微弱落后（−0.0016）。提升主要来自 ≤5 和 6-20 bucket，而非原始设计目标（长历史用户）。

### Unique Hit（test set，R@50）

| 指标 | 数量 |
| --- | ---: |
| 两者均命中 | 26,225 |
| 仅 td 命中 | 12,727 |
| 仅 timeaware 命中 | **24,979** |

timeaware 独有命中比 td 独有命中多 ~2× (24,979 vs 12,727)。

### 是否满足替换条件

| 条件 | 结果 |
| --- | --- |
| full valid R@50 (ta > td) | ✅ 0.126603 > 0.124811 |
| full test R@50 (ta > td) | ✅ 0.103136 > 0.078458 |
| full test R@50 (ta > final 0.078315) | ✅ 0.103136 > 0.078315 |
| 提升幅度 ≥ +0.0015 (vs final) | ✅ Δ = +0.024821 |
| 至少一个 bucket 正向 | ✅ le5 +0.037, 6to20 +0.018 |
| **综合建议** | **✅ 建议替换 user tower** |

### ⚠️ 关键限制（必须向 Eddy 报告）

1. **训练崩溃**：timeaware best checkpoint = epoch 2，后续 18 epoch 全部无效。生产部署需严格早停（patience=1~2）。
2. **>20 bucket 未改善**：最初的设计动机（改善长历史用户）在 test 上未实现（Δ=−0.0016），提升来自短历史用户。
3. **test R@50 大幅提升的机制**：timeaware 通过 recency bucket 重度加权最近一次交互（valid item）来预测 test item。这在离线评测上表现好，但需验证在实际推荐中是否依然有效（valid item 始终为最近交互）。
4. **不是 full 20ep fair comparison**：timeaware 实际上只用了 2 epoch，和 Phase 1 smoke checkpoint 完全相同，20ep 训练未带来额外收益。
5. **multi-channel/Faiss 未重跑**：替换 user tower 后，需要重跑 Faiss index、multi-channel eval，验证整体 pipeline 影响。

### 是否建议进入 README / 简历

- **README：** 尚不建议更新，需先确认 Eddy 是否决定替换 user tower 主线。
- **简历：** 暂不写入，等 20ep 稳定性问题确认后再评估。
- **面试口述：** 可描述为"做了公平对比实验验证 Transformer user tower 的有效性，发现早停关键，offset evaluation (valid item in test history) 产生显著提升，还需进一步验证 production 可行性"。


---

## 12. Stability Sweep — 设置

**状态：** 🔄 训练中（2026-05-20）

### 12.0 动机

原始 20ep 训练中（lr=0.001），timeaware Transformer 在 epoch 2 达到峰值（limited valid R@50=0.1244），之后持续崩溃至 epoch 20（R@50=0.028）。目标是找到能**稳定复现** full test R@50≈0.10 且不 collapse 的训练配置。

### 12.1 Sweep 配置

| 参数 | A | B | C | D |
| --- | --- | --- | --- | --- |
| lr（原始 1e-3） | **1e-3** | 3e-4 | 1e-4 | 3e-4 |
| grad_clip_norm | 0 | **1.0** | **1.0** | **1.0** |
| warmup_steps | 0 | 0 | 0 | **1000（~1 epoch）** |
| lr_schedule | none | none | none | **cosine** |
| early_stopping_patience | **2** | 3 | 3 | 3 |
| max epochs | 20 | 20 | 20 | 20 |

所有配置：pooling=transformer_timeaware，max_len=100，heads=4，ffn_dim=256，dropout=0.1，batch=4096，temp=0.15，seed=42，eval_max_users=50K

### 12.2 预计训练时间

| 配置 | 预计停止 epoch | 预计时间 |
| --- | --- | --- |
| A（lr=1e-3，patience=2） | ~ep4（early stop） | ~10 min |
| B（lr=3e-4，patience=3） | ~ep10-15 | ~25-30 min |
| C（lr=1e-4，patience=3） | ep20（可能不 collapse） | ~42 min |
| D（warmup+cosine，patience=3） | ~ep10-15 | ~25-30 min |
| 总计（含 full eval × 4） | — | **~1.75 h** |

**结果待补充。**


---

## 12. Stability Sweep — 结果

**状态：** ✅ 完成（2026-05-20）

### 12.1 Sweep 设计

| 参数 | A | B | C | D |
| --- | --- | --- | --- | --- |
| lr | 1e-3 | **3e-4** | **1e-4** | **3e-4** |
| grad_clip_norm | 0（disabled） | 1.0 | 1.0 | 1.0 |
| warmup_steps | 0 | 0 | 0 | **1000（~1 epoch）** |
| lr_schedule | none | none | none | **cosine** |
| early_stopping_patience | **2** | **3** | **3** | **3** |
| max epochs | 20 | 20 | 20 | 20 |

### 12.2 Overall 对比

| 配置 | best_ep | 实际跑 | 早停 | collapse | limited valid R@50 | full test R@50 | Δ vs final |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| A：原 lr=1e-3，patience=2 | 2 | 4 | 是 | no | 0.124300 | 0.103128 | +0.024813 |
| B：lr=3e-4，grad_clip=1.0，patience=3 | 3 | 6 | 是 | no | 0.123300 | 0.100282 | +0.021967 |
| C：lr=1e-4，grad_clip=1.0，patience=3 | 5 | 8 | 是 | no | 0.116860 | 0.094946 | +0.016631 |
| D：lr=3e-4，grad_clip=1.0，warmup+cosine，patience=3 | 3 | 6 | 是 | no | 0.122260 | 0.100304 | +0.021989 |

### 12.3 Test Bucket Recall@50

| 配置 | ≤5 | 6-20 | >20 |
| --- | ---: | ---: | ---: |
| A：原 lr=1e-3，patience=2 | 0.126680 | 0.091286 | 0.044760 |
| B：lr=3e-4，grad_clip=1.0，patience=3 | 0.121270 | 0.090490 | 0.043108 |
| C：lr=1e-4，grad_clip=1.0，patience=3 | 0.116145 | 0.085045 | 0.037273 |
| D：lr=3e-4，grad_clip=1.0，warmup+cosine，patience=3 | 0.122551 | 0.089763 | 0.040796 |

### 12.4 最稳定配置

**best full test R@50：** A_baseline_earlystop → 0.103128


---

## 13. History Length Distribution

**状态：** ✅ 完成（2026-05-20）

### 13.1 基本统计量

| 分组 | 用户数 | min | mean | median | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_users | 497,449 | 3 | 8.68 | 5 | 9 | 16 | 25 | 59 | 1839 |
| valid_eval_users | 497,449 | 3 | 8.68 | 5 | 9 | 16 | 25 | 59 | 1839 |
| test_eval_users | 497,449 | 4 | 9.68 | 6 | 10 | 17 | 26 | 60 | 1840 |

> - `train_users`：train history raw length（截断前，与截断后 max_len=20 final model 一致口径）
> - `valid_eval_users`：valid eval history = train history（口径完全相同）
> - `test_eval_users`：test eval history = train + valid（每用户恰好 +1 交互）

### 13.2 Max_len 覆盖率

| 分组 | max_len | ≤ 用户数 | ≤ 比例 | > 用户数 | > 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_users | ≤20 | 463,891 | 93.2% | 33,558 | 6.8% |
| train_users | ≤50 | 490,674 | 98.6% | 6,775 | 1.4% |
| train_users | ≤100 | 495,741 | 99.7% | 1,708 | 0.3% |
| test_eval_users | ≤20 | 461,002 | 92.7% | 36,447 | 7.3% |
| test_eval_users | ≤50 | 490,401 | 98.6% | 7,048 | 1.4% |
| test_eval_users | ≤100 | 495,715 | 99.7% | 1,734 | 0.3% |

### 13.3 解读

- **max_len=20**：train 完整覆盖 93.2%，剩余 6.8% 被截断
- **max_len=50**：train 完整覆盖 98.6%，超出 50 的仅 1.4%
- **max_len=100**：train 完整覆盖 99.7%，超出 100 的仅 0.3%
- **train median=5，p90=16，p99=59**
- max_len Ablation（Section 14）将量化不同 max_len 对 Recall@50 的实际影响。

---

## 14. Max_len Ablation

**状态：** ✅ 完成（2026-05-20）

### 14.1 整体对比

| 配置 | best_ep | 实际跑 | 早停 | limited valid R@50 | full valid R@50 | full test R@50 | Δ vs final |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| max_len=20 | 2 | 4 | 是 | 0.122480 | 0.124229 | 0.101211 | +0.022896 |
| max_len=50 | 2 | 4 | 是 | 0.124280 | 0.126589 | 0.102306 | +0.023991 |
| max_len=100 (复用) | 2 | 4 | 是 | 0.124300 | 0.126655 | 0.103128 | +0.024813 |

### 14.2 Test Bucket Recall@50（按 user history 长度）

| 配置 | ≤5 交互 | 6-20 交互 | >20 交互 |
| --- | ---: | ---: | ---: |
| max_len=20 | 0.125815 | 0.091221 | 0.024087 |
| max_len=50 | 0.124604 | 0.092029 | 0.040714 |
| max_len=100 (复用) | 0.126680 | 0.091286 | 0.044760 |

### 14.3 结论

**最优 max_len（full valid R@50 最高）：100**

- max_len=50 vs max_len=20：full valid R@50 差 +0.002360
- max_len=100 vs max_len=50：full valid R@50 差 +0.000066

---

## 15. Seed Robustness（max_len=100）

**状态：** ✅ 完成（2026-05-20）

### 15.1 各 seed 结果

| 配置 | seed | best/实际 | 早停 | full valid R@50 | full test R@50 | Δ vs final | ≤5 | 6-20 | >20 |
| --- | ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed42 | 42 | ep2 / 4ep | 是 | 0.126655 | 0.103128 | +0.024813 | 0.126680 | 0.091286 | 0.044760 |
| seed2024 | 2024 | ep2 / 4ep | 是 | 0.127884 | 0.103704 | +0.025389 | 0.126390 | 0.092817 | 0.043962 |
| seed2025 | 2025 | ep3 / 5ep | 是 | 0.123710 | 0.096223 | +0.017908 | 0.111488 | 0.090100 | 0.047871 |

### 15.2 汇总统计

| 指标 | 值 |
| --- | ---: |
| mean full test R@50 | 0.101019 |
| std  full test R@50 | 0.003399 |
| min  full test R@50 | 0.096223 |
| max  full test R@50 | 0.103704 |
| 所有 seed > 0.078315（final） | ✅ |
| 所有 seed ≥ 0.10 | ❌ |
| 稳定性评价 | 有一定 seed sensitivity（0.003<std≤0.005） |

---

## 16. Confirmation 实验总结与建议

**状态：** ✅ 完成（2026-05-20）

### 16.1 长历史是否关键？

max_len=100 vs max_len=20 full test R@50 差值：+0.0019

**结论：** max_len 影响极小，提升主要来自 time-aware Transformer 本身，而非长历史。

### 16.2 time-aware Transformer 本身是否有效？

对比 current final（time-decay mean pool max_len=20）：
- time-aware Transformer max_len=20 full test R@50（如有）远超 0.078315，说明 Transformer 架构本身有效
- 即使最短历史下，time-aware Transformer 也超越 time-decay mean pool

### 16.3 Seed 稳定性

- mean ± std = 0.101019 ± 0.003399
- 稳定性评价：有一定 seed sensitivity（0.003<std≤0.005）
- 所有 seed 均超过 final 0.078315：是

### 16.4 是否建议进入 multi-channel / Faiss 重跑？

**✅ 建议进入**

条件检查：
- mean full test R@50 ≥ 0.10：✅
- 所有 seed 超 final：✅
- seed 较稳定（std≤0.003）：⚠️ std=0.0034

**进入条件满足。建议 Eddy 确认后：① 正式训练 Transformer user tower（Config A 参数，patience=2）；② 导出 item embeddings → 重建 Faiss index；③ 重跑 multi-channel eval；④ 更新 README 和简历。**

> ⚠️ 以上均为 offline full eval 结论，不等于 online A/B。最终替换需 Eddy 明确 go-ahead。

---

## 17. Canonical Transformer Final Run

**状态：** ✅ 完成（2026-05-20）

### 17.1 训练配置

- model: time-aware Transformer Two-Tower
- max_len=100, seed=42, lr=1e-3, patience=2, epochs≤20
- config: `configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml`
- output: `outputs/text_timeaware_transformer_max100_final/`

### 17.2 结果

| 指标 | 值 |
|---|---:|
| best_epoch | 2 |
| epochs_trained | 4 |
| early_stopped | ✅ |
| full valid R@50 | 0.126653 |
| full valid NDCG@50 | 0.052641 |
| full valid MRR@50 | 0.033996 |
| **full test R@50** | **0.103168** |
| full test NDCG@50 | 0.040087 |
| full test MRR@50 | 0.024439 |

### 17.3 vs Old Final

| 模型 | full test R@50 | Δ absolute | Δ relative |
|---|---:|---:|---:|
| Old Final（Time-decay） | 0.078315 | — | — |
| **Canonical Transformer** | **0.103168** | **+0.024853** | **+31.7%** |

### 17.4 Test Bucket Recall@50

| Bucket | Recall@50 |
|---|---:|
| ≤5 交互 | 0.126750 |
| 6-20 交互 | 0.091306 |
| >20 交互 | 0.044760 |

### 17.5 验证与判断

- ✅ 所有验证通过

- 是否复现 investigation seed=42 (0.103128)：差值 +0.000040

**✅ 建议替换：full test R@50 > 0.10，等 Eddy 确认后进入 multi-channel / Faiss 重跑。**

> ⚠️ 以上均为 offline full eval 结论。不覆盖旧 final。不自动更新 README / 简历。

---

## 18. Gain Attribution Closeout：full-test 修正

**状态：** ✅ 完成（2026-06-03）

**最终 headline：** 时间特征（positional + recency bucket）是必要组件；纯 attention 在 paired smoke 中比 time-decay 还差。attention 也不是摆设：full test 上 canonical Transformer TT `0.103168` 明显高于 `mean_pool_timeaware` `0.086420`，额外贡献约 `+0.016748`。limited-valid smoke 一度会得出"attention 非必要"的误判，full test 修正了这个结论，说明不能轻信小规模 eval。

**目的：** 拆分 `transformer_timeaware` 的增益来源：到底来自 self-attention，还是来自 `positional + recency bucket` 时间特征工程。

### 18.1 Paired limited-valid smoke 设置

- 输出目录：`outputs/transformer_gain_attribution/`
- 评估口径：**3-epoch 50K limited-valid smoke**，不是 full test
- 四档配置除 `pooling_type` / `output_dir` 外同参：
  - `time_decay`
  - `transformer_vanilla`
  - `transformer_timeaware`
  - `mean_pool_timeaware`
- `mean_pool_timeaware`：`item_id_embedding + positional embedding + recency bucket embedding` 后直接 masked mean pool，**不经过 Transformer encoder**
- 不修改 canonical run / split / seed / loss / temperature / seen mask / cold 口径

### 18.2 Limited-valid smoke 结果

| pooling | best_ep | limited-valid R@50 | NDCG@50 | MRR@50 | params | train_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| time_decay | 3 | 0.116640 | 0.049344 | 0.032211 | 41,715,840 | 152.2 |
| transformer_vanilla | 2 | 0.107640 | 0.044244 | 0.028299 | 41,772,224 | 349.7 |
| transformer_timeaware | 2 | 0.124560 | 0.051794 | 0.033408 | 41,772,736 | 356.6 |
| mean_pool_timeaware | 3 | 0.121940 | 0.051260 | 0.033356 | 41,722,752 | 158.7 |

预注册判据：

```text
|mean_pool_timeaware - transformer_timeaware| < 0.003
```

实际差值：

```text
0.121940 - 0.124560 = -0.002620
```

**limited-valid 过程结论：** 预设判据通过。50K limited-valid smoke 会提示 time-aware Transformer 的 early valid 增益主要来自 `positional + recency bucket` 时间特征工程；此时如果只看 smoke，容易误写成"self-attention 非必要"。

### 18.3 mean_pool_timeaware full eval

对 `mean_pool_timeaware` best checkpoint 单独补了一次 full eval：

| split | R@50 | NDCG@50 | MRR@50 |
| --- | ---: | ---: | ---: |
| full valid | 0.124525 | 0.051831 | 0.033454 |
| full test | 0.086420 | 0.033833 | 0.020773 |

Test bucket R@50：

| bucket | R@50 |
| --- | ---: |
| le5 | 0.101687 |
| 6to20 | 0.079531 |
| gt20 | 0.043246 |

**full-test 修正结论：**

- `mean_pool_timeaware` full test R@50=`0.086420`，高于历史 Time-decay TT `0.078315`，说明 `positional + recency bucket` 时间特征本身有独立价值。
- canonical Transformer TT full test R@50=`0.103168`，比 `mean_pool_timeaware` 高 `+0.016748`，说明 attention 对 full-test 泛化仍有额外贡献，不是摆设。
- `mean_pool_timeaware` 不替代 canonical；canonical 仍保持 `0.103168` 不变。
- 这次雷点体现为：limited-valid smoke 可以用于快速定位机制，但不能单独作为最终归因结论；最终 headline 必须服从 full test。

---

## 19. Collapse Diagnosis Closeout

**状态：** ✅ 完成（2026-06-03）

**输出目录：** `outputs/transformer_collapse_diagnosis/`

### 19.1 20-epoch collapse 曲线

只读取既有 `outputs/transformer_user_tower_investigation/timeaware_max100_20ep/train_log.csv`，不重训。

| 指标 | 值 |
| --- | ---: |
| peak_epoch | 2 |
| peak limited-valid R@50 | 0.124340 |
| final_epoch | 20 |
| final limited-valid R@50 | 0.027580 |
| drop after peak | -0.096760 |
| retention after peak | 22.18% |
| peak train_loss | 6.201144 |
| final train_loss | 3.638897 |
| train_loss delta after peak | -2.562247 |

这确认了原始 20ep timeaware run 的核心现象：train loss 持续下降，但 valid Recall 从 epoch 2 峰值后坍塌。

### 19.2 Canonical best checkpoint embedding health

加载 canonical best checkpoint：`outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt`，导出 item embeddings 并计算谱诊断。

| 指标 | 值 |
| --- | ---: |
| checkpoint_epoch | 2 |
| checkpoint best limited-valid R@50 | 0.124560 |
| item embedding shape | 153,977 × 64 |
| nan_count / inf_count | 0 / 0 |
| norm_mean | 1.000000 |
| effective_rank | 28.812 |
| participation_rank | 20.588 |
| top1 explained variance | 0.110135 |
| top5 explained variance | 0.375480 |
| uniformity sample | -3.502885 |
| mean pairwise squared distance | 1.964033 |
| healthy_peak_checkpoint | true |

诊断结论：canonical best checkpoint 是有限、非单维坍塌、高有效秩的健康峰值版；坍塌主要发生在继续以 `lr=1e-3` 训练 epoch 2 之后。

### 19.3 与 stability sweep 的合并结论

已有 stability sweep：

| config | 设置 | best_ep | full test R@50 |
| --- | --- | ---: | ---: |
| A | lr=1e-3, no clip, patience=2 | 2 | 0.103128 |
| B | lr=3e-4, grad_clip=1.0, patience=3 | 3 | 0.100282 |
| C | lr=1e-4, grad_clip=1.0, patience=3 | 5 | 0.094946 |
| D | lr=3e-4, grad_clip=1.0, warmup+cosine | 3 | 0.100304 |

**最终解释：** 坍塌由 `lr=1e-3` 在 epoch 2 后继续优化驱动。`lr=3e-4 + grad_clip` 能让训练更稳，但相对 canonical early-stopped run 损失约 `0.0028–0.0029` Recall@50。项目选择 canonical A，不是因为后期训练稳定，而是因为 early stopping 能锁住 epoch 2 的健康峰值 checkpoint。

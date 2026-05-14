# 问题日志

## 加载 Amazon Reviews 2023 时出现 HuggingFace datasets 兼容性错误

- 严重程度：中等
- 状态：已解决

### 现象

```text
Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py
```

### 影响

此前 Amazon All_Beauty 数据检查没有完成，也没有生成检查报告。

### 根因假设

主环境缺少合适的 datasets 版本，并且新版 datasets 对 HuggingFace dataset script 支持存在兼容性问题。McAuley-Lab/Amazon-Reviews-2023 需要通过兼容的数据集加载机制读取。

### 已尝试的排查步骤

- 确认错误不是 Amazon 数据本身不可用，也不是服务器问题。
- 在 `/workspace/amazon-two-tower` 内创建项目独立虚拟环境 `.venv`。
- 在 `.venv` 中安装兼容依赖：
  - `datasets==2.17.0`
  - `huggingface_hub==0.36.2`
  - `pyyaml`
  - `pandas`
  - `pyarrow`
- 验证依赖版本：
  - python 3.12.13
  - datasets 2.17.0
  - huggingface_hub 0.36.2
  - pandas 3.0.2
  - pyarrow 24.0.0

### 最终解决方案

使用项目独立 `.venv`，并在该环境中固定兼容版本 `datasets==2.17.0` 与 `huggingface_hub==0.36.2`。

随后重新运行：

```bash
python scripts/inspect_amazon_dataset.py --config configs/amazon_all_beauty_phase1.yaml
```

脚本成功完成，生成 `outputs/inspection_all_beauty.md`。

### 解决结果

- loading strategy used：`full_load`
- review row count：701528
- meta row count：112590
- unique user_id count：631986
- unique parent_asin count：112565
- review 数据可用。
- meta 数据可用。

### 后续复用建议

- 运行 Amazon 项目代码时，进入 `/workspace/amazon-two-tower` 并激活 `.venv`。
- 不要提交 `.venv/`。
- 不要在未得到 Eddy 确认前修改全局 Python 环境或随意升级/降级依赖。

## All_Beauty 交互稀疏，无法作为 Phase 1 主实验数据集

- 严重程度：Phase 1 为高，Phase 0 为中等
- 状态：已解决 / 已决策
- 日期：2026-05-09

### 现象

运行 `scripts/analyze_interactions.py` 后发现，All_Beauty 在 `rating >= 4` 临时正样本视图中有 500107 条 interaction，但用户重复交互极少，迭代式 k-core 过滤后规模大幅下降：

- A 组 `user>=3,item>=3`：8657 interaction，1531 user，1694 item
- B 组 `user>=5,item>=5`：293 interaction，51 user，52 item
- C 组 `user>=10,item>=10`：10 interaction，1 user，1 item
- 93.28% 的用户只有 1 条正向交互
- 用户正向交互数 p50=1，p75=1，p90=1，p99=3
- 最宽松 k-core 下 test cold item 比例为 7.97%

### 影响

- 不适合用于生成简历里的主实验 Recall@50 数字。
- 双塔模型在这种数据上即使能训练，指标也可能很低且不可解释。
- All_Beauty 仍可作为 Phase 0 工程验证数据集，用于验证数据加载、inspection、interaction analysis、日志流程等。

### 初步原因判断

All_Beauty 品类具有一次性消费特征，很多用户只购买或评价一次就离开。问题不是数据加载失败，也不是参数设置错误，而是该品类天然用户重复交互不足。

### 已尝试的排查步骤

- 统计 rating 分布。
- 构造 `rating >= 4` 临时正样本视图。
- 统计用户和 item 正向交互数分布。
- 模拟 A/B/C 三组 k-core 阈值。
- 模拟 leave-one-out split 并统计 cold item 比例。

### 最终解决方案

将 All_Beauty 降级为 Phase 0 工程验证数据集；Phase 1 切换到更合适的大类目，但不直接锁 Electronics，需要先对候选品类做对比分析。

### 后续复用建议

- 明天优先扩展 `analyze_interactions.py` 支持多品类配置。
- 候选品类先比较 Electronics、Video_Games、Movies_and_TV。
- 生成 `outputs/category_comparison.md` 后，再决定 Phase 1 主数据集。

## 普通 `nohup` 启动 3-core preprocess 后进程未存活

- 严重程度：Low
- 状态：已解决
- 日期：2026-05-10

### 现象

启动 Movies_and_TV 3-core preprocess 后，第一次后台 PID `17961` 很快退出，`logs/preprocess_3core.log` 为空，`data/processed/movies_tv_3core/` 未创建。

### 报错原文或关键日志

没有 Python 报错输出。检查结果：

```text
logs/preprocess_3core.log size = 0
ps -p 17961 无运行进程
data/processed/movies_tv_3core 不存在
```

### 影响

普通后台启动没有成功保持进程运行，但没有写入预处理输出，也没有覆盖已有数据。

### 初步原因判断

当前工具 shell 退出后，普通后台子进程没有稳定脱离会话。

### 已尝试的排查步骤

- 检查 `logs/preprocess_3core.log` 文件大小。
- 检查 `data/processed/movies_tv_3core/` 是否创建。
- 检查是否存在残留 preprocess 进程。
- 确认 `.venv/bin/python` 可用。

### 最终解决方案

使用 `nohup setsid` 重新启动后台任务，使进程脱离当前 shell：

```bash
nohup setsid .venv/bin/python scripts/preprocess_amazon.py --config configs/preprocess_movies_tv_3core.yaml > logs/preprocess_3core.log 2>&1 < /dev/null &
```

新 PID 为 `18247`，日志已正常写入并显示 review 数据读取完成。该后台任务随后成功完成，生成 Movies_and_TV 3-core preprocess 结果。

### 后续复用建议

以后在当前工具环境中启动长时间后台任务时，优先使用 `nohup setsid ... > logs/<name>.log 2>&1 < /dev/null &`，并在 10-20 秒后检查日志和 PID。

## 项目 `.venv` 缺少 `torch`，Two-Tower smoke test 无法启动

- 严重程度：High
- 状态：已缓解 / Mitigated
- 日期：2026-05-10

### 现象

运行 ID-only Two-Tower smoke test 时，脚本在依赖检查阶段退出，没有进入训练。

### 报错原文或关键日志

```text
ERROR: 缺少依赖：torch。请先在项目 .venv 中安装 package：torch
```

### 影响

- `scripts/train_two_tower.py` 已通过语法检查，但 smoke test 未能运行。
- 不能启动 overnight 5 epoch training。
- 当前无法验证 loss、评估 mask、checkpoint 和 train log 输出。

### 初步原因判断

当前 `/workspace/amazon-two-tower/.venv` 中尚未安装 PyTorch。之前服务器全局环境曾验证过 PyTorch/CUDA 可用，但本项目现在要求运行 Amazon 代码时使用项目独立 `.venv`。

### 已尝试的排查步骤

- 使用 `.venv/bin/python -m py_compile scripts/train_two_tower.py` 验证脚本语法，通过。
- 使用 `.venv/bin/python scripts/train_two_tower.py --config configs/two_tower_movies_tv_5core.yaml --smoke_test` 运行 smoke test，失败于缺少 `torch`。

### 当前状态

已通过方案 C-1 缓解：重建项目 `.venv`，并使用 `--system-site-packages` 复用服务器全局 GPU PyTorch。新 `.venv` 仍是 Amazon 项目的唯一运行环境，不是直接切到全局 Python 跑训练。

### 后续复用建议

不要切换到全局 Python 偷跑训练。应先在 `.venv` 中安装合适版本的 PyTorch，并记录版本后再重新运行 smoke test。

### 追加排查记录

2026-05-10 尝试执行：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

pip 输出显示开始下载：

```text
Looking in indexes: https://download.pytorch.org/whl/cu126
Collecting torch
  Downloading torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (29 kB)
...
Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch)
  Downloading nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
```

随后安装进程长时间无输出，CPU 占用很低，`~/.cache/pip` 约 94MB。为避免无限等待，已终止该 pip 进程。终止后确认：

```text
torch check failed: ModuleNotFoundError("No module named 'torch'")
```

### 解决方案 C-1

2026-05-10 采用老师确认的方案 C-1：

- 备份旧 `.venv` 为 `.venv_backup_20260510_141029`。
- 使用 `/venv/main/bin/python -m venv .venv --system-site-packages` 重建项目 `.venv`。
- 不再在项目 `.venv` 中硬下载 `torch`、`nvidia-*`、`triton` 或 CUDA 大包。
- 新 `.venv` 继续作为 Amazon 项目唯一运行环境。
- `torch` 来自全局环境：
  - torch：`2.11.0+cu126`
  - torch file：`/venv/main/lib/python3.12/site-packages/torch/__init__.py`
  - cuda available：`True`
  - device：`NVIDIA GeForce RTX 3090`
- 项目数据依赖仍由 `.venv` 控制：
  - datasets：`2.17.0`
  - huggingface_hub：`0.36.2`
- GPU compute test 已通过。

后续可以在新 `.venv` 中重新运行 Two-Tower smoke test，但本次只做环境验证，没有启动训练。

## 读取 eval-only JSON 指标时裸 `python` 不存在

- 严重程度：Low
- 状态：已解决 / Resolved
- 日期：2026-05-10

### 现象

Two-Tower full valid / full test eval-only 已成功完成后，读取 `metrics_valid_full.json` 和 `metrics_test_full.json` 时并行执行了未激活 `.venv` 的裸 `python` 命令，命令失败。

### 报错原文或关键日志

```text
/bin/bash: line 1: python: command not found
```

### 影响

不影响 eval-only 运行结果。`outputs/two_tower_movies_tv_5core_full_eval/` 下的指标文件已经正常生成，只是第一次读取 JSON 的辅助命令失败。

### 初步原因判断

当前 shell 没有默认 `python` 命令；Amazon 项目应使用激活后的项目环境 `.venv`。

### 已尝试的排查步骤

- 确认 `outputs/two_tower_movies_tv_5core_full_eval/metrics_valid_full.json` 存在。
- 确认 `outputs/two_tower_movies_tv_5core_full_eval/metrics_test_full.json` 存在。
- 改用项目 `.venv` 后重新读取 JSON。

### 最终解决方案

使用以下命令读取指标：

```bash
source .venv/bin/activate && python -m json.tool outputs/two_tower_movies_tv_5core_full_eval/metrics_valid_full.json
source .venv/bin/activate && python -m json.tool outputs/two_tower_movies_tv_5core_full_eval/metrics_test_full.json
```

两条命令均成功。

### 后续复用建议

在 `/workspace/amazon-two-tower` 中运行 Python 命令时，先执行：

```bash
source .venv/bin/activate
```

## Movies_and_TV 5-core split 存在重复 user-item 跨 split

- 严重程度：High
- 状态：已解决 / Resolved
- 日期：2026-05-10

### 现象

Two-Tower full valid / full test evaluation 出现异常 gap：

- full valid `Recall@50=0.085741`
- full test `Recall@50=0.052154`

老师判断 valid 到 test 约 39% 的相对下降不太可能只由正常数据分布或 test 多 mask 一个 valid item 解释，因此优先做 bug diagnosis。

### 报错原文或关键日志

本次没有 Python exception。诊断脚本输出的关键数据为：

```text
test pair 出现在 train 中：2637
test pair 出现在 valid 中：10733
valid pair 出现在 train 中：2641
valid users：505425，其中单条 valid 的 users：505425
test users：505425，其中单条 test 的 users：505425
valid 中有但 test 中没有的 users：0
test 中有但 valid 中没有的 users：0
```

补充统计：

```text
valid/test same target users: 10733
test_in_train unique_users: 2637
test_in_valid unique_users: 10733
valid_in_train unique_users: 2641
train duplicate user-item rows beyond first: 54695
user-item pairs appearing in multiple splits: 10737
total unique user-item pairs: 5345014
```

### 影响

- 当前 Movies_and_TV 5-core train/valid/test 的 held-out target 与历史交互不是严格按 `(user_idx, item_idx)` 互斥。
- 同一用户同一 `parent_asin` 可能同时出现在 train / valid / test 中。
- 这会影响 ItemCF 和 Two-Tower 的 evaluation 语义，导致指标解释不够干净。
- 在修复或确认策略前，不建议继续启动 full training 或进入 text embedding。

### 初步原因判断

当前 preprocess 的 leave-one-out 是按 interaction 切分，可能没有在切分前对同一用户的重复 `parent_asin` 交互做去重或聚合。Amazon review 中同一用户可能对同一父商品多次出现交互记录，导致同一 `(user_idx, item_idx)` 跨 split。

### 已尝试的排查步骤

- 新增并运行只读诊断脚本：`scripts/diagnose_two_tower_eval_gap.py`
- 检查 `test` 的 `(user_idx, item_idx)` pair 是否出现在 `train` / `valid`。
- 检查 `valid` 的 `(user_idx, item_idx)` pair 是否出现在 `train`。
- 检查 valid/test 是否每个 user 各一条。
- 检查 valid/test user set 是否一致。
- 检查 `user_idx` / `item_idx` 是否越界。
- 小样本检查 test target 是否被 seen mask 误伤，以及 unmask 后 target 是否仍在候选集中。
- 比较 valid/test target popularity 分布。
- 小样本比较 `test mask=train seen` 和 `test mask=train+valid seen`。

### 当前状态

已解决。已按 Eddy 确认的策略修改 `scripts/preprocess_amazon.py`：在 `rating >= threshold` 过滤之前，对同一 `(user_id, parent_asin)` 按 `user_id`, `parent_asin`, `timestamp`, `original_row_idx` 稳定排序，并保留最新一条 interaction。

### 后续复用建议

后续处理 3-core 或其他品类时，也应在 `rating >= threshold` 过滤之前使用相同的 user-item 去重策略，避免同一 `(user_id, parent_asin)` 跨 train / valid / test split。

### 最终解决方案

2026-05-10 已重跑 Movies_and_TV 5-core preprocess，输出目录仍为 `data/processed/movies_tv_5core/`。

去重统计：

```text
n_interactions_before_dedup = 17328314
n_interactions_after_dedup = 17158519
dedup_removed_interactions = 169795
dedup_removal_ratio = 0.00979870286284055
```

新的 5-core 核心规模：

```text
n_users = 497449
n_items = 153977
n_interactions_total = 5314336
n_interactions_train = 4319438
n_interactions_valid = 497449
n_interactions_test = 497449
n_cold_items_in_valid = 312
n_cold_items_in_test = 979
cold_item_ratio_valid = 0.000627199974268719
cold_item_ratio_test = 0.0019680409449008844
```

clean split 验证结果：

```text
test_in_train = 0
test_in_valid = 0
valid_in_train = 0
valid_test_same_target_users = 0
train_duplicate_user_item_rows_beyond_first = 0
valid_duplicate_user_item_rows_beyond_first = 0
test_duplicate_user_item_rows_beyond_first = 0
user_item_pairs_appearing_in_multiple_splits = 0
```

当前不再继续使用旧 5-core preprocess 产物作为正式 baseline 输入。后续需要基于 clean 5-core 数据重跑 ItemCF 和 ID-only Two-Tower baseline。

### 结果归档

已基于 clean 5-core 数据完成 baseline 重建：

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

旧 ItemCF / Two-Tower 结果只保留为 bug diagnosis 记录，不再作为正式 baseline。下一步需要对 clean Two-Tower best checkpoint 跑 full valid / full test evaluation。

## Clean Two-Tower full valid / full test 仍存在明显 gap

- 严重程度：High
- 状态：Open
- 日期：2026-05-10

### 现象

clean Movies_and_TV 5-core 数据已经修复 split bug，并且 clean split 验证通过。但 clean ID-only Two-Tower 5 epoch checkpoint 做 full eval 后，valid / test 仍存在明显差距：

```text
full valid Recall@50 = 0.081591
full test Recall@50 = 0.046746
```

clean Two-Tower full test 也低于 clean ItemCF test baseline：

```text
clean ItemCF Recall@50 = 0.083570
clean Two-Tower Recall@50 = 0.046746
```

### 报错原文或关键日志

本次没有 Python exception。eval-only 正常完成：

```text
eval-only 完成：{"checkpoint": "outputs/two_tower_movies_tv_5core_clean_overnight/checkpoints/best_model.pt", "eval_split": "both", "output_dir": "outputs/two_tower_movies_tv_5core_clean_full_eval", "test_recall@50": 0.0467460269502689, "valid_recall@50": 0.08159119116058551}
```

### 影响

- clean Two-Tower 训练链路健康，但当前 full test 指标不能作为超过 ItemCF 的正式结论。
- 不建议直接启动 20/25/30 epoch full training。
- 需要先分析 clean valid-test gap，否则继续训练可能只是在扩大同一问题。

### 初步原因判断

旧 split 跨 split 重复 user-item 的 bug 已修复，因此当前 gap 需要重新诊断。可能方向包括：

- valid/test target popularity 或时间分布差异。
- ID-only batch negative 训练目标对下一跳 valid 更敏感，对更远的 test target 泛化不足。
- 5 epoch 训练尚未充分，但不能在未诊断前直接拉长训练。
- evaluation 口径虽然已对齐 `train_valid` test seen mask，但仍需复查 clean full eval 的 rank 分布。

### 已尝试的排查步骤

- 使用 clean checkpoint 执行 eval-only full valid / full test。
- 确认加载 checkpoint：`outputs/two_tower_movies_tv_5core_clean_overnight/checkpoints/best_model.pt`
- 确认 `eval_max_users=None`。
- 确认 full valid / full test 输出已生成：
  - `outputs/two_tower_movies_tv_5core_clean_full_eval/metrics_valid_full.json`
  - `outputs/two_tower_movies_tv_5core_clean_full_eval/metrics_test_full.json`
  - `outputs/two_tower_movies_tv_5core_clean_full_eval/two_tower_full_eval_report.md`

### 当前状态

Open。clean full eval 已完成，2026-05-11 已完成 clean gap diagnosis，但尚未关闭 issue。

### 后续复用建议

下一步先分析 clean valid-test gap，不要直接进入 20/25/30 epoch full training、text embedding、LogQ、temperature sweep 或 negative sampling ablation。

### 2026-05-11 追加诊断记录

本次新增并运行 `scripts/diagnose_clean_two_tower_gap.py`，只做诊断，不训练、不调参、不重跑正式 baseline。

生成输出：

```text
outputs/clean_two_tower_gap_diagnosis.md
outputs/clean_two_tower_gap_diagnosis.json
```

Two-Tower item popularity bucket `Recall@50`：

```text
<=5: valid=0.030506, test=0.017206
6-10: valid=0.047532, test=0.028045
11-20: valid=0.063839, test=0.039357
21-100: valid=0.083437, test=0.050934
101-500: valid=0.089584, test=0.051666
>500: valid=0.103644, test=0.056632
```

Two-Tower user history length bucket `Recall@50`：

```text
3-5: valid=0.097054, test=0.053495
6-10: valid=0.074147, test=0.043644
11-20: valid=0.053494, test=0.034030
21-50: valid=0.039707, test=0.028693
>50: valid=0.025755, test=0.022680
```

Two-Tower rank sanity check：

```text
valid Recall@20=0.059959, Recall@50=0.081591, Recall@100=0.102734
test Recall@20=0.032375, Recall@50=0.046746, Recall@100=0.061782
valid median rank=10024.000000
test median rank=16730.500000
```

ItemCF vs Two-Tower test hit overlap：

```text
both_hit=17272
itemcf_hit_only=24218
two_tower_hit_only=5936
both_miss=449044
ItemCF diagnostic Recall@50=0.083570
Two-Tower diagnostic Recall@50=0.046746
```

诊断结论：

- valid-test gap 在所有 item popularity bucket 中都存在；长尾 bucket 的绝对 Recall@50 最低，但 gap 不只集中在长尾。
- user history length 分桶中也普遍存在 gap；valid/test 的用户历史分布相同，因此 gap 不是由用户历史长度分布差异造成。
- ItemCF 的优势主要来自用户历史 item 的局部共现关系，`itemcf_hit_only` 的 target item popularity median 为 517。
- Two-Tower 仍有 ItemCF miss 但自己 hit 的样本，`two_tower_hit_only=5936`。
- rank sanity check 与已有 full eval 指标一致，本次未发现新的 evaluation bug 迹象。
- 当前更像是 ID-only 表达能力不足叠加 test target 更长尾、用户兴趣随时间漂移，而不是单纯训练轮数不足。
- issue 仍保持 Open，不建议直接进入 20/25/30 epoch full training。

本次运行期间出现一次 PyTorch warning：

```text
/workspace/amazon-two-tower/scripts/diagnose_clean_two_tower_gap.py:188: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:213.)
  user_tensor = torch.as_tensor(batch["user_idx"].to_numpy(dtype=np.int64), device=device)
```

处理结果：脚本已改为 `to_numpy(dtype=np.int64, copy=True)` 并通过 `py_compile`；未重新跑完整诊断。

### 2026-05-11 追加诊断 A：clean ItemCF full valid eval

本次目标：验证同一 clean split 上 ItemCF 是否也存在类似 valid-test gap。

本次新增配置：

```text
configs/itemcf_movies_tv_5core_clean_valid.yaml
```

本次输出目录：

```text
outputs/itemcf_movies_tv_5core_clean_valid/
```

本次没有启动 Two-Tower，没有训练，没有调参，没有覆盖 `outputs/itemcf_movies_tv_5core_clean/`。

valid eval seen mask 口径：

```text
eval_split=valid
eval_seen_filter=train
valid evaluation 过滤 train seen items
允许当前 valid target 作为候选
exclude is_cold_item_for_eval=True
```

clean ItemCF valid 指标：

```text
num_eval_users=497137
num_skipped_cold_users=312
num_no_recommendation_users=0
Recall@20=0.11312575809082809
Recall@50=0.1406976346560405
Recall@100=0.16352836340887925
NDCG@20=0.06192890723311102
NDCG@50=0.06740399170896363
NDCG@100=0.07110182453591324
MRR@20=0.04705190895483009
MRR@50=0.04793296612989793
MRR@100=0.048257025866777686
```

与 clean ItemCF test 对比：

```text
clean ItemCF valid Recall@50=0.1406976346560405
clean ItemCF test Recall@50=0.08357000422986283
absolute_drop=0.05712763042617767
relative_drop_from_valid=40.60%
```

诊断判断：

- ItemCF 在 clean valid/test 上也存在明显 gap，valid 到 test 的 `Recall@50` 相对下降约 40.60%。
- Two-Tower valid 到 test 的 `Recall@50` 相对下降约 42.71%，量级接近 ItemCF。
- 因此当前 valid-test gap 更可能来自 clean split 本身的难度差异或时间推进带来的目标变化，而不是 Two-Tower pipeline 特有 evaluation bug。
- Two-Tower test 仍明显低于 ItemCF test，Two-Tower 模型能力问题仍存在，但 valid-test gap 本身不能只归因于 Two-Tower pipeline。
- issue 仍保持 Open，不建议直接进入 20/25/30 epoch full training。
- 后续不建议继续把 Two-Tower eval bug 作为第一优先级。
- 下一阶段建议讨论 `text-enhanced item tower`、popularity correction、longer training ablation 的优先级。
- 当前更推荐先做 `text-enhanced item tower` 作为主线增强。

报告文案修正：

- 初次生成的 `outputs/itemcf_movies_tv_5core_clean_valid/itemcf_run_report.md` 复用了 test eval seen mask 文案。
- 已最小修改 `scripts/run_itemcf.py`，使报告按 `eval_split` 输出 seen mask 说明。
- 重新运行 clean ItemCF valid eval 后，报告已正确记录：`valid eval 过滤 train seen items，并允许当前 valid target 作为候选。`

### 2026-05-11 当前问题调整

根据 clean ItemCF full valid eval，当前 open issue 不再按 Two-Tower-specific evaluation bug 继续追。问题调整为：

```text
ID-only Two-Tower test baseline underperforms ItemCF
```

已确认事实：

```text
clean ItemCF valid Recall@50=0.140698
clean ItemCF test Recall@50=0.083570
clean ItemCF relative_drop_from_valid≈40.60%
clean ID-only Two-Tower valid Recall@50=0.081591
clean ID-only Two-Tower test Recall@50=0.046746
clean ID-only Two-Tower relative_drop_from_valid≈42.71%
```

判断：

- ItemCF 和 Two-Tower 在 clean split 上都有同量级 valid-test gap。
- 因此 large valid-test gap 不再像是 Two-Tower-specific evaluation bug。
- 更合理的解释是 leave-one-out + valid/test seen mask 口径下存在 split-level difficulty shift，test target 更难预测。
- 当前真正问题是 ID-only Two-Tower test `Recall@50=0.046746` 明显低于 clean ItemCF test `Recall@50=0.083570`。

下一步：

- 先做 clean ID-only Two-Tower 20 epoch baseline，验证 5 epoch 欠拟合假设。
- 暂不做 text-enhanced、LogQ、temperature sweep 或 negative sampling。
- issue 仍保持 Open，后续根据 20 epoch baseline 结果再判断是否需要进入 text-enhanced item tower 或其他增强方向。

### 2026-05-11 追加记录：clean ID-only Two-Tower 20 epoch baseline

本次目标：验证 clean ID-only Two-Tower 5 epoch 是否欠拟合。

本次新增配置：

```text
configs/two_tower_movies_tv_5core_clean_20epoch.yaml
```

本次输出：

```text
outputs/two_tower_movies_tv_5core_clean_20epoch/
logs/two_tower_clean_20epoch.log
```

本次没有做 text-enhanced、LogQ、temperature sweep、negative sampling，没有修改 preprocess 数据，没有重跑 ItemCF。

运行确认：

```text
device=cuda
n_users=497449
n_items=153977
train interactions=4319438
valid interactions=497449
test interactions=497449
epoch 1 first batch similarity min/max=-8.586546 / 8.787075
```

训练结果：

```text
best_epoch=18
best_valid_recall@50=0.091520
best checkpoint=outputs/two_tower_movies_tv_5core_clean_20epoch/checkpoints/best_model.pt
```

核心趋势：

```text
epoch 1 train_loss=9.428876 valid Recall@50=0.025940
epoch 5 train_loss=4.099209 valid Recall@50=0.081220
epoch 10 train_loss=3.609980 valid Recall@50=0.087960
epoch 15 train_loss=3.416934 valid Recall@50=0.089460
epoch 18 train_loss=3.345820 valid Recall@50=0.091520
epoch 20 train_loss=3.308424 valid Recall@50=0.090820
```

与 5 epoch 结果对比：

```text
5 epoch valid subset Recall@50=0.081220
20 epoch best valid subset Recall@50=0.091520
absolute_improvement=0.010300
relative_improvement≈12.68%
```

诊断判断：

- 20 epoch training 正常完成，未发现 `OOM`、`Killed`、`Traceback`、`nan`、`inf`、`FloatingPointError` 或 `RuntimeError`。
- train_loss 持续下降，说明 5 epoch 确实偏欠拟合。
- valid `Recall@50` 到 epoch 18 达到最高，epoch 19/20 略有回落，说明后期接近平台。
- 20 epoch best valid `Recall@50=0.091520` 仍明显低于 clean ItemCF valid `Recall@50=0.140698`，ID-only Two-Tower underperform 问题仍未解决。
- 下一步建议用 eval-only 跑 20 epoch best checkpoint 的 clean full valid / test，再判断 test 指标是否改善。

本次运行期间出现一次 PyTorch warning：

```text
/workspace/amazon-two-tower/scripts/train_two_tower.py:419: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:213.)
  user_tensor = torch.as_tensor(batch["user_idx"].to_numpy(dtype=np.int64), device=device)
```

### 2026-05-11 追加记录：20 epoch best checkpoint full eval

本次任务严格 eval-only，未 retrain，未修改 preprocess，未启动 text-enhanced、LogQ、temperature sweep、negative sampling 或 ItemCF。

加载 checkpoint：

```text
outputs/two_tower_movies_tv_5core_clean_20epoch/checkpoints/best_model.pt
```

该 checkpoint 对应 `best_epoch=18`。

输出目录：

```text
outputs/two_tower_movies_tv_5core_clean_20epoch_full_eval/
```

执行命令：

```bash
source .venv/bin/activate && python scripts/train_two_tower.py --config configs/two_tower_movies_tv_5core_clean_20epoch.yaml --eval_only --checkpoint outputs/two_tower_movies_tv_5core_clean_20epoch/checkpoints/best_model.pt --eval_split both --eval_output_dir outputs/two_tower_movies_tv_5core_clean_20epoch_full_eval
```

full valid 指标：

```text
num_eval_users=497137
num_skipped_cold_users=312
candidate_count_min=153907
candidate_count_max=153974
target_item_in_candidate_range=true
topk_shape=[256, 100]
eval_batch_size=256
Recall@20=0.06710826190768339
Recall@50=0.09214361433568614
Recall@100=0.1167726401374269
NDCG@50=0.039984435523071675
MRR@50=0.02665389295939516
```

full test 指标：

```text
num_eval_users=496470
num_skipped_cold_users=979
candidate_count_min=153906
candidate_count_max=153973
target_item_in_candidate_range=true
topk_shape=[256, 100]
eval_batch_size=256
Recall@20=0.036578242391282455
Recall@50=0.05319757487864322
Recall@100=0.07086228775152577
NDCG@50=0.021494396747563677
MRR@50=0.013541905091225163
```

对比结论：

- 20 epoch full valid `Recall@50=0.092144`。
- 20 epoch full test `Recall@50=0.053198`。
- valid/test gap 仍存在，relative drop 约 42.27%。
- 20 epoch full test 仍低于 clean ItemCF test `Recall@50=0.083570`。
- 20 epoch 相比 5 epoch 有提升，但提升有限；结合 epoch 14-20 平台期，结果支持 ID-only representation ceiling 判断。
- 不建议继续增加 epoch。
- 结果支持下一步转向 `text-enhanced item tower`。

本次 eval-only 期间出现同类 PyTorch warning：

```text
/workspace/amazon-two-tower/scripts/train_two_tower.py:419: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:213.)
  user_tensor = torch.as_tensor(batch["user_idx"].to_numpy(dtype=np.int64), device=device)
```

## 2026-05-11 - text-enhanced item tower 尚未实现（已解决）

状态：Resolved（M6.1 已实现 smoke test）

本次尝试：

- 检查 text-enhanced item tower 是否已有实现或部分实现。
- 检查范围包括 configs、`scripts/train_two_tower.py`、`scripts/preprocess_amazon.py`、processed clean data schema、`docs/data_notes.md`、`outputs/inspection_movies_and_tv.md`。

发现：

- 当前 `text-enhanced item tower` 尚未实现。
- 现有 Two-Tower training pipeline 是 ID-only，dataloader 只读取 `user_idx` / `item_idx`。
- 当前 model 只包含 user/item ID embedding，没有 item text/title/category/meta encoder 或 fusion。
- 当前 processed clean data 不包含 item text/meta feature artifact。
- raw metadata 中存在可用 item-side 字段：`title`、`description`、`features`、`categories`、`main_category`，可通过 `parent_asin` 与 processed item mapping 对齐。

当前状态：

- Pending。
- 需要先确认 item feature artifact、text representation、fusion 方式和 smoke test config，再启动 text-enhanced full training。

### 2026-05-11 更新

M6.1 已完成：

- `scripts/train_text_two_tower.py` 已实现，smoke test PASS。
- item feature artifact：`outputs/item_text_embeddings/movies_tv_5core/item_text_embedding.npy` 和 `item_has_text.npy` 已生成。
- 状态更新为 Resolved。

---

## 2026-05-11 - 38.3% item 缺少 title/description，text embedding 只能 fallback parent_asin

- 严重程度：中等
- 状态：Mitigated（通过 has_text mask 缓解）
- 日期：2026-05-11

### 现象

在 M5.2 生成全量 item text embedding 时，发现 153977 个 item 中有 58961 个（38.3%）同时缺少 title 和 description，只能用 parent_asin 作为文本输入。

### 影响

- 这 38.3% 的 item 的 text embedding 不含语义信息，接近随机方向。
- M5.5 pure text retrieval 证实：target has_text=0 item 的 Recall@50 在 valid 和 test 上只有约 0.0019 和 0.0016。
- 如果 M6 对所有 item 等权使用 text embedding，fallback embedding 会引入噪声，误导模型。

### 已确认数据

```text
title + description : 88243 / 153977 = 57.3%
title only          : 6772  / 153977 = 4.4%
description only    : 1     / 153977 = 0.0%
fallback (empty)    : 58961 / 153977 = 38.3%
```

target item has_text=0 ratio（M5.5 eval）：

```text
valid target has_text=0 ratio = 54.5%
test  target has_text=0 ratio = 54.5%
```

### 已缓解措施

M6.1 中通过 `has_text mask` 处理：对 `has_text=False` 的 item，text_proj 输出乘以 0（屏蔽），使其 text path 不参与 fusion。

具体：

```python
txt_proj = txt_proj * self._has_text[item_idx].unsqueeze(-1)
# _has_text: bool -> float, has_text=False 的 item text_proj 被置 0
```

### 后续复用建议

M6.2 正式训练使用 `use_has_text_mask: true`（已写入 config），无需额外操作。若后续引入新的 text 特征（categories、features 等），需同步更新 has_text mask 构建逻辑。

---

## 2026-05-11 - sentence-transformers 安装时 huggingface_hub 版本升级

- 严重程度：低
- 状态：观察中（当前无已知问题）
- 日期：2026-05-11

### 现象

安装 sentence-transformers 时，huggingface_hub 从 `0.36.2` 升级到 `1.14.0`，超出原先固定版本范围。

### 影响

- datasets `2.17.0` 内部依赖 huggingface_hub；版本不匹配理论上存在兼容风险。
- 当前实际运行未发现错误：`load_dataset` 和 sentence-transformers model loading 均正常。
- item text embedding（M5.2）和 pure text retrieval（M5.5）均正常完成。

### 当前状态

- 暂不回滚。
- 若后续 `load_dataset` 出现兼容性错误，优先考虑在 `.venv` 中 pin `huggingface_hub<1.0.0` 并重新安装。

### 后续复用建议

每次在 `.venv` 中安装新包时，检查 `huggingface_hub` 版本是否被升级，避免对 `datasets` 产生意外影响。

---

## 2026-05-11 - M6.1 train_text_two_tower.py 存在 UserWarning（非阻塞）

- 严重程度：低（不影响训练）
- 状态：Open（可后续修复）
- 日期：2026-05-11

### 现象

smoke test 运行期间出现 PyTorch warning：

```text
UserWarning: Converting a tensor with requires_grad=True to a scalar,
which will raise an error in a future PyTorch release.
```

发生位置：`scripts/train_text_two_tower.py` 中 `float(logits.min())` 和 `float(logits.max())` 诊断日志。

### 影响

- 不影响梯度计算、训练过程或 checkpoint 质量。
- 仅影响第一个 epoch 的诊断日志输出。
- M6.2 正式训练期间会持续出现该 warning，但不会导致训练失败。

### 当前状态

Open。smoke test 已确认此 warning 不阻塞训练，目前不修复。

### 建议修复方案

将相关行改为：

```python
float(logits.min().detach())
float(logits.max().detach())
```

后续有空时修复；修复后需重新 `py_compile` 确认语法。

---

## 2026-05-11 - Text-enhanced additive v1 增益有限，test 未超过 β 阈值

- 严重程度：中等（影响简历叙事，但不阻塞训练链路）
- 状态：Open
- 日期：2026-05-11

### 现象

M6.2 additive residual text-enhanced Two-Tower 20 epoch full eval 完成后，test Recall@50 仅比 ID-only Two-Tower 提升 +0.001363（+2.56%），按预设阈值属于 borderline γ 分支：

```text
ID-only Two-Tower 20ep full test Recall@50 = 0.053198
Additive text-enhanced 20ep full test Recall@50 = 0.054561
absolute_gain = +0.001363
relative_gain = +2.56%
branch = γ（test < 0.055，borderline：差距 0.000439）
```

### 根因分析

主要瓶颈是 **54.7% 的 test target item 没有真实文本**（has_text=0），text path 被 has_text mask 屏蔽，additive fusion 退化为 ID-only：

```text
has_text=1 target：224753 / 496470 = 45.27%  → Recall@50 = 0.070464
has_text=0 target：271717 / 496470 = 54.73%  → Recall@50 = 0.041407
```

has_text=1 group 的提升是真实的（text 有效），但 has_text=0 group 占多数，拉低了整体指标。

text embedding 覆盖率分布（全量 item）：

```text
title + description : 88243 / 153977 = 57.3%
title only          : 6772  / 153977 = 4.4%
description only    : 1     / 153977 = 0.0%
fallback (parent_asin) : 58961 / 153977 = 38.3%
```

### 三方对比（Recall@50）

| 方法 | valid | test |
| --- | ---: | ---: |
| clean ItemCF | 0.140698 | 0.083570 |
| ID-only Two-Tower 20ep | 0.092144 | 0.053198 |
| Additive text-enhanced 20ep | 0.093940 | 0.054561 |

两个 Two-Tower 版本均远低于 clean ItemCF，text 增益绝对量小。

### 影响

- 当前简历叙事不能写"text-enhanced Two-Tower 超过 ItemCF 或超过纯 ID 模型"。
- borderline γ 的定性是：text 有效但覆盖率限制了总体提升，不是架构错误。
- additive v1 仍作为第一个干净 text-enhanced baseline 保留，为后续 ablation 提供参照点。

### 潜在后续实验方向（待讨论）

1. **扩充 text 覆盖率**：补充 item categories / features 等字段，降低 has_text=0 比例。
2. **popularity correction / frequency-based re-weighting**：两个 Two-Tower 模型都低于 ItemCF，popularity bias 是潜在方向。
3. **更大 text encoder 或 fine-tune text projection**：当前 Linear(384→64) projection frozen text，可尝试更大 projection 或 unfreezing。
4. **LogQ correction**：解决 in-batch negative popularity bias。

### 后续复用建议

当前 additive v1 作为正式 text-enhanced baseline，输出目录固定为：

```text
outputs/text_two_tower_additive_movies_tv_5core_20epoch/
```

后续任何新的 text fusion 变体需使用新的 `output_dir`，不覆盖当前输出。

### 2026-05-11 D1+D2 诊断证据追加

**D1：ID-only has_text split**

| has_text | ID-only R@50 | Text-enhanced R@50 | delta |
| --- | ---: | ---: | ---: |
| 1（有文本） | 0.068173 | 0.070464 | +0.002291 |
| 0（无文本） | 0.040811 | 0.041407 | +0.000596 |

- has_text=1 group 增益为 +0.002291（+3.4%），text signal 有效。
- has_text=0 group 增益接近零（+0.000596），additive fusion 退化为 ID-only，mask 有效。

**D2：popularity bucket × model matrix（test Recall@50）**

| bucket | ItemCF | ID-only | Text-enhanced |
| --- | ---: | ---: | ---: |
| ≤5 | 0.040405 | 0.023284 | 0.024197 |
| 6–20 | 0.047940 | 0.043748 | 0.046240 |
| 21–100 | 0.060890 | 0.062918 | 0.064433 |
| >100 | 0.122522 | 0.054604 | 0.055465 |

- ItemCF 在 >100 热门 bucket（42.8% targets）上大幅领先（0.1225 vs 0.0546）。这是 ItemCF 整体优于 Two-Tower 的主要来源。
- Two-Tower 在 21–100 中等热度 bucket 持平或略优于 ItemCF。
- text-enhanced 在所有 bucket 上均小幅优于 ID-only。

**影响更新**

- 简历叙事应重点描述 popularity bucket 结构性差异，而非整体 Recall 数字。
- 当前最强叙事：ItemCF 依赖局部共现（热门 item 强）；Two-Tower 依赖全局 embedding（中等热度 item 有泛化优势）；text-enhanced 在有真实元数据的 item 上有实质增益。
- 推迟 hybrid retrieval 和进一步架构调参至 5/15 之后。

---

## 2026-05-13 - 新服务器缺少项目环境与大产物，需要重建 preprocess 链路

- 严重程度：High
- 状态：Partially Resolved（项目 `.venv` 和 clean 5-core processed data 已恢复；checkpoints / embeddings 仍缺失）
- 日期：2026-05-13

### 现象

新服务器接管检查确认：

```text
无 data/processed/
无 checkpoints
无 item_text_embedding.npy
无旧 HuggingFace cache
无项目 .venv
/venv/main/bin/python 存在，且系统环境中有 PyTorch/CUDA
```

### 影响

- 不能直接继续 Faiss benchmark。
- 不能直接导出 item tower embeddings 或 eval user embeddings。
- 不能直接复用历史 ID-only / text-enhanced checkpoint。
- 必须先恢复数据链路，至少重新生成 Movies_and_TV clean 5-core processed dataset。

### 已执行恢复步骤

按项目既有原则，在项目目录内重建 `.venv`，并复用系统 PyTorch/CUDA：

```bash
/venv/main/bin/python -m venv .venv --system-site-packages
```

安装 preprocess 最小依赖：

```bash
.venv/bin/python -m pip install 'datasets==2.17.0' 'huggingface_hub==0.36.2' pyyaml pandas pyarrow
```

验证结果：

```text
Python = 3.12.13
datasets = 2.17.0
huggingface_hub = 0.36.2
pandas = 3.0.3
pyarrow = 24.0.0
PyYAML = 6.0.3
torch = 2.11.0+cu128
torch file = /venv/main/lib/python3.12/site-packages/torch/__init__.py
cuda_available = True
GPU = NVIDIA GeForce RTX 3090
```

重新下载 / 加载 Amazon Reviews 2023 Movies_and_TV，并运行 clean 5-core preprocess：

```bash
HF_HOME=/workspace/.hf_home HF_DATASETS_CACHE=/workspace/.hf_home/datasets \
  .venv/bin/python scripts/preprocess_amazon.py \
  --config configs/preprocess_movies_tv_5core.yaml
```

### 恢复结果

preprocess 成功完成，输出目录：

```text
data/processed/movies_tv_5core/
```

已生成：

```text
README.md
id2item.json
id2user.json
item2id.json
stats.json
test.parquet
train.parquet
user2id.json
valid.parquet
```

核心规模：

```text
n_interactions_before_dedup = 17328314
n_interactions_after_dedup  = 17158519
dedup_removed_interactions  = 169795
n_users                     = 497449
n_items                     = 153977
n_interactions_total        = 5314336
n_interactions_train        = 4319438
n_interactions_valid        = 497449
n_interactions_test         = 497449
n_cold_items_in_valid       = 312
n_cold_items_in_test        = 979
cold_item_ratio_valid       = 0.000627199974268719
cold_item_ratio_test        = 0.0019680409449008844
```

split schema / row count 已验证：

```text
train.parquet rows = 4319438
valid.parquet rows = 497449, cold = 312
test.parquet  rows = 497449, cold = 979
user_idx range = 0..497448
item_idx range = 0..153976
```

### 当前未恢复部分

仍缺少：

```text
outputs/two_tower_movies_tv_5core_clean_20epoch/checkpoints/best_model.pt
outputs/text_two_tower_additive_movies_tv_5core_20epoch/checkpoints/best_model.pt
outputs/item_text_embeddings/movies_tv_5core/item_text_embedding.npy
导出的 item tower embeddings
导出的 eval user embeddings
ID-only / text-enhanced full eval outputs
```

### 后续状态更新

2026-05-13 后续已在新服务器重新生成 Movies_and_TV clean 5-core processed data；已重训并恢复 ID-only Two-Tower checkpoint；已完成 ID-only Faiss benchmark，并生成离线 benchmark embeddings。text-enhanced checkpoint 和 item text embeddings 仍未恢复，本阶段未处理。

### 后续复用建议

- 当前状态只恢复了数据链路，不应直接进入 Faiss benchmark。
- 下一步需要先恢复或重训 checkpoint。
- 若重训，仍应使用项目 `.venv`，不要直接切到全局 Python。
- `data/processed/`、`.venv/`、HF cache、checkpoints、embedding 均不提交 Git。

## 项目 `.venv` 初始缺少 `faiss`

- 严重程度：Low
- 状态：已解决
- 日期：2026-05-13

### 现象

启动 ID-only Two-Tower Faiss benchmark 前检查依赖时，项目 `.venv` 中无法导入 `faiss`。

### 报错原文或关键日志

```text
ModuleNotFoundError: No module named 'faiss'
```

### 影响

- 无法直接运行 Faiss FlatIP / IVF-Flat offline retrieval benchmark。
- 不影响已有 processed data、checkpoint 或 full eval metrics。

### 初步原因判断

新服务器项目 `.venv` 是按最小 preprocess / training 依赖恢复的，此前没有安装 Faiss。

### 已尝试的排查步骤

- 使用项目 `.venv` 执行 `import faiss` 检查，确认缺失。
- 遵守本轮要求，只安装 CPU 版 Faiss，不安装 GPU Faiss。

### 最终解决方案

只在项目 `.venv` 中安装：

```bash
.venv/bin/python -m pip install faiss-cpu
```

安装后确认：

```text
faiss-cpu = 1.13.2
Location = /workspace/amazon-two-tower/.venv/lib/python3.12/site-packages
```

### 后续复用建议

- Faiss benchmark 继续使用项目 `.venv`。
- 当前阶段优先使用 `faiss-cpu`；如后续需要 GPU Faiss，应单独评估 CUDA / PyTorch / Faiss 版本兼容性，并由 Eddy 确认后再安装。

## 直接 import benchmark 脚本重写报告时缺少 `PYTHONPATH=scripts`

- 严重程度：Low
- 状态：已解决
- 日期：2026-05-13

### 现象

Faiss nprobe sweep 完成后，为了只按既有 JSON 重写 markdown 报告顺序，曾用 `importlib` 直接加载 `scripts/benchmark_faiss_id_two_tower.py`。该命令没有设置 `PYTHONPATH=scripts`，导致脚本中的本地模块导入失败。

### 报错原文或关键日志

```text
ERROR: 缺少依赖：train_two_tower。请先在项目 .venv 中安装 package：train_two_tower
```

### 影响

- 不影响 nprobe sweep 结果。
- 不影响 `benchmark_results.json`。
- 只影响一次 markdown 报告重写命令。

### 初步原因判断

直接通过 `importlib.util.spec_from_file_location` 加载脚本时，当前 Python import path 没有包含 `scripts/`，因此无法解析同目录下的 `train_two_tower.py`。

### 已尝试的排查步骤

- 确认 benchmark sweep 已完成且 JSON 已写入。
- 使用正确的 import path 重新运行只重写报告的命令。

### 最终解决方案

使用：

```bash
PYTHONPATH=scripts .venv/bin/python - <<'PY'
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('bench', 'scripts/benchmark_faiss_id_two_tower.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
result = bench.read_json(Path('outputs/faiss_id_two_tower_clean_20epoch/benchmark_results.json'))
bench.write_report(Path('outputs/faiss_id_two_tower_clean_20epoch/benchmark_report.md'), result)
PY
```

报告已成功重写，未重新运行 benchmark。

### 后续复用建议

- 若从 repo root 直接 import `scripts/` 下的单文件脚本，应设置 `PYTHONPATH=scripts`，或优先通过脚本命令行入口执行。

## 2026-05-13 - Mean pooling 5epoch limited eval 完成，作为 ID-only 表达能力问题的用户塔增强对照

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-13

### 现象

此前 ID-only Two-Tower 5epoch limited valid baseline 为：

```text
Recall@50 = 0.0812
```

为验证轻量用户塔升级是否能改善 ID-only 表达能力，本轮运行 `user_id embedding + historical item mean pooling` 的 5epoch limited valid eval。

### 影响范围

- 只影响 mean pooling controlled experiment 记录。
- 未修改 processed data。
- 未运行 full valid/test。
- 未运行 20epoch。
- 未运行 text-enhanced、Faiss 新实验、Transformer、LogQ、负采样或 hybrid retrieval。

### 已尝试的排查步骤

使用配置：

```text
configs/two_tower_movies_tv_5core_mean_pool_5epoch.yaml
```

输出目录：

```text
outputs/user_mean_pool_5ep/
```

日志：

```text
logs/user_mean_pool_5ep.log
```

### 结果

```text
eval setting = limited valid, eval_max_users=50000
best_epoch = 4
best Recall@50 = 0.095680
Recall@20 = 0.069580
Recall@100 = 0.122080
NDCG@50 = 0.041792
MRR@50 = 0.028037
```

每轮 Recall@50：

```text
epoch 1 = 0.085020
epoch 2 = 0.093880
epoch 3 = 0.095340
epoch 4 = 0.095680
epoch 5 = 0.095320
```

### 客观判断

- Mean pooling 5epoch limited valid `Recall@50=0.095680` 高于 ID-only 5epoch limited valid `Recall@50=0.0812`。
- train_loss 正常下降。
- best epoch 出现在 epoch 4，epoch 5 略有回落。
- NDCG@50 / MRR@50 相比 ID-only 5epoch baseline 同步改善。
- 该结果支持把 mean pooling 作为用户塔轻量增强候选，但仍不是 full valid/test 结论。

### 后续复用建议

- 可以建议进入 20epoch，但只在 Eddy 明确批准后执行。
- 若进入 20epoch，继续保持 limited-valid 监控，不提前运行 full test。
- 不要把本结果写成超过 ItemCF、线上效果提升或最终收敛模型。

## 2026-05-13 - Mean pooling 20epoch limited eval 完成，full valid/test 暂缓

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-13

### 现象

Mean pooling 5epoch limited valid 已高于 ID-only 5epoch baseline：

```text
ID-only 5epoch limited valid Recall@50 = 0.0812
Mean pooling 5epoch limited valid Recall@50 = 0.095680
```

本轮按同一受控口径继续运行 mean pooling 20epoch limited valid eval，用于判断是否值得后续进入 full valid/test。

### 影响范围

- 只影响 mean pooling controlled experiment 记录。
- 未修改 processed data。
- 未运行 full valid/test。
- 未运行 text-enhanced、Transformer、LogQ、负采样、Faiss sweep 或 hybrid retrieval。
- 未修改 ID-only baseline checkpoint 或结果。

### 已尝试的排查步骤

使用配置：

```text
configs/two_tower_movies_tv_5core_mean_pool_20epoch.yaml
```

输出目录：

```text
outputs/user_mean_pool_20ep/
```

日志：

```text
logs/user_mean_pool_20ep.log
```

### 结果

```text
eval setting = limited valid, eval_max_users=50000
best_epoch = 6
best Recall@50 = 0.096580
Recall@20 = 0.069780
Recall@100 = 0.122120
NDCG@50 = 0.042543
MRR@50 = 0.028778
```

每轮 Recall@50：

```text
epoch 1 = 0.085020
epoch 2 = 0.093880
epoch 3 = 0.095340
epoch 4 = 0.095680
epoch 5 = 0.095320
epoch 6 = 0.096580
epoch 7 = 0.094740
epoch 8 = 0.095640
epoch 9 = 0.095600
epoch 10 = 0.096000
epoch 11 = 0.095380
epoch 12 = 0.095920
epoch 13 = 0.096180
epoch 14 = 0.095120
epoch 15 = 0.095380
epoch 16 = 0.095040
epoch 17 = 0.094900
epoch 18 = 0.095540
epoch 19 = 0.095100
epoch 20 = 0.094820
```

### 客观判断

- Mean pooling 20epoch limited valid `Recall@50=0.096580` 略高于 mean pooling 5epoch limited valid `Recall@50=0.095680`。
- 该结果高于 ID-only 5epoch limited valid `Recall@50=0.0812`。
- best epoch 出现在 epoch 6，后续 Recall@50 在相近区间波动；不能据此声称最终收敛或线上提升。
- 可以建议 5/14 运行 full valid/test，但只在 Eddy 明确批准后执行。

### 后续复用建议

- 若继续，应只跑 mean pooling 20epoch full valid/test，不扩展到新结构或调参。
- 不要把 limited valid 结果写成 full eval、线上效果或超过 ItemCF 的结论。

## 2026-05-13 - Mean pooling 20epoch full valid/test eval 完成

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-13

### 现象

Mean pooling 20epoch limited valid best checkpoint 为：

```text
checkpoint = outputs/user_mean_pool_20ep/checkpoints/best_model.pt
best_epoch = 6
limited valid best Recall@50 = 0.096580
```

本轮需要验证该 checkpoint 在 full valid/test offline evaluation 上是否仍然优于 ID-only 20epoch full eval baseline。

### 影响范围

- 只运行 eval-only full valid/test。
- 未继续训练。
- 未修改模型结构。
- 未调参。
- 未运行 Faiss、text-enhanced、Transformer、LogQ、负采样或 hybrid retrieval。

### 已尝试的排查步骤

为避免误触发训练，`scripts/train_mean_pool_two_tower.py` 增加了最小 eval-only 参数：

```text
--eval_only
--checkpoint
--eval_output_dir
--full_eval
```

执行：

```text
python scripts/train_mean_pool_two_tower.py \
  --config configs/two_tower_movies_tv_5core_mean_pool_20epoch.yaml \
  --eval_only \
  --full_eval \
  --checkpoint outputs/user_mean_pool_20ep/checkpoints/best_model.pt \
  --eval_output_dir outputs/user_mean_pool_20ep_full_eval
```

输出目录：

```text
outputs/user_mean_pool_20ep_full_eval/
```

日志：

```text
logs/user_mean_pool_20ep_full_eval.log
```

### 结果

Full valid：

```text
num_eval_users = 497137
num_skipped_cold_users = 312
Recall@20 = 0.069880
Recall@50 = 0.096309
Recall@100 = 0.122059
NDCG@50 = 0.041919
MRR@50 = 0.028022
```

Full test：

```text
num_eval_users = 496470
num_skipped_cold_users = 979
Recall@20 = 0.042462
Recall@50 = 0.061601
Recall@100 = 0.081028
NDCG@50 = 0.025176
MRR@50 = 0.016047
```

对比 ID-only 20epoch full eval：

```text
full valid Recall@50: 0.096309 vs 0.092144, absolute improvement +0.004166, relative improvement +4.52%
full valid NDCG@50:  0.041919 vs 0.039984, delta +0.001934
full valid MRR@50:   0.028022 vs 0.026654, delta +0.001368
full test Recall@50: 0.061601 vs 0.053198, absolute improvement +0.008403, relative improvement +15.80%
full test NDCG@50:  0.025176 vs 0.021494, delta +0.003681
full test MRR@50:   0.016047 vs 0.013542, delta +0.002505
```

### 客观判断

- Mean pooling 20epoch checkpoint 在 full valid/test offline eval 上保持了相对 ID-only 20epoch full baseline 的提升。
- 该结果支持把 `user_id embedding + historical item mean pooling` 记录为一个有效的用户塔轻量增强候选。
- 不能把该结果写成线上效果、超过 ItemCF 或最终模型一定更强。

### 后续复用建议

- 可以将 full valid/test 结果用于 5/15 简历中的 offline evaluation 对比。
- 若后续继续扩展，应先由 Eddy 明确批准实验边界。

## 2026-05-13 - Text + Mean Pooling combination experiment completed

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-13

### 现象

最终组合实验需要复用 text-enhanced item tower 与 mean-pooling user tower，但新服务器缺少历史 text embedding cache。

### 影响范围

- 若没有 `item_text_embedding.npy` 与 `item_has_text.npy`，组合模型无法按既有 text-enhanced 逻辑运行。
- 本轮没有改动已有 ID-only、text-enhanced、mean-pooling baseline 结果。
- 未运行 Faiss、Transformer、attention、LogQ、负采样或 hybrid retrieval。

### 处理

- 按既有 text embedding 规则重新生成 cache：

```text
outputs/item_text_embeddings/movies_tv_5core/item_text_embedding.npy
outputs/item_text_embeddings/movies_tv_5core/item_has_text.npy
```

- 生成结果：

```text
embedding shape = (153977, 384)
has_text = 95016 / 153977
fallback = 58961
```

- 新增独立组合脚本与配置：

```text
scripts/train_text_mean_pool_two_tower.py
configs/two_tower_movies_tv_5core_text_mean_pool_smoke.yaml
configs/two_tower_movies_tv_5core_text_mean_pool_5epoch.yaml
configs/two_tower_movies_tv_5core_text_mean_pool_20epoch.yaml
```

### 结果

Limited valid 50K：

```text
5epoch best epoch = 3
5epoch best Recall@50 = 0.099020
20epoch best epoch = 3
20epoch best Recall@50 = 0.099020
```

Full valid/test：

```text
full valid Recall@50 = 0.099628
full valid NDCG@50 = 0.042867
full valid MRR@50 = 0.028419

full test Recall@50 = 0.066042
full test NDCG@50 = 0.026751
full test MRR@50 = 0.016922
```

对比 Mean Pooling 20epoch full eval：

```text
full valid Recall@50: 0.099628 vs 0.096309, absolute delta +0.003319
full test Recall@50:  0.066042 vs 0.061601, absolute delta +0.004441
```

### 客观判断

- 组合模型在当前 offline full valid/test evaluation 中高于 mean pooling 单独模型。
- 该结果说明 text item signal 在 mean pooling user tower 基础上仍有边际收益。
- 不能把该结果写成线上效果、超过 ItemCF 或最终架构一定最优。

## 2026-05-13 - Text + Mean Pooling temperature sweep completed

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-13

### 现象

为 5/15 项目报告和简历补充 ablation，需要在 Text + Mean Pooling Two-Tower 架构上检查 softmax temperature τ 对 limited-valid retrieval metrics 的影响。

### 影响范围

- 只运行 5epoch limited valid eval，`eval_max_users=50000`。
- τ values: `0.05`, `0.07`, `0.10`, `0.15`。
- τ=0.07 复用已有 Text + Mean Pooling 5epoch limited valid 结果。
- 除 τ 外所有超参保持不变。
- 未运行 full valid/test，未运行 Faiss，未加入 Transformer、attention、LogQ 或其他结构。

### 结果

| temperature τ | valid Recall@20 | valid Recall@50 | valid Recall@100 | valid NDCG@50 | valid MRR@50 | best epoch | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 0.061800 | 0.088840 | 0.115280 | 0.037190 | 0.024179 | 2 | new run |
| 0.07 | 0.070440 | 0.099020 | 0.125780 | 0.042686 | 0.028408 | 3 | reused existing 5epoch result |
| 0.10 | 0.081560 | 0.111880 | 0.142600 | 0.048586 | 0.032416 | 5 | new run |
| 0.15 | 0.083240 | 0.117240 | 0.150240 | 0.049176 | 0.031891 | 4 | new run |

```text
Best τ = 0.15
Recall@50 absolute change vs τ=0.07 = +0.018220
Recall@50 relative change vs τ=0.07 = +18.40%
```

### 客观判断

- τ=0.15 achieved the best limited-valid Recall@50 in this sweep.
- A higher temperature improved the 5epoch limited-valid result for the current Text + Mean Pooling architecture.
- This remains an offline limited-valid ablation; it does not imply online improvement or full-test improvement.

## 2026-05-14 - Text + Mean Pooling extended temperature sweep completed

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-14

### 现象

上一轮 temperature sweep 中 τ=0.15 was the best under 5epoch limited-valid evaluation. 本轮继续测试更大的 τ=0.20 和 τ=0.30，确认是否还能继续提升。

### 影响范围

- 架构仍为 Text + Mean Pooling Two-Tower。
- 只改变 τ。
- τ=0.15 复用上一轮结果。
- τ=0.20 和 τ=0.30 是新 5epoch limited-valid runs。
- 除 τ 外所有超参保持不变。
- 未运行 full valid/test，未运行 Faiss，未加入 Transformer、attention、LogQ 或其他结构。

### 结果

| temperature τ | valid Recall@20 | valid Recall@50 | valid Recall@100 | valid NDCG@50 | valid MRR@50 | best epoch | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | 0.081560 | 0.111880 | 0.142600 | 0.048586 | 0.032416 | 5 | previous sweep reference |
| 0.15 | 0.083240 | 0.117240 | 0.150240 | 0.049176 | 0.031891 | 4 | previous best from initial sweep |
| 0.20 | 0.078500 | 0.113700 | 0.148660 | 0.046526 | 0.029581 | 4 | new run |
| 0.30 | 0.069300 | 0.102440 | 0.138040 | 0.040254 | 0.024696 | 2 | new run |

```text
Best τ = 0.15
Recall@50 absolute change vs τ=0.15 = +0.000000
Recall@50 relative change vs τ=0.15 = +0.00%
Best newly tested larger τ = 0.20
τ=0.20 Recall@50 delta vs τ=0.15 = -0.003540
```

### 客观判断

- τ=0.15 remains the best value among τ=0.10/0.15/0.20/0.30 under this 5epoch limited-valid setting.
- Larger temperatures did not continue improving Recall@50.
- This remains an offline limited-valid ablation; it does not imply online improvement or full-test improvement.

## 2026-05-14 - Text + Mean Pooling τ=0.15 20epoch full eval completed

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-14

### 现象

Temperature sweep selected τ=0.15 under 5epoch limited-valid evaluation. 本轮需要验证 τ=0.15 的 limited-valid 提升是否迁移到 20epoch full valid/test offline evaluation。

### 影响范围

- 只运行 Text + Mean Pooling Two-Tower τ=0.15 20epoch training。
- 训练期间使用 50K limited valid。
- 使用 best checkpoint 运行 full valid/test eval。
- 未改 batch size、learning rate、seed、history_max_len、history_weight、text embedding path、item fusion 或模型结构。
- 未运行 Faiss，未加入 Transformer、attention、LogQ 或其他结构。

### 结果

Limited valid best checkpoint：

```text
best_epoch = 20
Recall@20 = 0.085960
Recall@50 = 0.120180
Recall@100 = 0.154460
NDCG@50 = 0.050516
MRR@50 = 0.032826
```

Full valid/test：

```text
full valid:
  num_eval_users = 497137
  num_skipped_cold_users = 312
  Recall@20 = 0.087209
  Recall@50 = 0.122606
  Recall@100 = 0.157538
  NDCG@50 = 0.051329
  MRR@50 = 0.033241

full test:
  num_eval_users = 496470
  num_skipped_cold_users = 979
  Recall@20 = 0.051566
  Recall@50 = 0.076337
  Recall@100 = 0.102540
  NDCG@50 = 0.029987
  MRR@50 = 0.018414
```

Comparison vs Text + Mean Pooling τ=0.07 full eval：

```text
τ=0.07 full valid Recall@50 = 0.099628
τ=0.15 full valid Recall@50 = 0.122606
absolute change = +0.022978
relative change = +23.06%

τ=0.07 full test Recall@50 = 0.066042
τ=0.15 full test Recall@50 = 0.076337
absolute change = +0.010295
relative change = +15.59%
```

### 客观判断

- τ=0.15 full test Recall@50 is higher than τ=0.07 under the current offline full evaluation setup.
- This validates that the τ=0.15 limited-valid improvement transferred to full valid/test in this run.
- This is offline evaluation only; it does not imply online improvement, does not claim the model exceeds ItemCF, and does not prove global optimality.

## 2026-05-14 - Cold-start / long-tail item bucket evaluation completed

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-14

### 现象

需要在 full test split 上按 target item 的 train interaction count 切 popularity bucket，检查最终主模型 Text + Mean Pooling τ=0.15 在 low-frequency / long-tail / mid-frequency / head item 上的 Recall@50 表现。

### 影响范围

- 只做 evaluation，不重新训练。
- Text + Mean Pooling τ=0.15 使用已有 checkpoint：`outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt`。
- Bucket 定义基于 train split item interaction count：`<=5`、`6-20`、`21-100`、`>100`。
- `<=5` 是 low-frequency / cold-start-like item，不是 completely unseen item cold start。
- ItemCF / ID-only bucket metrics 复用 2026-05-11 D2 popularity bucket matrix docs 记录。
- Mean Pooling / Text+MP τ=0.15 bucket metrics 本轮从已有 checkpoints 计算。
- 未运行 Faiss，未做 Hard Negative Mining，未改模型结构。

### 输出

```text
outputs/cold_start_eval/results.md
outputs/cold_start_eval/results.json
```

### 结果

| bucket by train item count | test targets | ItemCF R@50 | ID-only R@50 | Mean Pool R@50 | Text+MP τ=0.15 R@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=5 | 35045 | 0.040405 | 0.023284 | 0.024083 | 0.031046 |
| 6-20 | 87067 | 0.047940 | 0.043748 | 0.046458 | 0.056933 |
| 21-100 | 161718 | 0.060890 | 0.062918 | 0.063703 | 0.079564 |
| >100 | 212640 | 0.122522 | 0.054604 | 0.060233 | 0.083277 |

### 客观判断

- Text+MP τ=0.15 在四个 bucket 均高于 Mean Pooling 和 ID-only。
- Text+MP τ=0.15 在 `>100` head bucket 上 Recall@50 最高，但仍低于 ItemCF。
- 在 `<=5` 和 `6-20` low-frequency / long-tail buckets 上，Text+MP τ=0.15 相比 Mean Pooling 分别提升 `+0.006962` 和 `+0.010475` Recall@50。
- 该结果是 offline test bucket evaluation，不代表线上效果，也不声称超过 ItemCF。

---

## 2026-05-13 - User history length bucket diagnostic 完成，docs 更新延迟补录

- 严重程度：Low
- 状态：Resolved
- 日期：2026-05-13（docs 于 2026-05-14 补录）

### 现象

item popularity bucket evaluation 完成后，需要继续对 full test non-cold users 按 train history length 切桶，诊断 Text + Mean Pooling τ=0.15 在不同用户历史长度下的 Recall 表现。评估脚本已正常运行完成，但 docs 更新未及时写入。

### 影响范围

- 只做 evaluation，不重新训练，不修改模型结构，不调参。
- 使用已有 checkpoint：`outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt`。
- 输出目录：`outputs/user_history_bucket_eval/`。
- 脚本：`scripts/eval_user_history_buckets.py`（已存在，未修改）。
- 未运行 Faiss，未做 Hard Negative Mining，未做 batch size sweep 或 temperature sweep。

### 结果

| user train history length | num_test_users | Recall@20 | Recall@50 | Recall@100 | NDCG@50 | MRR@50 | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | n/a | n/a | n/a | n/a | n/a | no non-cold test users |
| 1-2 | 0 | n/a | n/a | n/a | n/a | n/a | no non-cold test users |
| 3-5 | 271578 | 0.061456 | 0.086826 | 0.112903 | 0.035723 | 0.022782 | evaluated |
| 6-20 | 191450 | 0.042695 | 0.067401 | 0.094051 | 0.024778 | 0.014330 | evaluated |
| >20 | 33442 | 0.022038 | 0.042312 | 0.066982 | 0.013229 | 0.006328 | evaluated |

### 客观判断

1. 这是 user history length bucket diagnostic，不是解决 true new-user cold start。
2. 当前 processed 5-core full test non-cold users 中，train history length 0 和 1–2 桶均为空。
3. Text+MP τ=0.15 在 3–5 history bucket 最强，Recall@50 = `0.086826`。
4. Recall@50 不随 train history length 单调提升；长历史用户反而 Recall 更低。
5. 可能解释：long-history 用户兴趣更复杂，简单 mean pooling 把多兴趣平均成模糊向量，降低召回准确性。
6. 真正的新用户冷启动需要用户画像、初始兴趣、上下文特征或 popularity fallback，超出本实验范围。
7. 该诊断只说明当前模型边界，不应夸大为解决用户冷启动问题。

### 后续复用建议

- 本 issue 已关闭，只做记录用。
- 若后续要改善长历史用户召回，可考虑 attention pooling、时序建模或兴趣拆分，但需单独立项评估。

---

## 2026-05-14 - Hard Negative Mining smoke 完成（链路通过，1epoch 信号弱）

- 严重程度：Low
- 状态：Resolved（smoke 通过；是否继续 5epoch 待 Eddy 决定）
- 日期：2026-05-14

### 背景

当前最终主模型（Text + Mean Pooling τ=0.15，full test Recall@50=0.076337）已完成。为探索是否能进一步改善相似 item 之间的细粒度区分，本轮做 HNM smoke test。

**重要声明：**
- 这是 smoke test，不是最终实验。
- HNM 不替换当前主模型，不改变当前主模型结论。
- HNM 目的是强化相似 item 判别，不是解决 true new-user cold start。
- 只跑 1epoch limited eval；不做 20epoch、full valid/test、Faiss。

### Hard Negative 构造方式

- 使用已有 `item_text_embedding.npy`（153977 items, 384-dim sentence-transformer embeddings）。
- 训练前一次性 GPU 分块预建 HN table：每 item 存储 top-5 text-cosine-similar neighbors（排除自身）。
- 训练时每个正样本从 top-5 中选取 2 个有效 hard negatives：排除正样本自身、排除 user train history items。
- HN table 预建耗时 ~5s；99.9% 样本成功匹配 ≥2 个有效 hard negatives。

### 影响范围

- 只影响 smoke 记录。
- 新增独立脚本与配置，未修改主训练脚本。
- 不修改 ID-only、text-enhanced、mean pooling、Text+MP τ=0.15 任何已有 baseline 或 checkpoint。

### 脚本与配置

```text
scripts/train_text_mean_pool_hard_negative_smoke.py
configs/two_tower_movies_tv_5core_text_mean_pool_hnm_smoke.yaml
```

### HNM 超参

```text
lambda_hn = 0.1
hard_negatives_per_sample = 2
hn_top_k = 5
temperature = 0.15（与 baseline 一致）
epochs = 1
eval_max_users = 50000
```

### 1epoch limited valid 结果

| 对比 | Recall@50 | delta |
| --- | ---: | ---: |
| Text+MP τ=0.15 epoch 1（baseline） | 0.107460 | — |
| HNM smoke epoch 1 | 0.107840 | +0.000380 |

其余指标（epoch 1 HNM）：

```text
Recall@20  = 0.075240
Recall@50  = 0.107840
Recall@100 = 0.140740
NDCG@50    = 0.045099
MRR@50     = 0.029255
train_total_loss = 7.211303
train_main_loss  = 7.135476
train_hn_loss    = 0.758267
epoch_time       = 137.65s（约 baseline 的 1.77×）
```

### 客观判断

- Smoke 通过：无 OOM、nan、inf、Killed、Traceback。
- 1epoch Recall@50 delta = +0.000380（约 +0.04% 相对提升），几乎与 baseline 持平。
- 1epoch 信号太弱，不能作为 HNM 有效/无效的最终结论。
- epoch 开销约 1.77× baseline，主要来自 HNM 的二次 forward pass。

### 风险分析

| 风险 | 说明 |
| --- | --- |
| false negative risk | 99.9% 样本有效，0.1% 因 history 重叠跳过；低。has_text=0 items（38.3%）HN 质量存疑（text embedding 无语义）。 |
| 额外计算开销 | HN table 预建 ~5s；每 epoch 约 1.77× baseline。 |
| 短期效益不明显 | 1epoch delta ≈ 0；若 5epoch 仍无改善，降级为 future-work。 |

### 后续复用建议

- 若继续 5epoch，应先由 Eddy 确认。
- 5epoch 结果若仍无明显改善（建议阈值：vs baseline epoch5 Recall@50 提升 <0.002），则将 HNM 归为 future-work，不进入正式实验。
- 不要把 smoke 结果写成已超过 baseline 或已改善最终主模型的结论。

---

## 2026-05-14 - Model-based HNM smoke 完成（epoch1 Recall@50 低于 baseline，停止）

- 严重程度：Low
- 状态：Resolved（smoke 完成；不建议继续 5epoch）
- 日期：2026-05-14

### 背景

Text-based HNM smoke（使用 frozen text embeddings 选 hard negatives）在 epoch1 Recall@50 仅比 baseline 高 +0.000380（几乎持平），且 38.3% 无 text metadata 的 item 其 text HN 质量存疑。本轮改用当前最终模型（Text+MP τ=0.15，epoch=20）导出的 item embeddings 来构建 HN table，理论上更接近"模型真正容易混淆的 negatives"。

**重要声明：**
- 这是 smoke test，不是最终实验。
- 不替换当前主模型（full test Recall@50=0.076337）。
- HNM 目的是强化 item 细粒度区分，不是解决 true new-user cold start。
- 只跑 1epoch limited eval；不做 5epoch、full valid/test、Faiss。

### Hard Negative 构造方式

- 加载最终模型 checkpoint：`outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt`（epoch=20）。
- 导出所有 153977 个 item 的 L2 normalized item embeddings（dim=64）。
- 使用 **Faiss IndexFlatIP**（inner product = cosine for L2 normalized vectors）查找每个 item 的 top-50 nearest neighbors，排除自身。
- HN table：`(153977, 50)` int32，100% 有效条目。
- 训练时每个正样本从 top-50 中选取 2 个有效 hard negatives：排除正样本自身、排除 user train history items。
- **100%（4319438/4319438）的训练样本有效**（top_k=50 消除了 history 重叠导致的无效样本问题）。
- HN table 构建耗时：68.91s。

### 影响范围

- 只影响 smoke 记录。
- 新增独立脚本与配置，未修改主训练脚本。
- 不修改任何已有 baseline 或 checkpoint。

### 脚本与配置

```text
scripts/train_text_mean_pool_model_hard_negative_smoke.py
configs/two_tower_movies_tv_5core_text_mean_pool_model_hnm_smoke.yaml
```

### HNM 超参

```text
model_hn_checkpoint = outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt
lambda_hn = 0.1
hard_negatives_per_sample = 2
hn_top_k = 50（Faiss IndexFlatIP，model embedding space）
temperature = 0.15（与 baseline 一致）
epochs = 1
eval_max_users = 50000
```

### 1epoch limited valid 结果

| Model | epoch1 limited Recall@50 | delta vs baseline |
| --- | ---: | ---: |
| Baseline Text+MP τ=0.15 | 0.107460 | — |
| Text-based HNM smoke | 0.107840 | +0.000380 |
| **Model-based HNM smoke** | **0.105200** | **-0.002260** |

其余指标（epoch 1 Model-based HNM）：

```text
Recall@20  = 0.074420
Recall@50  = 0.105200
Recall@100 = 0.137280
NDCG@50    = 0.044351
MRR@50     = 0.028933
train_total_loss = 7.264517
train_main_loss  = 7.154447
train_hn_loss    = 1.100696（高于 text-based 0.758，model-based HN 更难）
epoch_time       = 199.67s（约 baseline 的 2.56×）
```

### 客观判断

1. Smoke 通过：无 OOM、nan、inf、Killed、Traceback；loss 下降正常。
2. 1epoch limited valid Recall@50 = `0.105200`，**低于** baseline `0.107460`，delta = `-0.002260`（约 -2.1% 相对下降）。
3. model-based hn_loss 均值 ≈ 1.10 高于 text-based hn_loss ≈ 0.76，确认 model-based HN 更难（模型更难区分）。但在 epoch1 阶段，过硬的 HN 可能与 main loss 梯度方向冲突，导致 epoch1 指标低于 baseline。
4. 有效 HN 覆盖率 100%，优于 text-based 99.9%；top_k=50 完全解决了 history 重叠问题。
5. **per task spec：epoch1 Recall@50 低于 baseline，不建议继续 5epoch**。
6. 当前最终主模型结论（full test Recall@50=0.076337）不变。

### 风险分析

| 风险 | 说明 |
| --- | --- |
| epoch1 Recall@50 低于 baseline | Model-based HN 在初始化阶段过硬；梯度干扰 main loss；需要减小 lambda_hn 或加 warmup。 |
| HN table 静态 | HN 来自 final model embeddings（训练前一次性导出），不随 training model 更新；属于 static HNM，非 online dynamic HNM。 |
| 额外计算开销 | HN table 构建 68.91s；每 epoch 约 2.56× baseline；显著高于 text-based 的 1.77×。 |
| false negative risk | 100% coverage，history 排除完整；false negative 风险低。 |

### 后续复用建议

- 本 smoke 结果（epoch1 Recall@50 < baseline）不支持继续 5epoch，按 task spec 停止。
- 若未来重试 model-based HNM，建议：
  1. 减小 lambda_hn（如 0.01 或 0.05）；
  2. 加 warmup（前 N epoch 不引入 HN loss）；
  3. 减少 hard_negatives_per_sample（如 1）；
  4. 或考虑 in-batch top-K 替代预计算静态 HN table（实现 online dynamic HNM）。
- 不要把本 smoke 结果写成 model-based HNM 无效的最终结论；epoch1 信号过早，仅说明当前 lambda_hn=0.1 在 epoch1 不适合。

## 2026-05-14 - Semi-hard HNM smoke 完成（epoch1 Recall@50 有边际改善，接近但未达到 0.109 阈值，不继续 5epoch）

- 严重程度：Low（exploratory experiment）
- 状态：Completed — smoke passed，不继续 5epoch

### 实验配置

```text
model_hn_checkpoint = outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt
hn_top_k_search = 300
semi_hard_start_rank = 50
semi_hard_end_rank = 200
n_candidates_per_item = 150
lambda_hn = 0.03
hard_negatives_per_sample = 2
temperature = 0.15
epochs = 1
eval_max_users = 50000
```

Semi-hard 核心思想：跳过每个 item 模型 embedding 空间中最相似的 top-50 neighbors（rank 0–49），改从 rank 50–199 中采样。这些 candidates 仍与 positive item 相似，但难度较低，预期减少 false negative 风险和初期梯度冲突。

### 构造结果

```text
HN table shape: (153977, 150) int32
HN table valid entries: 23096550 (100.0%)
HN table build time: 93.20s (Faiss IndexFlatIP, top-302 search)
Valid HN coverage (epoch 1): 4319438 / 4319438 (100%)
hn_loss avg: 0.996296
```

### 1epoch limited valid eval 结果

```text
Recall@20  = 0.076320
Recall@50  = 0.108840
Recall@100 = 0.140260
NDCG@50    = 0.045514
MRR@50     = 0.029514
epoch time = 336.48s（约 4.31× baseline ~78s）
```

### 四方对比（epoch1 limited Recall@50）

| Model | epoch1 Recall@50 | delta vs baseline |
| --- | ---: | ---: |
| Baseline Text+MP τ=0.15 | 0.107460 | — |
| Text-based HNM smoke | 0.107840 | +0.000380 |
| Model-based top-50 HNM smoke | 0.105200 | -0.002260 |
| **Semi-hard HNM smoke** | **0.108840** | **+0.001380** |

### 客观判断

1. Smoke **链路通过**：无 OOM、nan、inf、Killed、Traceback；training 稳定。
2. epoch1 Recall@50 = 0.108840 > baseline 0.107460（+0.001380 absolute，约 +1.28% relative），是四方对比中最高。
3. hn_loss ≈ 0.996，难度介于 text-based（0.758）和 model-based top-50（1.100）之间，符合 semi-hard 设计预期。
4. 按任务规范：0.108840 < 0.109000 建议阈值，不推荐立即继续 5epoch。
5. 额外计算开销：epoch 336s ≈ 4.31× baseline；主要瓶颈在 compute_hn_loss 中 150 candidates 的 Python 循环。

### 是否建议继续 5epoch

**不推荐立即继续。**

- Recall@50 未达 0.109 阈值；1epoch 信号改善有限。
- 每 epoch 4.31× 计算开销，5epoch ≈ 1680s，性价比不确定。
- 当前主线实验已完成；Semi-hard HNM 属于 exploratory future-work。

若将来继续：
- 可优先批量化 compute_hn_loss 中的 candidates 过滤逻辑以降低 epoch 开销；
- 或调整 rank band（如 50–100）减少 candidates 数量；
- 或适当增大 lambda_hn（如 0.05）观察多 epoch 收敛趋势。

### 输出文件（不提交）

```text
outputs/text_mean_pool_semi_hnm_smoke/
outputs/text_mean_pool_semi_hnm_smoke/run_config.json
outputs/text_mean_pool_semi_hnm_smoke/semi_hnm_smoke_results.json
outputs/text_mean_pool_semi_hnm_smoke/summary.json
outputs/text_mean_pool_semi_hnm_smoke/checkpoints/epoch_1.pt
logs/text_mean_pool_semi_hnm_smoke.log
```

### 新增文件（待提交）

```text
scripts/train_text_mean_pool_semi_hard_negative_smoke.py
configs/two_tower_movies_tv_5core_text_mean_pool_semi_hnm_smoke.yaml
```

## 2026-05-14 - Semi-hard HNM λ=0.01 smoke 完成（降低 lambda 无改善，HNM 系列停止）

- 严重程度：Low（final exploratory experiment）
- 状态：Completed — HNM 系列实验全部结束，归入 future work

### 背景

在 Semi-hard HNM λ=0.03（epoch1 Recall@50=0.108840，高于 baseline 0.107460 约 +1.28%）的基础上，验证进一步降低 lambda_hn 至 0.01 是否能减少对 main loss 的干扰，从而改善结果。

### 实验配置

唯一改动：`lambda_hn = 0.01`（vs 0.03）

```text
model_hn_checkpoint = outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt
hn_top_k_search = 300
semi_hard_start_rank = 50
semi_hard_end_rank = 200
n_candidates_per_item = 150
lambda_hn = 0.01
hard_negatives_per_sample = 2
temperature = 0.15
epochs = 1
eval_max_users = 50000
```

### 结果

```text
Recall@20  = 0.076040
Recall@50  = 0.107840
Recall@100 = 0.138960
NDCG@50    = 0.045336
MRR@50     = 0.029525
epoch time = 401.22s（约 5.15× baseline ~78s）
```

### 五方对比（epoch1 limited Recall@50）

| Model | Recall@50 | delta |
| --- | ---: | ---: |
| Baseline Text+MP τ=0.15 | 0.107460 | — |
| Text-based HNM smoke | 0.107840 | +0.000380 |
| Model-based top-50 HNM smoke | 0.105200 | -0.002260 |
| Semi-hard HNM λ=0.03 | 0.108840 | +0.001380 |
| **Semi-hard HNM λ=0.01** | **0.107840** | **+0.000380** |

### 客观判断

1. λ=0.01 结果（0.107840）**低于** λ=0.03（0.108840），降低 lambda 没有改善。
2. 效果与 text-based HNM（0.107840）持平，说明 lambda 过小导致 HN 辅助信号过弱。
3. 按决策规则：`0.107840 ≤ 0.108840` → 停止 HNM 系列实验。
4. 4 种 HNM 变体均未超过 0.109000 建议阈值；最优变体（semi-hard λ=0.03）改善仅 +1.28%，epoch 开销 4.31×。

### HNM 系列最终结论

所有 HNM smoke 实验（text-based / model-based top-50 / semi-hard λ=0.03 / semi-hard λ=0.01）已完成。无任何变体达到 0.109000 阈值。全部归入 future work，不纳入主模型路线。

当前最终主模型结论不变：Text + Mean Pooling τ=0.15，full test Recall@50=0.076337。

若将来重新评估 HNM，建议方向：
- 批量化 compute_hn_loss 降低 epoch 开销
- 动态 HN（online mining，随训练模型更新）
- curriculum：前若干 epoch 纯 main_loss，逐步引入 HN

### 新增文件（待提交）

```text
scripts/train_text_mean_pool_semi_hard_negative_lambda001_smoke.py
configs/two_tower_movies_tv_5core_text_mean_pool_semi_hnm_lambda001_smoke.yaml
```

## 2026-05-14 - Time-decay Mean Pooling smoke 完成

- 严重程度：N/A（exploratory smoke）
- 状态：完成 / 建议继续 5epoch（待用户批准）

### 背景

User history length bucket diagnostic 发现长历史用户 Recall@50 低于短历史用户（6–20: 0.067401，>20: 0.042312），推测 simple mean pooling 稀释多兴趣历史。尝试 time-decay weighted mean pooling：最近 item 权重 1.0，按 decay_rate=0.8 指数衰减。

### 唯一改动

仅将 user tower 的 `masked_mean` 改为 `masked_weighted_mean(decay_rate=0.8)`。item tower、temperature、batch_size、lr、seed 全部与 Text + Mean Pooling τ=0.15 主模型一致。无新可学习参数。

### 1epoch limited valid 结果

| 指标 | 值 |
| --- | ---: |
| Recall@20 | 0.081300 |
| **Recall@50** | **0.114420** |
| Recall@100 | 0.145780 |
| NDCG@50 | 0.048829 |
| MRR@50 | 0.032171 |
| epoch time | 87.18s (≈1.12× baseline) |

### 对比（Recall@50）

| Model | epoch1 Recall@50 | delta |
| --- | ---: | ---: |
| Baseline Text+MP τ=0.15 | 0.107460 | — |
| Semi-hard HNM λ=0.03（best HNM） | 0.108840 | +0.001380 |
| **Time-decay MP (decay=0.8)** | **0.114420** | **+0.006960** |

### 客观判断

1. epoch1 Recall@50 = 0.114420，**超过 0.109000 阈值**，是所有 smoke 中最强信号。
2. 计算开销极低（1.12× baseline），远低于 HNM 系列（4×）。
3. 按决策规则：**建议进入 5epoch 实验，须等待用户批准**。

### 新增文件

```text
scripts/train_text_time_decay_mean_pool_two_tower_smoke.py
configs/two_tower_movies_tv_5core_text_time_decay_mean_pool_smoke.yaml
```

## 2026-05-14 - Time-decay Mean Pooling 5epoch limited eval 完成

- 严重程度：N/A（exploratory 5epoch eval）
- 状态：完成 / 建议继续 20epoch + full eval（待用户批准）

### 背景

1epoch smoke 结果（Recall@50=0.114420）超过阈值，用户批准继续 5epoch。与 smoke 相比仅修改 `epochs: 5` 和 `output_dir`，其余参数不变。

### epoch 曲线

| epoch | Recall@50 |
| ---: | ---: |
| 1 | 0.114420 |
| **2** | **0.119840** |
| 3 | 0.119740 |
| 4 | 0.119120 |
| 5 | 0.118840 |

best epoch = 2（Recall@50 = 0.119840）。epoch 2 后小幅下滑，收敛快于 simple mean pooling。

### 最佳 epoch 结果（epoch 2）

| 指标 | 值 |
| --- | ---: |
| Recall@20 | 0.085080 |
| **Recall@50** | **0.119840** |
| Recall@100 | 0.153540 |
| NDCG@50 | 0.051130 |
| MRR@50 | 0.033675 |

### 对比（Recall@50）

| Model | eval | Recall@50 |
| --- | --- | ---: |
| Simple Text+MP τ=0.15 | 5ep limited | 0.117240 |
| Simple Text+MP τ=0.15 | 20ep limited | 0.120180 |
| **Time-decay decay=0.8** | **5ep limited (best ep2)** | **0.119840** |

### 客观判断

1. 5epoch best（0.119840）**超过** simple 5epoch（0.117240），与 simple 20epoch（0.120180）差距仅 -0.000340。
2. 按决策规则：结果在 0.117240–0.120180 区间，**建议继续 20epoch + full eval，须等待用户批准**。
3. epoch time ~87s/epoch，20epoch ≈ 29 分钟，开销可接受。

### 新增文件

```text
configs/two_tower_movies_tv_5core_text_time_decay_mean_pool_5epoch.yaml
```

---

## 2026-05-14 - Time-decay Mean Pooling 20epoch + full eval 完成

- 严重程度：N/A（探索实验，成功）
- 状态：已完成

### 实验内容

在 5epoch 确认有效（best epoch 2，Recall@50=0.119840）后，运行 20epoch 完整训练 + full valid/test evaluation。

### 20epoch 训练结果

best epoch = **17**，limited valid Recall@50 = **0.121140**。

epoch-by-epoch 曲线揭示"早期快速上升+平台期+二次抬升"模式：
- epoch 1-2 快速上升（局部峰 0.119840）
- epoch 3-15 平台震荡（0.118260–0.119780）
- epoch 16-17 第二次突破（0.120220→**0.121140**）
- epoch 18-20 小幅回落（0.120080–0.120620）

### full valid/test 结果

| Model | Full valid Recall@50 | Full test Recall@50 | Test NDCG@50 | Test MRR@50 |
| --- | ---: | ---: | ---: | ---: |
| Simple Text+MP τ=0.15 | 0.122606 | 0.076337 | 0.029987 | 0.018414 |
| **Time-decay decay=0.8** | **0.122626** | **0.078315** | **0.030862** | **0.019036** |

### 客观结论

1. Time-decay full test Recall@50 = **0.078315**，**超过** simple Text+MP τ=0.15（0.076337），绝对提升 +0.001978（+2.59%）。
2. test NDCG@50（+2.92%）和 MRR@50（+3.38%）均有改善。
3. 使用 best_model.pt（epoch 17），而非 final epoch（epoch 20）。
4. 没有 Faiss，没有 HNM，没有 decay_rate sweep。
5. **建议作为候选最终主模型**；建议重跑 user history bucket diagnostic。

### 新增/修改文件

```text
configs/two_tower_movies_tv_5core_text_time_decay_mean_pool_20epoch.yaml  (新增)
scripts/train_text_time_decay_mean_pool_two_tower_smoke.py               (修改：新增 --eval_only/--full_eval 支持)
docs/daily_logs/2026-05-13.md                                            (修改：追加 20epoch 结果)
docs/issue_log.md                                                        (本条目)
```

---

## 2026-05-14 - Time-decay user history bucket diagnostic 完成

- 严重程度：N/A（诊断评估，成功）
- 状态：已完成

### 实验内容

重跑 user history length bucket diagnostic，使用 Time-decay Text+MP τ=0.15 best checkpoint（epoch 17），与 simple mean pooling bucket 结果对比，验证 time-decay 是否改善长历史（>20）用户桶。

### 结果

| user train history length | Simple Text+MP τ=0.15 R@50 | Time-decay Text+MP τ=0.15 R@50 | absolute change | relative change |
| --- | ---: | ---: | ---: | ---: |
| 3-5 | 0.086826 | 0.088792 | +0.001966 | +2.26% |
| 6-20 | 0.067401 | 0.069397 | +0.001996 | +2.96% |
| **>20** | **0.042312** | **0.044286** | **+0.001974** | **+4.66%** |

### 客观结论

1. **所有三个 evaluated bucket 均提升**，time-decay 是更稳健的 user history aggregation。
2. **>20 bucket 提升幅度最大（+4.66% relative）**，验证了 time-decay 的设计动机。
3. 未重新训练；使用 best_model.pt（epoch 17）。
4. 没有 Faiss，没有 HNM，没有 decay_rate sweep。
5. 这不是 true new-user cold-start 评估（0 和 1-2 bucket 为空）。

### 新增文件

```text
scripts/eval_user_history_buckets_time_decay.py  (新增)
outputs/user_history_bucket_eval_time_decay/results.json  (不提交)
outputs/user_history_bucket_eval_time_decay/results.md    (不提交)
docs/daily_logs/2026-05-13.md                             (修改：追加 diagnostic 结果)
docs/issue_log.md                                         (本条目)
```

---

## 2026-05-14 - Item Popularity Bucket Feature Smoke 完成（结果低于 baseline）

- 严重程度：N/A（探索实验，已按规则停止）
- 状态：已完成（不继续）

### 实验内容

在 Time-decay Text+MP τ=0.15 基础上，item tower 新增 popularity bucket embedding（4 buckets × 64 dim），验证是否带来提升。只跑 1epoch limited eval。

### 结果

| Model | epoch1 limited Recall@50 |
| --- | ---: |
| Time-decay Text+MP τ=0.15（baseline） | 0.114420 |
| Time-decay + item popularity (pop_weight=0.1) | **0.113600** |

delta = −0.000820（−0.72%）。

### 客观结论

1. 1epoch Recall@50 = 0.113600 **低于** time-decay baseline（0.114420）。
2. 按决策规则（< 0.114420 → 停止），**不进入 5epoch**。
3. 当前最终主模型仍为 Time-decay Text+MP τ=0.15（full test Recall@50 = 0.078315）。
4. 没有 full valid/test，没有 Faiss，没有 HNM，没有 popularity_weight sweep。

### 新增文件

```text
scripts/train_text_time_decay_popularity_two_tower_smoke.py  (新增)
configs/two_tower_movies_tv_5core_text_time_decay_popularity_smoke.yaml  (新增)
docs/daily_logs/2026-05-13.md  (修改：追加 popularity smoke 结果)
docs/issue_log.md  (本条目)
```

---

## [2026-05-14] Faiss Benchmark — Time-decay Text+MP Two-Tower τ=0.15（COMPLETED）

**Checkpoint:** `outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt`（epoch=17）

**任务：** 对最终主模型运行 offline Faiss IndexFlatIP 检索 benchmark，验证检索指标与 full eval 一致，并记录延迟/吞吐量。

**Faiss 检索指标（全量 test，496,470 用户）：**

| Metric | Faiss | Full Eval | Delta |
| --- | ---: | ---: | ---: |
| Recall@20 | 0.052724 | — | — |
| Recall@50 | **0.078315** | **0.078315** | **0.000000** |
| Recall@100 | 0.104792 | — | — |
| NDCG@50 | 0.030862 | — | — |
| MRR@50 | 0.019036 | — | — |

**一致性：** Faiss Recall@50 = full eval（delta=0）✓

**延迟/吞吐量：**

| 指标 | 值 |
| --- | ---: |
| 吞吐量 | 1,165 users/s |
| 平均延迟 | 0.858 ms/user |
| 搜索总时间 | 426.15 s |

**结论：** IndexFlatIP 协议正确，最终模型检索性能已全量验证。

**新增文件：**

```text
scripts/benchmark_faiss_time_decay_text_mean_pool.py  (新增)
docs/daily_logs/2026-05-13.md  (修改：追加 Faiss benchmark 结果)
docs/issue_log.md  (本条目)
```

---

## [2026-05-14] Faiss IVF Benchmark — Time-decay Text+MP Two-Tower τ=0.15（COMPLETED）

**Checkpoint:** `outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt`（epoch=17）

**任务：** 对最终主模型运行 offline Faiss IndexIVFFlat benchmark，对比 recall / speed trade-off。不重新训练。

**IVF 参数：** IndexIVFFlat，nlist=4096，nprobe=32 / 64，K_SEARCH=300。

**Embeddings：** 复用 FlatIP run 已保存的 item_embeddings.npy + test_user_embeddings.npy（一致性已验证）。

**结果：**

| Metric | FlatIP (exact) | IVF nprobe=32 | IVF nprobe=64 |
| --- | ---: | ---: | ---: |
| **Recall@50** | **0.078315** | **0.078172** | **0.078365** |
| Recall@20 | 0.052724 | 0.052662 | 0.052720 |
| Recall@100 | 0.104792 | 0.104979 | 0.105030 |
| NDCG@50 | 0.030862 | 0.030803 | 0.030872 |
| MRR@50 | 0.019036 | 0.018995 | 0.019038 |

**与 FlatIP 对比（nprobe=32）：** delta = −0.000143（−0.18%）

**延迟：**

| 指标 | FlatIP | IVF nprobe=32 |
| --- | ---: | ---: |
| Search time | 426.15 s | 17.05 s |
| 吞吐量 | 1,165 users/s | 29,114 users/s |
| 平均延迟 | 0.858 ms/user | 0.034 ms/user |
| Speedup | — | **25.0×** |

**结论 A：** IVF nprobe=32 Recall@50 仅损失 0.18%，搜索速度提升 25.0×。IVF 可替代 FlatIP 用于离线批量检索。

**新增文件：**

```text
scripts/benchmark_faiss_ivf_time_decay_text_mean_pool.py  (新增)
docs/daily_logs/2026-05-13.md  (修改：追加 IVF benchmark 结果)
docs/issue_log.md  (本条目)
```

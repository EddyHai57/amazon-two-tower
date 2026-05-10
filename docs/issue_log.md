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

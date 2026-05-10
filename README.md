# amazon-two-tower

目标：基于 Amazon Reviews 2023 构建 two-tower retrieval baseline。All_Beauty 仅保留为 Phase 0 工程验证数据集；Phase 1 主实验数据集确定为 Movies_and_TV，先完成 ID-only 基线，再在后续阶段加入文本 embedding。

## 数据集选择

- All_Beauty 已用于 Phase 0 工程验证，验证数据加载、inspection、interaction analysis 和日志流程。
- All_Beauty 不作为 Phase 1 主实验数据集，原因是交互极度稀疏，`user>=3,item>=3` 后只剩 8657 条 interaction。
- Movies_and_TV 作为 Phase 1 主实验数据集：
  - `k-core(3,3)`：8025936 interactions
  - `k-core(5,5)`：5413083 interactions
  - full_load 和 interaction analysis 已成功完成，工程风险可控。
- Video_Games 作为 fallback 数据集。
- Electronics 因 inspection 阶段下载 timeout 暂缓，不作为 Phase 1 主数据集。

## Phase 1 成功标准

项目处于 Phase 1，直到以下条件全部满足：

- 存在 train/valid/test 划分，并且每个用户的 test interactions 都严格晚于 train+valid，无时间泄漏。
- ItemCF baseline 在 test set 上产出可复现的 Recall@50。
- 最简单的 ID-only two-tower baseline 在同一个 test set 上产出 Recall@50，可与 ItemCF 正面对比。
- 所有预处理参数，例如 min interactions 和 rating threshold，都放在配置文件中，而不是硬编码在脚本里。

## 运行数据检查

```bash
cd /workspace/amazon-two-tower
python scripts/inspect_amazon_dataset.py --config configs/amazon_movies_and_tv_phase1.yaml
```

检查报告会写入：

```text
outputs/inspection_movies_and_tv.md
```

## 目录结构

```text
amazon-two-tower/
  configs/
    amazon_movies_and_tv_phase1.yaml
  data/
  logs/
  outputs/
  scripts/
    inspect_amazon_dataset.py
  src/
    __init__.py
```

# amazon-two-tower

目标：基于 Amazon Reviews 2023 All_Beauty 构建 two-tower retrieval baseline，先完成 Phase 1 的 ID-only 基线，再在后续阶段加入文本嵌入。

## Phase 1 成功标准

项目处于 Phase 1，直到以下条件全部满足：

- 存在 train/valid/test 划分，并且每个用户的 test interactions 都严格晚于 train+valid，无时间泄漏。
- ItemCF baseline 在 test set 上产出可复现的 Recall@50。
- 最简单的 ID-only two-tower baseline 在同一个 test set 上产出 Recall@50，可与 ItemCF 正面对比。
- 所有预处理参数，例如 min interactions 和 rating threshold，都放在配置文件中，而不是硬编码在脚本里。

## 运行数据检查

```bash
cd /workspace/amazon-two-tower
python scripts/inspect_amazon_dataset.py --config configs/amazon_all_beauty_phase1.yaml
```

检查报告会写入：

```text
outputs/inspection_all_beauty.md
```

## 目录结构

```text
amazon-two-tower/
  configs/
    amazon_all_beauty_phase1.yaml
  data/
  logs/
  outputs/
  scripts/
    inspect_amazon_dataset.py
  src/
    __init__.py
```

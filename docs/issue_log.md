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

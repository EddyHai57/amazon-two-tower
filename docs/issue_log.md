# 问题日志

## 加载 Amazon Reviews 2023 时出现 HuggingFace datasets 兼容性错误

- 严重程度：中等
- 状态：打开

### 现象

```text
Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py
```

### 影响

Amazon All_Beauty 数据检查没有完成，也没有生成检查报告。

### 根因假设

当前安装的 datasets 版本移除了对数据集加载脚本的支持，而 McAuley-Lab/Amazon-Reviews-2023 需要该机制加载。

### 下一步诊断

用 `python3` 检查当前安装的 datasets 版本，再决定是否使用兼容的隔离环境，或改用另一个官方加载方式。

未经 Eddy 确认，不升级或降级 datasets。

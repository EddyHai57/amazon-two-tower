# 项目日志指南

## 日志目的

项目日志用于记录：

- 每天做了什么。
- 遇到什么问题。
- 问题如何解决。
- 做了哪些设计决策。
- 哪些结果可以写进简历或面试复盘。

## 日志文件结构

本项目维护三个核心日志：

```text
docs/daily_logs/YYYY-MM-DD.md
docs/issue_log.md
docs/decision_log.md
```

## Daily Log 模板

```markdown
# YYYY-MM-DD 日志

## 今日目标

## 已完成事项

## 关键命令

## 产出文件

## 遇到的问题

## 当前 Git 状态

## 今日结论

## 下一步最小动作
```

## Issue Log 模板

```markdown
## Issue 编号：ISSUE-YYYYMMDD-编号

### 问题标题

### 发生时间

### 严重等级

Low / Medium / High / Critical

### 当前状态

Open / Investigating / Resolved / Deferred

### 现象

### 报错原文或关键日志

### 影响范围

### 初步原因判断

### 已尝试的排查步骤

### 最终解决方案

### 后续复用建议
```

## Decision Log 模板

```markdown
## Decision 编号：DECISION-YYYYMMDD-编号

### 决策时间

### 决策主题

### 可选方案

### 最终选择

### 选择原因

### 对实验可比性的影响

### 对后续开发的影响
```

## 记录规则

- 每次完成一个有意义的任务后，必须更新 daily log。
- 每次出现报错、失败命令、环境问题、数据问题、兼容性问题，必须更新 issue_log。
- 只有 Eddy 明确确认的设计选择，才能写入 decision_log。
- 不要把 Codex 自己的猜测写成最终决策。
- 失败命令不能隐藏，必须记录关键报错。
- 日志必须使用简体中文。
- 日志应该简洁，但要足够支持后续复盘和面试讲述。

## Git 规则

- 日志文件属于项目文档，可以提交到 Git。
- 大型数据、缓存、模型、embedding 不允许提交。
- 每次重要阶段完成后，应先运行 `git status`，再由 Eddy 确认是否 commit。
- 如果任务明确要求 commit，则按要求提交。

## 当前已知问题初始化规则

如果 `docs/issue_log.md` 不存在，需要创建该文件，并写入当前已知问题：

````markdown
## HuggingFace datasets 加载 Amazon Reviews 2023 兼容性错误

- 严重等级：Medium
- 状态：Open

### 现象

运行 inspection 脚本时出现：

```text
Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py
```

### 影响

Amazon All_Beauty 数据尚未成功加载，inspection_all_beauty.md 尚未生成。

### 初步原因

当前环境缺少合适版本的 datasets，且新版 datasets 可能不再支持该数据集所需的远程 dataset script。

### 下一步

在项目独立 `.venv` 中安装兼容版本依赖，先验证 datasets 和 huggingface_hub 版本，再重新运行 inspection 脚本。

### 注意

不要在未得到 Eddy 确认前修改全局 Python 环境或随意升级/降级依赖。
````

## 当前设计决策初始化规则

如果 `docs/decision_log.md` 不存在，需要创建该文件，并记录当前已确认决策：

```markdown
- Amazon 项目必须独立于 tianchi-two-tower，路径为 `/workspace/amazon-two-tower`。
- Codex 负责执行实现，Eddy 负责最终设计决策。
- 业务参数必须放在配置文件中，不允许硬编码在脚本里。
- Phase 1 先完成数据切分、ItemCF 基线、仅使用 ID 的双塔基线，再进入文本嵌入和消融实验。
- 遇到会影响实验可比性的决策，Codex 必须先停下来询问 Eddy。
```

## 执行后验证

完成日志相关修改后运行：

```bash
find /workspace/amazon-two-tower/docs -maxdepth 3 -type f | sort
sed -n '1,220p' /workspace/amazon-two-tower/docs/LOGGING_GUIDE.md
git status --short
```

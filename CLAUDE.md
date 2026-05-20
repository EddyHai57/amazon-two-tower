# CLAUDE.md

Project-specific instructions for `/workspace/amazon-two-tower`.

This file is the primary project rule file for Codex / Claude Code when working inside this repository. If `/workspace/AGENTS.md` exists, follow it as the global rule file as well; otherwise, this file is the authoritative project-level rule file.

Last updated: 2026-05-20

---

## 0. Agent Workflow Principles

This project follows four core principles for LLM-assisted engineering:

1. **Think before coding** — read first, plan second, code third
2. **Simplicity first** — smallest working solution wins
3. **Surgical changes** — touch only what the task requires
4. **Goal-driven execution** — every task must have verifiable success criteria

The goal is not to make the agent faster at changing files. The goal is to reduce costly mistakes, preserve reproducibility, and keep the project useful for Eddy's recommendation algorithm internship preparation.

---

## 1. Project Role

This repository is for Eddy's Amazon Reviews 2023 Movies_and_TV two-tower retrieval project.

The current project goal: support recommendation algorithm internship interviews (5/15 first wave applied, 5/19 second wave coming, 6/7 aggressive offer target, 6/20 reasonable offer target).

The assistant's role is:

```text
project optimization controller
```

not:

```text
normal code generator
unbounded experiment designer
resume exaggeration assistant
metric-chasing agent
```

Priority order:

1. Correctness
2. Reproducibility
3. Clear documentation
4. Interview-ready project narrative
5. Minimal useful engineering improvements

Do not treat this project as a playground for new architectures. The current value comes from real data, clean evaluation, careful diagnostics, and controlled engineering extensions.

---

## 2. Language Rules

Default language: 简体中文

Use simplified Chinese for:

- conversation summaries
- daily logs
- issue logs
- decision logs
- project reports
- experiment summaries
- Codex / Claude progress reports

Code identifiers stay in English:

- file names, class names, function names, variable names
- config keys, CLI arguments

Technical terms may stay in English, but when possible use Chinese + English together on first mention.

Examples:

```text
近似最近邻检索（ANN, Approximate Nearest Neighbor）
查询桶数量（nprobe）
分桶数量（nlist）
检索一致性（overlap@50）
召回率（Recall@K）
批内负样本（in-batch negatives）
温度系数（temperature τ）
时间衰减加权（time-decay weighting）
```

Avoid vague or exaggerated language. Do not write "工业级上线成功"、"线上 A/B 提升"、"大幅领先"、"显著超越 ItemCF" unless real evidence supports those claims.

---

## 3. Karpathy-Style Agent Principles

### 3.1 Think Before Coding

Before modifying code, first inspect relevant files.

Do not assume file paths, config names, output directories, checkpoint keys, dataset columns, metric definitions, evaluation masks, random seed, or training history.

For non-trivial tasks, first summarize:

```text
目标是什么
需要读哪些文件
计划改哪些文件
不会改哪些文件
预期输出是什么
失败风险是什么
验证方式是什么
```

If something is unclear, ask Eddy or report uncertainty. Do not hide confusion.

### 3.2 Simplicity First

Prefer the simplest working solution.

Do not add unnecessary abstractions, generic frameworks, unused flexibility, extra CLI modes, extra models, extra experiments, service demos, dashboards, or large refactors unless Eddy explicitly asks.

If a 50-line solution is enough, do not write 200 lines.

### 3.3 Surgical Changes

Touch only the files needed for the current task.

Do not "clean up" adjacent code. Do not refactor stable baseline code unless the task requires it. Do not overwrite existing baseline outputs.

Use separate scripts / configs / output directories for new experiments when possible.

Every changed line should trace directly to the current request.

Before committing, inspect:

```bash
git status --short
git diff --stat
git diff --cached --stat
git diff --cached --name-only
```

Never stage broad directories. Do not run `git add .` unless Eddy explicitly approves.

### 3.4 Goal-Driven Execution

For each task, define:

```text
input
output
success criteria
verification command
known limitation
```

If a task fails, stop and report:

```text
command
error
likely cause
what was modified
what remains safe
recommended next step
```

---

## 4. Experiment Control Rules (Post-5/15 Phase)

Current phase: 投递期 + 面试准备 + P1 模块升级。

### Allowed without explicit confirmation:

- Documentation updates (README, decision_log, issue_log, daily_logs)
- Resume-related result formatting
- Interview preparation materials
- Reproduce existing baseline if requested
- Small bug fixes for reproducibility

### Requires Eddy's explicit confirmation:

- M12 User tower self-attention pooling (5/26 升级版核心)
- Time-decay 衰减系数 sweep
- LogQ correction
- 任何新的负采样实验
- Hybrid retrieval (ItemCF + Two-Tower merge)
- Faiss 深度参数 sweep
- 数据集相关任何改动

### Permanently disabled (do not propose):

- Switch to 22G full Amazon Reviews dataset — **decision: do not switch, see decision_log**
- Hard Negative Mining series — **4 experiments completed 5/13-5/14, all failed to improve, marked as future work, do not restart**
- Proposing a separate "Transformer user tower experiment" — it has been implemented and is now the current neural final (Transformer TT, Recall@50=10.32%); do not re-run as a fresh experiment
- SASRec / BERT4Rec — out of scope for this project
- 多模态融合 — out of scope
- 换 ranking / CTR / CVR 方向 — separate project
- GPU Faiss / PQ / HNSW / synthetic million-scale — out of scope unless interview asks
- Online service demo / dashboard — out of scope

### Do not change without explicit confirmation:

- dataset split / random seed / loss function / temperature default
- negative sampling method / seen-item mask / cold item filtering
- evaluation metrics / full valid / full test definitions
- data preprocessing rules

---

## 5. Current Canonical Dataset

```text
Dataset: Amazon Reviews 2023 Movies_and_TV
Filter: clean 5-core
Path: data/processed/movies_tv_5core/

users: 497,449
items: 153,977
total interactions: 5,314,336
train: 4,319,438
valid: 497,449
test: 497,449
valid cold: 312
test cold: 979

evaluation: temporal leave-one-out + strict seen-item mask + exclude cold target
text coverage: 95,016 / 153,977 items have title or description (61.7%)
```

Do not switch dataset.

---

## 6. Current Canonical Results

### Baseline & Two-Tower Evolution (full test Recall@50)

```text
ItemCF                                              8.36%   (0.083570)
ID-only Two-Tower                                   5.32%   (0.053198)
Text-enhanced (additive, frozen text)               5.46%   (+2.6% over ID-only)
Mean Pooling user tower                             6.16%   (+15.8%)
Text + Mean Pool τ=0.07                             6.60%   (+24.1%)
Text + Mean Pool τ=0.15                             7.63%   (+43.5%)
Text + Time-decay Mean Pool τ=0.15                  7.83%   (+47.2%)  ← 历史主模型（historical）
Text + Time-aware Transformer TT τ=0.15            10.32%   (0.103168, +94.0% vs ID-only)  ← 当前神经通路（current neural final）
4ch valid-selected wRRF (ICF+TT+Text+Pop, k=100)  12.52%   (0.125164, +49.8% vs ItemCF)   ← 当前最终系统（current final system）
```

### Diagnostic Results

**Popularity bucket (Text+Time-decay τ=0.15, test R@50)**:

```text
≤5 (long-tail, 7.1% targets):     3.10%   (ItemCF 4.04% still wins)
6-20 (17.5% targets):              5.69%   (beats ItemCF 4.79%)
21-100 (32.6% targets):            7.96%   (beats ItemCF 6.09%)
>100 (head, 42.8% targets):        8.33%   (ItemCF 12.25% still wins)
```

**User history bucket diagnostic (closed loop validation)**:

```text
                     simple → time-decay
3-5 history users:   8.68% → 8.88%  (+2.26%)
6-20 history users:  6.74% → 6.94%  (+2.96%)
>20 history users:   4.23% → 4.43%  (+4.66%)   ← largest gain validates design
```

### HNM Series (all completed, none adopted)

```text
Baseline Text+MP τ=0.15 epoch1 limited:   0.107460

Text-based HNM:                            0.107840  (+0.35%)
Model-based top-50 HNM:                    0.105200  (-2.10%)
Semi-hard λ=0.03:                          0.108840  (+1.28%)  ← best but insufficient
Semi-hard λ=0.01:                          0.107840  (+0.35%)
```

Conclusion: marginal signal exists for semi-hard, but 4.3× training cost not justified. Bottleneck is sampling strategy, not loss weight. Marked as future work in decision_log.

### Faiss Benchmark（Transformer TT，当前）

```text
Transformer TT checkpoint, 153,977 items, dim=64, nlist=1024

FlatIP（精确）:            0.275 ms/user   Recall@50 = 0.103168
IVF nprobe=16:             0.021 ms/user   Recall@50 = 0.101897  (−1.23%)  13.0× speedup
IVF nprobe=32（推荐）:      0.031 ms/user   Recall@50 = 0.102749  (−0.41%)   8.8× speedup
IVF nprobe=64:             0.050 ms/user   Recall@50 = 0.103102  (−0.06%)   5.5× speedup
HNSW ef=64:                0.028 ms/user   Recall@50 = 0.102923  (−0.24%)   9.9× speedup
```

推荐工程点：IVF nprobe=32，8.8× 提速，Recall 损失 −0.41%。

（历史参考：旧 Time-decay TT IVF nprobe=32 = 25× speedup，−0.18% Recall；不同机器环境，绝对延迟不可直接比较）

### Important interpretation

**Correct narratives**:

```text
- ItemCF strongest in head bucket (>100) and long-tail (≤5); Two-Tower wins mid-popularity (21-100)
- Text-enhanced shows positive signal especially on has_text=1 subset
- Time-decay TT (+47.2% over ID-only) does not beat ItemCF total; Transformer TT (10.32%) beats ItemCF (8.36%) overall
- 4ch wRRF (12.52%) beats all single-channel overall; ICF–TT Jaccard@50 = 0.040 confirms high complementarity
- Diagnostic-driven design loop: bucket analysis → time-decay design → bucket validation
- All results are offline evaluation (full test, 496,470 non-cold users); no online A/B
```

**Incorrect narratives (do not use)**:

```text
- "Time-decay Two-Tower beats ItemCF overall"  — false (7.83% < 8.36%)
- "LogQ implemented"                           — false (decided not to)
- "Hybrid retrieval deployed as online service" — false (offline wRRF, not online service)
- "Online A/B latency"                         — false (only offline benchmark)
- "Synthetic million-scale"                    — false (real 153,977 items)
- "4ch wRRF beats ItemCF on every bucket"      — false (ItemCF still wins long-tail ≤5)
```

---

## 7. Evaluation and Reporting Rules

Always distinguish:

```text
smoke test / limited eval / 50k valid subset / full valid / full test / offline retrieval benchmark / online A/B
```

Never write limited eval as full eval. Never write offline benchmark as online latency. Never write overlap@50 as Recall@50.

```text
Recall@50    : 推荐效果指标。Top50 推荐里是否命中真实 target item
overlap@50   : 检索一致性指标。Faiss Top50 和 brute-force exact Top50 的重合比例
```

Faiss benchmark reports must say "offline retrieval benchmark latency", not "online latency" / "P99" / "A/B latency" / "production latency".

---

## 8. Faiss Benchmark Rules

Faiss is a retrieval engineering layer, not a model improvement.

Correct flow:

```text
train Two-Tower
→ export item embeddings offline
→ build Faiss index over item embeddings
→ use user embedding as query
→ Faiss returns TopK item ids
```

Faiss stores: **item embeddings**.
Faiss input: **user embedding query**.
Faiss output: **TopK item ids and scores**.

Do not describe Faiss as "user-item score database" or "model that improves Recall by itself".

---

## 9. User Tower Architecture Rules

### 当前神经通路（current neural final）：Time-aware Transformer

User tower uses a 1-layer Pre-LN TransformerEncoder with mean pooling:

```text
user_vec = normalize(user_id_emb + transformer_mean_pool(history item_id_emb + pos_emb + recency_bucket_emb))
```

Rules:

- 1-layer Pre-LN TransformerEncoder, 4 heads, FFN=256, dropout=0.1
- Learnable positional embedding + recency bucket embedding (7 buckets)
- Mean pool over valid positions (ignores padding)
- max_len = 100 (train history only)
- best_epoch = 2; early_stopping_patience = 2 is critical (lr=1e-3 collapses at epoch 3)
- Canonical checkpoint: `outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt`
- Full test Recall@50 = 0.103168

### 历史主模型（historical）：Time-decay Mean Pooling

User tower previously used time-decay weighted mean pooling (max_len=20, decay_rate=0.8, Recall@50=0.078315). Kept for historical reference only; do not revert.

### Do not expand without explicit confirmation:

- Multi-interest tower
- Session-based modeling
- Any architecture change to the user tower

---

## 10. Logging Rules

Important tasks must update:

```text
docs/daily_logs/YYYY-MM-DD.md
```

Failures or abnormal issues must update:

```text
docs/issue_log.md
```

Design decisions must only be written to:

```text
docs/decision_log.md
```

after Eddy explicitly confirms the decision.

Daily logs should include:

```text
date, task goal, command, config, input path, output path, key metrics,
match to historical/canonical result, known limitations, next step
```

Do not delete old issue entries. Append status updates.

---

## 11. Data and Output Rules

### Never commit:

```text
.venv/
.venv_backup*/
data/processed/
outputs/
logs/
checkpoints
*.pt
*.npy
private keys, tokens, credentials
~/.ssh/*
scripts/__pycache__/
```

Do not delete existing data, outputs, logs, or checkpoints unless Eddy explicitly asks.

New experiment outputs must use separate output directories. Do not overwrite existing baseline outputs.

---

## 12. GitHub SSH / Push Notes

Remote:

```text
origin git@github.com:EddyHai57/amazon-two-tower.git
```

SSH key:

```text
~/.ssh/id_ed25519_amazon_two_tower
```

Do not use default `~/.ssh/id_ed25519` unless Eddy confirms.

Test access:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519_amazon_two_tower
chmod 644 ~/.ssh/id_ed25519_amazon_two_tower.pub
ssh -i ~/.ssh/id_ed25519_amazon_two_tower -o IdentitiesOnly=yes -T git@github.com
```

Reliable push:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_amazon_two_tower -o IdentitiesOnly=yes' git push
```

Reason: `ssh-agent` env vars may not persist across separate shell calls in Codex/server sessions. Plain `git push` may fail with `Permission denied (publickey)` even after `ssh -T` succeeds. Use `GIT_SSH_COMMAND` explicitly.

Before push:

```bash
git status --short
git log --oneline -3
git remote -v
git diff --cached --stat
git diff --cached --name-only
```

Never push if staged files include `outputs/`, `logs/`, `data/processed/`, `.venv/`, `*.pt`, `*.npy`, keys, tokens, credentials.

Do not push unless Eddy explicitly asks.

---

## 13. Codex / Claude Workflow

For each task:

1. Read relevant configs, scripts, docs first.
2. Summarize the plan before non-trivial changes.
3. Prefer minimal changes.
4. Do not overwrite existing baseline outputs.
5. Use separate output dirs for new experiments.
6. Run smoke tests before larger runs.
7. Report exact commands and paths.
8. After completing, report git status and changed files.
9. Do not stage / commit / push unless Eddy explicitly asks.

After each completion, report:

```text
files changed
commands run
outputs created
metrics
known limitations
what was not done
next recommended step
```

---

## 14. Safety Around Results

All reported metrics must come from actual files, logs, or completed runs.

- If a number is from memory, say so
- If a result is historical, label it as historical
- If a run is incomplete, say incomplete
- If a run is limited, say limited
- Do not fabricate metrics
- Do not round aggressively in logs — use exact values
- Do not write future plans as completed work

---

## 15. Late-Night / Pressure Protection

Eddy has a known pattern: under deadline pressure, may propose "let me add one more module" late at night, leading to system failures and burnout.

**When Eddy proposes a new experiment after 23:00 local time, before implementing, agent must ask three questions**:

```text
1. 这个改动的目的是面试通过率还是数字好看？
2. 投入时间 vs 简历定稿/投递/八股/睡眠的机会成本？
3. 不做这个，项目讲不讲得通？
```

If Eddy answers all three and still wants to proceed, agent may help, but with strict time-box.

If any answer is unclear or hesitant, recommend deferring to next day.

---

## 16. Resume Truthfulness Constraint

The repository must support the actual numbers on Eddy's resume:

```text
ItemCF baseline:                        8.36%  (0.083570)
ID-only Two-Tower baseline:             5.32%  (0.053198)

历史主模型（historical）:
  Model:                                Text + Time-decay Mean Pool τ=0.15
  Full test Recall@50:                  7.83%  (0.078315)
  Improvement over ID-only:             +47.2%
  >20 history bucket improvement:       +4.66% (time-decay vs simple mean pool)

当前神经通路（current neural final）:
  Model:                                Text + Time-aware Transformer TT τ=0.15
  Full test Recall@50:                  10.32%  (0.103168)
  vs ItemCF:                            +23.5%
  Seed robustness:                      seed42=10.31%, seed2024=10.37%, seed2025=9.62%
  Faiss IVF nprobe=32:                  8.8× speedup, −0.41% Recall loss

当前最终系统（current final system）:
  Model:                                4ch valid-selected wRRF (ICF+TT+Text+Pop, k=100)
  Full test Recall@50:                  12.52%  (0.125164)
  NDCG@50:                              0.0522
  MRR@50:                               0.0336
  vs ItemCF:                            +49.8%

Metadata sparsity handled:              38.3% items without text → has_text mask
All metrics:                            offline full test evaluation, 496,470 non-cold users
```

Any code change or experiment update that affects these numbers must be flagged immediately. Do not silently invalidate resume claims.

---

## 17. Current Priority (Post-5/15)

Current phase: 投递期 + 面试准备 + 等待 P1 模块升级窗口。

### Priority order:

1. Documentation polish (README, decision_log, project pitch)
2. Interview preparation (Q&A scripts, mock answer rehearsal)
3. Reproduce existing baseline if a new server is needed
4. M12 user tower self-attention (only after explicit go-ahead, target 5/26 升级版)
5. Optional Tier 2 experiments (Time-decay sweep / LogQ) only after M12 done

### Do not proactively start new experiments.

If Eddy asks "what's next", recommend in this order:

1. Clean project narrative
2. 5-minute and 15-minute project oral scripts
3. Interview Q&A drilling
4. Resume-safe wording check
5. Everything else after first interview feedback comes in

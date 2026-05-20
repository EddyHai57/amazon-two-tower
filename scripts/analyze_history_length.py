#!/usr/bin/env python3
"""Part 1: Raw user history length distribution.

History length口径：
  - train_users        : train history（raw，截断前）
  - valid_eval_users   : valid eval 时 history = train history（口径相同）
  - test_eval_users    : test eval 时 history = train + valid（每用户 +1 交互）

Outputs:
  outputs/transformer_user_tower_investigation/history_length_distribution.json
  outputs/transformer_user_tower_investigation/history_length_distribution.csv
Appends:
  docs/reports/transformer_user_tower_investigation.md  (Section 13)
  docs/daily_logs/<today>.md                            (Part 18)
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR    = Path("data/processed/movies_tv_5core")
OUT_DIR     = Path("outputs/transformer_user_tower_investigation")
REPORT_PATH = Path("docs/reports/transformer_user_tower_investigation.md")
DAILY_PATH  = Path("docs/daily_logs") / f"{date.today()}.md"


def compute_stats(lengths: np.ndarray, label: str) -> dict:
    n = int(len(lengths))
    d: dict = {
        "label":     label,
        "num_users": n,
        "min":       int(lengths.min()),
        "mean":      round(float(lengths.mean()), 2),
        "median":    int(np.median(lengths)),
        "p75":       int(np.percentile(lengths, 75)),
        "p90":       int(np.percentile(lengths, 90)),
        "p95":       int(np.percentile(lengths, 95)),
        "p99":       int(np.percentile(lengths, 99)),
        "max":       int(lengths.max()),
    }
    for t in [20, 50, 100]:
        le = int((lengths <= t).sum())
        gt = n - le
        d[f"users_le{t}"] = le
        d[f"users_gt{t}"] = gt
        d[f"pct_le{t}"]   = round(le / n * 100, 2)
        d[f"pct_gt{t}"]   = round(gt / n * 100, 2)
    return d


def main() -> None:
    logging.info("Loading data from %s", DATA_DIR)
    train_df = pd.read_parquet(DATA_DIR / "train.parquet",
                               columns=["user_idx", "item_idx", "timestamp"])
    valid_df = pd.read_parquet(DATA_DIR / "valid.parquet",
                               columns=["user_idx", "item_idx"])
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet",
                               columns=["user_idx", "item_idx"])

    # ── Train raw history ────────────────────────────────────────────────────
    train_counts = train_df.groupby("user_idx")["item_idx"].count()
    train_lens = train_counts.values.astype(np.int64)  # all users have > 0

    # ── Valid eval raw history = train history ────────────────────────────────
    valid_users = valid_df["user_idx"].unique()
    valid_lens = train_counts.reindex(valid_users, fill_value=0).values.astype(np.int64)

    # ── Test eval raw history = train + valid ─────────────────────────────────
    # Each user has exactly 1 valid interaction, so test_len = train_len + 1
    test_users = test_df["user_idx"].unique()
    tv_counts = (
        pd.concat([train_df[["user_idx", "item_idx"]], valid_df[["user_idx", "item_idx"]]],
                  ignore_index=True)
        .groupby("user_idx")["item_idx"].count()
    )
    test_lens = tv_counts.reindex(test_users, fill_value=0).values.astype(np.int64)

    # Filter out any zero-length entries (shouldn't occur in 5-core)
    train_stats = compute_stats(train_lens[train_lens > 0], "train_users")
    valid_stats = compute_stats(valid_lens[valid_lens > 0], "valid_eval_users")
    test_stats  = compute_stats(test_lens[test_lens  > 0], "test_eval_users")
    all_stats   = [train_stats, valid_stats, test_stats]

    # ── Save JSON ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "history_length_distribution.json"
    json_path.write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Wrote %s", json_path)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "history_length_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_stats.keys()))
        w.writeheader()
        for s in all_stats:
            w.writerow(s)
    logging.info("Wrote %s", csv_path)

    # ── Report section ────────────────────────────────────────────────────────
    def srow(s: dict) -> str:
        return (
            f"| {s['label']} | {s['num_users']:,} | {s['min']} | {s['mean']} "
            f"| {s['median']} | {s['p75']} | {s['p90']} | {s['p95']} "
            f"| {s['p99']} | {s['max']} |\n"
        )

    def trow(s: dict, t: int) -> str:
        return (
            f"| {s['label']} | ≤{t} | {s[f'users_le{t}']:,} | {s[f'pct_le{t}']:.1f}% "
            f"| {s[f'users_gt{t}']:,} | {s[f'pct_gt{t}']:.1f}% |\n"
        )

    ts = train_stats
    tst = test_stats
    section = f"""
---

## 13. History Length Distribution

**状态：** ✅ 完成（{date.today()}）

### 13.1 基本统计量

| 分组 | 用户数 | min | mean | median | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{srow(train_stats)}{srow(valid_stats)}{srow(test_stats)}
> - `train_users`：train history raw length（截断前，与截断后 max_len=20 final model 一致口径）
> - `valid_eval_users`：valid eval history = train history（口径完全相同）
> - `test_eval_users`：test eval history = train + valid（每用户恰好 +1 交互）

### 13.2 Max_len 覆盖率

| 分组 | max_len | ≤ 用户数 | ≤ 比例 | > 用户数 | > 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
{trow(ts, 20)}{trow(ts, 50)}{trow(ts, 100)}{trow(tst, 20)}{trow(tst, 50)}{trow(tst, 100)}
### 13.3 解读

- **max_len=20**：train 完整覆盖 {ts['pct_le20']:.1f}%，剩余 {ts['pct_gt20']:.1f}% 被截断
- **max_len=50**：train 完整覆盖 {ts['pct_le50']:.1f}%，超出 50 的仅 {ts['pct_gt50']:.1f}%
- **max_len=100**：train 完整覆盖 {ts['pct_le100']:.1f}%，超出 100 的仅 {ts['pct_gt100']:.1f}%
- **train median={ts['median']}，p90={ts['p90']}，p99={ts['p99']}**
- max_len Ablation（Section 14）将量化不同 max_len 对 Recall@50 的实际影响。
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(section)
    logging.info("Appended Section 13 to %s", REPORT_PATH)

    # ── Daily log ─────────────────────────────────────────────────────────────
    daily = f"""
---

## Part 18：History Length Distribution 分析

**状态：** ✅

| max_len | train ≤ 覆盖率 | test_eval ≤ 覆盖率 |
| ---: | ---: | ---: |
| 20 | {ts['pct_le20']:.1f}% | {tst['pct_le20']:.1f}% |
| 50 | {ts['pct_le50']:.1f}% | {tst['pct_le50']:.1f}% |
| 100 | {ts['pct_le100']:.1f}% | {tst['pct_le100']:.1f}% |

train: mean={ts['mean']}, median={ts['median']}, p90={ts['p90']}, p99={ts['p99']}, max={ts['max']}

输出：`{json_path}`，`{csv_path}`
"""
    with DAILY_PATH.open("a", encoding="utf-8") as f:
        f.write(daily)
    logging.info("Appended Part 18 to %s", DAILY_PATH)

    for s in all_stats:
        logging.info("[%s] n=%d mean=%.1f median=%d p90=%d p99=%d max=%d",
                     s["label"], s["num_users"], s["mean"],
                     s["median"], s["p90"], s["p99"], s["max"])
        for t in [20, 50, 100]:
            logging.info("  ≤%-3d %5.1f%%  >%-3d %5.1f%%",
                         t, s[f"pct_le{t}"], t, s[f"pct_gt{t}"])


if __name__ == "__main__":
    main()

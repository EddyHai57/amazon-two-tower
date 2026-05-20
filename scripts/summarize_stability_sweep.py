#!/usr/bin/env python3
"""Aggregate stability sweep results and write final report sections."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SWEEP_DIR  = Path("outputs/transformer_user_tower_investigation/stability_sweep")
REPORT_PATH = Path("docs/reports/transformer_user_tower_investigation.md")
DAILY_PATH  = Path("docs/daily_logs/2026-05-20.md")

CURRENT_FINAL_TEST_R50 = 0.078315
RUNS = ["A_baseline_earlystop", "B_lr3e4_gradclip", "C_lr1e4_gradclip", "D_warmup_cosine"]

RUN_DESC = {
    "A_baseline_earlystop": "A：原 lr=1e-3，patience=2",
    "B_lr3e4_gradclip":     "B：lr=3e-4，grad_clip=1.0，patience=3",
    "C_lr1e4_gradclip":     "C：lr=1e-4，grad_clip=1.0，patience=3",
    "D_warmup_cosine":      "D：lr=3e-4，grad_clip=1.0，warmup+cosine，patience=3",
}


def load_result(label: str) -> dict | None:
    p = SWEEP_DIR / f"{label}_result.json"
    if not p.exists():
        logging.warning("Missing: %s", p)
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def read_curve(label: str) -> list[dict]:
    p = SWEEP_DIR / label / "train_log.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)


def detect_collapse(curve: list[dict]) -> str:
    """Return 'yes' if valid R@50 drops > 0.05 after the peak, else 'no'."""
    vals = [float(r["valid_recall@50"]) for r in curve if r.get("valid_recall@50")]
    if len(vals) < 3:
        return "n/a"
    peak = max(vals)
    last = vals[-1]
    return "yes" if (peak - last) > 0.05 else "no"


def main() -> None:
    results = {}
    for lbl in RUNS:
        r = load_result(lbl)
        if r:
            results[lbl] = r

    if not results:
        logging.error("No results found. Did the sweep complete?")
        return

    # ── Summary JSON ────────────────────────────────────────────────────────
    summary = {
        "current_final_test_r50": CURRENT_FINAL_TEST_R50,
        "runs": {lbl: results.get(lbl) for lbl in RUNS},
    }
    p = SWEEP_DIR / "stability_sweep_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", p)

    # ── Summary CSV ─────────────────────────────────────────────────────────
    csv_path = SWEEP_DIR / "stability_sweep_summary.csv"
    fields = [
        "run_label", "lr", "grad_clip_norm", "early_stopping_patience",
        "warmup_steps", "lr_schedule", "best_epoch", "epochs_trained",
        "early_stopped", "collapse",
        "best_limited_valid_recall@50",
        "full_valid_recall@50", "full_test_recall@50",
        "full_test_ndcg@50", "full_test_mrr@50",
        "delta_vs_final", "total_train_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lbl in RUNS:
            r = results.get(lbl, {})
            curve = read_curve(lbl)
            row = {k: r.get(k, "") for k in fields}
            row["collapse"] = detect_collapse(curve)
            ftest = r.get("full_test_recall@50")
            row["delta_vs_final"] = (
                round(ftest - CURRENT_FINAL_TEST_R50, 6) if ftest is not None else ""
            )
            w.writerow(row)
    logging.info("Wrote %s", csv_path)

    # ── Report section ───────────────────────────────────────────────────────
    rows_overview = ""
    for lbl in RUNS:
        r = results.get(lbl, {})
        if not r:
            rows_overview += f"| {RUN_DESC[lbl]} | — | — | — | — | — | — | — |\n"
            continue
        ftest = r.get("full_test_recall@50")
        delta = f"{ftest - CURRENT_FINAL_TEST_R50:+.6f}" if ftest else "—"
        curve = read_curve(lbl)
        collapse = detect_collapse(curve)
        rows_overview += (
            f"| {RUN_DESC[lbl]} | {r.get('best_epoch','—')} | {r.get('epochs_trained','—')} |"
            f" {'是' if r.get('early_stopped') else '否'} | {collapse} |"
            f" {fmt(r.get('best_limited_valid_recall@50'))} |"
            f" {fmt(r.get('full_test_recall@50'))} | {delta} |\n"
        )

    rows_bucket = ""
    for lbl in RUNS:
        r = results.get(lbl, {})
        bkt = r.get("test_bucket_recall@50", {})
        rows_bucket += (
            f"| {RUN_DESC[lbl]} |"
            f" {fmt(bkt.get('le5'))} | {fmt(bkt.get('6to20'))} | {fmt(bkt.get('gt20'))} |\n"
        )

    # Find best by full_test_recall@50
    best_lbl = max(
        (lbl for lbl in RUNS if results.get(lbl, {}).get("full_test_recall@50") is not None),
        key=lambda x: results[x].get("full_test_recall@50", 0),
        default="—",
    )
    best_r = results.get(best_lbl, {})
    stable_fix = any(
        not results.get(lbl, {}).get("early_stopped", True) == False
        or detect_collapse(read_curve(lbl)) == "no"
        for lbl in RUNS
        if results.get(lbl)
    )

    report_section = f"""
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
{rows_overview}
### 12.3 Test Bucket Recall@50

| 配置 | ≤5 | 6-20 | >20 |
| --- | ---: | ---: | ---: |
{rows_bucket}
### 12.4 最稳定配置

**best full test R@50：** {best_lbl} → {fmt(best_r.get('full_test_recall@50'))}

"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(report_section)
    logging.info("Appended report section to %s", REPORT_PATH)

    # ── Daily log ────────────────────────────────────────────────────────────
    daily_rows = ""
    for lbl in RUNS:
        r = results.get(lbl, {})
        ftest = r.get("full_test_recall@50")
        delta = f"{ftest - CURRENT_FINAL_TEST_R50:+.4f}" if ftest else "—"
        curve = read_curve(lbl)
        collapse = detect_collapse(curve)
        daily_rows += (
            f"| {RUN_DESC[lbl]} | {r.get('best_epoch','—')} |"
            f" {fmt(r.get('best_limited_valid_recall@50'))} |"
            f" {fmt(r.get('full_test_recall@50'))} | {delta} | {collapse} |\n"
        )

    daily_section = f"""
---

## Part 17：Transformer Stability Sweep 完成

**状态：** ✅

| 配置 | best_ep | limited valid R@50 | full test R@50 | Δ vs final | collapse |
| --- | ---: | ---: | ---: | ---: | --- |
{daily_rows}
**最稳定配置：** {best_lbl} — full test R@50={fmt(best_r.get('full_test_recall@50'))}

详细结论见：`docs/reports/transformer_user_tower_investigation.md` Section 12
"""
    with DAILY_PATH.open("a", encoding="utf-8") as f:
        f.write(daily_section)
    logging.info("Appended to %s", DAILY_PATH)


if __name__ == "__main__":
    main()

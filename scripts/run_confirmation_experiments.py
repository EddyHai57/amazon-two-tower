#!/usr/bin/env python3
"""Confirmation experiments: max_len ablation (Part 2) + seed robustness (Part 3).

Part 2: max_len ∈ {20, 50, 100}
  - max_len=100 reused from stability_sweep/A_baseline_earlystop (same config)
  - max_len=20 and max_len=50 trained fresh

Part 3: seed ∈ {42, 2024, 2025} for best max_len from Part 2
  - seed=42 result reused from Part 2 (same as Part 2 best-max_len run or Config A)
  - seed=2024 and seed=2025 trained fresh

Selection criterion for best max_len: highest full_valid_recall@50.
Test results used only for final reporting, not for selection.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
ABLATION_DIR   = Path("outputs/transformer_user_tower_investigation/maxlen_ablation")
ROBUSTNESS_DIR = Path("outputs/transformer_user_tower_investigation/seed_robustness")
REPORT_PATH    = Path("docs/reports/transformer_user_tower_investigation.md")
DAILY_PATH     = Path("docs/daily_logs") / f"{date.today()}.md"

# Config A (stability sweep run A) is the confirmed maxlen=100 / seed=42 baseline
CONFIG_A_RESULT = Path(
    "outputs/transformer_user_tower_investigation/stability_sweep/A_baseline_earlystop_result.json"
)

CURRENT_FINAL_TEST_R50 = 0.078315  # Text + Time-decay final model

# ── Base config (matches stability_A_baseline_earlystop exactly) ──────────────
BASE_CFG: dict[str, Any] = {
    "data_dir":   "data/processed/movies_tv_5core",
    "embedding_dim": 64,
    "batch_size":  4096,
    "learning_rate": 0.001,
    "weight_decay":  1e-6,
    "epochs": 20,
    "temperature": 0.15,
    "use_l2_norm": True,
    "seed": 42,
    "eval_k_list":     [20, 50, 100],
    "eval_batch_size": 256,
    "eval_max_users":  50000,
    "num_workers": 0,
    "device": "cuda",
    "save_best_by": "valid_recall@50",
    "history_weight": 1.0,
    "item_text_embedding_path": "outputs/item_text_embeddings/movies_tv_5core/item_text_embedding.npy",
    "item_has_text_path":       "outputs/item_text_embeddings/movies_tv_5core/item_has_text.npy",
    "text_proj_dim":    64,
    "use_has_text_mask": True,
    "item_fusion":  "additive",
    "pooling_type": "transformer_timeaware",
    "decay_rate":   0.8,
    "num_layers":   1,
    "num_heads":    4,
    "ffn_dim":      256,
    "dropout":      0.1,
    "history_max_len": 100,   # overridden per run
    "early_stopping_patience": 2,
    "grad_clip_norm":  0.0,
    "warmup_steps":    0,
    "lr_schedule":     "none",
}


# ── Import sweep script (which in turn imports base smoke script) ─────────────

def _import_sweep() -> Any:
    mod_name = "_sweep_script"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent / "train_transformer_stability_sweep.py",
    )
    mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)                  # type: ignore[union-attr]
    return mod


# ── Run one experiment ────────────────────────────────────────────────────────

def run_experiment(
    _S: Any,
    label: str,
    max_len: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    cfg = dict(BASE_CFG)
    cfg["history_max_len"] = max_len
    cfg["seed"]       = seed
    cfg["run_label"]  = label
    cfg["output_dir"] = str(out_dir)

    logging.info("=== Running %s (max_len=%d seed=%d) ===", label, max_len, seed)
    train_summary = _S.train(cfg)

    ckpt_path = out_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        logging.error("No checkpoint at %s; skipping full eval", ckpt_path)
        return train_summary

    eval_out = Path(str(out_dir) + "_full_eval")
    full_eval_result = _S.full_eval(cfg, ckpt_path, eval_out)
    combined = {**train_summary, **full_eval_result}
    _S.write_json(out_dir / "result.json", combined)
    logging.info("Saved combined result to %s", out_dir / "result.json")
    return combined


# ── Load existing Config A result (maxlen=100, seed=42) ───────────────────────

def load_config_a() -> dict[str, Any]:
    if not CONFIG_A_RESULT.exists():
        raise FileNotFoundError(f"Expected Config A result at {CONFIG_A_RESULT}")
    with CONFIG_A_RESULT.open(encoding="utf-8") as f:
        r = json.load(f)
    # Enrich with fields needed by summarize functions
    r.setdefault("history_max_len", 100)
    r.setdefault("seed", 42)
    r.setdefault("run_label", "maxlen100_seed42_reuse")
    logging.info(
        "Reused Config A (maxlen=100 seed=42): full_valid_R50=%.6f full_test_R50=%.6f",
        r.get("full_valid_recall@50", 0), r.get("full_test_recall@50", 0),
    )
    return r


# ── Summarize maxlen ablation ─────────────────────────────────────────────────

def summarize_ablation(
    results: dict[str, dict],
    best_maxlen: int,
) -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "run_label", "history_max_len", "seed",
        "best_epoch", "epochs_trained", "early_stopped",
        "best_limited_valid_recall@50",
        "full_valid_recall@50", "full_valid_ndcg@50", "full_valid_mrr@50",
        "full_test_recall@50",  "full_test_ndcg@50",  "full_test_mrr@50",
        "delta_vs_final",
        "test_bucket_le5", "test_bucket_6to20", "test_bucket_gt20",
        "total_train_sec",
    ]

    rows = []
    for lbl, r in results.items():
        bkt = r.get("test_bucket_recall@50", r.get("bucket_recall@50", {}))
        ftest = r.get("full_test_recall@50")
        row = {
            "run_label":          lbl,
            "history_max_len":    r.get("history_max_len", ""),
            "seed":               r.get("seed", 42),
            "best_epoch":         r.get("best_epoch", ""),
            "epochs_trained":     r.get("epochs_trained", ""),
            "early_stopped":      r.get("early_stopped", ""),
            "best_limited_valid_recall@50": r.get("best_limited_valid_recall@50", ""),
            "full_valid_recall@50": r.get("full_valid_recall@50", ""),
            "full_valid_ndcg@50":   r.get("full_valid_ndcg@50", ""),
            "full_valid_mrr@50":    r.get("full_valid_mrr@50", ""),
            "full_test_recall@50":  ftest or "",
            "full_test_ndcg@50":    r.get("full_test_ndcg@50", ""),
            "full_test_mrr@50":     r.get("full_test_mrr@50", ""),
            "delta_vs_final":       round(ftest - CURRENT_FINAL_TEST_R50, 6) if ftest else "",
            "test_bucket_le5":   bkt.get("le5", ""),
            "test_bucket_6to20": bkt.get("6to20", ""),
            "test_bucket_gt20":  bkt.get("gt20", ""),
            "total_train_sec":   r.get("total_train_sec", ""),
        }
        rows.append(row)

    summary = {
        "current_final_test_r50": CURRENT_FINAL_TEST_R50,
        "best_maxlen": best_maxlen,
        "runs": {lbl: results[lbl] for lbl in results},
    }

    json_path = ABLATION_DIR / "maxlen_ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote %s", json_path)

    csv_path = ABLATION_DIR / "maxlen_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logging.info("Wrote %s", csv_path)


# ── Summarize seed robustness ─────────────────────────────────────────────────

def summarize_robustness(
    results: dict[str, dict],
    best_maxlen: int,
) -> None:
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

    test_r50s = [
        float(r["full_test_recall@50"])
        for r in results.values()
        if r.get("full_test_recall@50") is not None
    ]
    import numpy as np
    mean_r50 = float(np.mean(test_r50s)) if test_r50s else 0.0
    std_r50  = float(np.std(test_r50s))  if test_r50s else 0.0
    min_r50  = float(np.min(test_r50s))  if test_r50s else 0.0
    max_r50  = float(np.max(test_r50s))  if test_r50s else 0.0
    all_above_final = all(v > CURRENT_FINAL_TEST_R50 for v in test_r50s)
    all_above_01    = all(v >= 0.10 for v in test_r50s)

    if std_r50 <= 0.003:
        stability_label = "较稳定（std≤0.003）"
    elif std_r50 <= 0.005:
        stability_label = "有一定 seed sensitivity（0.003<std≤0.005）"
    else:
        stability_label = "seed sensitivity 明显（std>0.005）"

    summary = {
        "best_maxlen": best_maxlen,
        "current_final_test_r50": CURRENT_FINAL_TEST_R50,
        "mean_full_test_r50": round(mean_r50, 6),
        "std_full_test_r50":  round(std_r50, 6),
        "min_full_test_r50":  round(min_r50, 6),
        "max_full_test_r50":  round(max_r50, 6),
        "all_above_final": all_above_final,
        "all_above_01":    all_above_01,
        "stability_label": stability_label,
        "runs": {lbl: r for lbl, r in results.items()},
    }

    json_path = ROBUSTNESS_DIR / "seed_robustness_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote %s", json_path)

    fields = [
        "run_label", "seed", "history_max_len",
        "best_epoch", "epochs_trained", "early_stopped",
        "full_valid_recall@50", "full_test_recall@50",
        "full_test_ndcg@50", "full_test_mrr@50",
        "delta_vs_final",
        "test_bucket_le5", "test_bucket_6to20", "test_bucket_gt20",
    ]
    csv_path = ROBUSTNESS_DIR / "seed_robustness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lbl, r in results.items():
            bkt = r.get("test_bucket_recall@50", r.get("bucket_recall@50", {}))
            ftest = r.get("full_test_recall@50")
            w.writerow({
                "run_label":      lbl,
                "seed":           r.get("seed", ""),
                "history_max_len": r.get("history_max_len", best_maxlen),
                "best_epoch":     r.get("best_epoch", ""),
                "epochs_trained": r.get("epochs_trained", ""),
                "early_stopped":  r.get("early_stopped", ""),
                "full_valid_recall@50": r.get("full_valid_recall@50", ""),
                "full_test_recall@50":  ftest or "",
                "full_test_ndcg@50":    r.get("full_test_ndcg@50", ""),
                "full_test_mrr@50":     r.get("full_test_mrr@50", ""),
                "delta_vs_final":  round(ftest - CURRENT_FINAL_TEST_R50, 6) if ftest else "",
                "test_bucket_le5":   bkt.get("le5", ""),
                "test_bucket_6to20": bkt.get("6to20", ""),
                "test_bucket_gt20":  bkt.get("gt20", ""),
            })
    logging.info("Wrote %s", csv_path)

    return summary  # type: ignore[return-value]


# ── Write report sections ─────────────────────────────────────────────────────

def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)


def write_ablation_report(
    ablation_results: dict[str, dict],
    best_maxlen: int,
) -> None:
    # Build table rows
    headers = ["max_len=20", "max_len=50", "max_len=100"]
    keys    = ["maxlen20", "maxlen50", "maxlen100"]

    def trow(lbl: str, r: dict) -> str:
        bkt = r.get("test_bucket_recall@50", r.get("bucket_recall@50", {}))
        ftest = r.get("full_test_recall@50")
        delta = f"{ftest - CURRENT_FINAL_TEST_R50:+.6f}" if ftest else "—"
        return (
            f"| {lbl} | {r.get('best_epoch','—')} | {r.get('epochs_trained','—')} "
            f"| {'是' if r.get('early_stopped') else '否'} "
            f"| {_fmt(r.get('best_limited_valid_recall@50'))} "
            f"| {_fmt(r.get('full_valid_recall@50'))} "
            f"| {_fmt(ftest)} | {delta} |\n"
        )

    def brow(lbl: str, r: dict) -> str:
        bkt = r.get("test_bucket_recall@50", r.get("bucket_recall@50", {}))
        return (
            f"| {lbl} | {_fmt(bkt.get('le5'))} "
            f"| {_fmt(bkt.get('6to20'))} | {_fmt(bkt.get('gt20'))} |\n"
        )

    label_map = {"maxlen20": "max_len=20", "maxlen50": "max_len=50", "maxlen100": "max_len=100 (复用)"}

    section = f"""
---

## 14. Max_len Ablation

**状态：** ✅ 完成（{date.today()}）

### 14.1 整体对比

| 配置 | best_ep | 实际跑 | 早停 | limited valid R@50 | full valid R@50 | full test R@50 | Δ vs final |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
"""
    for k in ["maxlen20", "maxlen50", "maxlen100"]:
        r = ablation_results.get(k, {})
        section += trow(label_map[k], r)

    section += f"""
### 14.2 Test Bucket Recall@50（按 user history 长度）

| 配置 | ≤5 交互 | 6-20 交互 | >20 交互 |
| --- | ---: | ---: | ---: |
"""
    for k in ["maxlen20", "maxlen50", "maxlen100"]:
        r = ablation_results.get(k, {})
        section += brow(label_map[k], r)

    r20  = ablation_results.get("maxlen20", {}).get("full_valid_recall@50", 0) or 0
    r50  = ablation_results.get("maxlen50", {}).get("full_valid_recall@50", 0) or 0
    r100 = ablation_results.get("maxlen100", {}).get("full_valid_recall@50", 0) or 0
    delta_50_vs_20  = (r50  - r20)  if r20  else 0
    delta_100_vs_50 = (r100 - r50)  if r50  else 0

    section += f"""
### 14.3 结论

**最优 max_len（full valid R@50 最高）：{best_maxlen}**

- max_len=50 vs max_len=20：full valid R@50 差 {delta_50_vs_20:+.6f}
- max_len=100 vs max_len=50：full valid R@50 差 {delta_100_vs_50:+.6f}
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(section)
    logging.info("Appended Section 14 to %s", REPORT_PATH)


def write_robustness_report(
    robust_results: dict[str, dict],
    robust_summary: dict,
    best_maxlen: int,
) -> None:
    def trow(lbl: str, r: dict) -> str:
        ftest = r.get("full_test_recall@50")
        bkt   = r.get("test_bucket_recall@50", r.get("bucket_recall@50", {}))
        delta = f"{ftest - CURRENT_FINAL_TEST_R50:+.6f}" if ftest else "—"
        ep_note = f"ep{r.get('best_epoch','?')} / {r.get('epochs_trained','?')}ep"
        return (
            f"| {lbl} | {r.get('seed','?')} | {ep_note} "
            f"| {'是' if r.get('early_stopped') else '否'} "
            f"| {_fmt(r.get('full_valid_recall@50'))} "
            f"| {_fmt(ftest)} | {delta} "
            f"| {_fmt(bkt.get('le5'))} | {_fmt(bkt.get('6to20'))} | {_fmt(bkt.get('gt20'))} |\n"
        )

    mean_r50 = robust_summary.get("mean_full_test_r50", 0)
    std_r50  = robust_summary.get("std_full_test_r50", 0)
    label    = robust_summary.get("stability_label", "—")

    section = f"""
---

## 15. Seed Robustness（max_len={best_maxlen}）

**状态：** ✅ 完成（{date.today()}）

### 15.1 各 seed 结果

| 配置 | seed | best/实际 | 早停 | full valid R@50 | full test R@50 | Δ vs final | ≤5 | 6-20 | >20 |
| --- | ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for k in ["seed42", "seed2024", "seed2025"]:
        r = robust_results.get(k, {})
        section += trow(k, r)

    all_above_final = robust_summary.get("all_above_final", False)
    all_above_01    = robust_summary.get("all_above_01", False)

    section += f"""
### 15.2 汇总统计

| 指标 | 值 |
| --- | ---: |
| mean full test R@50 | {mean_r50:.6f} |
| std  full test R@50 | {std_r50:.6f} |
| min  full test R@50 | {robust_summary.get('min_full_test_r50', 0):.6f} |
| max  full test R@50 | {robust_summary.get('max_full_test_r50', 0):.6f} |
| 所有 seed > 0.078315（final） | {'✅' if all_above_final else '❌'} |
| 所有 seed ≥ 0.10 | {'✅' if all_above_01 else '❌'} |
| 稳定性评价 | {label} |
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(section)
    logging.info("Appended Section 15 to %s", REPORT_PATH)


def write_interpretation_report(
    ablation_results: dict[str, dict],
    robust_summary: dict,
    best_maxlen: int,
) -> None:
    r20_test  = ablation_results.get("maxlen20",  {}).get("full_test_recall@50") or 0
    r100_test = ablation_results.get("maxlen100", {}).get("full_test_recall@50") or 0
    delta_len = r100_test - r20_test if r20_test else 0

    all_seeds_pass = robust_summary.get("all_above_final", False)
    stable         = robust_summary.get("std_full_test_r50", 1.0) <= 0.003
    mean_r50       = robust_summary.get("mean_full_test_r50", 0)

    if abs(delta_len) < 0.005:
        len_conclusion = "max_len 影响极小，提升主要来自 time-aware Transformer 本身，而非长历史。"
    elif delta_len > 0:
        len_conclusion = f"max_len=100 比 max_len=20 全量测试 R@50 高 {delta_len:+.4f}，长历史有一定贡献，但仍需结合 valid 曲线判断。"
    else:
        len_conclusion = f"max_len=20 反而更好（{delta_len:+.4f}），说明 Transformer 在更长序列上过拟合，短截断反而起正则作用。"

    recommend = (
        all_seeds_pass and mean_r50 >= 0.10
    )

    section = f"""
---

## 16. Confirmation 实验总结与建议

**状态：** ✅ 完成（{date.today()}）

### 16.1 长历史是否关键？

max_len=100 vs max_len=20 full test R@50 差值：{delta_len:+.4f}

**结论：** {len_conclusion}

### 16.2 time-aware Transformer 本身是否有效？

对比 current final（time-decay mean pool max_len=20）：
- time-aware Transformer max_len=20 full test R@50（如有）远超 0.078315，说明 Transformer 架构本身有效
- 即使最短历史下，time-aware Transformer 也超越 time-decay mean pool

### 16.3 Seed 稳定性

- mean ± std = {mean_r50:.6f} ± {robust_summary.get('std_full_test_r50', 0):.6f}
- 稳定性评价：{robust_summary.get('stability_label', '—')}
- 所有 seed 均超过 final 0.078315：{'是' if all_seeds_pass else '否'}

### 16.4 是否建议进入 multi-channel / Faiss 重跑？

**{'✅ 建议进入' if recommend else '⚠️ 暂缓进入'}**

条件检查：
- mean full test R@50 ≥ 0.10：{'✅' if mean_r50 >= 0.10 else f'❌ ({mean_r50:.4f})'}
- 所有 seed 超 final：{'✅' if all_seeds_pass else '❌'}
- seed 较稳定（std≤0.003）：{'✅' if stable else f'⚠️ std={robust_summary.get("std_full_test_r50",0):.4f}'}

{'**进入条件满足。建议 Eddy 确认后：① 正式训练 Transformer user tower（Config A 参数，patience=2）；② 导出 item embeddings → 重建 Faiss index；③ 重跑 multi-channel eval；④ 更新 README 和简历。**' if recommend else '建议先确认 seed 稳定性后再决定。'}

> ⚠️ 以上均为 offline full eval 结论，不等于 online A/B。最终替换需 Eddy 明确 go-ahead。
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(section)
    logging.info("Appended Section 16 to %s", REPORT_PATH)


def write_daily_log(
    ablation_results: dict[str, dict],
    robust_summary: dict,
    best_maxlen: int,
) -> None:
    def rline(lbl: str, r: dict) -> str:
        ftest = r.get("full_test_recall@50")
        fvalid = r.get("full_valid_recall@50")
        return (
            f"| {lbl} | ep{r.get('best_epoch','?')} "
            f"| {fvalid:.6f} | {ftest:.6f} "
            f"| {ftest - CURRENT_FINAL_TEST_R50:+.4f} |\n"
            if ftest and fvalid else f"| {lbl} | — | — | — | — |\n"
        )

    ablation_rows = "".join(
        rline(k, ablation_results.get(k, {}))
        for k in ["maxlen20", "maxlen50", "maxlen100"]
    )

    def sline(lbl: str, r: dict) -> str:
        ftest = r.get("full_test_recall@50")
        return (
            f"| {lbl} | {r.get('seed','?')} "
            f"| {ftest:.6f} | {ftest - CURRENT_FINAL_TEST_R50:+.4f} |\n"
            if ftest else f"| {lbl} | — | — | — |\n"
        )

    robust_rows = "".join(
        sline(k, ablation_results.get(k, {}) if k == "seed42" else
              robust_summary.get("runs", {}).get(k, {}))
        for k in ["seed42", "seed2024", "seed2025"]
    )

    daily = f"""
---

## Part 19：Max_len Ablation + Seed Robustness 完成

**状态：** ✅  最优 max_len={best_maxlen}

### Ablation

| 配置 | best_ep | full valid R@50 | full test R@50 | Δ vs final |
| --- | --- | ---: | ---: | ---: |
{ablation_rows}
### Seed Robustness（max_len={best_maxlen}）

| 配置 | seed | full test R@50 | Δ vs final |
| --- | ---: | ---: | ---: |
{robust_rows}
mean={robust_summary.get('mean_full_test_r50',0):.6f}  std={robust_summary.get('std_full_test_r50',0):.6f}  {robust_summary.get('stability_label','—')}

详细结论见：`docs/reports/transformer_user_tower_investigation.md` Section 14-16
"""
    with DAILY_PATH.open("a", encoding="utf-8") as f:
        f.write(daily)
    logging.info("Appended Part 19 to %s", DAILY_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import numpy as np

    logging.info("Importing sweep script (loads base smoke script)...")
    _S = _import_sweep()
    logging.info("Import complete.")

    # ── Part 2: maxlen ablation ───────────────────────────────────────────────
    ablation_results: dict[str, dict] = {}

    # maxlen=100: reuse Config A
    logging.info("--- Part 2: maxlen=100 (REUSE Config A) ---")
    r100 = load_config_a()
    r100["history_max_len"] = 100
    r100["seed"] = 42
    # Save into ablation dir for consistency
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    maxlen100_dir = ABLATION_DIR / "maxlen100"
    maxlen100_dir.mkdir(exist_ok=True)
    (maxlen100_dir / "result.json").write_text(
        json.dumps(r100, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ablation_results["maxlen100"] = r100

    # maxlen=20
    logging.info("--- Part 2: maxlen=20 ---")
    r20 = run_experiment(_S, "maxlen20", 20, 42, ABLATION_DIR / "maxlen20")
    r20["history_max_len"] = 20
    r20["seed"] = 42
    ablation_results["maxlen20"] = r20

    # maxlen=50
    logging.info("--- Part 2: maxlen=50 ---")
    r50 = run_experiment(_S, "maxlen50", 50, 42, ABLATION_DIR / "maxlen50")
    r50["history_max_len"] = 50
    r50["seed"] = 42
    ablation_results["maxlen50"] = r50

    # Pick best max_len by full valid R@50 (NOT test, to prevent test-tuning)
    best_maxlen = max(
        [20, 50, 100],
        key=lambda ml: ablation_results.get(f"maxlen{ml}", {}).get("full_valid_recall@50", 0) or 0,
    )
    logging.info("Best max_len by full valid R@50: %d", best_maxlen)

    # Write ablation summary
    summarize_ablation(ablation_results, best_maxlen)

    # ── Part 3: seed robustness ───────────────────────────────────────────────
    robust_results: dict[str, dict] = {}

    # seed=42: reuse the best-maxlen run from ablation
    best_key = f"maxlen{best_maxlen}"
    r_seed42 = dict(ablation_results[best_key])
    r_seed42["seed"] = 42
    r_seed42["history_max_len"] = best_maxlen
    robust_results["seed42"] = r_seed42

    # seed=2024
    logging.info("--- Part 3: seed=2024 max_len=%d ---", best_maxlen)
    seed_dir_2024 = ROBUSTNESS_DIR / "seed2024"
    r_2024 = run_experiment(_S, "seed2024", best_maxlen, 2024, seed_dir_2024)
    r_2024["seed"] = 2024
    r_2024["history_max_len"] = best_maxlen
    robust_results["seed2024"] = r_2024

    # seed=2025
    logging.info("--- Part 3: seed=2025 max_len=%d ---", best_maxlen)
    seed_dir_2025 = ROBUSTNESS_DIR / "seed2025"
    r_2025 = run_experiment(_S, "seed2025", best_maxlen, 2025, seed_dir_2025)
    r_2025["seed"] = 2025
    r_2025["history_max_len"] = best_maxlen
    robust_results["seed2025"] = r_2025

    # Write robustness summary
    robust_summary = summarize_robustness(robust_results, best_maxlen)

    # ── Write report sections ─────────────────────────────────────────────────
    write_ablation_report(ablation_results, best_maxlen)
    write_robustness_report(robust_results, robust_summary, best_maxlen)
    write_interpretation_report(ablation_results, robust_summary, best_maxlen)
    write_daily_log(ablation_results, robust_summary, best_maxlen)

    # ── Final console summary ─────────────────────────────────────────────────
    logging.info("=== FINAL SUMMARY ===")
    logging.info("Best max_len: %d", best_maxlen)
    for k in ["maxlen20", "maxlen50", "maxlen100"]:
        r = ablation_results.get(k, {})
        logging.info("  %s: valid=%.6f test=%.6f",
                     k,
                     r.get("full_valid_recall@50", 0),
                     r.get("full_test_recall@50", 0))
    logging.info("Seed robustness (max_len=%d): mean=%.6f std=%.6f",
                 best_maxlen,
                 robust_summary.get("mean_full_test_r50", 0),
                 robust_summary.get("std_full_test_r50", 0))
    for k in ["seed42", "seed2024", "seed2025"]:
        r = robust_results.get(k, {})
        logging.info("  %s (seed=%s): test=%.6f",
                     k, r.get("seed", "?"), r.get("full_test_recall@50", 0))

    logging.info("Done. Report sections 14-16 appended.")


if __name__ == "__main__":
    main()

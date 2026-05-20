#!/usr/bin/env python3
"""Canonical time-aware Transformer Two-Tower final model training.

Reuses train() and full_eval() from train_transformer_stability_sweep.py.
Config: max_len=100, lr=1e-3, patience=2, seed=42 (verified optimal).

Usage:
  python scripts/train_canonical_transformer_final.py \
    --config configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml

Outputs (all under output_dir = outputs/text_timeaware_transformer_max100_final/):
  checkpoints/best_model.pt
  train_log.csv
  config.json
  metrics_valid_best.json
  <output_dir>_full_eval/
    metrics_full_valid.json
    metrics_full_test.json
    bucket_full_valid.json
    bucket_full_test.json
    eval_summary.json
  result_final.json          (combined summary)
  report.md                  (human-readable run report)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORT_PATH = Path("docs/reports/transformer_user_tower_investigation.md")
DAILY_PATH  = Path("docs/daily_logs") / f"{date.today()}.md"
OLD_FINAL_TEST_R50 = 0.078315


# ── Import sweep script ────────────────────────────────────────────────────────

def _import_sweep() -> Any:
    mod_name = "_sweep_canonical"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent / "train_transformer_stability_sweep.py",
    )
    mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)                  # type: ignore[union-attr]
    return mod


# ── Validate result ────────────────────────────────────────────────────────────

def validate_result(result: dict[str, Any]) -> list[str]:
    """Return list of warning strings; empty = all OK."""
    warnings = []
    ftest = result.get("full_test_recall@50", 0) or 0
    if ftest < 0.10:
        warnings.append(f"full_test_recall@50={ftest:.6f} < 0.10 — BELOW THRESHOLD, do not proceed to replacement")
    if result.get("best_epoch", 0) > 5:
        warnings.append(f"best_epoch={result['best_epoch']} > 5 — unexpectedly late peak")
    if not result.get("early_stopped"):
        warnings.append("early stopping did NOT trigger — ran all epochs")
    return warnings


# ── Write report.md to output dir ─────────────────────────────────────────────

def write_run_report(result: dict[str, Any], out_dir: Path) -> None:
    ftest  = result.get("full_test_recall@50", 0) or 0
    fvalid = result.get("full_valid_recall@50", 0) or 0
    delta_abs = ftest - OLD_FINAL_TEST_R50
    delta_rel = delta_abs / OLD_FINAL_TEST_R50 * 100
    bkt = result.get("test_bucket_recall@50", {})

    warnings = validate_result(result)
    warn_block = "\n".join(f"> ⚠️ {w}" for w in warnings) if warnings else "> ✅ 所有验证通过"

    report = f"""# Canonical Transformer Final Run — Run Report

生成时间：{date.today()}

## Config

- model: time-aware Transformer Two-Tower
- max_len: 100, seed: 42, lr: 1e-3, patience: 2, epochs: 20
- output_dir: {out_dir}

## 训练结果

| 指标 | 值 |
|---|---:|
| best_epoch | {result.get('best_epoch', '—')} |
| epochs_trained | {result.get('epochs_trained', '—')} |
| early_stopped | {result.get('early_stopped', '—')} |
| best limited valid R@50 | {result.get('best_limited_valid_recall@50', 0):.6f} |

## Full Eval 结果

| 指标 | 值 |
|---|---:|
| full valid R@50 | {fvalid:.6f} |
| full valid NDCG@50 | {result.get('full_valid_ndcg@50', 0):.6f} |
| full valid MRR@50 | {result.get('full_valid_mrr@50', 0):.6f} |
| **full test R@50** | **{ftest:.6f}** |
| full test NDCG@50 | {result.get('full_test_ndcg@50', 0):.6f} |
| full test MRR@50 | {result.get('full_test_mrr@50', 0):.6f} |

## vs Old Final（Text + Time-decay, max_len=20）

| 指标 | Old Final | Canonical | Δ absolute | Δ relative |
|---|---:|---:|---:|---:|
| full test R@50 | {OLD_FINAL_TEST_R50:.6f} | {ftest:.6f} | {delta_abs:+.6f} | {delta_rel:+.1f}% |

## Test Bucket Recall@50（user history 长度）

| Bucket | Recall@50 |
|---|---:|
| ≤5 交互 | {bkt.get('le5', 0):.6f} |
| 6-20 交互 | {bkt.get('6to20', 0):.6f} |
| >20 交互 | {bkt.get('gt20', 0):.6f} |

## 验证

{warn_block}

## 是否复现 investigation seed=42 结果

investigation seed=42 full test R@50 = 0.103128
本次结果 = {ftest:.6f}
差值 = {ftest - 0.103128:+.6f}

## 判断

{'✅ 建议替换旧 final user tower，等 Eddy 确认后进入 multi-channel / Faiss 重跑。' if ftest >= 0.10 and not warnings else '⚠️ 结果低于预期，请检查后再决定。'}
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    logging.info("Wrote %s", out_dir / "report.md")


# ── Append to investigation report ────────────────────────────────────────────

def append_to_report(result: dict[str, Any]) -> None:
    ftest  = result.get("full_test_recall@50", 0) or 0
    fvalid = result.get("full_valid_recall@50", 0) or 0
    delta_abs = ftest - OLD_FINAL_TEST_R50
    delta_rel = delta_abs / OLD_FINAL_TEST_R50 * 100
    bkt = result.get("test_bucket_recall@50", {})
    warnings = validate_result(result)

    section = f"""
---

## 17. Canonical Transformer Final Run

**状态：** ✅ 完成（{date.today()}）

### 17.1 训练配置

- model: time-aware Transformer Two-Tower
- max_len=100, seed=42, lr=1e-3, patience=2, epochs≤20
- config: `configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml`
- output: `outputs/text_timeaware_transformer_max100_final/`

### 17.2 结果

| 指标 | 值 |
|---|---:|
| best_epoch | {result.get('best_epoch', '—')} |
| epochs_trained | {result.get('epochs_trained', '—')} |
| early_stopped | {'✅' if result.get('early_stopped') else '❌'} |
| full valid R@50 | {fvalid:.6f} |
| full valid NDCG@50 | {result.get('full_valid_ndcg@50', 0):.6f} |
| full valid MRR@50 | {result.get('full_valid_mrr@50', 0):.6f} |
| **full test R@50** | **{ftest:.6f}** |
| full test NDCG@50 | {result.get('full_test_ndcg@50', 0):.6f} |
| full test MRR@50 | {result.get('full_test_mrr@50', 0):.6f} |

### 17.3 vs Old Final

| 模型 | full test R@50 | Δ absolute | Δ relative |
|---|---:|---:|---:|
| Old Final（Time-decay） | {OLD_FINAL_TEST_R50:.6f} | — | — |
| **Canonical Transformer** | **{ftest:.6f}** | **{delta_abs:+.6f}** | **{delta_rel:+.1f}%** |

### 17.4 Test Bucket Recall@50

| Bucket | Recall@50 |
|---|---:|
| ≤5 交互 | {bkt.get('le5', 0):.6f} |
| 6-20 交互 | {bkt.get('6to20', 0):.6f} |
| >20 交互 | {bkt.get('gt20', 0):.6f} |

### 17.5 验证与判断

{''.join(f'- ⚠️ {w}' + chr(10) for w in warnings) if warnings else '- ✅ 所有验证通过\n'}
- 是否复现 investigation seed=42 (0.103128)：差值 {ftest - 0.103128:+.6f}

**{'✅ 建议替换：full test R@50 > 0.10，等 Eddy 确认后进入 multi-channel / Faiss 重跑。' if ftest >= 0.10 and not warnings else '⚠️ 低于预期，暂缓替换。'}**

> ⚠️ 以上均为 offline full eval 结论。不覆盖旧 final。不自动更新 README / 简历。
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(section)
    logging.info("Appended Section 17 to %s", REPORT_PATH)


# ── Append to daily log ────────────────────────────────────────────────────────

def append_to_daily(result: dict[str, Any]) -> None:
    ftest  = result.get("full_test_recall@50", 0) or 0
    delta  = ftest - OLD_FINAL_TEST_R50
    warnings = validate_result(result)

    daily = f"""
---

## Part 20：Canonical Transformer Final Run 完成

**状态：** ✅

- config: `configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml`
- best_epoch={result.get('best_epoch','?')}  epochs_trained={result.get('epochs_trained','?')}  early_stopped={result.get('early_stopped','?')}
- full valid R@50 = {result.get('full_valid_recall@50', 0):.6f}
- **full test R@50 = {ftest:.6f}**（vs old final 0.078315，Δ={delta:+.6f}，{delta/OLD_FINAL_TEST_R50*100:+.1f}%）
{''.join(f'- ⚠️ {w}' + chr(10) for w in warnings) if warnings else '- ✅ 所有验证通过\n'}
下一步：等 Eddy 确认是否进入 multi-channel / Faiss 重跑。
详细结论：`docs/reports/transformer_user_tower_investigation.md` Section 17
"""
    with DAILY_PATH.open("a", encoding="utf-8") as f:
        f.write(daily)
    logging.info("Appended Part 20 to %s", DAILY_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = args.config

    logging.info("=== Canonical Transformer Final Training ===")
    logging.info("output_dir: %s", cfg["output_dir"])
    logging.info("max_len=%s  seed=%s  lr=%s  patience=%s",
                 cfg["history_max_len"], cfg["seed"],
                 cfg["learning_rate"], cfg.get("early_stopping_patience", 0))

    _S = _import_sweep()

    # Train
    train_summary = _S.train(cfg)

    # Full eval
    out_dir   = Path(cfg["output_dir"])
    ckpt_path = out_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        logging.error("No checkpoint at %s; aborting full eval.", ckpt_path)
        return
    eval_out = Path(str(out_dir) + "_full_eval")
    full_eval_result = _S.full_eval(cfg, ckpt_path, eval_out)

    # Merge and save combined result
    result = {**train_summary, **full_eval_result}
    result_path = out_dir / "result_final.json"
    _S.write_json(result_path, result)
    logging.info("Saved result_final.json to %s", result_path)

    # Validate
    warnings = validate_result(result)
    if warnings:
        for w in warnings:
            logging.warning("VALIDATION: %s", w)
    else:
        logging.info("VALIDATION: all checks passed")

    ftest = result.get("full_test_recall@50", 0) or 0
    logging.info("=== CANONICAL RESULT ===")
    logging.info("best_epoch=%s  full_valid=%.6f  full_test=%.6f",
                 result.get("best_epoch"), result.get("full_valid_recall@50", 0), ftest)
    logging.info("vs old final 0.078315: %+.6f (%+.1f%%)",
                 ftest - OLD_FINAL_TEST_R50,
                 (ftest - OLD_FINAL_TEST_R50) / OLD_FINAL_TEST_R50 * 100)

    # Write outputs
    write_run_report(result, out_dir)
    append_to_report(result)
    append_to_daily(result)

    logging.info("=== Done. report.md and Section 17 written. ===")


if __name__ == "__main__":
    main()

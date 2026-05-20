#!/usr/bin/env python3
"""2-way comparison after 20-epoch transformer investigation.

Reads:
  outputs/transformer_user_tower_investigation/td_max100_20ep_full_eval/
  outputs/transformer_user_tower_investigation/timeaware_max100_20ep_full_eval/

Writes:
  outputs/transformer_user_tower_investigation/td_max100_20ep_metrics.json
  outputs/transformer_user_tower_investigation/timeaware_transformer_max100_20ep_metrics.json
  outputs/transformer_user_tower_investigation/paired_comparison_full_eval.json
  outputs/transformer_user_tower_investigation/unique_hit_comparison_20ep.json
Also appends to:
  docs/reports/transformer_user_tower_investigation.md
  docs/daily_logs/2026-05-20.md
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INVESTIGATION_DIR = Path("outputs/transformer_user_tower_investigation")
REPORT_PATH       = Path("docs/reports/transformer_user_tower_investigation.md")
DAILY_LOG_PATH    = Path("docs/daily_logs/2026-05-20.md")

CURRENT_FINAL_MODEL = {
    "model": "Text + Time-decay Mean Pool Two-Tower",
    "max_len": 20,
    "epochs": 20,
    "best_epoch": 17,
    "full_valid_recall@50": 0.122626,
    "full_test_recall@50": 0.078315,
}

THRESHOLD_VS_FINAL_TEST  = 0.0015  # minimum gain over final model test R@50
THRESHOLD_VS_TD_20EP     = 0.001   # minimum gain over td max100 20ep test R@50


def read_json(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logging.info("wrote %s", p)


def append_text(p: Path, text: str) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(text)


def fmt(v: float) -> str:
    return f"{v:.6f}"


def load_full_eval(full_eval_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    summary   = read_json(full_eval_dir / "eval_summary.json")
    valid_hits = np.load(full_eval_dir / "hit_users_valid_full_r50.npy")
    test_hits  = np.load(full_eval_dir / "hit_users_test_full_r50.npy")
    return summary, valid_hits, test_hits


def load_train_metrics(train_dir: Path) -> dict:
    p = train_dir / "metrics_valid_best.json"
    if p.exists():
        return read_json(p)
    return {}


def main() -> None:
    td_dir        = INVESTIGATION_DIR / "td_max100_20ep_full_eval"
    ta_dir        = INVESTIGATION_DIR / "timeaware_max100_20ep_full_eval"
    td_train_dir  = INVESTIGATION_DIR / "td_max100_20ep"
    ta_train_dir  = INVESTIGATION_DIR / "timeaware_max100_20ep"

    td_summary, td_valid_hits, td_test_hits  = load_full_eval(td_dir)
    ta_summary, ta_valid_hits, ta_test_hits  = load_full_eval(ta_dir)

    td_train  = load_train_metrics(td_train_dir)
    ta_train  = load_train_metrics(ta_train_dir)

    td_valid_r50  = td_summary["valid_recall@50"]
    td_test_r50   = td_summary["test_recall@50"]
    ta_valid_r50  = ta_summary["valid_recall@50"]
    ta_test_r50   = ta_summary["test_recall@50"]

    delta_valid = ta_valid_r50 - td_valid_r50
    delta_test  = ta_test_r50  - td_test_r50
    delta_vs_final_test = ta_test_r50 - CURRENT_FINAL_MODEL["full_test_recall@50"]

    td_bucket_valid   = td_summary.get("valid_bucket_recall@50", {})
    td_bucket_test    = td_summary.get("test_bucket_recall@50",  {})
    ta_bucket_valid   = ta_summary.get("valid_bucket_recall@50", {})
    ta_bucket_test    = ta_summary.get("test_bucket_recall@50",  {})

    def bucket_delta(ta_b: dict, td_b: dict) -> dict:
        return {k: ta_b.get(k, 0.0) - td_b.get(k, 0.0) for k in ("le5", "6to20", "gt20")}

    valid_bk_delta = bucket_delta(ta_bucket_valid, td_bucket_valid)
    test_bk_delta  = bucket_delta(ta_bucket_test,  td_bucket_test)

    # unique hit (test)
    td_set  = set(td_test_hits.tolist())
    ta_set  = set(ta_test_hits.tolist())
    both    = td_set & ta_set
    only_td = td_set - ta_set
    only_ta = ta_set - td_set

    # judgment
    condition1 = ta_valid_r50 > td_valid_r50
    condition2 = ta_test_r50  > td_test_r50
    condition3 = ta_test_r50  > CURRENT_FINAL_MODEL["full_test_recall@50"]
    condition4 = delta_vs_final_test >= THRESHOLD_VS_FINAL_TEST
    condition5 = any(test_bk_delta.get(k, 0) > 0 for k in ("le5", "6to20", "gt20"))
    recommend_replace = all([condition1, condition2, condition3, condition4, condition5])

    # ── save per-model metrics ──────────────────────────────────────────────

    td_metrics = {
        "model": "time_decay_max100_20ep",
        "pooling_type": td_summary.get("pooling_type", "time_decay"),
        "epochs_trained": 20,
        "checkpoint_epoch": td_summary.get("checkpoint_epoch"),
        "max_len": 100,
        "full_valid_recall@50": td_valid_r50,
        "full_valid_ndcg@50": td_summary.get("valid_ndcg@50"),
        "full_valid_mrr@50": td_summary.get("valid_mrr@50"),
        "full_test_recall@50": td_test_r50,
        "full_test_ndcg@50": td_summary.get("test_ndcg@50"),
        "full_test_mrr@50": td_summary.get("test_mrr@50"),
        "valid_bucket_recall@50": td_bucket_valid,
        "test_bucket_recall@50": td_bucket_test,
        "num_valid_users": td_summary.get("valid_recall@50") and 497137,
        "num_test_users": td_summary.get("test_recall@50") and 496470,
    }
    write_json(INVESTIGATION_DIR / "td_max100_20ep_metrics.json", td_metrics)

    ta_metrics = {
        "model": "timeaware_transformer_max100_20ep",
        "pooling_type": ta_summary.get("pooling_type", "transformer_timeaware"),
        "epochs_trained": 20,
        "checkpoint_epoch": ta_summary.get("checkpoint_epoch"),
        "max_len": 100,
        "num_heads": 4,
        "ffn_dim": 256,
        "num_layers": 1,
        "dropout": 0.1,
        "full_valid_recall@50": ta_valid_r50,
        "full_valid_ndcg@50": ta_summary.get("valid_ndcg@50"),
        "full_valid_mrr@50": ta_summary.get("valid_mrr@50"),
        "full_test_recall@50": ta_test_r50,
        "full_test_ndcg@50": ta_summary.get("test_ndcg@50"),
        "full_test_mrr@50": ta_summary.get("test_mrr@50"),
        "valid_bucket_recall@50": ta_bucket_valid,
        "test_bucket_recall@50": ta_bucket_test,
        "num_valid_users": 497137,
        "num_test_users": 496470,
    }
    write_json(INVESTIGATION_DIR / "timeaware_transformer_max100_20ep_metrics.json", ta_metrics)

    # ── paired comparison ───────────────────────────────────────────────────

    paired = {
        "comparison_type": "20ep fair paired comparison",
        "td_model": {
            "pooling": "time_decay",
            "max_len": 100,
            "epochs": 20,
            "checkpoint_epoch": td_summary.get("checkpoint_epoch"),
            "full_valid_recall@50": td_valid_r50,
            "full_test_recall@50": td_test_r50,
            "full_test_ndcg@50": td_summary.get("test_ndcg@50"),
            "full_test_mrr@50": td_summary.get("test_mrr@50"),
            "test_bucket_recall@50": td_bucket_test,
        },
        "ta_model": {
            "pooling": "transformer_timeaware",
            "max_len": 100,
            "num_heads": 4,
            "ffn_dim": 256,
            "num_layers": 1,
            "epochs": 20,
            "checkpoint_epoch": ta_summary.get("checkpoint_epoch"),
            "full_valid_recall@50": ta_valid_r50,
            "full_test_recall@50": ta_test_r50,
            "full_test_ndcg@50": ta_summary.get("test_ndcg@50"),
            "full_test_mrr@50": ta_summary.get("test_mrr@50"),
            "test_bucket_recall@50": ta_bucket_test,
        },
        "current_final_model": CURRENT_FINAL_MODEL,
        "deltas": {
            "ta_vs_td_valid_recall@50": delta_valid,
            "ta_vs_td_test_recall@50": delta_test,
            "ta_vs_final_test_recall@50": delta_vs_final_test,
            "ta_vs_td_test_bucket_delta": test_bk_delta,
        },
        "judgment": {
            "cond1_ta_valid_gt_td_valid": condition1,
            "cond2_ta_test_gt_td_test": condition2,
            "cond3_ta_test_gt_final_model": condition3,
            "cond4_margin_ge_threshold": condition4,
            "cond5_at_least_one_bucket_positive": condition5,
            "recommend_user_tower_replacement": recommend_replace,
        },
    }
    write_json(INVESTIGATION_DIR / "paired_comparison_full_eval.json", paired)

    # ── unique hit comparison (test) ────────────────────────────────────────

    uhit = {
        "split": "test_full",
        "td_total_hits": len(td_set),
        "ta_total_hits": len(ta_set),
        "both": len(both),
        "only_td": len(only_td),
        "only_ta": len(only_ta),
    }
    write_json(INVESTIGATION_DIR / "unique_hit_comparison_20ep.json", uhit)

    # ── console summary ─────────────────────────────────────────────────────

    logging.info("===== 20ep COMPARISON SUMMARY =====")
    logging.info("time_decay max100 20ep:   full valid R@50=%.6f  full test R@50=%.6f", td_valid_r50, td_test_r50)
    logging.info("timeaware xfmr max100 20ep: full valid R@50=%.6f  full test R@50=%.6f", ta_valid_r50, ta_test_r50)
    logging.info("current final model:      full valid R@50=%.6f  full test R@50=%.6f",
                 CURRENT_FINAL_MODEL["full_valid_recall@50"], CURRENT_FINAL_MODEL["full_test_recall@50"])
    logging.info("delta valid R@50 (ta-td): %+.6f", delta_valid)
    logging.info("delta test  R@50 (ta-td): %+.6f", delta_test)
    logging.info("delta test  R@50 (ta-final): %+.6f", delta_vs_final_test)
    logging.info("test bucket delta (ta-td): le5=%+.6f  6to20=%+.6f  gt20=%+.6f",
                 test_bk_delta.get("le5", 0), test_bk_delta.get("6to20", 0), test_bk_delta.get("gt20", 0))
    logging.info("unique hit (test): both=%d  only_td=%d  only_ta=%d", len(both), len(only_td), len(only_ta))
    logging.info("recommend_user_tower_replacement: %s", recommend_replace)
    logging.info("===================================")

    # ── append to investigation report ──────────────────────────────────────

    def yes_no(b: bool) -> str:
        return "✅ 是" if b else "❌ 否"

    report_section = f"""
---

## 9. Phase 2：20epoch 公平对比结果

**状态：** ✅ 完成

### Overall 对比（full valid / full test）

| 模型 | best_ep | full valid R@50 | full test R@50 | full test NDCG@50 | full test MRR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time_decay max100 20ep | {td_summary.get('checkpoint_epoch')} | {fmt(td_valid_r50)} | {fmt(td_test_r50)} | {fmt(td_summary.get('test_ndcg@50', 0))} | {fmt(td_summary.get('test_mrr@50', 0))} |
| timeaware Transformer max100 20ep | {ta_summary.get('checkpoint_epoch')} | {fmt(ta_valid_r50)} | {fmt(ta_test_r50)} | {fmt(ta_summary.get('test_ndcg@50', 0))} | {fmt(ta_summary.get('test_mrr@50', 0))} |
| **Δ (ta − td)** | — | **{fmt(delta_valid):>10}** | **{fmt(delta_test):>10}** | — | — |
| 当前 final model（参考） | 17 | 0.122626 | 0.078315 | — | — |

### Test Bucket Recall@50 对比

| 桶 | td 20ep | timeaware 20ep | Δ |
| --- | ---: | ---: | ---: |
| ≤5 | {fmt(td_bucket_test.get('le5', 0))} | {fmt(ta_bucket_test.get('le5', 0))} | {fmt(test_bk_delta.get('le5', 0))} |
| 6-20 | {fmt(td_bucket_test.get('6to20', 0))} | {fmt(ta_bucket_test.get('6to20', 0))} | {fmt(test_bk_delta.get('6to20', 0))} |
| >20 | {fmt(td_bucket_test.get('gt20', 0))} | {fmt(ta_bucket_test.get('gt20', 0))} | {fmt(test_bk_delta.get('gt20', 0))} |

### Unique Hit（test set, R@50）

| 指标 | 数量 |
| --- | ---: |
| 两者均命中 | {len(both)} |
| 仅 td 命中 | {len(only_td)} |
| 仅 timeaware 命中 | {len(only_ta)} |

### 是否建议替换 user tower

| 条件 | 结论 |
| --- | --- |
| ta full valid R@50 > td full valid R@50 | {yes_no(condition1)} |
| ta full test R@50 > td full test R@50 | {yes_no(condition2)} |
| ta full test R@50 > final model 0.078315 | {yes_no(condition3)} |
| 提升幅度 ≥ +{THRESHOLD_VS_FINAL_TEST:.4f}（vs final model） | {yes_no(condition4)} |
| 至少一个 bucket 正向提升 | {yes_no(condition5)} |
| **综合建议：替换 user tower** | {yes_no(recommend_replace)} |

"""
    append_text(REPORT_PATH, report_section)
    logging.info("Appended results to %s", REPORT_PATH)

    # ── append to daily log ─────────────────────────────────────────────────

    daily_section = f"""
---

## Part 15：Transformer Investigation Phase 2 完成

**状态：** ✅ 20ep 公平对比完成

| 模型 | full valid R@50 | full test R@50 |
| --- | ---: | ---: |
| time_decay max100 20ep | {fmt(td_valid_r50)} | {fmt(td_test_r50)} |
| timeaware Transformer max100 20ep | {fmt(ta_valid_r50)} | {fmt(ta_test_r50)} |
| Δ (ta − td) | {fmt(delta_valid)} | {fmt(delta_test)} |
| 当前 final model（参考） | 0.122626 | 0.078315 |

**建议替换 user tower：** {yes_no(recommend_replace)}

详细结论见：`docs/reports/transformer_user_tower_investigation.md`

"""
    append_text(DAILY_LOG_PATH, daily_section)
    logging.info("Appended results to %s", DAILY_LOG_PATH)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Time-decay Text + Mean Pooling test metrics by user train-history length buckets.

Diagnostic-only: loads the Time-decay Text+MP tau=0.15 best checkpoint and groups
full-test non-cold targets by each user's train-history length.  Compares results
against the previously measured simple mean pooling bucket metrics to validate
whether time-decay pooling improves the long-history (>20) user bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_text_time_decay_mean_pool_two_tower_smoke as time_decay_model  # noqa: E402


BUCKETS = [
    ("0", lambda length: length == 0, "no train history; not true new-user cold start in this processed split"),
    ("1-2", lambda length: 1 <= length <= 2, "very short train history"),
    ("3-5", lambda length: 3 <= length <= 5, "short train history"),
    ("6-20", lambda length: 6 <= length <= 20, "medium train history"),
    (">20", lambda length: length > 20, "long train history"),
]

# Simple mean pooling bucket Recall@50 reference values (from prior diagnostic)
SIMPLE_MEAN_POOL_BUCKET_R50 = {
    "0": None,
    "1-2": None,
    "3-5": 0.086826,
    "6-20": 0.067401,
    ">20": 0.042312,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-decay user history length bucket diagnostic.")
    parser.add_argument(
        "--config",
        default="configs/two_tower_movies_tv_5core_text_time_decay_mean_pool_20epoch.yaml",
        help="Time-decay config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt",
        help="Time-decay best checkpoint path.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/user_history_bucket_eval_time_decay",
        help="Output directory.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bucket_for_length(length: int) -> str:
    for name, predicate, _ in BUCKETS:
        if predicate(length):
            return name
    raise ValueError(f"Unexpected history length: {length}")


def prepare_eval_targets(bundle: time_decay_model.DataBundle) -> pd.DataFrame:
    train_lengths = bundle.train_df.groupby("user_idx").size().to_dict()
    eval_targets = bundle.test_df[~bundle.test_df["is_cold_item_for_eval"].astype(bool)].copy()
    eval_targets["train_history_len"] = eval_targets["user_idx"].map(
        lambda user: int(train_lengths.get(int(user), 0))
    )
    eval_targets["bucket"] = eval_targets["train_history_len"].map(bucket_for_length)
    return eval_targets


def encode_all_items_cpu(model: Any, num_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + 65536, num_items)
            item_idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(item_idx).detach().cpu())
    return torch.cat(chunks, dim=0)


def empty_bucket_metric(description: str, note: str) -> dict[str, Any]:
    return {
        "description": description,
        "num_test_users": 0,
        "num_targets": 0,
        "recall@20": None,
        "recall@50": None,
        "recall@100": None,
        "ndcg@50": None,
        "mrr@50": None,
        "note": note,
    }


def evaluate_by_user_history_bucket(
    model: Any,
    config: dict[str, Any],
    stats: dict[str, Any],
    eval_targets: pd.DataFrame,
    history_matrix: np.ndarray,
    seen_items: dict[int, set[int]],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    num_items = int(stats["n_items"])
    eval_batch_size = int(config["eval_batch_size"])
    item_emb_cpu = encode_all_items_cpu(model, num_items, device)
    max_k = 100

    sums: dict[str, dict[str, float]] = {}
    user_sets: dict[str, set[int]] = {}
    for bucket_name, _, _ in BUCKETS:
        sums[bucket_name] = {
            "count": 0.0,
            "recall@20": 0.0,
            "recall@50": 0.0,
            "recall@100": 0.0,
            "ndcg@50": 0.0,
            "mrr@50": 0.0,
        }
        user_sets[bucket_name] = set()

    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_targets), eval_batch_size):
            batch = eval_targets.iloc[start : start + eval_batch_size]
            users_np = batch["user_idx"].to_numpy(dtype=np.int64, copy=True)
            targets_np = batch["item_idx"].to_numpy(dtype=np.int64, copy=True)
            user_tensor = torch.as_tensor(users_np, device=device)
            target_tensor = torch.as_tensor(targets_np, dtype=torch.long, device=device)
            history_tensor = torch.as_tensor(history_matrix[users_np], dtype=torch.long, device=device)
            user_emb = model.encode_users(user_tensor, history_tensor)
            item_emb = item_emb_cpu.to(device)
            scores = (user_emb @ item_emb.T) / float(config["temperature"])

            row_indices = torch.arange(scores.shape[0], device=device)
            target_scores = scores[row_indices, target_tensor].clone()
            for row_pos, (user_idx, target_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)
            ):
                seen = seen_items.get(int(user_idx), set())
                if seen:
                    scores[row_pos, torch.as_tensor(list(seen), dtype=torch.long, device=device)] = -torch.inf
                scores[row_pos, int(target_item)] = target_scores[row_pos]

            topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
            for user_idx, target_item, bucket, rec_items in zip(
                users_np, targets_np, batch["bucket"].tolist(), topk, strict=True
            ):
                bucket_sums = sums[bucket]
                bucket_sums["count"] += 1.0
                user_sets[bucket].add(int(user_idx))
                matched = np.where(rec_items == int(target_item))[0]
                if matched.size == 0:
                    continue
                rank = int(matched[0]) + 1
                if rank <= 20:
                    bucket_sums["recall@20"] += 1.0
                if rank <= 50:
                    bucket_sums["recall@50"] += 1.0
                    bucket_sums["ndcg@50"] += 1.0 / math.log2(rank + 1)
                    bucket_sums["mrr@50"] += 1.0 / rank
                if rank <= 100:
                    bucket_sums["recall@100"] += 1.0

    metrics: dict[str, dict[str, Any]] = {}
    for bucket_name, _, description in BUCKETS:
        count = int(sums[bucket_name]["count"])
        if count == 0:
            metrics[bucket_name] = empty_bucket_metric(description, "n/a: no non-cold test users in this bucket")
            continue
        metrics[bucket_name] = {
            "description": description,
            "num_test_users": len(user_sets[bucket_name]),
            "num_targets": count,
            "recall@20": sums[bucket_name]["recall@20"] / count,
            "recall@50": sums[bucket_name]["recall@50"] / count,
            "recall@100": sums[bucket_name]["recall@100"] / count,
            "ndcg@50": sums[bucket_name]["ndcg@50"] / count,
            "mrr@50": sums[bucket_name]["mrr@50"] / count,
            "note": "evaluated",
        }
    return metrics


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def fmt_rel(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def build_comparison_rows(metrics: dict[str, dict[str, Any]]) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for bucket_name in ("3-5", "6-20", ">20"):
        simple_r50 = SIMPLE_MEAN_POOL_BUCKET_R50[bucket_name]
        td_r50 = metrics[bucket_name].get("recall@50")
        if simple_r50 is None or td_r50 is None:
            rows.append((bucket_name, fmt_metric(simple_r50), fmt_metric(td_r50), "n/a", "n/a"))
        else:
            delta = td_r50 - simple_r50
            rel = delta / simple_r50 * 100.0
            rows.append((bucket_name, fmt_metric(simple_r50), fmt_metric(td_r50), fmt_delta(delta), fmt_rel(rel)))
    return rows


def build_diagnostic_analysis(metrics: dict[str, dict[str, Any]]) -> str:
    bucket_gt20 = metrics[">20"]
    bucket_620 = metrics["6-20"]
    bucket_35 = metrics["3-5"]

    td_gt20 = bucket_gt20.get("recall@50")
    td_620 = bucket_620.get("recall@50")
    td_35 = bucket_35.get("recall@50")

    simple_gt20 = SIMPLE_MEAN_POOL_BUCKET_R50[">20"]
    simple_620 = SIMPLE_MEAN_POOL_BUCKET_R50["6-20"]
    simple_35 = SIMPLE_MEAN_POOL_BUCKET_R50["3-5"]

    lines = []

    # >20 bucket
    if td_gt20 is not None and simple_gt20 is not None:
        delta_gt20 = td_gt20 - simple_gt20
        rel_gt20 = delta_gt20 / simple_gt20 * 100.0
        if delta_gt20 > 0:
            lines.append(
                f"**>20 bucket IMPROVED**: Time-decay Recall@50={td_gt20:.6f} vs simple={simple_gt20:.6f} "
                f"(absolute {delta_gt20:+.6f}, relative {rel_gt20:+.2f}%). "
                "This supports the design motivation: simple mean pooling dilutes long multi-interest histories, "
                "while time-decay down-weighting older items helps focus on recent preferences."
            )
        else:
            lines.append(
                f"**>20 bucket did NOT improve**: Time-decay Recall@50={td_gt20:.6f} vs simple={simple_gt20:.6f} "
                f"(absolute {delta_gt20:+.6f}, relative {rel_gt20:+.2f}%). "
                "The overall full-test improvement (+2.59%) is therefore not primarily driven by long-history user repair."
            )

    # 6-20 bucket
    if td_620 is not None and simple_620 is not None:
        delta_620 = td_620 - simple_620
        rel_620 = delta_620 / simple_620 * 100.0
        if delta_620 > 0:
            lines.append(
                f"**6-20 bucket IMPROVED**: Time-decay Recall@50={td_620:.6f} vs simple={simple_620:.6f} "
                f"(absolute {delta_620:+.6f}, relative {rel_620:+.2f}%)."
            )
        else:
            lines.append(
                f"**6-20 bucket did NOT improve**: Time-decay Recall@50={td_620:.6f} vs simple={simple_620:.6f} "
                f"(absolute {delta_620:+.6f}, relative {rel_620:+.2f}%)."
            )

    # 3-5 bucket
    if td_35 is not None and simple_35 is not None:
        delta_35 = td_35 - simple_35
        rel_35 = delta_35 / simple_35 * 100.0
        if delta_35 > 0:
            lines.append(
                f"**3-5 bucket IMPROVED**: Time-decay Recall@50={td_35:.6f} vs simple={simple_35:.6f} "
                f"(absolute {delta_35:+.6f}, relative {rel_35:+.2f}%)."
            )
        elif delta_35 < 0:
            lines.append(
                f"**3-5 bucket DECLINED**: Time-decay Recall@50={td_35:.6f} vs simple={simple_35:.6f} "
                f"(absolute {delta_35:+.6f}, relative {rel_35:+.2f}%). "
                "Time-decay may over-emphasise recent items for short-history users who have limited history coverage."
            )
        else:
            lines.append(f"**3-5 bucket UNCHANGED**: Recall@50={td_35:.6f}.")

    lines.append(
        "Note: this is NOT a true new-user cold-start evaluation. "
        "In this processed 5-core split, the 0 and 1-2 train-history buckets contain no non-cold test users. "
        "True zero-history cold-start would require separate onboarding or content-based fallback strategies."
    )

    return " ".join(lines)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["bucket_metrics"]
    comparison_rows = build_comparison_rows(metrics)

    lines = [
        "# Time-decay User History Length Bucket Diagnostic",
        "",
        "## Task",
        "",
        "Evaluate Time-decay Text + Mean Pooling tau=0.15 on full test non-cold users grouped by user train history length.",
        "Compare against previously measured simple mean pooling bucket metrics to validate the time-decay design motivation.",
        "This is an offline diagnostic only — no retraining, no Faiss, no hard negatives.",
        "",
        "## Bucket Definition",
        "",
        "| user train history length | description |",
        "| --- | --- |",
    ]
    for bucket_name, _, description in BUCKETS:
        lines.append(f"| {bucket_name} | {description} |")

    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- config: `{payload['config']}`",
            f"- checkpoint: `{payload['checkpoint']}`",
            f"- checkpoint epoch: {payload.get('checkpoint_epoch', 'unknown')}",
            "- bucket source: train split user history length.",
            "- eval split: full test non-cold users.",
            "- user tower history for test: train + valid history (matches existing full-test protocol).",
            "- seen-item filter for test: train + valid items masked.",
            "- No retraining, no Faiss, no hard negative mining, no decay_rate sweep.",
            "",
            "## Table 1: Time-decay Results",
            "",
            "| user train history length | num_test_users | Recall@20 | Recall@50 | Recall@100 | NDCG@50 | MRR@50 | note |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for bucket_name, _, _ in BUCKETS:
        row = metrics[bucket_name]
        lines.append(
            f"| {bucket_name} | {row['num_test_users']} | {fmt_metric(row['recall@20'])} | "
            f"{fmt_metric(row['recall@50'])} | {fmt_metric(row['recall@100'])} | "
            f"{fmt_metric(row['ndcg@50'])} | {fmt_metric(row['mrr@50'])} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Table 2: Simple vs Time-decay Recall@50 Comparison",
            "",
            "| user train history length | Simple Text+MP τ=0.15 R@50 | Time-decay Text+MP τ=0.15 R@50 | absolute change | relative change |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket_name, simple_r50, td_r50, delta, rel in comparison_rows:
        lines.append(f"| {bucket_name} | {simple_r50} | {td_r50} | {delta} | {rel} |")

    lines.extend(
        [
            "",
            "## Diagnostic Analysis",
            "",
            payload["diagnostic_analysis"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = time_decay_model.load_config(config_path)
    config["config_path"] = str(config_path)
    time_decay_model.require_config(config)
    config["eval_max_users"] = None
    time_decay_model.set_seed(int(config["seed"]))

    device = time_decay_model.resolve_device(str(config["device"]))
    bundle = time_decay_model.load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    history_max_len = int(config["history_max_len"])
    test_history_frame = pd.concat(
        [bundle.train_df, bundle.valid_df[time_decay_model.TRAIN_COLUMNS]], ignore_index=True
    )
    test_history_matrix, _ = time_decay_model.build_history_matrix(test_history_frame, num_users, history_max_len)
    train_seen = time_decay_model.build_seen_items(bundle.train_df)
    test_seen = time_decay_model.merge_seen_items(train_seen, bundle.valid_df)

    model = time_decay_model.build_model(config, bundle.stats, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    checkpoint_epoch = ckpt.get("epoch", "unknown")
    logging.info(
        "checkpoint loaded: epoch=%s %s=%.6f",
        checkpoint_epoch,
        ckpt.get("best_metric_name", "best_metric"),
        float(ckpt.get("best_metric_value", 0.0)),
    )

    eval_targets = prepare_eval_targets(bundle)
    logging.info("full test non-cold targets=%s", len(eval_targets))

    bucket_metrics = evaluate_by_user_history_bucket(
        model=model,
        config=config,
        stats=bundle.stats,
        eval_targets=eval_targets,
        history_matrix=test_history_matrix,
        seen_items=test_seen,
        device=device,
    )

    diagnostic_analysis = build_diagnostic_analysis(bucket_metrics)

    payload = {
        "task": "user_history_length_bucket_diagnostic_time_decay",
        "model": "Time-decay Text + Mean Pooling Two-Tower tau=0.15 decay_rate=0.8",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "split": "test",
        "eval_targets": "full test non-cold users",
        "metric_set": ["recall@20", "recall@50", "recall@100", "ndcg@50", "mrr@50"],
        "protocol": {
            "bucket_source": "train split user history length",
            "test_history": "train + valid history, matching existing full-test protocol",
            "seen_item_filter": "train + valid items masked for test retrieval",
            "offline_only": True,
            "no_retraining": True,
        },
        "simple_mean_pool_reference_r50": SIMPLE_MEAN_POOL_BUCKET_R50,
        "bucket_metrics": bucket_metrics,
        "diagnostic_analysis": diagnostic_analysis,
    }
    write_json(output_dir / "results.json", payload)
    write_report(output_dir / "results.md", payload)
    logging.info("wrote %s and %s", output_dir / "results.json", output_dir / "results.md")


if __name__ == "__main__":
    main()

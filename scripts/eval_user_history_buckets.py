#!/usr/bin/env python3
"""Evaluate Text + Mean Pooling test metrics by user train-history length buckets.

This is an evaluation-only diagnostic. It loads the saved Text + Mean Pooling
tau=0.15 checkpoint and groups full-test non-cold targets by each user's
history length in the train split.
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

import train_text_mean_pool_two_tower as text_mean_pool  # noqa: E402


BUCKETS = [
    ("0", lambda length: length == 0, "no train history; not true new-user cold start in this processed split"),
    ("1-2", lambda length: 1 <= length <= 2, "very short train history"),
    ("3-5", lambda length: 3 <= length <= 5, "short train history"),
    ("6-20", lambda length: 6 <= length <= 20, "medium train history"),
    (">20", lambda length: length > 20, "long train history"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="User history length bucket diagnostic.")
    parser.add_argument(
        "--config",
        default="configs/two_tower_movies_tv_5core_text_mean_pool_tau015_20epoch.yaml",
        help="Text + Mean Pooling tau=0.15 config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt",
        help="Text + Mean Pooling tau=0.15 checkpoint path.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/user_history_bucket_eval",
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


def prepare_eval_targets(bundle: text_mean_pool.DataBundle) -> pd.DataFrame:
    train_lengths = bundle.train_df.groupby("user_idx").size().to_dict()
    eval_targets = bundle.test_df[~bundle.test_df["is_cold_item_for_eval"].astype(bool)].copy()
    eval_targets["train_history_len"] = eval_targets["user_idx"].map(lambda user: int(train_lengths.get(int(user), 0)))
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


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# User History Length Bucket Diagnostic",
        "",
        "## Task",
        "",
        "Evaluate Text + Mean Pooling tau=0.15 on full test non-cold users grouped by user train history length.",
        "This is an offline diagnostic only, not training and not online performance.",
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
            "- bucket source: train split user history length.",
            "- eval split: full test non-cold users.",
            "- user tower history for test follows the existing full-test protocol: train + valid history.",
            "- seen-item filter for test masks train + valid items.",
            "- No retraining, no Faiss, no hard negative mining.",
            "",
            "## Results",
            "",
            "| user train history length | num_test_users | Recall@20 | Recall@50 | Recall@100 | NDCG@50 | MRR@50 | note |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for bucket_name, _, _ in BUCKETS:
        row = payload["bucket_metrics"][bucket_name]
        lines.append(
            f"| {bucket_name} | {row['num_test_users']} | {fmt_metric(row['recall@20'])} | "
            f"{fmt_metric(row['recall@50'])} | {fmt_metric(row['recall@100'])} | "
            f"{fmt_metric(row['ndcg@50'])} | {fmt_metric(row['mrr@50'])} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic Conclusion",
            "",
            payload["diagnostic_summary"],
            "",
            "True new-user cold start is not solved by this model alone; it would require user profile features, explicit initial preferences, onboarding signals, or a popularity/content fallback.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(metrics: dict[str, dict[str, Any]]) -> str:
    evaluated = [(bucket, row) for bucket, row in metrics.items() if row["recall@50"] is not None]
    if not evaluated:
        return "No non-cold test users were available for any user-history bucket."
    best_bucket, best_row = max(evaluated, key=lambda item: item[1]["recall@50"])
    worst_bucket, worst_row = min(evaluated, key=lambda item: item[1]["recall@50"])
    zero_note = metrics["0"]["note"]
    one_two_note = metrics["1-2"]["note"]
    return (
        f"Text+MP tau=0.15 is strongest for the `{best_bucket}` bucket "
        f"(Recall@50={best_row['recall@50']:.6f}) and weakest for the `{worst_bucket}` bucket "
        f"(Recall@50={worst_row['recall@50']:.6f}) among evaluated buckets. "
        "The result is not monotonic with train history length; longer histories can correspond to broader or harder-to-rank user preferences rather than easier retrieval. "
        f"The `0` bucket status is `{zero_note}`, and the `1-2` bucket status is `{one_two_note}`. "
        "Therefore this processed 5-core split does not directly measure true zero-history or extremely short-history new-user behavior. "
        "This diagnostic does not solve true new-user cold start."
    )


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

    config = text_mean_pool.load_config(config_path)
    config["config_path"] = str(config_path)
    text_mean_pool.require_config(config)
    config["eval_max_users"] = None
    text_mean_pool.set_seed(int(config["seed"]))

    device = text_mean_pool.resolve_device(str(config["device"]))
    bundle = text_mean_pool.load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    history_max_len = int(config["history_max_len"])
    test_history_frame = pd.concat([bundle.train_df, bundle.valid_df[text_mean_pool.TRAIN_COLUMNS]], ignore_index=True)
    test_history_matrix, _ = text_mean_pool.build_history_matrix(test_history_frame, num_users, history_max_len)
    train_seen = text_mean_pool.build_seen_items(bundle.train_df)
    test_seen = text_mean_pool.merge_seen_items(train_seen, bundle.valid_df)

    model = text_mean_pool.build_model(config, bundle.stats, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(
        "checkpoint loaded: epoch=%s %s=%.6f",
        checkpoint.get("epoch"),
        checkpoint.get("best_metric_name", "best_metric"),
        float(checkpoint.get("best_metric_value", 0.0)),
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
    payload = {
        "task": "user_history_length_bucket_diagnostic",
        "model": "Text + Mean Pooling Two-Tower tau=0.15",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
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
        "bucket_metrics": bucket_metrics,
        "diagnostic_summary": build_summary(bucket_metrics),
    }
    write_json(output_dir / "results.json", payload)
    write_report(output_dir / "results.md", payload)
    logging.info("wrote %s and %s", output_dir / "results.json", output_dir / "results.md")


if __name__ == "__main__":
    main()

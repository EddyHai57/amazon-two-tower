#!/usr/bin/env python3
"""Evaluate test Recall@50 by target-item train-count buckets.

This script is evaluation-only. It reuses the existing Mean Pooling and
Text + Mean Pooling model definitions, loads saved checkpoints, and reports
bucketed test Recall@50 without retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_mean_pool_two_tower as mean_pool  # noqa: E402
import train_text_mean_pool_two_tower as text_mean_pool  # noqa: E402


BUCKETS = [
    ("<=5", lambda count: count <= 5, "cold-start-like / very low-frequency item"),
    ("6-20", lambda count: 6 <= count <= 20, "long-tail item"),
    ("21-100", lambda count: 21 <= count <= 100, "mid-frequency item"),
    (">100", lambda count: count > 100, "head item"),
]

REUSED_DOC_BASELINES = {
    "<=5": {"itemcf": 0.040405, "id_only": 0.023284},
    "6-20": {"itemcf": 0.047940, "id_only": 0.043748},
    "21-100": {"itemcf": 0.060890, "id_only": 0.062918},
    ">100": {"itemcf": 0.122522, "id_only": 0.054604},
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    config_path: Path
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-start / long-tail bucket Recall@50 evaluation.")
    parser.add_argument("--output_dir", default="outputs/cold_start_eval", help="Output directory.")
    parser.add_argument(
        "--mean_pool_config",
        default="configs/two_tower_movies_tv_5core_mean_pool_20epoch.yaml",
        help="Mean Pooling config path.",
    )
    parser.add_argument(
        "--mean_pool_checkpoint",
        default="outputs/user_mean_pool_20ep/checkpoints/best_model.pt",
        help="Mean Pooling checkpoint path.",
    )
    parser.add_argument(
        "--text_mean_pool_config",
        default="configs/two_tower_movies_tv_5core_text_mean_pool_tau015_20epoch.yaml",
        help="Text + Mean Pooling tau=0.15 config path.",
    )
    parser.add_argument(
        "--text_mean_pool_checkpoint",
        default="outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt",
        help="Text + Mean Pooling tau=0.15 checkpoint path.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bucket_for_count(count: int) -> str:
    for name, predicate, _ in BUCKETS:
        if predicate(count):
            return name
    raise ValueError(f"Unexpected item count: {count}")


def build_eval_targets(bundle: Any) -> pd.DataFrame:
    item_counts = bundle.train_df["item_idx"].value_counts().to_dict()
    eval_targets = bundle.test_df[~bundle.test_df["is_cold_item_for_eval"].astype(bool)].copy()
    eval_targets["train_item_count"] = eval_targets["item_idx"].map(lambda item: int(item_counts.get(int(item), 0)))
    eval_targets["bucket"] = eval_targets["train_item_count"].map(bucket_for_count)
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


def load_model(spec: ModelSpec, stats: dict[str, Any]) -> tuple[Any, dict[str, Any], torch.device]:
    if spec.kind == "mean_pool":
        config = mean_pool.load_config(spec.config_path)
        config["config_path"] = str(spec.config_path)
        mean_pool.require_config(config)
        device = mean_pool.resolve_device(str(config["device"]))
        model = mean_pool.MeanPoolTwoTower(
            num_users=int(stats["n_users"]),
            num_items=int(stats["n_items"]),
            embedding_dim=int(config["embedding_dim"]),
            use_l2_norm=bool(config["use_l2_norm"]),
            history_weight=float(config["history_weight"]),
        ).to(device)
    elif spec.kind == "text_mean_pool":
        config = text_mean_pool.load_config(spec.config_path)
        config["config_path"] = str(spec.config_path)
        text_mean_pool.require_config(config)
        device = text_mean_pool.resolve_device(str(config["device"]))
        model = text_mean_pool.build_model(config, stats, device)
    else:
        raise ValueError(f"Unsupported model kind: {spec.kind}")

    checkpoint = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(
        "%s checkpoint loaded: epoch=%s %s=%.6f",
        spec.name,
        checkpoint.get("epoch"),
        checkpoint.get("best_metric_name", "best_metric"),
        float(checkpoint.get("best_metric_value", 0.0)),
    )
    return model, config, device


def evaluate_model_by_bucket(
    spec: ModelSpec,
    bundle: Any,
    eval_targets: pd.DataFrame,
    history_matrix: np.ndarray,
    seen_items: dict[int, set[int]],
) -> dict[str, Any]:
    model, config, device = load_model(spec, bundle.stats)
    num_items = int(bundle.stats["n_items"])
    eval_batch_size = int(config["eval_batch_size"])
    item_emb_cpu = encode_all_items_cpu(model, num_items, device)

    hits = {name: 0 for name, _, _ in BUCKETS}
    counts = {name: 0 for name, _, _ in BUCKETS}
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

            top50 = torch.topk(scores, k=50, dim=1).indices.cpu().numpy()
            for target_item, bucket, rec_items in zip(targets_np, batch["bucket"].tolist(), top50, strict=True):
                counts[bucket] += 1
                if int(target_item) in rec_items:
                    hits[bucket] += 1

    bucket_metrics = {}
    for bucket_name, _, description in BUCKETS:
        count = counts[bucket_name]
        bucket_metrics[bucket_name] = {
            "description": description,
            "hits@50": hits[bucket_name],
            "num_targets": count,
            "recall@50": hits[bucket_name] / count if count else 0.0,
        }
    return bucket_metrics


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["results_table"]
    counts = payload["bucket_counts"]
    lines = [
        "# Cold-Start / Long-Tail Item Bucket Evaluation",
        "",
        "## Task",
        "",
        "Evaluate full test Recall@50 by target-item train interaction count bucket.",
        "This is offline test bucket evaluation only, not online performance.",
        "",
        "## Bucket Definition",
        "",
        "| bucket by train item count | description | test target count | target ratio |",
        "| --- | --- | ---: | ---: |",
    ]
    for bucket_name, _, description in BUCKETS:
        entry = counts[bucket_name]
        lines.append(
            f"| {bucket_name} | {description} | {entry['num_targets']} | {entry['target_ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "The `<=5` bucket is low-frequency / cold-start-like, not completely unseen item cold start.",
            "",
            "## Recall@50 Results",
            "",
            "| bucket by train item count | ItemCF R@50 | ID-only R@50 | Mean Pool R@50 | Text+MP tau=0.15 R@50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['bucket']} | {row['itemcf_recall@50']:.6f} | {row['id_only_recall@50']:.6f} | "
            f"{row['mean_pool_recall@50']:.6f} | {row['text_mean_pool_tau015_recall@50']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- ItemCF and ID-only bucket metrics are reused from the existing D2 popularity bucket matrix recorded in docs.",
            "- Mean Pooling and Text+MP tau=0.15 bucket metrics were computed in this run from saved checkpoints.",
            "- User tower history for this bucket evaluation uses train split only; seen-item filtering for test masks train+valid items, matching full-test candidate filtering.",
            "",
            "## Diagnostic Analysis",
            "",
            payload["diagnostic_summary"],
            "",
            "## Safety Notes",
            "",
            "- Offline test bucket evaluation only.",
            "- Not online performance.",
            "- No retraining, no Faiss, no hard negative mining.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(
    eval_targets: pd.DataFrame,
    mean_pool_metrics: dict[str, Any],
    text_mean_pool_metrics: dict[str, Any],
) -> dict[str, Any]:
    total = len(eval_targets)
    bucket_counts = {}
    results_table = []
    for bucket_name, _, description in BUCKETS:
        target_count = int((eval_targets["bucket"] == bucket_name).sum())
        bucket_counts[bucket_name] = {
            "description": description,
            "num_targets": target_count,
            "target_ratio": target_count / total if total else 0.0,
        }
        results_table.append(
            {
                "bucket": bucket_name,
                "num_targets": target_count,
                "itemcf_recall@50": REUSED_DOC_BASELINES[bucket_name]["itemcf"],
                "id_only_recall@50": REUSED_DOC_BASELINES[bucket_name]["id_only"],
                "mean_pool_recall@50": mean_pool_metrics[bucket_name]["recall@50"],
                "text_mean_pool_tau015_recall@50": text_mean_pool_metrics[bucket_name]["recall@50"],
            }
        )

    text_best_bucket = max(results_table, key=lambda row: row["text_mean_pool_tau015_recall@50"])["bucket"]
    text_beats_mean = all(row["text_mean_pool_tau015_recall@50"] > row["mean_pool_recall@50"] for row in results_table)
    text_beats_id = all(row["text_mean_pool_tau015_recall@50"] > row["id_only_recall@50"] for row in results_table)
    text_beats_itemcf = all(row["text_mean_pool_tau015_recall@50"] > row["itemcf_recall@50"] for row in results_table)
    cold_delta = results_table[0]["text_mean_pool_tau015_recall@50"] - results_table[0]["mean_pool_recall@50"]
    long_tail_delta = results_table[1]["text_mean_pool_tau015_recall@50"] - results_table[1]["mean_pool_recall@50"]
    head_gap_itemcf = results_table[-1]["text_mean_pool_tau015_recall@50"] - results_table[-1]["itemcf_recall@50"]

    diagnostic_summary = (
        f"Text+MP tau=0.15 is strongest in the {text_best_bucket} bucket. "
        f"It {'exceeds' if text_beats_mean else 'does not exceed'} Mean Pooling in every bucket and "
        f"{'exceeds' if text_beats_id else 'does not exceed'} ID-only in every bucket. "
        f"It {'exceeds' if text_beats_itemcf else 'does not exceed'} ItemCF in every bucket; the head-item "
        f"gap vs ItemCF is {head_gap_itemcf:+.6f}. "
        f"For low-frequency buckets, Text+MP tau=0.15 changes Recall@50 vs Mean Pooling by "
        f"{cold_delta:+.6f} in <=5 and {long_tail_delta:+.6f} in 6-20. "
        "This supports an offline view of better two-tower generalization on several non-head buckets, "
        "while the test distribution remains strongly affected by head items."
    )

    return {
        "task": "cold_start_long_tail_item_bucket_eval",
        "split": "test",
        "metric": "recall@50",
        "num_eval_targets": int(total),
        "bucket_definition": {
            name: {"description": description, "predicate": name}
            for name, _, description in BUCKETS
        },
        "protocol": {
            "item_counts_source": "train split item interaction counts",
            "eval_targets": "full test non-cold users",
            "user_history": "train split only",
            "seen_item_filter": "train + valid items masked for test retrieval",
            "offline_only": True,
        },
        "sources": {
            "itemcf": "docs/daily_logs/2026-05-11.md D2 popularity bucket matrix",
            "id_only": "docs/daily_logs/2026-05-11.md D2 popularity bucket matrix",
            "mean_pool": "computed from outputs/user_mean_pool_20ep/checkpoints/best_model.pt",
            "text_mean_pool_tau015": "computed from outputs/text_mean_pool_tau015_20ep/checkpoints/best_model.pt",
        },
        "bucket_counts": bucket_counts,
        "results_table": results_table,
        "model_bucket_metrics": {
            "mean_pool": mean_pool_metrics,
            "text_mean_pool_tau015": text_mean_pool_metrics,
        },
        "diagnostic_summary": diagnostic_summary,
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)

    mean_spec = ModelSpec(
        name="Mean Pooling Two-Tower",
        kind="mean_pool",
        config_path=Path(args.mean_pool_config),
        checkpoint_path=Path(args.mean_pool_checkpoint),
    )
    text_mean_spec = ModelSpec(
        name="Text + Mean Pooling tau=0.15",
        kind="text_mean_pool",
        config_path=Path(args.text_mean_pool_config),
        checkpoint_path=Path(args.text_mean_pool_checkpoint),
    )
    for spec in [mean_spec, text_mean_spec]:
        if not spec.config_path.exists():
            raise FileNotFoundError(f"Missing config: {spec.config_path}")
        if not spec.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {spec.checkpoint_path}")

    config = mean_pool.load_config(mean_spec.config_path)
    bundle = mean_pool.load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    history_max_len = int(config["history_max_len"])
    train_history_matrix, _ = mean_pool.build_history_matrix(bundle.train_df, num_users, history_max_len)
    train_seen = mean_pool.build_seen_items(bundle.train_df)
    test_seen = mean_pool.merge_seen_items(train_seen, bundle.valid_df)
    eval_targets = build_eval_targets(bundle)
    logging.info("test non-cold targets=%s", len(eval_targets))

    mean_metrics = evaluate_model_by_bucket(mean_spec, bundle, eval_targets, train_history_matrix, test_seen)
    text_mean_metrics = evaluate_model_by_bucket(text_mean_spec, bundle, eval_targets, train_history_matrix, test_seen)
    payload = build_payload(eval_targets, mean_metrics, text_mean_metrics)

    write_json(output_dir / "results.json", payload)
    write_report(output_dir / "results.md", payload)
    logging.info("wrote %s and %s", output_dir / "results.json", output_dir / "results.md")


if __name__ == "__main__":
    main()

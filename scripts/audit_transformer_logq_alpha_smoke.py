#!/usr/bin/env python3
"""Limited-valid exact Top50 exposure audit for one LogQ alpha smoke checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import train_transformer_stability_sweep as stability
from audit_transformer_logq_effect import (
    aggregate_exposure_metrics,
    aggregate_popularity_bucket_recall,
    compute_ranking_metrics,
    encode_exact_top50,
    load_model,
)


def select_limited_valid_eval(
    valid_df: pd.DataFrame,
    max_users: int,
) -> pd.DataFrame:
    non_cold = valid_df[~valid_df["is_cold_item_for_eval"].astype(bool)].copy()
    return non_cold.head(max_users).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bundle = stability.load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_matrix = stability.build_history_matrix(
        bundle.train_df,
        num_users,
        int(config["history_max_len"]),
    )
    seen_items = stability.build_seen_items(bundle.train_df)
    eval_df = select_limited_valid_eval(bundle.valid_df, int(config["eval_max_users"]))
    targets = eval_df["item_idx"].to_numpy(dtype=np.int64)
    item_popularity = np.bincount(
        bundle.train_df["item_idx"].to_numpy(dtype=np.int64),
        minlength=num_items,
    )
    device = stability.resolve_device(str(config["device"]))
    model = load_model(
        Path(args.config),
        Path(args.checkpoint),
        bundle.stats,
        device,
    )
    topk = encode_exact_top50(
        model,
        eval_df,
        history_matrix,
        seen_items,
        num_items,
        device,
        int(config["eval_batch_size"]),
        float(config["temperature"]),
    )
    ranking_metrics = compute_ranking_metrics(topk, targets)
    hit_mask = ranking_metrics.pop("hit_mask")
    result: dict[str, Any] = {
        "eval_protocol": {
            "split": "limited_valid",
            "history_frame": "train",
            "seen_mask": "build_seen_items(train)",
            "exclude_cold_target": True,
            "num_eval_users": int(len(eval_df)),
            "topk": 50,
            "retrieval": "exact inner product",
        },
        "variant": {
            "variant_name": config["variant_name"],
            "logq_alpha": float(config.get("logq_alpha", 1.0)),
            "q_mode": str(config.get("q_mode", "empirical")),
        },
        "ranking_metrics": ranking_metrics,
        "target_item_popularity_bucket_recall": aggregate_popularity_bucket_recall(
            targets,
            hit_mask,
            item_popularity,
        ),
        "exposure_metrics": aggregate_exposure_metrics(topk, item_popularity),
    }
    np.save(output_dir / "top50.npy", topk)
    stability.write_json(output_dir / "audit_summary.json", result)


if __name__ == "__main__":
    main()

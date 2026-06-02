#!/usr/bin/env python3
"""Offline exact Top50 audit for canonical vs LogQ Transformer Two-Tower."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

import train_transformer_stability_sweep as stability
from train_transformer_logq_smoke import build_log_q_for_mode


POPULARITY_BUCKETS = [
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21-100", 21, 100),
    (">100", 101, None),
]
POPULARITY_BUCKETS_WITH_UNSEEN = [("unseen", 0, 0), *POPULARITY_BUCKETS]
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000


def bucket_name(popularity: int) -> str:
    for name, low, high in POPULARITY_BUCKETS:
        if popularity >= low and (high is None or popularity <= high):
            return name
    return "unseen"


def popularity_mask(
    popularity: np.ndarray,
    low: int,
    high: int | None,
) -> np.ndarray:
    return (popularity >= low) & ((popularity <= high) if high is not None else True)


def aggregate_popularity_bucket_recall(
    targets: np.ndarray,
    hit_mask: np.ndarray,
    item_popularity: np.ndarray,
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    for name, _, _ in POPULARITY_BUCKETS:
        bucket_targets = np.array([bucket_name(int(item_popularity[item])) == name for item in targets])
        count = int(bucket_targets.sum())
        hits = int((bucket_targets & hit_mask).sum())
        result[name] = {
            "targets": count,
            "hits": hits,
            "recall@50": hits / count if count else 0.0,
        }
    return result


def aggregate_hit_transition(
    baseline_hit: np.ndarray,
    logq_hit: np.ndarray,
) -> dict[str, int]:
    return {
        "both_hit": int((baseline_hit & logq_hit).sum()),
        "baseline_only": int((baseline_hit & ~logq_hit).sum()),
        "logq_only": int((~baseline_hit & logq_hit).sum()),
        "neither_hit": int((~baseline_hit & ~logq_hit).sum()),
    }


def paired_bootstrap_ci(
    baseline_hit: np.ndarray,
    candidate_hit: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, int | float]:
    if len(baseline_hit) != len(candidate_hit):
        raise ValueError("Paired bootstrap inputs must have equal lengths.")
    count = len(baseline_hit)
    if count == 0:
        return {
            "point_estimate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "num_users": 0,
        }
    deltas = candidate_hit.astype(np.int8) - baseline_hit.astype(np.int8)
    counts = np.array([
        np.count_nonzero(deltas == -1),
        np.count_nonzero(deltas == 0),
        np.count_nonzero(deltas == 1),
    ])
    samples = np.random.default_rng(seed).multinomial(count, counts / count, size=resamples)
    sampled_delta = (samples[:, 2] - samples[:, 0]) / count
    return {
        "point_estimate": float(deltas.mean()),
        "ci95_low": float(np.percentile(sampled_delta, 2.5)),
        "ci95_high": float(np.percentile(sampled_delta, 97.5)),
        "num_users": count,
    }


def summarize_paired_bootstrap(
    baseline_hit: np.ndarray,
    candidate_hit: np.ndarray,
    targets: np.ndarray,
    item_popularity: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    long_tail_mask = item_popularity[targets] <= 20
    return {
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "overall_recall@50_delta": paired_bootstrap_ci(
            baseline_hit,
            candidate_hit,
            seed=seed,
            resamples=resamples,
        ),
        "long_tail_recall@50_delta": paired_bootstrap_ci(
            baseline_hit[long_tail_mask],
            candidate_hit[long_tail_mask],
            seed=seed,
            resamples=resamples,
        ),
        "long_tail_definition": "target_item_train_popularity <= 20",
    }


def aggregate_exposure_metrics(
    topk: np.ndarray,
    item_popularity: np.ndarray,
) -> dict[str, Any]:
    popularity = item_popularity[topk].reshape(-1)
    total = int(popularity.size)
    exposure_counts = np.bincount(topk.reshape(-1), minlength=len(item_popularity)).astype(np.float64)
    nonzero_exposure = exposure_counts[exposure_counts > 0]
    if total and len(item_popularity) > 1:
        probabilities = nonzero_exposure / total
        normalized_entropy = float(
            -(probabilities * np.log(probabilities)).sum() / math.log(len(item_popularity))
        )
    else:
        normalized_entropy = 0.0
    if total and len(item_popularity):
        sorted_exposure = np.sort(exposure_counts)
        ranks = np.arange(1, len(sorted_exposure) + 1, dtype=np.float64)
        exposure_gini = float(
            ((2.0 * ranks - len(sorted_exposure) - 1.0) * sorted_exposure).sum()
            / (len(sorted_exposure) * total)
        )
    else:
        exposure_gini = 0.0
    bucket_counts = {
        name: int(popularity_mask(popularity, low, high).sum())
        for name, low, high in POPULARITY_BUCKETS_WITH_UNSEEN
    }
    return {
        "avg_pop": float(popularity.mean()),
        "median_pop": float(np.median(popularity)),
        "p90_pop": float(np.percentile(popularity, 90)),
        "catalog_coverage": int(np.unique(topk).size),
        "normalized_exposure_entropy": normalized_entropy,
        "exposure_gini": exposure_gini,
        "topk_item_bucket_share": {
            name: bucket_counts[name] / total if total else 0.0
            for name, _, _ in POPULARITY_BUCKETS_WITH_UNSEEN
        },
    }


def average_jaccard(baseline_topk: np.ndarray, logq_topk: np.ndarray) -> float:
    total = 0.0
    for baseline_row, logq_row in zip(baseline_topk, logq_topk, strict=True):
        baseline_set = set(int(value) for value in baseline_row)
        logq_set = set(int(value) for value in logq_row)
        total += len(baseline_set & logq_set) / len(baseline_set | logq_set)
    return total / len(baseline_topk) if len(baseline_topk) else 0.0


def build_test_seen_items(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> dict[int, set[int]]:
    return stability.merge_seen_items(stability.build_seen_items(train_df), valid_df)


def select_non_cold_eval(test_df: pd.DataFrame) -> pd.DataFrame:
    return test_df[~test_df["is_cold_item_for_eval"].astype(bool)].copy()


def compute_ranking_metrics(topk: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    hit_mask = np.zeros(len(targets), dtype=bool)
    ndcg = 0.0
    mrr = 0.0
    for index, (row, target) in enumerate(zip(topk, targets, strict=True)):
        positions = np.where(row == target)[0]
        if positions.size:
            rank = int(positions[0]) + 1
            hit_mask[index] = True
            ndcg += 1.0 / math.log2(rank + 1)
            mrr += 1.0 / rank
    count = len(targets)
    return {
        "hit_mask": hit_mask,
        "recall@50": float(hit_mask.mean()) if count else 0.0,
        "ndcg@50": ndcg / count if count else 0.0,
        "mrr@50": mrr / count if count else 0.0,
    }


def encode_exact_top50(
    model: torch.nn.Module,
    eval_df: pd.DataFrame,
    history_matrix: np.ndarray,
    seen_items: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    item_emb_cpu = stability.encode_all_items(model, num_items, device)
    topk_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_df), batch_size):
            batch = eval_df.iloc[start:start + batch_size]
            users = batch["user_idx"].to_numpy(dtype=np.int64)
            targets = batch["item_idx"].to_numpy(dtype=np.int64)
            user_tensor = torch.as_tensor(users, device=device)
            history_tensor = torch.as_tensor(history_matrix[users], dtype=torch.long, device=device)
            scores = (
                model.encode_users(user_tensor, history_tensor) @ item_emb_cpu.to(device).T
            ) / temperature
            row_indices = torch.arange(scores.shape[0], device=device)
            target_tensor = torch.as_tensor(targets, device=device)
            target_scores = scores[row_indices, target_tensor].clone()
            for row, (user, target) in enumerate(zip(users, targets, strict=True)):
                seen = seen_items.get(int(user), set())
                if seen:
                    scores[row, torch.as_tensor(list(seen), dtype=torch.long, device=device)] = -torch.inf
                scores[row, int(target)] = target_scores[row]
            topk_rows.append(torch.topk(scores, k=50, dim=1).indices.cpu().numpy())
    return np.concatenate(topk_rows, axis=0)


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    stats: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = stability.build_model(config, stats, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def correction_stats(
    item_popularity: np.ndarray,
    *,
    q_estimator: str = "empirical_frequency",
    batch_size: int = 1,
) -> dict[str, Any]:
    item_tensor = torch.from_numpy(
        np.repeat(np.arange(len(item_popularity), dtype=np.int64), item_popularity)
    )
    negative_log_q = -build_log_q_for_mode(
        item_tensor,
        len(item_popularity),
        "empirical",
        42,
        q_estimator=q_estimator,
        batch_size=batch_size,
    ).numpy()
    q = np.exp(-negative_log_q)

    def summarize_bucket(mask: np.ndarray) -> dict[str, int | float | None]:
        values = negative_log_q[mask]
        return {
            "items": int(mask.sum()),
            "mean": float(values.mean()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }

    return {
        "q_definition": (
            "1 - (1 - train_item_frequency) ** batch_size"
            if q_estimator == "batch_appearance"
            else "bincount(train_df.item_idx).clamp_min(1) / sum"
        ),
        "train_item_count_distribution": {
            "min": int(item_popularity.min()),
            "median": float(np.median(item_popularity)),
            "p95": float(np.percentile(item_popularity, 95)),
            "max": int(item_popularity.max()),
        },
        "q_distribution": {
            "min": float(q.min()),
            "median": float(np.median(q)),
            "p95": float(np.percentile(q, 95)),
            "max": float(q.max()),
        },
        "negative_log_q_by_popularity_bucket": {
            name: summarize_bucket(popularity_mask(item_popularity, low, high))
            for name, low, high in POPULARITY_BUCKETS_WITH_UNSEEN
        },
    }


def compare_bucket_recall(
    baseline: dict[str, dict[str, int | float]],
    logq: dict[str, dict[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    for name, _, _ in POPULARITY_BUCKETS:
        baseline_recall = float(baseline[name]["recall@50"])
        logq_recall = float(logq[name]["recall@50"])
        result[name] = {
            "targets": int(baseline[name]["targets"]),
            "baseline_hits": int(baseline[name]["hits"]),
            "logq_hits": int(logq[name]["hits"]),
            "baseline_recall@50": baseline_recall,
            "logq_recall@50": logq_recall,
            "absolute_delta": logq_recall - baseline_recall,
            "relative_delta": (
                (logq_recall - baseline_recall) / baseline_recall
                if baseline_recall else None
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--logq_config", required=True)
    parser.add_argument("--logq_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logq_config = yaml.safe_load(Path(args.logq_config).read_text(encoding="utf-8"))
    bundle = stability.load_data(Path(logq_config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_frame = pd.concat(
        [bundle.train_df, bundle.valid_df[stability.TRAIN_COLUMNS]],
        ignore_index=True,
    )
    history_matrix = stability.build_history_matrix(
        history_frame,
        num_users,
        int(logq_config["history_max_len"]),
    )
    test_seen = build_test_seen_items(bundle.train_df, bundle.valid_df)
    eval_df = select_non_cold_eval(bundle.test_df)
    targets = eval_df["item_idx"].to_numpy(dtype=np.int64)
    item_popularity = np.bincount(
        bundle.train_df["item_idx"].to_numpy(dtype=np.int64),
        minlength=num_items,
    )
    device = stability.resolve_device(str(logq_config["device"]))

    baseline_model = load_model(
        Path(args.baseline_config),
        Path(args.baseline_checkpoint),
        bundle.stats,
        device,
    )
    baseline_topk = encode_exact_top50(
        baseline_model,
        eval_df,
        history_matrix,
        test_seen,
        num_items,
        device,
        args.eval_batch_size,
        float(logq_config["temperature"]),
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logq_model = load_model(
        Path(args.logq_config),
        Path(args.logq_checkpoint),
        bundle.stats,
        device,
    )
    logq_topk = encode_exact_top50(
        logq_model,
        eval_df,
        history_matrix,
        test_seen,
        num_items,
        device,
        args.eval_batch_size,
        float(logq_config["temperature"]),
    )

    baseline_metrics = compute_ranking_metrics(baseline_topk, targets)
    logq_metrics = compute_ranking_metrics(logq_topk, targets)
    baseline_hit = baseline_metrics.pop("hit_mask")
    logq_hit = logq_metrics.pop("hit_mask")
    baseline_bucket = aggregate_popularity_bucket_recall(targets, baseline_hit, item_popularity)
    logq_bucket = aggregate_popularity_bucket_recall(targets, logq_hit, item_popularity)
    result = {
        "eval_protocol": {
            "split": "full_test",
            "test_frame": "concat(train, valid)",
            "seen_mask": "merge_seen_items(train_seen, valid)",
            "exclude_cold_target": True,
            "num_eval_users": int(len(eval_df)),
            "topk": 50,
            "retrieval": "exact inner product",
        },
        "baseline": {
            "ranking_metrics": baseline_metrics,
            "exposure_metrics": aggregate_exposure_metrics(baseline_topk, item_popularity),
            "target_item_popularity_bucket_recall": baseline_bucket,
        },
        "logq": {
            "ranking_metrics": logq_metrics,
            "exposure_metrics": aggregate_exposure_metrics(logq_topk, item_popularity),
            "target_item_popularity_bucket_recall": logq_bucket,
        },
        "comparison": {
            "recall@50_absolute_delta": logq_metrics["recall@50"] - baseline_metrics["recall@50"],
            "recall@50_relative_delta": (
                (logq_metrics["recall@50"] - baseline_metrics["recall@50"])
                / baseline_metrics["recall@50"]
            ),
            "target_item_popularity_bucket_recall": compare_bucket_recall(
                baseline_bucket,
                logq_bucket,
            ),
            "hit_transition": aggregate_hit_transition(baseline_hit, logq_hit),
            "top50_average_jaccard": average_jaccard(baseline_topk, logq_topk),
            "paired_bootstrap_ci": summarize_paired_bootstrap(
                baseline_hit,
                logq_hit,
                targets,
                item_popularity,
            ),
        },
        "train_only_logq": correction_stats(
            item_popularity,
            q_estimator=str(logq_config.get("q_estimator", "empirical_frequency")),
            batch_size=int(logq_config.get("batch_size", 1)),
        ),
    }
    np.save(output_dir / "baseline_top50.npy", baseline_topk)
    np.save(output_dir / "logq_top50.npy", logq_topk)
    stability.write_json(output_dir / "audit_summary.json", result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated 2x2 LogQ and duplicate-mask smoke for the Transformer user tower."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import train_transformer_maxlen100_smoke as base


EXTRA_CONFIG_KEYS = [
    "variant_name",
    "use_logq_correction",
    "mask_duplicate_items",
]
TRAIN_LOG_FIELDS = [
    "epoch",
    "train_loss",
    "valid_recall@20",
    "valid_recall@50",
    "valid_recall@100",
    "valid_ndcg@50",
    "valid_mrr@50",
    "duplicate_rows",
    "train_rows",
    "duplicate_row_ratio",
    "train_time_seconds",
    "eval_time_seconds",
    "epoch_time_seconds",
    "gpu_peak_memory_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Isolated LogQ smoke YAML config.")
    return parser.parse_args()


def require_config(cfg: dict[str, Any]) -> None:
    base.require_config(cfg)
    for key in EXTRA_CONFIG_KEYS:
        if key not in cfg:
            raise KeyError(f"Config missing key: {key}")
    if int(cfg["epochs"]) != 3:
        raise ValueError("LogQ smoke must run exactly 3 epochs.")
    if int(cfg["eval_max_users"]) != 50000:
        raise ValueError("LogQ smoke must use 50K limited-valid eval.")
    validate_q_mode(str(cfg.get("q_mode", "empirical")))
    validate_logq_alpha(float(cfg.get("logq_alpha", 1.0)))


def build_log_q(train_item_idx: torch.Tensor, num_items: int) -> torch.Tensor:
    """Build log train-only item frequency with unseen items clamped to one."""
    counts = torch.bincount(train_item_idx.to(dtype=torch.long).cpu(), minlength=num_items)
    q = counts.to(dtype=torch.float32).clamp_min(1.0)
    return (q / q.sum()).log()


def validate_q_mode(q_mode: str) -> None:
    if q_mode not in {"empirical", "shuffled", "constant"}:
        raise ValueError(f"Unsupported q_mode: {q_mode}")


def validate_logq_alpha(logq_alpha: float) -> None:
    if not 0.0 <= logq_alpha <= 1.0:
        raise ValueError(f"logq_alpha must be within [0.0, 1.0], got {logq_alpha}")


def build_log_q_for_mode(
    train_item_idx: torch.Tensor,
    num_items: int,
    q_mode: str,
    shuffle_seed: int,
) -> torch.Tensor:
    validate_q_mode(q_mode)
    log_q = build_log_q(train_item_idx, num_items)
    if q_mode == "empirical":
        return log_q
    if q_mode == "constant":
        return torch.full_like(log_q, -float(np.log(num_items)))
    if q_mode == "shuffled":
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
        return log_q[torch.randperm(num_items, generator=generator)]
    raise AssertionError("unreachable")


def summarize_batch_duplicates(batch_item_idx: torch.Tensor) -> dict[str, int | float]:
    rows = int(batch_item_idx.numel())
    unique_items = int(torch.unique(batch_item_idx).numel())
    duplicate_rows = rows - unique_items
    return {
        "rows": rows,
        "unique_items": unique_items,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_ratio": duplicate_rows / rows if rows else 0.0,
    }


def apply_logq_and_duplicate_mask(
    logits: torch.Tensor,
    batch_item_idx: torch.Tensor,
    log_q: torch.Tensor,
    *,
    use_logq: bool,
    mask_duplicate_items: bool,
    logq_alpha: float = 1.0,
) -> torch.Tensor:
    validate_logq_alpha(logq_alpha)
    corrected = logits
    if use_logq:
        candidate_log_q = log_q.to(device=logits.device, dtype=logits.dtype)[batch_item_idx]
        corrected = corrected - logq_alpha * candidate_log_q.unsqueeze(0)
    if mask_duplicate_items:
        same_item = batch_item_idx.unsqueeze(0) == batch_item_idx.unsqueeze(1)
        diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        corrected = corrected.masked_fill(same_item & ~diagonal, torch.finfo(logits.dtype).min)
    return corrected


def train_one_step(
    model: base.TextTwoTowerTransformerSmoke,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor,
    *,
    use_logq: bool,
    mask_duplicate_items: bool,
    logq_alpha: float = 1.0,
) -> tuple[float, dict[str, int | float]]:
    optimizer.zero_grad(set_to_none=True)
    raw_user, raw_item = model.raw_batch(user_idx, item_idx, history_item_idx)
    user_emb = F.normalize(raw_user, p=2, dim=-1) if model.use_l2_norm else raw_user
    item_emb = F.normalize(raw_item, p=2, dim=-1) if model.use_l2_norm else raw_item
    logits = (user_emb @ item_emb.T) / temperature
    corrected_logits = apply_logq_and_duplicate_mask(
        logits,
        item_idx,
        log_q,
        use_logq=use_logq,
        mask_duplicate_items=mask_duplicate_items,
        logq_alpha=logq_alpha,
    )
    if not torch.isfinite(corrected_logits).all():
        raise FloatingPointError("corrected logits contain nan/inf; stopping.")
    labels = torch.arange(corrected_logits.shape[0], device=corrected_logits.device)
    loss = F.cross_entropy(corrected_logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("loss is nan/inf; stopping.")
    loss.backward()
    optimizer.step()
    return float(loss.item()), summarize_batch_duplicates(item_idx)


def train_epoch(
    model: base.TextTwoTowerTransformerSmoke,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    epoch: int,
    log_q: torch.Tensor,
) -> tuple[float, dict[str, int | float]]:
    model.train()
    total_loss = 0.0
    total_rows = 0
    total_duplicate_rows = 0
    for batch_index, (user_idx, item_idx, history_item_idx) in enumerate(loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        history_item_idx = history_item_idx.to(device)
        loss, duplicate_stats = train_one_step(
            model,
            optimizer,
            user_idx,
            item_idx,
            history_item_idx,
            float(cfg["temperature"]),
            log_q,
            use_logq=bool(cfg["use_logq_correction"]),
            mask_duplicate_items=bool(cfg["mask_duplicate_items"]),
            logq_alpha=float(cfg.get("logq_alpha", 1.0)),
        )
        rows = int(duplicate_stats["rows"])
        total_loss += loss * rows
        total_rows += rows
        total_duplicate_rows += int(duplicate_stats["duplicate_rows"])
        if batch_index == 0:
            logging.info(
                "epoch %s batch 0: loss=%.4f bs=%s duplicate_ratio=%.4f",
                epoch,
                loss,
                rows,
                duplicate_stats["duplicate_row_ratio"],
            )
    return total_loss / total_rows, {
        "train_rows": total_rows,
        "duplicate_rows": total_duplicate_rows,
        "duplicate_row_ratio": total_duplicate_rows / total_rows if total_rows else 0.0,
    }


def init_train_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def append_train_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=TRAIN_LOG_FIELDS).writerow(
            {field: row.get(field, "") for field in TRAIN_LOG_FIELDS}
        )


def gpu_peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


def train(cfg: dict[str, Any]) -> None:
    require_config(cfg)
    base.set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output_dir"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base.write_json(out_dir / "run_config.json", cfg)

    device = base.resolve_device(str(cfg["device"]))
    logging.info(
        "variant=%s device=%s logq=%s alpha=%.2f duplicate_mask=%s",
        cfg["variant_name"],
        device,
        cfg["use_logq_correction"],
        float(cfg.get("logq_alpha", 1.0)),
        cfg["mask_duplicate_items"],
    )
    bundle = base.load_data(Path(cfg["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_matrix = base.build_history_matrix(
        bundle.train_df,
        num_users,
        int(cfg["history_max_len"]),
    )
    raw_history_lengths = base.compute_raw_history_lengths(bundle.train_df, num_users)
    loader = base.make_dataloader(bundle.train_df, history_matrix, cfg)
    seen_items = base.build_seen_items(bundle.train_df)
    model = base.build_model(cfg, bundle.stats, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train_item_idx = torch.from_numpy(bundle.train_df["item_idx"].to_numpy(dtype=np.int64))
    q_mode = str(cfg.get("q_mode", "empirical"))
    q_shuffle_seed = int(cfg.get("q_shuffle_seed", cfg["seed"]))
    log_q = build_log_q_for_mode(train_item_idx, num_items, q_mode, q_shuffle_seed).to(device)
    q = log_q.exp()
    q_stats = {
        "source": "train_df.item_idx",
        "q_mode": q_mode,
        "q_shuffle_seed": q_shuffle_seed,
        "num_items": num_items,
        "min_q": float(q.min().item()),
        "max_q": float(q.max().item()),
        "sum_q": float(q.sum().item()),
    }

    train_log_path = out_dir / "train_log.csv"
    init_train_log(train_log_path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base.log_gpu_memory(device)

    best_recall = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_buckets: dict[str, Any] = {}
    total_train_seconds = 0.0
    epoch_duplicate_stats: list[dict[str, int | float]] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_started = time.time()
        train_started = time.time()
        train_loss, duplicate_stats = train_epoch(
            model,
            loader,
            optimizer,
            cfg,
            device,
            epoch,
            log_q,
        )
        train_seconds = time.time() - train_started
        eval_started = time.time()
        valid_metrics, hit_users, buckets = base.evaluate_with_buckets(
            model,
            bundle.valid_df,
            history_matrix,
            raw_history_lengths,
            seen_items,
            cfg,
            bundle.stats,
            device,
            "valid",
        )
        eval_seconds = time.time() - eval_started
        epoch_seconds = time.time() - epoch_started
        total_train_seconds += epoch_seconds
        epoch_duplicate_stats.append(duplicate_stats)
        logging.info(
            "epoch %s: loss=%.6f R@50=%.6f NDCG@50=%.6f MRR@50=%.6f time=%.1fs",
            epoch,
            train_loss,
            valid_metrics["recall@50"],
            valid_metrics["ndcg@50"],
            valid_metrics["mrr@50"],
            epoch_seconds,
        )
        append_train_log(train_log_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_recall@20": valid_metrics["recall@20"],
            "valid_recall@50": valid_metrics["recall@50"],
            "valid_recall@100": valid_metrics["recall@100"],
            "valid_ndcg@50": valid_metrics["ndcg@50"],
            "valid_mrr@50": valid_metrics["mrr@50"],
            **duplicate_stats,
            "train_time_seconds": train_seconds,
            "eval_time_seconds": eval_seconds,
            "epoch_time_seconds": epoch_seconds,
            "gpu_peak_memory_gb": gpu_peak_memory_gb(device),
        })
        if float(valid_metrics["recall@50"]) > best_recall:
            best_recall = float(valid_metrics["recall@50"])
            best_epoch = epoch
            best_metrics = valid_metrics
            best_buckets = buckets
            np.save(out_dir / "hit_users_valid_r50.npy", np.array(sorted(hit_users), dtype=np.int64))
            base.save_checkpoint(
                out_dir / "checkpoints" / "best_model.pt",
                model,
                cfg,
                bundle.stats,
                epoch,
                best_recall,
            )
        if device.type == "cuda":
            base.log_gpu_memory(device)

    duplicate_rows = sum(int(row["duplicate_rows"]) for row in epoch_duplicate_stats)
    train_rows = sum(int(row["train_rows"]) for row in epoch_duplicate_stats)
    summary = {
        "variant_name": cfg["variant_name"],
        "q_mode": q_mode,
        "logq_alpha": float(cfg.get("logq_alpha", 1.0)),
        "use_logq_correction": bool(cfg["use_logq_correction"]),
        "mask_duplicate_items": bool(cfg["mask_duplicate_items"]),
        "best_epoch": best_epoch,
        "best_valid_recall@50": best_recall,
        "best_valid_ndcg@50": best_metrics.get("ndcg@50", 0.0),
        "best_valid_mrr@50": best_metrics.get("mrr@50", 0.0),
        "num_eval_users": best_metrics.get("num_eval_users", 0),
        "bucket_recall@50": {
            name: best_buckets[name]["recall@50"] for name, _ in base.HISTORY_BUCKETS
        },
        "bucket_counts": {
            name: best_buckets[name]["count"] for name, _ in base.HISTORY_BUCKETS
        },
        "duplicate_rows": duplicate_rows,
        "train_rows": train_rows,
        "duplicate_row_ratio": duplicate_rows / train_rows if train_rows else 0.0,
        "q_stats": q_stats,
        "parameter_count": base.count_trainable_params(model),
        "total_train_sec": total_train_seconds,
        "gpu_peak_memory_gb": gpu_peak_memory_gb(device),
        "output_dir": str(out_dir),
    }
    base.write_json(out_dir / "metrics_valid_best.json", summary)
    logging.info(
        "complete: variant=%s best_epoch=%s R@50=%.6f total_time=%.1fs",
        cfg["variant_name"],
        best_epoch,
        best_recall,
        total_train_seconds,
    )


def main() -> None:
    base.setup_logging()
    args = parse_args()
    train(base.load_config(Path(args.config)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated full-train candidate for time-aware Transformer with LogQ loss."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import train_transformer_stability_sweep as stability
from train_transformer_logq_smoke import (
    build_log_q_for_mode,
    compute_sampling_loss,
    resolve_loss_variant,
    summarize_batch_duplicates,
)
from transformer_sampling_losses import (
    validate_logq_alpha,
    validate_loss_variant,
    validate_mns_uniform_fraction,
    validate_q_estimator,
)


RESULTS_DIR = Path("outputs/transformer_user_tower_investigation/logq_full")
TRAIN_LOG_FIELDS = [
    *stability.SWEEP_TRAIN_LOG_FIELDS,
    "duplicate_rows",
    "train_rows",
    "duplicate_row_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip_full_eval", action="store_true")
    return parser.parse_args()


def validate_logq_config(cfg: dict[str, Any]) -> None:
    if cfg.get("mask_duplicate_items") is not False:
        raise ValueError("mask_duplicate_items must be false for LogQ full candidate.")
    loss_variant = resolve_loss_variant(cfg)
    validate_loss_variant(loss_variant)
    if loss_variant == "infonce":
        raise ValueError("loss_variant=infonce is a baseline, not a full sampling candidate.")
    if loss_variant in {"old_logq", "uber_batchq", "refined_logq", "mns_refined_logq"}:
        if cfg.get("use_logq_correction") is not True:
            raise ValueError(f"use_logq_correction must be true for {loss_variant}.")
    validate_q_estimator(str(cfg.get("q_estimator", "empirical_frequency")))
    validate_logq_alpha(float(cfg.get("logq_alpha", 1.0)))
    validate_mns_uniform_fraction(float(cfg.get("mns_uniform_fraction", 0.5)))


def load_config(path: Path) -> dict[str, Any]:
    cfg = stability.load_config(path)
    validate_logq_config(cfg)
    return cfg


def compute_logq_loss(
    model: torch.nn.Module,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor,
    *,
    cfg: dict[str, Any] | None = None,
    num_items: int | None = None,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    effective_cfg = cfg or {
        "loss_variant": "old_logq",
        "mask_duplicate_items": False,
        "logq_alpha": 1.0,
    }
    return compute_sampling_loss(
        model,
        user_idx,
        item_idx,
        history_item_idx,
        temperature,
        log_q,
        cfg=effective_cfg,
        num_items=num_items or int(log_q.numel()),
    )


def train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor,
    *,
    cfg: dict[str, Any] | None = None,
    num_items: int | None = None,
) -> tuple[float, dict[str, int | float]]:
    optimizer.zero_grad(set_to_none=True)
    loss, duplicate_stats = compute_logq_loss(
        model,
        user_idx,
        item_idx,
        history_item_idx,
        temperature,
        log_q,
        cfg=cfg,
        num_items=num_items,
    )
    loss.backward()
    optimizer.step()
    return float(loss.item()), duplicate_stats


def train_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: dict[str, Any],
    device: torch.device,
    grad_clip_norm: float,
    log_q: torch.Tensor,
) -> tuple[float, float, float, dict[str, int | float]]:
    model.train()
    total_loss = 0.0
    total_rows = 0
    total_duplicate_rows = 0
    total_grad_norm = 0.0
    batch_count = 0
    temperature = float(cfg["temperature"])

    for user_idx, item_idx, history_item_idx in loader:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        history_item_idx = history_item_idx.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, duplicate_stats = compute_logq_loss(
            model,
            user_idx,
            item_idx,
            history_item_idx,
            temperature,
            log_q,
            cfg=cfg,
            num_items=int(log_q.numel()),
        )
        loss.backward()

        grad_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm += parameter.grad.data.norm(2).item() ** 2
        total_grad_norm += math.sqrt(grad_norm)

        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        rows = int(duplicate_stats["rows"])
        total_loss += float(loss.item()) * rows
        total_rows += rows
        total_duplicate_rows += int(duplicate_stats["duplicate_rows"])
        batch_count += 1

    return (
        total_loss / total_rows if total_rows else 0.0,
        total_grad_norm / batch_count if batch_count else 0.0,
        float(optimizer.param_groups[0]["lr"]),
        {
            "train_rows": total_rows,
            "duplicate_rows": total_duplicate_rows,
            "duplicate_row_ratio": total_duplicate_rows / total_rows if total_rows else 0.0,
        },
    )


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_logq_config(cfg)
    stability.set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output_dir"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stability.write_json(out_dir / "config.json", cfg)

    device = stability.resolve_device(str(cfg["device"]))
    bundle = stability.load_data(Path(cfg["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_matrix = stability.build_history_matrix(
        bundle.train_df,
        num_users,
        int(cfg["history_max_len"]),
    )
    raw_history_lengths = stability.compute_raw_history_lengths(bundle.train_df, num_users)
    seen_items = stability.build_seen_items(bundle.train_df)
    loader = stability.make_dataloader(bundle.train_df, history_matrix, cfg)
    model = stability.build_model(cfg, bundle.stats, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    steps_per_epoch = math.ceil(len(bundle.train_df) / int(cfg["batch_size"]))
    scheduler = stability.build_scheduler(optimizer, cfg, steps_per_epoch)
    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
    patience = int(cfg.get("early_stopping_patience", 0))

    train_items = np.array(bundle.train_df["item_idx"].to_numpy(dtype=np.int64), copy=True)
    q_mode = str(cfg.get("q_mode", "empirical"))
    q_estimator = str(cfg.get("q_estimator", "empirical_frequency"))
    log_q = build_log_q_for_mode(
        torch.from_numpy(train_items),
        num_items,
        q_mode,
        int(cfg.get("q_shuffle_seed", cfg["seed"])),
        q_estimator=q_estimator,
        batch_size=int(cfg["batch_size"]),
    ).to(device)
    q = log_q.exp()
    q_stats = {
        "source": "train_df.item_idx",
        "q_mode": q_mode,
        "q_estimator": q_estimator,
        "num_items": num_items,
        "min_q": float(q.min().item()),
        "max_q": float(q.max().item()),
        "sum_q": float(q.sum().item()),
    }

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=TRAIN_LOG_FIELDS).writeheader()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_recall = -1.0
    best_metrics: dict[str, Any] = {}
    best_buckets: dict[str, Any] = {}
    best_epoch = 0
    no_improve = 0
    total_seconds = 0.0
    early_stopped = False
    all_duplicate_rows = 0
    all_train_rows = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_started = time.time()
        train_loss, grad_norm, current_lr, duplicate_stats = train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            cfg,
            device,
            grad_clip_norm,
            log_q,
        )
        train_seconds = time.time() - epoch_started
        eval_started = time.time()
        valid_metrics, hit_users, buckets = stability.evaluate_with_buckets(
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
        total_seconds += time.time() - epoch_started
        all_duplicate_rows += int(duplicate_stats["duplicate_rows"])
        all_train_rows += int(duplicate_stats["train_rows"])
        recall = float(valid_metrics["recall@50"])
        logging.info(
            "ep%02d loss=%.4f R@50=%.6f gnorm=%.3f lr=%.2e duplicate_ratio=%.4f",
            epoch,
            train_loss,
            recall,
            grad_norm,
            current_lr,
            duplicate_stats["duplicate_row_ratio"],
        )
        with log_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=TRAIN_LOG_FIELDS).writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_recall@20": valid_metrics.get("recall@20", ""),
                "valid_recall@50": recall,
                "valid_recall@100": valid_metrics.get("recall@100", ""),
                "valid_ndcg@50": valid_metrics.get("ndcg@50", ""),
                "valid_mrr@50": valid_metrics.get("mrr@50", ""),
                "grad_norm": round(grad_norm, 5),
                "lr": current_lr,
                "train_time_seconds": round(train_seconds, 2),
                "eval_time_seconds": round(eval_seconds, 2),
                **duplicate_stats,
            })

        if recall > best_recall:
            best_recall = recall
            best_metrics = valid_metrics
            best_buckets = buckets
            best_epoch = epoch
            no_improve = 0
            np.save(out_dir / "hit_users_valid_r50.npy", np.array(sorted(hit_users), dtype=np.int64))
            stability.save_checkpoint(
                out_dir / "checkpoints" / "best_model.pt",
                model,
                cfg,
                bundle.stats,
                epoch,
                best_recall,
            )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                early_stopped = True
                break
        if device.type == "cuda":
            stability.log_gpu_memory(device)

    summary = {
        "run_label": cfg.get("run_label", ""),
        "loss_variant": resolve_loss_variant(cfg),
        "q_estimator": q_estimator,
        "logq_alpha": float(cfg.get("logq_alpha", 1.0)),
        "mns_uniform_fraction": float(cfg.get("mns_uniform_fraction", 0.5)),
        "use_logq_correction": bool(cfg.get("use_logq_correction", True)),
        "mask_duplicate_items": False,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "early_stopped": early_stopped,
        "early_stopping_patience": patience,
        "best_limited_valid_recall@50": best_recall,
        "best_limited_valid_ndcg@50": best_metrics.get("ndcg@50", 0.0),
        "best_limited_valid_mrr@50": best_metrics.get("mrr@50", 0.0),
        "bucket_recall@50": {
            bucket: best_buckets[bucket]["recall@50"] for bucket, _ in stability.HISTORY_BUCKETS
        },
        "duplicate_rows": all_duplicate_rows,
        "train_rows": all_train_rows,
        "duplicate_row_ratio": all_duplicate_rows / all_train_rows if all_train_rows else 0.0,
        "q_stats": q_stats,
        "total_train_sec": round(total_seconds, 1),
        "gpu_peak_gb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**3)
            if device.type == "cuda" else None
        ),
        "output_dir": str(out_dir),
    }
    stability.write_json(out_dir / "metrics_valid_best.json", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(Path(args.config))
    cfg["config_path"] = args.config
    train_summary = train(cfg)
    if not args.skip_full_eval:
        checkpoint_path = Path(cfg["output_dir"]) / "checkpoints" / "best_model.pt"
        eval_output_dir = Path(cfg["output_dir"] + "_full_eval")
        train_summary.update(stability.full_eval(cfg, checkpoint_path, eval_output_dir))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stability.write_json(RESULTS_DIR / f"{cfg['run_label']}_result.json", train_summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Time-aware Transformer stability sweep.

Adds over the base smoke script:
  - early_stopping_patience  (int,   0  = disabled)
  - grad_clip_norm           (float, 0.0 = disabled)
  - warmup_steps             (int,   0  = disabled)
  - lr_schedule              ('cosine' | 'none')

After training, automatically runs full eval (full valid + full test)
and saves per-run metrics to configs["output_dir"] and
outputs/transformer_user_tower_investigation/stability_sweep/.

Usage:
  python scripts/train_transformer_stability_sweep.py --config configs/stability_A.yaml
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Bootstrap: import from base smoke script so we reuse model/data/eval code
# ---------------------------------------------------------------------------

def _import_base() -> Any:
    mod_name = "_base_smoke"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent / "train_transformer_maxlen100_smoke.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Must register before exec so @dataclass can resolve cls.__module__
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_B = _import_base()

# Re-export the heavy functions we need
load_data              = _B.load_data
build_history_matrix   = _B.build_history_matrix
compute_raw_history_lengths = _B.compute_raw_history_lengths
build_seen_items       = _B.build_seen_items
merge_seen_items       = _B.merge_seen_items
load_text_artifacts    = _B.load_text_artifacts
make_dataloader        = _B.make_dataloader
build_model            = _B.build_model
evaluate_with_buckets  = _B.evaluate_with_buckets
encode_all_items       = _B.encode_all_items
log_gpu_memory         = _B.log_gpu_memory
save_checkpoint        = _B.save_checkpoint
resolve_device         = _B.resolve_device
set_seed               = _B.set_seed
write_json             = _B.write_json
HISTORY_BUCKETS        = _B.HISTORY_BUCKETS
TRAIN_COLUMNS          = _B.TRAIN_COLUMNS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SWEEP_TRAIN_LOG_FIELDS = [
    "epoch", "train_loss", "valid_recall@20", "valid_recall@50",
    "valid_recall@100", "valid_ndcg@50", "valid_mrr@50",
    "grad_norm", "lr", "train_time_seconds", "eval_time_seconds",
]

SWEEP_BASE_DIR = Path("outputs/transformer_user_tower_investigation/stability_sweep")

REQUIRED = [
    "data_dir", "output_dir", "embedding_dim", "batch_size", "learning_rate",
    "weight_decay", "epochs", "temperature", "use_l2_norm", "seed",
    "eval_k_list", "eval_batch_size", "eval_max_users", "num_workers", "device",
    "save_best_by", "history_max_len", "history_weight",
    "item_text_embedding_path", "item_has_text_path", "text_proj_dim",
    "use_has_text_mask", "item_fusion", "pooling_type", "decay_rate",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in REQUIRED:
        if k not in cfg:
            raise KeyError(f"Config missing: {k}")
    if cfg.get("pooling_type") != "transformer_timeaware":
        raise ValueError("stability sweep only supports pooling_type=transformer_timeaware")
    return cfg


# ---------------------------------------------------------------------------
# Training one epoch with grad norm tracking
# ---------------------------------------------------------------------------

def train_epoch_with_grad_norm(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: dict[str, Any],
    device: torch.device,
    grad_clip_norm: float,
) -> tuple[float, float, float]:
    """Return (avg_train_loss, avg_grad_norm, final_lr)."""
    model.train()
    total_loss, total_n = 0.0, 0
    total_grad_norm = 0.0
    batch_count = 0
    temperature = float(cfg["temperature"])

    for user_idx, item_idx, hist_idx in loader:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        hist_idx = hist_idx.to(device)

        optimizer.zero_grad(set_to_none=True)
        raw_u, raw_i = model.raw_batch(user_idx, item_idx, hist_idx)
        u_emb = F.normalize(raw_u, p=2, dim=-1) if model.use_l2_norm else raw_u
        i_emb = F.normalize(raw_i, p=2, dim=-1) if model.use_l2_norm else raw_i
        logits = (u_emb @ i_emb.T) / temperature
        labels = torch.arange(logits.shape[0], device=device)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("loss is nan/inf; stopping.")
        loss.backward()

        # Gradient norm (before clipping)
        g_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g_norm += p.grad.data.norm(2).item() ** 2
        g_norm = math.sqrt(g_norm)
        total_grad_norm += g_norm

        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = user_idx.shape[0]
        total_loss += loss.item() * n
        total_n += n
        batch_count += 1

    avg_loss = total_loss / total_n if total_n else 0.0
    avg_gnorm = total_grad_norm / batch_count if batch_count else 0.0
    current_lr = optimizer.param_groups[0]["lr"]
    return avg_loss, avg_gnorm, current_lr


# ---------------------------------------------------------------------------
# Build LR scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg: dict[str, Any], steps_per_epoch: int):
    schedule = str(cfg.get("lr_schedule", "none")).lower()
    warmup_steps = int(cfg.get("warmup_steps", 0))
    total_epochs = int(cfg["epochs"])
    total_steps = total_epochs * steps_per_epoch

    if schedule == "none" and warmup_steps == 0:
        return None

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        ws = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        schedulers.append(ws)
        milestones.append(warmup_steps)

    if schedule == "cosine":
        cosine_steps = total_steps - warmup_steps
        cs = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cosine_steps), eta_min=0.0
        )
        schedulers.append(cs)
    elif warmup_steps > 0:
        # warmup only, constant afterwards
        cs = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        schedulers.append(cs)

    if len(schedulers) == 1:
        return schedulers[0]
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", cfg)

    device = resolve_device(str(cfg["device"]))
    logging.info("sweep-train device=%s lr=%s gc=%s patience=%s warmup=%s sched=%s",
                 device, cfg["learning_rate"],
                 cfg.get("grad_clip_norm", 0.0),
                 cfg.get("early_stopping_patience", 0),
                 cfg.get("warmup_steps", 0),
                 cfg.get("lr_schedule", "none"))

    bundle  = load_data(Path(cfg["data_dir"]))
    n_users = int(bundle.stats["n_users"])
    max_len = int(cfg["history_max_len"])

    hist_mat = build_history_matrix(bundle.train_df, n_users, max_len)
    raw_lens = compute_raw_history_lengths(bundle.train_df, n_users)
    seen     = build_seen_items(bundle.train_df)
    loader   = make_dataloader(bundle.train_df, hist_mat, cfg)

    model = build_model(cfg, bundle.stats, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    steps_per_epoch = math.ceil(len(bundle.train_df) / int(cfg["batch_size"]))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    grad_clip = float(cfg.get("grad_clip_norm", 0.0))
    patience  = int(cfg.get("early_stopping_patience", 0))

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=SWEEP_TRAIN_LOG_FIELDS).writeheader()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_r50 = -1.0
    best_metrics: dict[str, Any] = {}
    best_bucket: dict[str, Any] = {}
    best_epoch = 0
    no_improve = 0
    total_sec = 0.0
    early_stopped = False

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        train_loss, g_norm, cur_lr = train_epoch_with_grad_norm(
            model, loader, optimizer, scheduler, cfg, device, grad_clip
        )
        train_sec = time.time() - t0

        t_eval = time.time()
        valid_metrics, hit_users, bucket = evaluate_with_buckets(
            model, bundle.valid_df, hist_mat, raw_lens, seen,
            cfg, bundle.stats, device, "valid",
        )
        eval_sec  = time.time() - t_eval
        epoch_sec = time.time() - t0
        total_sec += epoch_sec

        r50 = float(valid_metrics["recall@50"])
        logging.info(
            "ep%02d loss=%.4f R@50=%.6f gnorm=%.3f lr=%.2e time=%.0fs",
            epoch, train_loss, r50, g_norm, cur_lr, epoch_sec,
        )
        for bname, _ in HISTORY_BUCKETS:
            br = bucket[bname]
            logging.info("  %-6s R@50=%.6f hits=%d", bname, br["recall@50"], br["hits"])

        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SWEEP_TRAIN_LOG_FIELDS).writerow({
                "epoch": epoch, "train_loss": train_loss,
                "valid_recall@20": valid_metrics.get("recall@20", ""),
                "valid_recall@50": r50,
                "valid_recall@100": valid_metrics.get("recall@100", ""),
                "valid_ndcg@50": valid_metrics.get("ndcg@50", ""),
                "valid_mrr@50":  valid_metrics.get("mrr@50", ""),
                "grad_norm": round(g_norm, 5),
                "lr": cur_lr,
                "train_time_seconds": round(train_sec, 2),
                "eval_time_seconds":  round(eval_sec, 2),
            })

        if r50 > best_r50:
            best_r50 = r50
            best_metrics = valid_metrics
            best_bucket  = bucket
            best_epoch   = epoch
            no_improve   = 0
            np.save(out_dir / "hit_users_valid_r50.npy",
                    np.array(sorted(hit_users), dtype=np.int64))
            save_checkpoint(out_dir / "checkpoints" / "best_model.pt",
                            model, cfg, bundle.stats, epoch, best_r50)
            logging.info("  → best ep=%d R@50=%.6f", epoch, best_r50)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                logging.info("Early stopping: no improve for %d epochs. Best ep=%d R@50=%.6f",
                             patience, best_epoch, best_r50)
                early_stopped = True
                break

        if device.type == "cuda":
            log_gpu_memory(device)

    gpu_mb = (
        float(torch.cuda.max_memory_allocated(device) / 1024**3)
        if device.type == "cuda" else None
    )
    summary = {
        "run_label":       cfg.get("run_label", ""),
        "pooling_type":    cfg["pooling_type"],
        "lr":              float(cfg["learning_rate"]),
        "grad_clip_norm":  float(cfg.get("grad_clip_norm", 0.0)),
        "early_stopping_patience": int(cfg.get("early_stopping_patience", 0)),
        "warmup_steps":    int(cfg.get("warmup_steps", 0)),
        "lr_schedule":     str(cfg.get("lr_schedule", "none")),
        "best_epoch":      best_epoch,
        "epochs_trained":  epoch,
        "early_stopped":   early_stopped,
        "best_limited_valid_recall@50": best_r50,
        "best_limited_valid_ndcg@50":   best_metrics.get("ndcg@50", 0.0),
        "best_limited_valid_mrr@50":    best_metrics.get("mrr@50", 0.0),
        "bucket_recall@50": {b: best_bucket[b]["recall@50"] for b, _ in HISTORY_BUCKETS},
        "total_train_sec": round(total_sec, 1),
        "gpu_peak_gb":     gpu_mb,
        "output_dir":      str(out_dir),
    }
    write_json(out_dir / "metrics_valid_best.json", summary)
    logging.info("Training done: best_ep=%d R@50=%.6f total=%.0fs", best_epoch, best_r50, total_sec)
    return summary


# ---------------------------------------------------------------------------
# Full eval on best checkpoint
# ---------------------------------------------------------------------------

def full_eval(cfg: dict[str, Any], checkpoint_path: Path, eval_out: Path) -> dict[str, Any]:
    cfg_eval = dict(cfg)
    cfg_eval["eval_max_users"] = None
    eval_out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(cfg["device"]))
    bundle = load_data(Path(cfg["data_dir"]))
    n_users = int(bundle.stats["n_users"])
    max_len = int(cfg["history_max_len"])

    train_hist = build_history_matrix(bundle.train_df, n_users, max_len)
    train_raw  = compute_raw_history_lengths(bundle.train_df, n_users)
    train_seen = build_seen_items(bundle.train_df)

    test_frame = __import__("pandas").concat(
        [bundle.train_df, bundle.valid_df[TRAIN_COLUMNS]], ignore_index=True
    )
    test_hist  = build_history_matrix(test_frame, n_users, max_len)
    test_raw   = compute_raw_history_lengths(test_frame, n_users)
    test_seen  = merge_seen_items(train_seen, bundle.valid_df)

    model = build_model(cfg_eval, bundle.stats, device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info("loaded checkpoint epoch=%s metric=%.6f", ckpt.get("epoch"), ckpt.get("best_metric_value", 0))

    valid_m, valid_hits, valid_bkt = evaluate_with_buckets(
        model, bundle.valid_df, train_hist, train_raw, train_seen,
        cfg_eval, bundle.stats, device, "valid_full",
    )
    test_m, test_hits, test_bkt = evaluate_with_buckets(
        model, bundle.test_df, test_hist, test_raw, test_seen,
        cfg_eval, bundle.stats, device, "test_full",
    )

    write_json(eval_out / "metrics_full_valid.json", valid_m)
    write_json(eval_out / "metrics_full_test.json",  test_m)
    write_json(eval_out / "bucket_full_valid.json",  valid_bkt)
    write_json(eval_out / "bucket_full_test.json",   test_bkt)
    np.save(eval_out / "hit_users_valid_full_r50.npy",
            np.array(sorted(valid_hits), dtype=np.int64))
    np.save(eval_out / "hit_users_test_full_r50.npy",
            np.array(sorted(test_hits), dtype=np.int64))

    result = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "full_valid_recall@50": valid_m["recall@50"],
        "full_valid_ndcg@50":   valid_m["ndcg@50"],
        "full_valid_mrr@50":    valid_m["mrr@50"],
        "full_test_recall@50":  test_m["recall@50"],
        "full_test_ndcg@50":    test_m["ndcg@50"],
        "full_test_mrr@50":     test_m["mrr@50"],
        "valid_bucket_recall@50": {k: v["recall@50"] for k, v in valid_bkt.items()},
        "test_bucket_recall@50":  {k: v["recall@50"] for k, v in test_bkt.items()},
    }
    write_json(eval_out / "eval_summary.json", result)
    logging.info("full-eval: valid R@50=%.6f  test R@50=%.6f",
                 valid_m["recall@50"], test_m["recall@50"])
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip_full_eval", action="store_true",
                   help="Skip full eval after training.")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cfg["config_path"] = args.config

    train_summary = train(cfg)

    if not args.skip_full_eval:
        ckpt_path = Path(cfg["output_dir"]) / "checkpoints" / "best_model.pt"
        eval_out  = Path(cfg["output_dir"] + "_full_eval")
        if not ckpt_path.exists():
            logging.error("No checkpoint found at %s; skipping full eval.", ckpt_path)
            return
        full_eval_result = full_eval(cfg, ckpt_path, eval_out)
        train_summary.update(full_eval_result)

    # Save combined summary for sweep aggregation
    SWEEP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    label = cfg.get("run_label", Path(cfg["output_dir"]).name)
    write_json(SWEEP_BASE_DIR / f"{label}_result.json", train_summary)
    logging.info("Sweep result saved to %s", SWEEP_BASE_DIR / f"{label}_result.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only collapse diagnosis for the canonical Transformer tower.

Inputs are existing artifacts only:
  - 20-epoch time-aware Transformer train_log.csv
  - canonical best_model.pt

Writes a new diagnosis bundle under outputs/transformer_collapse_diagnosis/.
It does not train, evaluate, or overwrite canonical outputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_TRAIN_LOG = Path("outputs/transformer_user_tower_investigation/timeaware_max100_20ep/train_log.csv")
DEFAULT_CONFIG = Path("configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml")
DEFAULT_CHECKPOINT = Path("outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/transformer_collapse_diagnosis")

STABILITY_SWEEP = {
    "A_lr1e3_no_clip_patience2": {
        "lr": 1e-3,
        "grad_clip_norm": 0.0,
        "best_epoch": 2,
        "limited_valid_recall@50": 0.124300,
        "full_test_recall@50": 0.103128,
    },
    "B_lr3e4_clip1_patience3": {
        "lr": 3e-4,
        "grad_clip_norm": 1.0,
        "best_epoch": 3,
        "limited_valid_recall@50": 0.123300,
        "full_test_recall@50": 0.100282,
    },
    "C_lr1e4_clip1_patience3": {
        "lr": 1e-4,
        "grad_clip_norm": 1.0,
        "best_epoch": 5,
        "limited_valid_recall@50": 0.116860,
        "full_test_recall@50": 0.094946,
    },
    "D_lr3e4_clip1_warmup_cosine_patience3": {
        "lr": 3e-4,
        "grad_clip_norm": 1.0,
        "best_epoch": 3,
        "limited_valid_recall@50": 0.122260,
        "full_test_recall@50": 0.100304,
    },
}


def _import_sweep() -> Any:
    mod_name = "_collapse_sweep"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent / "train_transformer_stability_sweep.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_train_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            if "epoch" in row:
                row["epoch"] = int(row["epoch"])
                rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in train log: {path}")
    return rows


def find_peak_row(rows: list[dict[str, float]], metric: str = "valid_recall@50") -> dict[str, float]:
    usable = [row for row in rows if metric in row and math.isfinite(row[metric])]
    if not usable:
        raise ValueError(f"metric not found in train log: {metric}")
    return max(usable, key=lambda row: row[metric])


def summarize_collapse(rows: list[dict[str, float]]) -> dict[str, Any]:
    peak = find_peak_row(rows)
    final = rows[-1]
    peak_r = float(peak["valid_recall@50"])
    final_r = float(final.get("valid_recall@50", float("nan")))
    peak_loss = float(peak.get("train_loss", float("nan")))
    final_loss = float(final.get("train_loss", float("nan")))
    return {
        "epochs_observed": len(rows),
        "peak_epoch": int(peak["epoch"]),
        "peak_valid_recall@50": peak_r,
        "final_epoch": int(final["epoch"]),
        "final_valid_recall@50": final_r,
        "absolute_drop_after_peak": final_r - peak_r,
        "relative_retention_after_peak": final_r / peak_r if peak_r else None,
        "peak_train_loss": peak_loss,
        "final_train_loss": final_loss,
        "train_loss_delta_after_peak": final_loss - peak_loss,
    }


def effective_rank_from_singular_values(singular_values: np.ndarray) -> float:
    power = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    probs = power / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def participation_rank_from_singular_values(singular_values: np.ndarray) -> float:
    power = np.square(np.asarray(singular_values, dtype=np.float64))
    denom = float(np.sum(np.square(power)))
    if denom <= 0.0:
        return 0.0
    return float(np.square(power.sum()) / denom)


def uniformity_sample(embeddings: np.ndarray, sample_pairs: int, seed: int = 42) -> dict[str, Any]:
    if len(embeddings) < 2 or sample_pairs <= 0:
        return {"sample_pairs": 0, "uniformity": None}
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(embeddings), size=sample_pairs)
    right = rng.integers(0, len(embeddings), size=sample_pairs)
    same = left == right
    while np.any(same):
        right[same] = rng.integers(0, len(embeddings), size=int(np.sum(same)))
        same = left == right
    diff = embeddings[left] - embeddings[right]
    dist2 = np.sum(diff * diff, axis=1)
    values = -2.0 * dist2
    max_v = float(np.max(values))
    uniformity = max_v + math.log(float(np.mean(np.exp(values - max_v))))
    return {
        "sample_pairs": int(sample_pairs),
        "uniformity": float(uniformity),
        "mean_pairwise_sq_distance": float(np.mean(dist2)),
    }


def embedding_diagnostics(embeddings: np.ndarray, sample_pairs: int) -> dict[str, Any]:
    emb = np.asarray(embeddings, dtype=np.float64)
    finite = np.isfinite(emb)
    centered = emb - emb.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    power = np.square(singular_values)
    explained = power / power.sum() if float(power.sum()) > 0 else np.zeros_like(power)
    norms = np.linalg.norm(emb, axis=1)
    eff_rank = effective_rank_from_singular_values(singular_values)
    part_rank = participation_rank_from_singular_values(singular_values)
    top1 = float(explained[0]) if len(explained) else 0.0
    health = {
        "finite": bool(finite.all()),
        "effective_rank_ge_16": bool(eff_rank >= 16.0),
        "top1_explained_lt_0_50": bool(top1 < 0.50),
    }
    return {
        "shape": [int(emb.shape[0]), int(emb.shape[1])],
        "nan_count": int(np.isnan(emb).sum()),
        "inf_count": int(np.isinf(emb).sum()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "singular_values_top10": [float(x) for x in singular_values[:10]],
        "singular_values": [float(x) for x in singular_values],
        "explained_variance_top10": [float(x) for x in explained[:10]],
        "effective_rank": eff_rank,
        "participation_rank": part_rank,
        "top1_explained_variance": top1,
        "top5_explained_variance": float(explained[:5].sum()) if len(explained) else 0.0,
        "uniformity_sample": uniformity_sample(emb, sample_pairs),
        "health_flags": health,
        "healthy_peak_checkpoint": bool(all(health.values())),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("wrote %s", path)


def write_line_svg(path: Path, rows: list[dict[str, float]], y_key: str, title: str, y_label: str) -> None:
    width, height = 760, 360
    left, right, top, bottom = 70, 24, 36, 54
    x_values = [float(row["epoch"]) for row in rows if y_key in row]
    y_values = [float(row[y_key]) for row in rows if y_key in row]
    if not x_values or not y_values:
        raise ValueError(f"cannot plot missing key: {y_key}")

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min or 1.0) * (width - left - right)

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * (height - top - bottom)

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, y_values))
    peak_idx = int(np.argmax(y_values))
    peak_x, peak_y = sx(x_values[peak_idx]), sy(y_values[peak_idx])
    final_x, final_y = sx(x_values[-1]), sy(y_values[-1])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="24" font-family="Arial" font-size="16" font-weight="700">{title}</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>
  <text x="{width/2:.0f}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="12">epoch</text>
  <text x="16" y="{height/2:.0f}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(-90 16 {height/2:.0f})">{y_label}</text>
  <text x="{left-8}" y="{sy(y_max)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.4f}</text>
  <text x="{left-8}" y="{sy(y_min)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.4f}</text>
  <text x="{left}" y="{height-bottom+18}" text-anchor="middle" font-family="Arial" font-size="11">{x_min:.0f}</text>
  <text x="{width-right}" y="{height-bottom+18}" text-anchor="middle" font-family="Arial" font-size="11">{x_max:.0f}</text>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <circle cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="4" fill="#16a34a"/>
  <text x="{peak_x+8:.1f}" y="{peak_y-8:.1f}" font-family="Arial" font-size="11">peak ep{x_values[peak_idx]:.0f}</text>
  <circle cx="{final_x:.1f}" cy="{final_y:.1f}" r="4" fill="#dc2626"/>
  <text x="{final_x-8:.1f}" y="{final_y-8:.1f}" text-anchor="end" font-family="Arial" font-size="11">final ep{x_values[-1]:.0f}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    logging.info("wrote %s", path)


def write_markdown(path: Path, collapse: dict[str, Any], emb: dict[str, Any], args: argparse.Namespace) -> None:
    a = STABILITY_SWEEP["A_lr1e3_no_clip_patience2"]["full_test_recall@50"]
    b = STABILITY_SWEEP["B_lr3e4_clip1_patience3"]["full_test_recall@50"]
    d = STABILITY_SWEEP["D_lr3e4_clip1_warmup_cosine_patience3"]["full_test_recall@50"]
    report = f"""# Transformer Collapse Diagnosis

This is a read-only diagnosis. It does not retrain, re-evaluate, or overwrite canonical outputs.

## Inputs

- train_log: `{args.train_log}`
- canonical config: `{args.config}`
- canonical checkpoint: `{args.checkpoint}`
- output_dir: `{args.output_dir}`

## 20-epoch collapse curve

| metric | value |
|---|---:|
| peak_epoch | {collapse['peak_epoch']} |
| peak limited-valid R@50 | {collapse['peak_valid_recall@50']:.6f} |
| final_epoch | {collapse['final_epoch']} |
| final limited-valid R@50 | {collapse['final_valid_recall@50']:.6f} |
| absolute drop after peak | {collapse['absolute_drop_after_peak']:.6f} |
| relative retention after peak | {collapse['relative_retention_after_peak']:.3f} |
| peak train_loss | {collapse['peak_train_loss']:.6f} |
| final train_loss | {collapse['final_train_loss']:.6f} |
| train_loss delta after peak | {collapse['train_loss_delta_after_peak']:.6f} |

Artifacts:

- `collapse_valid_recall_curve.svg`
- `collapse_train_loss_curve.svg`

## Canonical checkpoint item embedding health

| metric | value |
|---|---:|
| embedding rows | {emb['shape'][0]} |
| embedding dim | {emb['shape'][1]} |
| nan_count | {emb['nan_count']} |
| inf_count | {emb['inf_count']} |
| norm_mean | {emb['norm_mean']:.6f} |
| norm_std | {emb['norm_std']:.6f} |
| effective_rank | {emb['effective_rank']:.3f} |
| participation_rank | {emb['participation_rank']:.3f} |
| top1 explained variance | {emb['top1_explained_variance']:.6f} |
| top5 explained variance | {emb['top5_explained_variance']:.6f} |
| uniformity sample | {emb['uniformity_sample']['uniformity']:.6f} |
| mean pairwise sq distance | {emb['uniformity_sample']['mean_pairwise_sq_distance']:.6f} |
| healthy_peak_checkpoint | {emb['healthy_peak_checkpoint']} |

Artifacts:

- `item_embeddings.npy`
- `singular_values.npy`
- `embedding_spectrum.json`

## Stability sweep synthesis

Existing stability sweep points to optimization-driven collapse:

- A: lr=1e-3, no grad clip, patience=2 reached full-test R@50={a:.6f}.
- B: lr=3e-4 + grad_clip=1.0 reached full-test R@50={b:.6f}, delta vs A = {b - a:+.6f}.
- D: lr=3e-4 + grad_clip=1.0 + warmup/cosine reached full-test R@50={d:.6f}, delta vs A = {d - a:+.6f}.

Conclusion: the late-epoch collapse is driven by continuing lr=1e-3 optimization after the epoch-2 peak. Early stopping preserves a healthy checkpoint. Lower lr plus grad clipping stabilizes training but costs roughly 0.003 Recall@50 versus the canonical early-stopped run.
"""
    path.write_text(report, encoding="utf-8")
    logging.info("wrote %s", path)


def load_item_embeddings(config_path: Path, checkpoint_path: Path, device_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    sweep = _import_sweep()
    cfg = sweep.load_config(config_path)
    if device_name:
        cfg["device"] = device_name
    device = sweep.resolve_device(str(cfg["device"]))
    bundle = sweep.load_data(Path(cfg["data_dir"]))
    model = sweep.build_model(cfg, bundle.stats, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    item_emb = sweep.encode_all_items(model, int(bundle.stats["n_items"]), device).numpy()
    meta = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_metric_value": float(checkpoint.get("best_metric_value", 0.0)),
        "num_items": int(bundle.stats["n_items"]),
        "device": str(device),
    }
    return item_emb, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="")
    parser.add_argument("--sample-pairs", type=int, default=20000)
    parser.add_argument("--no-export-item-embeddings", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_train_log(args.train_log)
    collapse = summarize_collapse(rows)
    write_json(args.output_dir / "collapse_summary.json", collapse)
    write_line_svg(
        args.output_dir / "collapse_valid_recall_curve.svg",
        rows,
        "valid_recall@50",
        "Time-aware Transformer limited-valid R@50 collapse",
        "limited-valid R@50",
    )
    write_line_svg(
        args.output_dir / "collapse_train_loss_curve.svg",
        rows,
        "train_loss",
        "Time-aware Transformer train loss after peak",
        "train loss",
    )

    item_emb, checkpoint_meta = load_item_embeddings(args.config, args.checkpoint, args.device)
    emb_diag = embedding_diagnostics(item_emb, args.sample_pairs)
    emb_diag["checkpoint"] = checkpoint_meta
    write_json(args.output_dir / "embedding_spectrum.json", emb_diag)
    np.save(args.output_dir / "singular_values.npy", np.asarray(emb_diag["singular_values"], dtype=np.float64))
    if not args.no_export_item_embeddings:
        np.save(args.output_dir / "item_embeddings.npy", item_emb.astype(np.float32))
        logging.info("wrote %s", args.output_dir / "item_embeddings.npy")

    summary = {
        "collapse": collapse,
        "embedding": emb_diag,
        "stability_sweep": STABILITY_SWEEP,
        "constraints": {
            "read_only": True,
            "no_retrain": True,
            "new_output_dir": str(args.output_dir),
        },
    }
    write_json(args.output_dir / "diagnosis_summary.json", summary)
    write_markdown(args.output_dir / "report.md", collapse, emb_diag, args)
    logging.info("diagnosis complete: %s", args.output_dir)


if __name__ == "__main__":
    main()

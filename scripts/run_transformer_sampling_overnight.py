#!/usr/bin/env python3
"""Run the isolated industrial sampling validation queue with resumable outputs."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from select_transformer_sampling_candidates import select_candidates


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_DIR = Path("outputs/transformer_sampling_industrial_selection")
BASELINE_FULL_CONFIG = Path(
    "configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml"
)
BASELINE_FULL_CHECKPOINT = Path(
    "outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt"
)
SMOKE_SPECS = [
    ("baseline-infonce", Path("configs/transformer_logq_alpha_smoke_000.yaml")),
    ("empirical-oldlogq-alpha025", Path("configs/transformer_logq_alpha_smoke_025.yaml")),
    ("uber-batchq-alpha025", Path("configs/transformer_sampling_smoke_uber025.yaml")),
    ("uber-batchq-alpha100", Path("configs/transformer_sampling_smoke_uber100.yaml")),
    ("refined-logq", Path("configs/transformer_sampling_smoke_refined.yaml")),
    ("mns5050", Path("configs/transformer_sampling_smoke_mns5050.yaml")),
    ("mns5050-refined-logq", Path("configs/transformer_sampling_smoke_mns5050_refined.yaml")),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run(command: list[str]) -> None:
    logging.info("RUN %s", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def ensure_empty_or_missing(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {path}")


def ensure_smoke(variant: str, config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    smoke_output = Path(config["output_dir"])
    checkpoint = smoke_output / "checkpoints" / "best_model.pt"
    summary = smoke_output / "metrics_valid_best.json"
    if not summary.exists() or not checkpoint.exists():
        run([sys.executable, "scripts/train_transformer_logq_smoke.py", "--config", str(config_path)])
    audit_output = Path("outputs/transformer_sampling_industrial_smoke_audit") / variant
    audit_summary = audit_output / "audit_summary.json"
    if not audit_summary.exists():
        ensure_empty_or_missing(audit_output)
        run([
            sys.executable,
            "scripts/audit_transformer_logq_alpha_smoke.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(audit_output),
        ])
    audit = read_json(audit_summary)
    buckets = audit["target_item_popularity_bucket_recall"]
    exposure = audit["exposure_metrics"]
    return {
        "variant": variant,
        "smoke_config": str(config_path),
        "smoke_output_dir": str(smoke_output),
        "recall@50": float(audit["ranking_metrics"]["recall@50"]),
        "coverage": int(exposure["catalog_coverage"]),
        "head_share": float(exposure["topk_item_bucket_share"][">100"]),
        "non_head_bucket_recall": {
            bucket: float(buckets[bucket]["recall@50"])
            for bucket in ("1-5", "6-20", "21-100")
        },
        "audit_summary": str(audit_summary),
    }


def build_full_config(
    variant: str,
    smoke_config_path: Path,
    generated_dir: Path,
) -> Path:
    full = read_yaml(BASELINE_FULL_CONFIG)
    smoke = read_yaml(smoke_config_path)
    full.update({
        "run_label": f"transformer_sampling_full_{variant}",
        "output_dir": f"outputs/text_timeaware_transformer_sampling_full/{variant}",
        "loss_variant": smoke.get("loss_variant", "old_logq"),
        "use_logq_correction": bool(smoke.get("use_logq_correction", True)),
        "mask_duplicate_items": False,
        "q_mode": str(smoke.get("q_mode", "empirical")),
        "q_estimator": str(smoke.get("q_estimator", "empirical_frequency")),
        "logq_alpha": float(smoke.get("logq_alpha", 1.0)),
        "mns_uniform_fraction": float(smoke.get("mns_uniform_fraction", 0.5)),
    })
    path = generated_dir / f"full_{variant}.yaml"
    write_yaml(path, full)
    return path


def build_multichannel_config(
    variant: str,
    full_config_path: Path,
    checkpoint: Path,
    generated_dir: Path,
) -> Path:
    config = read_yaml(Path("configs/multichannel_transformer_final.yaml"))
    config.update({
        "train_config": str(full_config_path),
        "checkpoint": str(checkpoint),
        "output_dir": f"outputs/multichannel_transformer_sampling/{variant}",
    })
    path = generated_dir / f"multichannel_{variant}.yaml"
    write_yaml(path, config)
    return path


def ensure_full_and_downstream(
    candidate: dict[str, Any],
    generated_dir: Path,
) -> dict[str, Any]:
    variant = str(candidate["variant"])
    full_config = build_full_config(variant, Path(candidate["smoke_config"]), generated_dir)
    full_cfg = read_yaml(full_config)
    full_output = Path(full_cfg["output_dir"])
    checkpoint = full_output / "checkpoints" / "best_model.pt"
    eval_summary = Path(str(full_output) + "_full_eval") / "eval_summary.json"
    if not checkpoint.exists() or not eval_summary.exists():
        run([sys.executable, "scripts/train_transformer_logq_full.py", "--config", str(full_config)])
    full_eval = read_json(eval_summary)

    full_audit_dir = Path("outputs/transformer_sampling_full_audit") / variant
    full_audit_summary = full_audit_dir / "audit_summary.json"
    if not full_audit_summary.exists():
        ensure_empty_or_missing(full_audit_dir)
        run([
            sys.executable,
            "scripts/audit_transformer_logq_effect.py",
            "--baseline_config",
            str(BASELINE_FULL_CONFIG),
            "--baseline_checkpoint",
            str(BASELINE_FULL_CHECKPOINT),
            "--logq_config",
            str(full_config),
            "--logq_checkpoint",
            str(checkpoint),
            "--output_dir",
            str(full_audit_dir),
        ])

    multichannel_config = build_multichannel_config(variant, full_config, checkpoint, generated_dir)
    multichannel_output = Path(read_yaml(multichannel_config)["output_dir"])
    final_test = multichannel_output / "final_test_metrics.json"
    if not final_test.exists():
        ensure_empty_or_missing(multichannel_output)
        run([
            sys.executable,
            "scripts/run_multichannel_valid_selected.py",
            "--config",
            str(multichannel_config),
        ])

    faiss_output = Path("outputs/faiss_transformer_sampling") / variant
    faiss_summary = faiss_output / "faiss_benchmark_results.json"
    if not faiss_summary.exists():
        ensure_empty_or_missing(faiss_output)
        run([
            sys.executable,
            "scripts/benchmark_faiss_transformer_two_tower.py",
            "--config",
            str(full_config),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(faiss_output),
            "--expected_r50",
            str(full_eval["full_test_recall@50"]),
        ])

    return {
        "variant": variant,
        "full_config": str(full_config),
        "full_eval_summary": str(eval_summary),
        "full_audit_summary": str(full_audit_summary),
        "multichannel_final_test": str(final_test),
        "faiss_summary": str(faiss_summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated_config_dir",
        default="/workspace/server-logs/transformer_sampling_generated_configs",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    generated_dir = Path(args.generated_config_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    smoke_results = [ensure_smoke(variant, config) for variant, config in SMOKE_SPECS]
    baseline = next(item for item in smoke_results if item["variant"] == "baseline-infonce")
    if abs(float(baseline["recall@50"]) - 0.124460) > 0.0005:
        raise RuntimeError(f"Baseline smoke gate failed: {baseline['recall@50']}")
    write_json(SELECTION_DIR / "smoke_matrix.json", {"results": smoke_results})

    selection = select_candidates(smoke_results, baseline_variant="baseline-infonce")
    write_json(SELECTION_DIR / "selection.json", selection)
    downstream = [
        ensure_full_and_downstream(candidate, generated_dir)
        for candidate in selection["unique_full_candidates"]
    ]
    write_json(SELECTION_DIR / "overnight_summary.json", {
        "selection": selection,
        "downstream": downstream,
    })
    logging.info("QUEUE COMPLETE: %s", SELECTION_DIR / "overnight_summary.json")


if __name__ == "__main__":
    main()

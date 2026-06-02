#!/usr/bin/env python3
"""Run isolated low-alpha Uber batchQ smokes and exposure audits."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path("outputs/transformer_sampling_uber_lowalpha_audit")
SUMMARY_PATH = Path("outputs/transformer_sampling_uber_lowalpha_summary/summary.json")
SMOKE_SPECS = [
    ("baseline-infonce", Path("configs/transformer_logq_alpha_smoke_000.yaml")),
    ("uber-batchq-alpha025", Path("configs/transformer_sampling_smoke_uber025.yaml")),
    ("uber-batchq-alpha010", Path("configs/transformer_sampling_smoke_uber010.yaml")),
    ("uber-batchq-alpha005", Path("configs/transformer_sampling_smoke_uber005.yaml")),
    ("uber-batchq-alpha015", Path("configs/transformer_sampling_smoke_uber015.yaml")),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    logging.info("RUN %s", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def ensure_empty_or_missing(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {path}")


def summarize_audit(variant: str, config_path: Path, audit_summary: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    audit = read_json(audit_summary)
    exposure = audit["exposure_metrics"]
    buckets = audit["target_item_popularity_bucket_recall"]
    return {
        "variant": variant,
        "smoke_config": str(config_path),
        "audit_summary": str(audit_summary),
        "logq_alpha": float(config.get("logq_alpha", 0.0)),
        "recall@50": float(audit["ranking_metrics"]["recall@50"]),
        "coverage": int(exposure["catalog_coverage"]),
        "head_share": float(exposure["topk_item_bucket_share"][">100"]),
        "normalized_exposure_entropy": float(exposure["normalized_exposure_entropy"]),
        "exposure_gini": float(exposure["exposure_gini"]),
        "target_bucket_recall": {
            bucket: float(buckets[bucket]["recall@50"])
            for bucket in ("1-5", "6-20", "21-100", ">100")
        },
    }


def ensure_smoke_and_audit(variant: str, config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    smoke_output = Path(config["output_dir"])
    checkpoint = smoke_output / "checkpoints" / "best_model.pt"
    metrics_summary = smoke_output / "metrics_valid_best.json"
    if not checkpoint.exists() or not metrics_summary.exists():
        run([sys.executable, "scripts/train_transformer_logq_smoke.py", "--config", str(config_path)])

    audit_output = AUDIT_ROOT / variant
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
    return summarize_audit(variant, config_path, audit_summary)


def evaluate_candidate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    bucket_delta = {
        bucket: candidate["target_bucket_recall"][bucket] - baseline["target_bucket_recall"][bucket]
        for bucket in ("1-5", "6-20", "21-100", ">100")
    }
    constraints = {
        "recall@50": candidate["recall@50"] >= baseline["recall@50"] + 0.002,
        "coverage": candidate["coverage"] >= baseline["coverage"] * 0.95,
        "head_share": candidate["head_share"] <= 0.30,
        "bucket_1-5": bucket_delta["1-5"] >= -0.002,
        "bucket_6-20": bucket_delta["6-20"] >= -0.002,
        "bucket_21-100": bucket_delta["21-100"] >= 0.0,
        "normalized_exposure_entropy": (
            candidate["normalized_exposure_entropy"]
            >= baseline["normalized_exposure_entropy"] * 0.95
        ),
        "exposure_gini": candidate["exposure_gini"] <= baseline["exposure_gini"] * 1.05,
    }
    return {
        **candidate,
        "recall@50_delta": candidate["recall@50"] - baseline["recall@50"],
        "target_bucket_recall_delta": bucket_delta,
        "passes_gate": all(constraints.values()),
        "failed_constraints": [name for name, passed in constraints.items() if not passed],
    }


def select_pareto_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [candidate for candidate in candidates if candidate["passes_gate"]]
    if not passing:
        return None
    max_recall = max(candidate["recall@50"] for candidate in passing)
    near_best = [candidate for candidate in passing if max_recall - candidate["recall@50"] <= 0.001]
    return min(near_best, key=lambda candidate: candidate["logq_alpha"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if SUMMARY_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {SUMMARY_PATH}")

    summaries = [
        ensure_smoke_and_audit(variant, config_path)
        for variant, config_path in SMOKE_SPECS
    ]
    baseline = summaries[0]
    candidates = [evaluate_candidate(summary, baseline) for summary in summaries[2:]]
    payload = {
        "baseline": baseline,
        "reference_alpha025": summaries[1],
        "candidates": candidates,
        "selected_pareto_candidate": select_pareto_candidate(candidates),
        "selection_rule": {
            "recall@50_delta_min": 0.002,
            "coverage_ratio_min": 0.95,
            "head_share_max": 0.30,
            "bucket_1-5_delta_min": -0.002,
            "bucket_6-20_delta_min": -0.002,
            "bucket_21-100_delta_min": 0.0,
            "normalized_exposure_entropy_ratio_min": 0.95,
            "exposure_gini_ratio_max": 1.05,
            "near_best_recall_tolerance": 0.001,
        },
    }
    write_json(SUMMARY_PATH, payload)
    logging.info("WROTE %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()

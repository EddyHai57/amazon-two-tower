#!/usr/bin/env python3
"""Run gated full validation for the Uber batchQ alpha=0.10 candidate."""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("outputs/transformer_sampling_uber_alpha010_full_validation")
GENERATED_CONFIG_DIR = ROOT / "generated_configs"
BASELINE_CONFIG = Path("configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml")
BASELINE_CHECKPOINT = Path("outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt")
BASELINE_FULL_TEST_RECALL = {
    42: 0.10316836868290129,
    2024: 0.103704,
    2025: 0.096223,
}


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


def build_full_config(*, alpha: float, seed: int, label: str) -> Path:
    config = read_yaml(BASELINE_CONFIG)
    config.update({
        "run_label": label,
        "output_dir": f"outputs/text_timeaware_transformer_sampling_full/{label}",
        "loss_variant": "uber_batchq",
        "use_logq_correction": True,
        "mask_duplicate_items": False,
        "q_mode": "empirical",
        "q_estimator": "batch_appearance",
        "logq_alpha": alpha,
        "mns_uniform_fraction": 0.5,
        "seed": seed,
    })
    path = GENERATED_CONFIG_DIR / f"{label}.yaml"
    write_yaml(path, config)
    return path


def ensure_full_eval(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    output_dir = Path(config["output_dir"])
    checkpoint = output_dir / "checkpoints" / "best_model.pt"
    summary_path = Path(str(output_dir) + "_full_eval") / "eval_summary.json"
    if checkpoint.exists() != summary_path.exists():
        raise RuntimeError(f"Incomplete full run artifacts: {output_dir}")
    if not summary_path.exists():
        run([sys.executable, "scripts/train_transformer_logq_full.py", "--config", str(config_path)])
    return read_json(summary_path)


def ensure_seed42_audit(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    checkpoint = Path(config["output_dir"]) / "checkpoints" / "best_model.pt"
    output_dir = ROOT / "seed42_effect_audit"
    summary_path = output_dir / "audit_summary.json"
    if not summary_path.exists():
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {output_dir}")
        run([
            sys.executable,
            "scripts/audit_transformer_logq_effect.py",
            "--baseline_config",
            str(BASELINE_CONFIG),
            "--baseline_checkpoint",
            str(BASELINE_CHECKPOINT),
            "--logq_config",
            str(config_path),
            "--logq_checkpoint",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
        ])
    return read_json(summary_path)


def evaluate_gate0(full_test_recall: float) -> dict[str, Any]:
    delta = full_test_recall - BASELINE_FULL_TEST_RECALL[42]
    return {
        "name": "gate0_alpha000_sanity",
        "canonical_full_test_recall@50": BASELINE_FULL_TEST_RECALL[42],
        "alpha000_full_test_recall@50": full_test_recall,
        "absolute_delta": delta,
        "tolerance": 0.001,
        "passes_gate": abs(delta) < 0.001,
    }


def combine_long_tail_recall(buckets: dict[str, dict[str, int | float]]) -> float:
    hits = int(buckets["1-5"]["hits"]) + int(buckets["6-20"]["hits"])
    targets = int(buckets["1-5"]["targets"]) + int(buckets["6-20"]["targets"])
    return hits / targets if targets else 0.0


def evaluate_gate1(audit: dict[str, Any]) -> dict[str, Any]:
    baseline = audit["baseline"]
    candidate = audit["logq"]
    baseline_buckets = baseline["target_item_popularity_bucket_recall"]
    candidate_buckets = candidate["target_item_popularity_bucket_recall"]
    bucket_delta = {
        bucket: float(candidate_buckets[bucket]["recall@50"]) - float(baseline_buckets[bucket]["recall@50"])
        for bucket in ("1-5", "6-20", "21-100", ">100")
    }
    exposure = candidate["exposure_metrics"]
    baseline_coverage = int(baseline["exposure_metrics"]["catalog_coverage"])
    constraints = {
        **{f"bucket_{bucket}": bucket_delta[bucket] >= 0.0 for bucket in bucket_delta},
        "coverage": int(exposure["catalog_coverage"]) >= baseline_coverage * 0.95,
        "head_share": float(exposure["topk_item_bucket_share"][">100"]) < 0.30,
        "exposure_gini": float(exposure["exposure_gini"]) < 0.70,
    }
    return {
        "name": "gate1_alpha010_seed42_full_audit",
        "passes_gate": all(constraints.values()),
        "failed_constraints": [name for name, passed in constraints.items() if not passed],
        "ranking_metrics": candidate["ranking_metrics"],
        "exposure_metrics": exposure,
        "target_item_popularity_bucket_recall": candidate_buckets,
        "target_item_popularity_bucket_recall_delta": bucket_delta,
        "long_tail_recall@50": combine_long_tail_recall(candidate_buckets),
        "baseline_long_tail_recall@50": combine_long_tail_recall(baseline_buckets),
    }


def evaluate_gate2(candidate_recall: dict[int, float]) -> dict[str, Any]:
    paired_delta = {
        seed: candidate_recall[seed] - BASELINE_FULL_TEST_RECALL[seed]
        for seed in (42, 2024, 2025)
    }
    constraints = {
        **{f"paired_delta_seed{seed}": paired_delta[seed] > 0.0 for seed in paired_delta},
        "candidate_min_gt_canonical_min": (
            min(candidate_recall.values()) > min(BASELINE_FULL_TEST_RECALL.values())
        ),
    }
    return {
        "name": "gate2_alpha010_multiseed",
        "passes_gate": all(constraints.values()),
        "failed_constraints": [name for name, passed in constraints.items() if not passed],
        "baseline_full_test_recall@50": BASELINE_FULL_TEST_RECALL,
        "candidate_full_test_recall@50": candidate_recall,
        "paired_delta": paired_delta,
        "baseline_mean": statistics.mean(BASELINE_FULL_TEST_RECALL.values()),
        "baseline_std": statistics.pstdev(BASELINE_FULL_TEST_RECALL.values()),
        "candidate_mean": statistics.mean(candidate_recall.values()),
        "candidate_std": statistics.pstdev(candidate_recall.values()),
        "baseline_min": min(BASELINE_FULL_TEST_RECALL.values()),
        "candidate_min": min(candidate_recall.values()),
    }


def stop_after_gate(gate: dict[str, Any], payload: dict[str, Any]) -> None:
    payload["status"] = f"stopped_after_{gate['name']}"
    payload["failed_gate"] = gate
    write_json(ROOT / "final_summary.json", payload)
    logging.error("STOP %s", payload["status"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload: dict[str, Any] = {
        "candidate": "uber_batchq_alpha010",
        "scope": "full_test_and_multiseed_only",
        "downstream_not_started": ["4ch", "faiss", "canonical_replacement"],
    }

    gate0_config = build_full_config(alpha=0.0, seed=42, label="uber-batchq-alpha000-sanity-seed42")
    gate0_eval = ensure_full_eval(gate0_config)
    gate0 = evaluate_gate0(float(gate0_eval["full_test_recall@50"]))
    payload["gate0"] = gate0
    write_json(ROOT / "gate0_alpha000_sanity.json", gate0)
    if not gate0["passes_gate"]:
        stop_after_gate(gate0, payload)
        return

    seed42_config = build_full_config(alpha=0.10, seed=42, label="uber-batchq-alpha010-seed42")
    seed42_eval = ensure_full_eval(seed42_config)
    payload["seed42_full_eval"] = seed42_eval
    gate1 = evaluate_gate1(ensure_seed42_audit(seed42_config))
    payload["gate1"] = gate1
    write_json(ROOT / "gate1_alpha010_seed42_full_audit.json", gate1)
    if not gate1["passes_gate"]:
        stop_after_gate(gate1, payload)
        return

    multiseed_eval = {42: seed42_eval}
    for seed in (2024, 2025):
        config = build_full_config(alpha=0.10, seed=seed, label=f"uber-batchq-alpha010-seed{seed}")
        multiseed_eval[seed] = ensure_full_eval(config)
    payload["multiseed_full_eval"] = multiseed_eval
    gate2 = evaluate_gate2({
        seed: float(summary["full_test_recall@50"])
        for seed, summary in multiseed_eval.items()
    })
    payload["gate2"] = gate2
    write_json(ROOT / "gate2_alpha010_multiseed.json", gate2)
    payload["status"] = "completed_all_gates" if gate2["passes_gate"] else "stopped_after_gate2"
    write_json(ROOT / "final_summary.json", payload)
    logging.info("WROTE %s", ROOT / "final_summary.json")


if __name__ == "__main__":
    main()

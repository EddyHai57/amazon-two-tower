#!/usr/bin/env python3
"""Select balanced and Recall upper-bound candidates from audited sampling smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIN_RECALL = 0.124460
MIN_COVERAGE = 120315
MAX_HEAD_SHARE = 0.50
MIN_NON_HEAD_BUCKETS = 2
NON_HEAD_BUCKETS = ("1-5", "6-20", "21-100")


def passes_balanced_health(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    non_head_passes = sum(
        float(candidate["non_head_bucket_recall"][bucket])
        >= float(baseline["non_head_bucket_recall"][bucket])
        for bucket in NON_HEAD_BUCKETS
    )
    return (
        float(candidate["recall@50"]) > MIN_RECALL
        and int(candidate["coverage"]) >= MIN_COVERAGE
        and float(candidate["head_share"]) <= MAX_HEAD_SHARE
        and non_head_passes >= MIN_NON_HEAD_BUCKETS
    )


def select_candidates(results: list[dict[str, Any]], baseline_variant: str) -> dict[str, Any]:
    baseline = next(item for item in results if item["variant"] == baseline_variant)
    candidates: list[dict[str, Any]] = []
    for item in results:
        enriched = dict(item)
        enriched["balanced_health_pass"] = passes_balanced_health(item, baseline)
        candidates.append(enriched)
    experiment_candidates = [item for item in candidates if item["variant"] != baseline_variant]
    balanced = max(
        (item for item in experiment_candidates if item["balanced_health_pass"]),
        key=lambda item: float(item["recall@50"]),
        default=None,
    )
    recall_upper = max(experiment_candidates, key=lambda item: float(item["recall@50"]))
    unique = []
    for item in (balanced, recall_upper):
        if item is not None and all(existing["variant"] != item["variant"] for existing in unique):
            unique.append(item)
    return {
        "thresholds": {
            "min_recall@50": MIN_RECALL,
            "min_coverage": MIN_COVERAGE,
            "max_head_share": MAX_HEAD_SHARE,
            "min_non_head_buckets_not_below_baseline": MIN_NON_HEAD_BUCKETS,
        },
        "baseline": baseline,
        "all_candidates": candidates,
        "balanced_candidate": balanced,
        "recall_upper_bound_candidate": recall_upper,
        "unique_full_candidates": unique,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline_variant", default="baseline-infonce")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    selected = select_candidates(payload["results"], args.baseline_variant)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

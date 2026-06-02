from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_transformer_sampling_uber_alpha010_full_validation import (  # noqa: E402
    BASELINE_CHECKPOINTS,
    BASELINE_FULL_TEST_RECALL,
    evaluate_audit_gate,
    evaluate_gate0,
    evaluate_gate2,
    get_baseline_checkpoint,
)


def make_audit(
    *,
    recall: float = 0.120000,
    coverage: int = 150000,
    head_share: float = 0.28,
    gini: float = 0.68,
    bucket_1to5: float = 0.030000,
    bucket_6to20: float = 0.070000,
    bucket_21to100: float = 0.110000,
    bucket_gt100: float = 0.150000,
    ci95_low: float = 0.001,
) -> dict:
    baseline_buckets = {
        "1-5": {"recall@50": 0.025000, "targets": 100, "hits": 3},
        "6-20": {"recall@50": 0.060000, "targets": 200, "hits": 12},
        "21-100": {"recall@50": 0.100000, "targets": 300, "hits": 30},
        ">100": {"recall@50": 0.140000, "targets": 400, "hits": 56},
    }
    candidate_buckets = {
        "1-5": {"recall@50": bucket_1to5, "targets": 100, "hits": 3},
        "6-20": {"recall@50": bucket_6to20, "targets": 200, "hits": 14},
        "21-100": {"recall@50": bucket_21to100, "targets": 300, "hits": 33},
        ">100": {"recall@50": bucket_gt100, "targets": 400, "hits": 60},
    }
    return {
        "baseline": {
            "ranking_metrics": {"recall@50": BASELINE_FULL_TEST_RECALL[42]},
            "exposure_metrics": {"catalog_coverage": 152691},
            "target_item_popularity_bucket_recall": baseline_buckets,
        },
        "logq": {
            "ranking_metrics": {
                "recall@50": recall,
                "ndcg@50": 0.05,
                "mrr@50": 0.03,
            },
            "exposure_metrics": {
                "catalog_coverage": coverage,
                "topk_item_bucket_share": {">100": head_share},
                "exposure_gini": gini,
            },
            "target_item_popularity_bucket_recall": candidate_buckets,
        },
        "comparison": {
            "paired_bootstrap_ci": {
                "overall_recall@50_delta": {
                    "point_estimate": recall - BASELINE_FULL_TEST_RECALL[42],
                    "ci95_low": ci95_low,
                    "ci95_high": 0.02,
                    "num_users": 1000,
                },
            },
        },
    }


class Gate0Test(unittest.TestCase):
    def test_accepts_alpha_zero_when_full_test_matches_canonical(self) -> None:
        result = evaluate_gate0(BASELINE_FULL_TEST_RECALL[42] + 0.0004)

        self.assertTrue(result["passes_gate"])

    def test_rejects_alpha_zero_when_gap_reaches_tolerance(self) -> None:
        result = evaluate_gate0(BASELINE_FULL_TEST_RECALL[42] + 0.001)

        self.assertFalse(result["passes_gate"])


class Gate1Test(unittest.TestCase):
    def test_accepts_seed42_when_all_full_audit_constraints_pass(self) -> None:
        result = evaluate_audit_gate(make_audit(), seed=42)

        self.assertTrue(result["passes_gate"])
        self.assertEqual(result["failed_constraints"], [])
        self.assertAlmostEqual(result["long_tail_recall@50"], 17 / 300)

    def test_rejects_seed42_when_tail_or_exposure_constraints_fail(self) -> None:
        result = evaluate_audit_gate(
            make_audit(bucket_1to5=0.020000, head_share=0.31, gini=0.71),
            seed=42,
        )

        self.assertFalse(result["passes_gate"])
        self.assertIn("bucket_1-5", result["failed_constraints"])
        self.assertIn("head_share", result["failed_constraints"])
        self.assertIn("exposure_gini", result["failed_constraints"])

    def test_rejects_seed42_when_bootstrap_ci_includes_zero(self) -> None:
        result = evaluate_audit_gate(make_audit(ci95_low=0.0), seed=42)

        self.assertFalse(result["passes_gate"])
        self.assertIn("overall_recall_bootstrap_ci", result["failed_constraints"])


class BaselineCheckpointRoutingTest(unittest.TestCase):
    def test_routes_each_seed_to_historical_baseline_checkpoint(self) -> None:
        for seed in (42, 2024, 2025):
            self.assertEqual(get_baseline_checkpoint(seed), BASELINE_CHECKPOINTS[seed])

    def test_rejects_unknown_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported baseline seed"):
            get_baseline_checkpoint(7)


class Gate2Test(unittest.TestCase):
    def test_accepts_multiseed_when_each_seed_audit_passes(self) -> None:
        result = evaluate_gate2({
            42: evaluate_audit_gate(make_audit(recall=0.120000), seed=42),
            2024: evaluate_audit_gate(make_audit(recall=0.121000), seed=2024),
            2025: evaluate_audit_gate(make_audit(recall=0.119000), seed=2025),
        })

        self.assertTrue(result["passes_gate"])
        self.assertGreater(result["candidate_std"], 0.0)

    def test_rejects_multiseed_when_one_seed_audit_fails(self) -> None:
        result = evaluate_gate2({
            42: evaluate_audit_gate(make_audit(recall=0.120000), seed=42),
            2024: evaluate_audit_gate(make_audit(recall=0.121000, head_share=0.31), seed=2024),
            2025: evaluate_audit_gate(make_audit(recall=0.119000), seed=2025),
        })

        self.assertFalse(result["passes_gate"])
        self.assertIn("seed2024_audit_gate", result["failed_constraints"])


if __name__ == "__main__":
    unittest.main()

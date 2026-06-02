from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_transformer_sampling_uber_alpha010_full_validation import (  # noqa: E402
    BASELINE_FULL_TEST_RECALL,
    evaluate_gate0,
    evaluate_gate1,
    evaluate_gate2,
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
        result = evaluate_gate1(make_audit())

        self.assertTrue(result["passes_gate"])
        self.assertEqual(result["failed_constraints"], [])
        self.assertAlmostEqual(result["long_tail_recall@50"], 17 / 300)

    def test_rejects_seed42_when_tail_or_exposure_constraints_fail(self) -> None:
        result = evaluate_gate1(make_audit(bucket_1to5=0.020000, head_share=0.31, gini=0.71))

        self.assertFalse(result["passes_gate"])
        self.assertIn("bucket_1-5", result["failed_constraints"])
        self.assertIn("head_share", result["failed_constraints"])
        self.assertIn("exposure_gini", result["failed_constraints"])


class Gate2Test(unittest.TestCase):
    def test_accepts_multiseed_when_each_paired_delta_is_positive(self) -> None:
        result = evaluate_gate2({
            42: 0.120000,
            2024: 0.121000,
            2025: 0.119000,
        })

        self.assertTrue(result["passes_gate"])
        self.assertGreater(result["candidate_std"], 0.0)

    def test_rejects_multiseed_when_one_seed_does_not_improve(self) -> None:
        result = evaluate_gate2({
            42: 0.120000,
            2024: BASELINE_FULL_TEST_RECALL[2024],
            2025: 0.119000,
        })

        self.assertFalse(result["passes_gate"])
        self.assertIn("paired_delta_seed2024", result["failed_constraints"])


if __name__ == "__main__":
    unittest.main()

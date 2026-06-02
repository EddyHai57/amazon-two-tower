from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_transformer_sampling_uber_lowalpha import (  # noqa: E402
    evaluate_candidate,
    select_pareto_candidate,
)


def make_summary(
    *,
    variant: str,
    alpha: float,
    recall: float,
    coverage: int = 138000,
    head_share: float = 0.28,
    entropy: float = 0.96,
    gini: float = 0.52,
    bucket_1to5: float = 0.032,
    bucket_6to20: float = 0.072,
    bucket_21to100: float = 0.115,
    bucket_gt100: float = 0.170,
) -> dict:
    return {
        "variant": variant,
        "logq_alpha": alpha,
        "recall@50": recall,
        "coverage": coverage,
        "head_share": head_share,
        "normalized_exposure_entropy": entropy,
        "exposure_gini": gini,
        "target_bucket_recall": {
            "1-5": bucket_1to5,
            "6-20": bucket_6to20,
            "21-100": bucket_21to100,
            ">100": bucket_gt100,
        },
    }


BASELINE = make_summary(
    variant="baseline-infonce",
    alpha=0.0,
    recall=0.124460,
    coverage=141547,
    head_share=0.2097,
    entropy=1.0,
    gini=0.50,
    bucket_1to5=0.032455,
    bucket_6to20=0.072903,
    bucket_21to100=0.114014,
    bucket_gt100=0.160671,
)


class CandidateGateTest(unittest.TestCase):
    def test_accepts_candidate_that_meets_all_constraints(self) -> None:
        candidate = make_summary(
            variant="uber-batchq-alpha010",
            alpha=0.10,
            recall=0.130000,
        )

        result = evaluate_candidate(candidate, BASELINE)

        self.assertTrue(result["passes_gate"])
        self.assertEqual(result["failed_constraints"], [])

    def test_rejects_candidate_when_long_tail_and_exposure_constraints_fail(self) -> None:
        candidate = make_summary(
            variant="uber-batchq-alpha015",
            alpha=0.15,
            recall=0.140000,
            head_share=0.35,
            bucket_1to5=0.029,
        )

        result = evaluate_candidate(candidate, BASELINE)

        self.assertFalse(result["passes_gate"])
        self.assertIn("head_share", result["failed_constraints"])
        self.assertIn("bucket_1-5", result["failed_constraints"])


class ParetoSelectionTest(unittest.TestCase):
    def test_prefers_smaller_alpha_when_recall_is_within_tolerance(self) -> None:
        candidates = [
            evaluate_candidate(
                make_summary(variant="uber-batchq-alpha005", alpha=0.05, recall=0.130000),
                BASELINE,
            ),
            evaluate_candidate(
                make_summary(variant="uber-batchq-alpha010", alpha=0.10, recall=0.130800),
                BASELINE,
            ),
        ]

        selected = select_pareto_candidate(candidates)

        self.assertEqual(selected["variant"], "uber-batchq-alpha005")

    def test_returns_none_when_no_candidate_passes_gate(self) -> None:
        candidates = [
            evaluate_candidate(
                make_summary(
                    variant="uber-batchq-alpha015",
                    alpha=0.15,
                    recall=0.140000,
                    head_share=0.40,
                ),
                BASELINE,
            ),
        ]

        self.assertIsNone(select_pareto_candidate(candidates))


if __name__ == "__main__":
    unittest.main()

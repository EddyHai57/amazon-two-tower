from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from select_transformer_sampling_candidates import select_candidates  # noqa: E402


def make_result(
    variant: str,
    recall: float,
    coverage: int,
    head_share: float,
    buckets: tuple[float, float, float],
) -> dict:
    return {
        "variant": variant,
        "smoke_config": f"configs/{variant}.yaml",
        "recall@50": recall,
        "coverage": coverage,
        "head_share": head_share,
        "non_head_bucket_recall": {
            "1-5": buckets[0],
            "6-20": buckets[1],
            "21-100": buckets[2],
        },
    }


class CandidateSelectionTest(unittest.TestCase):
    def test_selects_balanced_and_recall_upper_bound(self) -> None:
        baseline = make_result("baseline", 0.124460, 141547, 0.21, (0.032, 0.073, 0.114))
        balanced = make_result("balanced", 0.164, 130000, 0.40, (0.035, 0.070, 0.130))
        head_heavy = make_result("head-heavy", 0.190, 60000, 0.75, (0.010, 0.050, 0.120))

        selected = select_candidates([baseline, balanced, head_heavy], baseline_variant="baseline")

        self.assertEqual(selected["balanced_candidate"]["variant"], "balanced")
        self.assertEqual(selected["recall_upper_bound_candidate"]["variant"], "head-heavy")
        self.assertTrue(selected["balanced_candidate"]["balanced_health_pass"])
        self.assertFalse(selected["recall_upper_bound_candidate"]["balanced_health_pass"])

    def test_collapses_duplicate_candidates(self) -> None:
        baseline = make_result("baseline", 0.124460, 141547, 0.21, (0.032, 0.073, 0.114))
        winner = make_result("winner", 0.164, 130000, 0.40, (0.035, 0.070, 0.130))

        selected = select_candidates([baseline, winner], baseline_variant="baseline")

        self.assertEqual([item["variant"] for item in selected["unique_full_candidates"]], ["winner"])


if __name__ == "__main__":
    unittest.main()

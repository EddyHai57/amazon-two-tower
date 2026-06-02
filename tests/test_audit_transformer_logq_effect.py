from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_transformer_logq_effect import (  # noqa: E402
    aggregate_exposure_metrics,
    aggregate_hit_transition,
    aggregate_popularity_bucket_recall,
    average_jaccard,
    build_test_seen_items,
    correction_stats,
    select_non_cold_eval,
)


class PopularityBucketRecallTest(unittest.TestCase):
    def test_counts_targets_and_hits_by_train_only_popularity(self) -> None:
        targets = np.array([0, 1, 2, 3])
        hit_mask = np.array([True, False, True, True])
        popularity = np.array([3, 10, 40, 200])

        result = aggregate_popularity_bucket_recall(targets, hit_mask, popularity)

        self.assertEqual(result["1-5"]["targets"], 1)
        self.assertEqual(result["1-5"]["hits"], 1)
        self.assertEqual(result["6-20"]["hits"], 0)
        self.assertEqual(result["21-100"]["recall@50"], 1.0)
        self.assertEqual(result[">100"]["recall@50"], 1.0)


class HitTransitionTest(unittest.TestCase):
    def test_separates_both_and_exclusive_hits(self) -> None:
        result = aggregate_hit_transition(
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
        )

        self.assertEqual(result, {
            "both_hit": 1,
            "baseline_only": 1,
            "logq_only": 1,
            "neither_hit": 1,
        })


class ExposureMetricsTest(unittest.TestCase):
    def test_reports_popularity_distribution_and_catalog_coverage(self) -> None:
        topk = np.array([[0, 1], [2, 3]])
        popularity = np.array([1, 10, 200, 0])

        result = aggregate_exposure_metrics(topk, popularity)

        self.assertEqual(result["catalog_coverage"], 4)
        self.assertEqual(result["topk_item_bucket_share"]["1-5"], 0.25)
        self.assertEqual(result["topk_item_bucket_share"]["6-20"], 0.25)
        self.assertEqual(result["topk_item_bucket_share"][">100"], 0.25)
        self.assertEqual(result["topk_item_bucket_share"]["unseen"], 0.25)

    def test_reports_uniform_catalog_exposure_entropy_and_gini(self) -> None:
        topk = np.array([[0, 1], [2, 3]])
        popularity = np.array([1, 10, 40, 200])

        result = aggregate_exposure_metrics(topk, popularity)

        self.assertAlmostEqual(result["normalized_exposure_entropy"], 1.0)
        self.assertAlmostEqual(result["exposure_gini"], 0.0)

    def test_exposure_concentration_includes_zero_exposure_catalog_items(self) -> None:
        topk = np.array([[0, 0], [0, 0]])
        popularity = np.array([1, 10, 40, 200])

        result = aggregate_exposure_metrics(topk, popularity)

        self.assertAlmostEqual(result["normalized_exposure_entropy"], 0.0)
        self.assertAlmostEqual(result["exposure_gini"], 0.75)


class CorrectionStatsTest(unittest.TestCase):
    def test_reports_train_item_counts_and_unseen_items(self) -> None:
        result = correction_stats(np.array([0, 1, 5, 30, 200]))

        self.assertEqual(result["train_item_count_distribution"]["min"], 0)
        self.assertEqual(result["train_item_count_distribution"]["max"], 200)
        self.assertEqual(
            result["negative_log_q_by_popularity_bucket"]["unseen"]["items"],
            1,
        )


class JaccardTest(unittest.TestCase):
    def test_averages_per_user_topk_overlap(self) -> None:
        baseline = np.array([[1, 2], [3, 4]])
        logq = np.array([[2, 5], [3, 4]])

        self.assertAlmostEqual(average_jaccard(baseline, logq), (1 / 3 + 1.0) / 2)


class EvalProtocolTest(unittest.TestCase):
    def test_test_seen_items_merge_train_and_valid(self) -> None:
        train = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [1, 2, 3]})
        valid = pd.DataFrame({"user_idx": [0, 1], "item_idx": [4, 5]})

        seen = build_test_seen_items(train, valid)

        self.assertEqual(seen[0], {1, 2, 4})
        self.assertEqual(seen[1], {3, 5})

    def test_select_non_cold_eval_excludes_cold_target(self) -> None:
        test = pd.DataFrame({
            "user_idx": [0, 1],
            "item_idx": [2, 3],
            "is_cold_item_for_eval": [False, True],
        })

        selected = select_non_cold_eval(test)

        self.assertEqual(selected["user_idx"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()

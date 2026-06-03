from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_transformer_collapse import (  # noqa: E402
    effective_rank_from_singular_values,
    find_peak_row,
    participation_rank_from_singular_values,
    read_train_log,
    summarize_collapse,
    uniformity_sample,
)


class CollapseTrainLogTest(unittest.TestCase):
    def test_summarize_collapse_uses_peak_and_final_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_log.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "valid_recall@50"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"epoch": 1, "train_loss": 6.8, "valid_recall@50": 0.114},
                        {"epoch": 2, "train_loss": 6.2, "valid_recall@50": 0.124},
                        {"epoch": 3, "train_loss": 5.9, "valid_recall@50": 0.100},
                    ]
                )

            rows = read_train_log(path)
            peak = find_peak_row(rows)
            summary = summarize_collapse(rows)

        self.assertEqual(peak["epoch"], 2)
        self.assertEqual(summary["peak_epoch"], 2)
        self.assertEqual(summary["final_epoch"], 3)
        self.assertAlmostEqual(summary["absolute_drop_after_peak"], -0.024)
        self.assertAlmostEqual(summary["train_loss_delta_after_peak"], -0.3)


class EmbeddingMathTest(unittest.TestCase):
    def test_effective_and_participation_rank_for_equal_singular_values(self) -> None:
        singular_values = np.ones(4)

        self.assertAlmostEqual(effective_rank_from_singular_values(singular_values), 4.0, places=6)
        self.assertAlmostEqual(participation_rank_from_singular_values(singular_values), 4.0, places=6)

    def test_uniformity_sample_returns_finite_value(self) -> None:
        embeddings = np.eye(4, dtype=np.float64)
        result = uniformity_sample(embeddings, sample_pairs=20, seed=123)

        self.assertEqual(result["sample_pairs"], 20)
        self.assertTrue(np.isfinite(result["uniformity"]))
        self.assertGreater(result["mean_pairwise_sq_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()

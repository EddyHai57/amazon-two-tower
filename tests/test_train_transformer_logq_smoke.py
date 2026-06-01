from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_transformer_logq_smoke import (  # noqa: E402
    apply_logq_and_duplicate_mask,
    build_log_q,
    summarize_batch_duplicates,
)


class BuildLogQTest(unittest.TestCase):
    def test_build_log_q_uses_train_frequency_and_clamps_unseen_items(self) -> None:
        log_q = build_log_q(torch.tensor([0, 0, 1, 2, 2, 2]), num_items=4)

        expected_q = torch.tensor([2.0, 1.0, 3.0, 1.0]) / 7.0
        torch.testing.assert_close(log_q.exp(), expected_q)


class ApplyCorrectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        self.items = torch.tensor([2, 5, 2])
        self.log_q = torch.log(torch.tensor([0.05, 0.10, 0.20, 0.05, 0.10, 0.50]))

    def test_logq_subtracts_candidate_column_log_probability(self) -> None:
        corrected = apply_logq_and_duplicate_mask(
            self.logits,
            self.items,
            self.log_q,
            use_logq=True,
            mask_duplicate_items=False,
        )

        expected = self.logits - self.log_q[self.items].unsqueeze(0)
        torch.testing.assert_close(corrected, expected)

    def test_duplicate_mask_blocks_same_item_off_diagonal_only(self) -> None:
        corrected = apply_logq_and_duplicate_mask(
            self.logits,
            self.items,
            self.log_q,
            use_logq=False,
            mask_duplicate_items=True,
        )

        mask_value = torch.finfo(self.logits.dtype).min
        self.assertEqual(float(corrected[0, 2]), mask_value)
        self.assertEqual(float(corrected[2, 0]), mask_value)
        self.assertEqual(float(corrected[0, 0]), 1.0)
        self.assertEqual(float(corrected[2, 2]), 9.0)

    def test_no_correction_returns_original_logits(self) -> None:
        corrected = apply_logq_and_duplicate_mask(
            self.logits,
            self.items,
            self.log_q,
            use_logq=False,
            mask_duplicate_items=False,
        )

        torch.testing.assert_close(corrected, self.logits)

    def test_combined_path_masks_duplicates_and_keeps_finite_logits(self) -> None:
        corrected = apply_logq_and_duplicate_mask(
            self.logits,
            self.items,
            self.log_q,
            use_logq=True,
            mask_duplicate_items=True,
        )

        self.assertTrue(bool(torch.isfinite(corrected).all()))
        self.assertEqual(float(corrected[0, 2]), torch.finfo(self.logits.dtype).min)
        self.assertEqual(float(corrected[2, 0]), torch.finfo(self.logits.dtype).min)
        self.assertAlmostEqual(
            float(corrected[0, 1]),
            float(self.logits[0, 1] - self.log_q[self.items[1]]),
        )


class DuplicateStatsTest(unittest.TestCase):
    def test_duplicate_stats_count_extra_rows(self) -> None:
        stats = summarize_batch_duplicates(torch.tensor([2, 5, 2, 7, 7, 7]))

        self.assertEqual(stats["rows"], 6)
        self.assertEqual(stats["unique_items"], 3)
        self.assertEqual(stats["duplicate_rows"], 3)
        self.assertAlmostEqual(stats["duplicate_row_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()

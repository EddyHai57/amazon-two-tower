from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformer_sampling_losses import (  # noqa: E402
    apply_old_logq_and_duplicate_mask,
    build_item_log_q,
    compute_mns_softmax_loss,
    compute_refined_logq_loss,
    sample_mns_candidate_ids,
)
from train_transformer_logq_smoke import compute_sampling_loss  # noqa: E402


class TinyTwoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(4, 4)
        self.item_emb = nn.Embedding(8, 4)
        self.use_l2_norm = True

    def raw_batch(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del history_item_idx
        return self.user_emb(user_idx), self.item_emb(item_idx)

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_emb(item_idx), p=2, dim=-1)


class BuildItemLogQTest(unittest.TestCase):
    def test_empirical_frequency_clamps_unseen_items(self) -> None:
        log_q = build_item_log_q(
            torch.tensor([0, 0, 1, 2, 2, 2]),
            num_items=4,
            q_estimator="empirical_frequency",
            batch_size=4,
        )

        expected = torch.tensor([2.0, 1.0, 3.0, 1.0]) / 7.0
        torch.testing.assert_close(log_q.exp(), expected)

    def test_batch_appearance_uses_uber_probability(self) -> None:
        log_q = build_item_log_q(
            torch.tensor([0, 0, 1, 2, 2, 2]),
            num_items=4,
            q_estimator="batch_appearance",
            batch_size=4,
        )

        frequency = torch.tensor([2.0, 1.0, 3.0, 1.0]) / 7.0
        expected = 1.0 - torch.pow(1.0 - frequency, 4)
        torch.testing.assert_close(log_q.exp(), expected)


class OldLogQTest(unittest.TestCase):
    def test_old_logq_matches_column_correction(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        items = torch.tensor([0, 1])
        log_q = torch.log(torch.tensor([0.25, 0.75]))

        corrected = apply_old_logq_and_duplicate_mask(
            logits,
            items,
            log_q,
            use_logq=True,
            mask_duplicate_items=False,
            logq_alpha=0.25,
        )

        torch.testing.assert_close(
            corrected,
            logits - 0.25 * log_q[items].unsqueeze(0),
        )


class RefinedLogQTest(unittest.TestCase):
    def test_refined_loss_matches_paper_formula(self) -> None:
        positive_scores = torch.tensor([2.0, 1.0])
        negative_scores = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
        positive_items = torch.tensor([0, 1])
        negative_items = torch.tensor([0, 1])
        log_q = torch.log(torch.tensor([0.25, 0.75]))

        loss = compute_refined_logq_loss(
            positive_scores,
            negative_scores,
            positive_items,
            negative_items,
            log_q,
        )

        q_prime = torch.tensor([1.0, 1.0])
        corrected_negative = torch.tensor([0.0, 0.5]) - torch.log(q_prime)
        original = corrected_negative - positive_scores
        sample_weight = torch.sigmoid(corrected_negative - torch.log(torch.tensor(2.0)) - positive_scores)
        expected = (sample_weight.detach() * original).mean()
        torch.testing.assert_close(loss, expected)

    def test_refined_loss_masks_target_collisions(self) -> None:
        loss = compute_refined_logq_loss(
            torch.tensor([1.0]),
            torch.tensor([[5.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.log(torch.tensor([0.5, 0.5])),
        )

        self.assertTrue(bool(torch.isfinite(loss)))


class MixedNegativeSamplingTest(unittest.TestCase):
    def test_candidate_sampling_is_reproducible(self) -> None:
        items = torch.tensor([0, 1, 2, 3])
        first = sample_mns_candidate_ids(
            items,
            num_items=10,
            uniform_fraction=0.5,
            generator=torch.Generator().manual_seed(2026),
        )
        second = sample_mns_candidate_ids(
            items,
            num_items=10,
            uniform_fraction=0.5,
            generator=torch.Generator().manual_seed(2026),
        )

        torch.testing.assert_close(first, second)
        self.assertEqual(first.numel(), items.numel())

    def test_mns_masks_target_collisions(self) -> None:
        positive_scores = torch.tensor([1.0])
        negative_scores = torch.tensor([[100.0, 0.0]])
        loss = compute_mns_softmax_loss(
            positive_scores,
            negative_scores,
            torch.tensor([0]),
            torch.tensor([0, 1]),
        )

        expected = F.cross_entropy(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        torch.testing.assert_close(loss, expected)


class SamplingDispatchTest(unittest.TestCase):
    def test_all_loss_variants_produce_finite_single_batch_loss(self) -> None:
        torch.manual_seed(42)
        model = TinyTwoTower()
        users = torch.tensor([0, 1, 2, 3])
        items = torch.tensor([0, 1, 2, 3])
        histories = torch.zeros((4, 1), dtype=torch.long)
        log_q = build_item_log_q(
            torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7]),
            num_items=8,
            q_estimator="empirical_frequency",
            batch_size=4,
        )

        for variant in ("infonce", "old_logq", "uber_batchq", "refined_logq", "mns", "mns_refined_logq"):
            with self.subTest(variant=variant):
                loss, stats = compute_sampling_loss(
                    model,
                    users,
                    items,
                    histories,
                    temperature=0.15,
                    log_q=log_q,
                    cfg={
                        "loss_variant": variant,
                        "mask_duplicate_items": False,
                        "logq_alpha": 0.25,
                        "mns_uniform_fraction": 0.5,
                    },
                    num_items=8,
                )

                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertEqual(stats["rows"], 4)


if __name__ == "__main__":
    unittest.main()

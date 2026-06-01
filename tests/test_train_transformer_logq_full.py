from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_transformer_logq_full import (  # noqa: E402
    train_one_step,
    validate_logq_config,
)
from train_transformer_logq_smoke import build_log_q  # noqa: E402


class TinyTwoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(3, 4)
        self.item_emb = nn.Embedding(3, 4)
        self.use_l2_norm = True

    def raw_batch(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del history_item_idx
        return self.user_emb(user_idx), self.item_emb(item_idx)


class ValidateConfigTest(unittest.TestCase):
    def test_accepts_logq_only(self) -> None:
        validate_logq_config({
            "use_logq_correction": True,
            "mask_duplicate_items": False,
        })

    def test_rejects_duplicate_mask_for_full_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "mask_duplicate_items"):
            validate_logq_config({
                "use_logq_correction": True,
                "mask_duplicate_items": True,
            })

    def test_rejects_disabled_logq_for_full_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "use_logq_correction"):
            validate_logq_config({
                "use_logq_correction": False,
                "mask_duplicate_items": False,
            })


class TrainStepTest(unittest.TestCase):
    def test_single_batch_logq_loss_is_finite(self) -> None:
        torch.manual_seed(42)
        model = TinyTwoTower()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 1, 2])
        histories = torch.zeros((3, 1), dtype=torch.long)
        log_q = build_log_q(torch.tensor([0, 0, 1, 2]), num_items=3)

        loss, duplicate_stats = train_one_step(
            model,
            optimizer,
            users,
            items,
            histories,
            temperature=0.15,
            log_q=log_q,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(duplicate_stats["rows"], 3)
        self.assertEqual(duplicate_stats["duplicate_rows"], 0)


if __name__ == "__main__":
    unittest.main()

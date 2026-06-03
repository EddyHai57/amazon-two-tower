from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_transformer_maxlen100_smoke import (  # noqa: E402
    TextTwoTowerTransformerSmoke,
    require_config,
)


def base_config(pooling_type: str) -> dict[str, object]:
    return {
        "data_dir": "data/processed/movies_tv_5core",
        "output_dir": "outputs/transformer_gain_attribution/test",
        "embedding_dim": 4,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.000001,
        "epochs": 3,
        "temperature": 0.15,
        "use_l2_norm": True,
        "seed": 42,
        "eval_k_list": [20, 50],
        "eval_batch_size": 2,
        "eval_max_users": 10,
        "num_workers": 0,
        "device": "cpu",
        "save_best_by": "valid_recall@50",
        "history_max_len": 5,
        "history_weight": 1.0,
        "item_text_embedding_path": "unused.npy",
        "item_has_text_path": "unused.npy",
        "text_proj_dim": 4,
        "use_has_text_mask": True,
        "item_fusion": "additive",
        "pooling_type": pooling_type,
        "decay_rate": 0.8,
    }


class MeanPoolTimeawareConfigTest(unittest.TestCase):
    def test_mean_pool_timeaware_is_valid_pooling_type(self) -> None:
        require_config(base_config("mean_pool_timeaware"))


class MeanPoolTimeawareBehaviorTest(unittest.TestCase):
    def test_mean_pool_timeaware_adds_position_and_recency_without_transformer_encoder(self) -> None:
        model = TextTwoTowerTransformerSmoke(
            num_users=2,
            num_items=5,
            embedding_dim=4,
            text_emb=torch.zeros(5, 4),
            has_text=torch.ones(5),
            text_proj_dim=4,
            use_l2_norm=False,
            use_has_text_mask=True,
            history_weight=1.0,
            pooling_type="mean_pool_timeaware",
            decay_rate=0.8,
            max_len=3,
            num_heads=2,
            ffn_dim=8,
            dropout=0.0,
            num_layers=1,
        )
        self.assertFalse(hasattr(model, "transformer_encoder"))
        self.assertTrue(hasattr(model, "mean_pool_timeaware_encoder"))

        with torch.no_grad():
            model.item_id_embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0],
                        [0.0, 0.0, 0.0, 4.0],
                    ]
                )
            )
            model.mean_pool_timeaware_encoder.pos_embedding.weight.zero_()
            model.mean_pool_timeaware_encoder.recency_embedding.weight.zero_()
            model.mean_pool_timeaware_encoder.pos_embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.1, 0.0, 0.0, 0.0],
                        [0.0, 0.2, 0.0, 0.0],
                        [0.0, 0.0, 0.3, 0.0],
                    ]
                )
            )
            model.mean_pool_timeaware_encoder.recency_embedding.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.4])
            model.mean_pool_timeaware_encoder.recency_embedding.weight[1] = torch.tensor([0.0, 0.0, 0.5, 0.0])
            model.mean_pool_timeaware_encoder.recency_embedding.weight[2] = torch.tensor([0.0, 0.6, 0.0, 0.0])

        history = torch.tensor([[1, 2, 3], [4, -1, -1]])
        user_emb = torch.zeros(2, 4)
        pooled = model._pool_history(history, user_emb)

        expected_first = torch.stack(
            [
                torch.tensor([1.1, 0.6, 0.0, 0.0]),
                torch.tensor([0.0, 2.2, 0.5, 0.0]),
                torch.tensor([0.0, 0.0, 3.3, 0.4]),
            ]
        ).mean(dim=0)
        expected_second = torch.tensor([0.1, 0.0, 0.0, 4.4])
        expected = torch.stack([expected_first, expected_second])

        torch.testing.assert_close(pooled, expected)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_transformer_logq_alpha_smoke import select_limited_valid_eval  # noqa: E402


class LimitedValidEvalSelectionTest(unittest.TestCase):
    def test_excludes_cold_targets_before_taking_head_users(self) -> None:
        valid = pd.DataFrame({
            "user_idx": [0, 1, 2, 3],
            "item_idx": [10, 11, 12, 13],
            "is_cold_item_for_eval": [True, False, False, False],
        })

        selected = select_limited_valid_eval(valid, max_users=2)

        self.assertEqual(selected["user_idx"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()

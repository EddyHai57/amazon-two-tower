#!/usr/bin/env python3
"""Semi-hard HNM lambda=0.01 smoke — thin wrapper over the lambda-generic implementation.

This variant uses lambda_hn=0.01 (vs 0.03 in the baseline semi-hard smoke).
All logic lives in train_text_mean_pool_semi_hard_negative_smoke; this file
exists for per-variant reproducibility and is paired with its own config:
  configs/two_tower_movies_tv_5core_text_mean_pool_semi_hnm_lambda001_smoke.yaml

The only behavioral difference from the lambda=0.03 run is lambda_hn, which is
read from the YAML config — no code change required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_text_mean_pool_semi_hard_negative_smoke import main  # noqa: E402

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RRF k sensitivity check (valid set only).

Fixed: text_w=0.3, pop_w=0.5, icf_w=1.0, tt_w=1.0
k values: 100, 150, 200, 300

Generates valid candidates once, then evaluates each k.
No test eval. No config changes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from run_multichannel_valid_selected import generate_candidates, load_shared_resources
from run_multichannel_retrieval_v3 import run_eval_with_diversity, weighted_rrf_merge_n


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    setup_logging()

    config_path = Path("configs/multichannel_valid_selected.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path("outputs/k_sensitivity_check/k_sensitivity_valid.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    K_VALUES = [100, 150, 200, 300]
    ICF_W, TT_W, TEXT_W, POP_W = 1.0, 1.0, 0.3, 0.5
    weights = [ICF_W, TT_W, TEXT_W, POP_W]

    k_list = [int(k) for k in config["eval_k_list"]]
    rrf_top_n = int(config["rrf_top_n"])
    raw_buckets = config.get("popularity_buckets", [[1, 5], [6, 20], [21, 100], [101, None]])
    pop_buckets = [(int(lo), None if hi is None else int(hi)) for lo, hi in raw_buckets]

    logging.info("=== RRF k Sensitivity Check (valid set only) ===")
    logging.info("Fixed weights: icf=%.1f tt=%.1f text=%.1f pop=%.1f", ICF_W, TT_W, TEXT_W, POP_W)
    logging.info("k values: %s", K_VALUES)

    # Load shared resources (model, data, ItemCF, embeddings)
    shared = load_shared_resources(config)
    eval_df = shared["valid_eval_df"]
    eval_targets = eval_df.copy()
    item_popularity = shared["item_popularity"]

    # Generate valid candidates once
    logging.info("[Phase] Generating valid candidates (once for all k)...")
    icf_cands, tt_cands, text_cands, pop_cands = generate_candidates(
        shared, eval_df,
        shared["valid_seen"], shared["icf_valid_seen"],
        shared["valid_history_matrix"], config, "Valid"
    )
    logging.info("[Phase] Candidates ready. Running k sweep...")

    results = []
    for k in K_VALUES:
        name = f"wrrf_k{k}_text{TEXT_W:.1f}_pop{POP_W:.1f}"
        res = run_eval_with_diversity(
            name, eval_targets,
            lambda u, w=weights, kk=k: weighted_rrf_merge_n(
                [icf_cands.get(u, []), tt_cands.get(u, []),
                 text_cands.get(u, []), pop_cands.get(u, [])],
                w, kk, rrf_top_n,
            ),
            k_list, item_popularity, pop_buckets,
        )
        res["wrrf_k"] = k
        res["text_w"] = TEXT_W
        res["pop_w"] = POP_W
        m = res["metrics"]
        bk = res.get("bucket_breakdown", {})
        logging.info(
            "k=%3d  Recall@50=%.6f  NDCG@50=%.6f  MRR@50=%.6f  avg_pop=%.0f  coverage=%d"
            "  ≤5=%.4f  6-20=%.4f  21-100=%.4f  >100=%.4f",
            k,
            m["recall@50"], m["ndcg@50"], m["mrr@50"],
            m["avg_rec_popularity"], m["item_coverage"],
            bk.get("1-5", {}).get("recall@50", 0),
            bk.get("6-20", {}).get("recall@50", 0),
            bk.get("21-100", {}).get("recall@50", 0),
            bk.get(">100", {}).get("recall@50", 0),
        )
        results.append(res)

    # Print summary table
    logging.info("")
    logging.info("=== Summary Table (valid set, text=0.3, pop=0.5) ===")
    logging.info("%-8s %-10s %-10s %-10s %-8s %-10s %-8s %-8s %-8s %-8s",
                 "k", "R@50", "NDCG@50", "MRR@50", "avg_pop", "coverage",
                 "≤5", "6-20", "21-100", ">100")
    for res in results:
        k = res["wrrf_k"]
        m = res["metrics"]
        bk = res.get("bucket_breakdown", {})
        logging.info(
            "%-8d %-10.6f %-10.6f %-10.6f %-8.1f %-10d %-8.4f %-8.4f %-8.4f %-8.4f",
            k,
            m["recall@50"], m["ndcg@50"], m["mrr@50"],
            m["avg_rec_popularity"], m["item_coverage"],
            bk.get("1-5", {}).get("recall@50", 0),
            bk.get("6-20", {}).get("recall@50", 0),
            bk.get("21-100", {}).get("recall@50", 0),
            bk.get(">100", {}).get("recall@50", 0),
        )

    with open(output_path, "w") as f:
        json.dump({"k_values": K_VALUES, "fixed_weights": {
            "icf_w": ICF_W, "tt_w": TT_W, "text_w": TEXT_W, "pop_w": POP_W,
        }, "results": results}, f, indent=2)
    logging.info("Saved to %s", output_path)
    logging.info("=== k sensitivity check DONE ===")


if __name__ == "__main__":
    main()

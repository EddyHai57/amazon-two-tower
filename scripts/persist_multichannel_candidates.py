#!/usr/bin/env python3
"""
Candidate persistence + overlap@100 + RRF attribution audit.

Purpose:
  - Persist 4-channel top-200 test candidates for reproducibility
  - Compute overlap@50 and overlap@100 (overlap@100 was previously unavailable)
  - Rebuild final RRF top-50 and verify Recall@50 == 0.104776
  - Compute RRF score contribution share per channel
  - Compute hit attribution (multi-source + fractional credit + exclusive)
  - No training, no weight tuning, no overwriting existing results.

Frozen config (from valid-selected):
  ItemCF w=1.0, TwoTower w=1.0, Text w=0.3, Popularity w=0.5, k=100, top_n=50

Output:
  outputs/multichannel_valid_selected/candidates_top200/
  outputs/multichannel_valid_selected/candidate_audit/
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_multichannel_valid_selected import generate_candidates, load_shared_resources
from run_multichannel_retrieval import (
    TRAIN_COLUMNS,
    aggregate_metrics,
    compute_metrics,
)
from run_multichannel_retrieval_v3 import weighted_rrf_merge_n

# ── constants ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/multichannel_valid_selected.yaml")
BASE_DIR = Path("outputs/multichannel_valid_selected")
CANDS_DIR = BASE_DIR / "candidates_top200"
AUDIT_DIR = BASE_DIR / "candidate_audit"

# Frozen final config
FROZEN = dict(icf_w=1.0, tt_w=1.0, text_w=0.3, pop_w=0.5, rrf_k=100, top_n=50)
WEIGHTS = [FROZEN["icf_w"], FROZEN["tt_w"], FROZEN["text_w"], FROZEN["pop_w"]]
CHANNELS = ["icf", "tt", "text", "pop"]
K_RRF = FROZEN["rrf_k"]
TOP_N = FROZEN["top_n"]

# Expected final metrics (for verification)
EXPECTED_R50 = 0.10477571655890587
EXPECTED_NDCG50 = 0.04159936572342471
EXPECTED_MRR50 = 0.02565650645006445

# Previous overlap@50 from v2 analysis (for alignment check)
PREV_OVERLAP_50 = {
    "icf-tt": 0.076189, "icf-text": 0.008302, "icf-pop": 0.038155,
    "tt-text": 0.013534, "tt-pop": 0.003273, "text-pop": 0.000447,
}
PREV_UNIQUE_HITS_50 = {
    "icf": 9979, "tt": 14624, "text": 4788, "pop": 13816,
}

SAMPLE_USERS = 1000  # users to save in detail attribution parquet


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def dicts_to_matrix(cands_dict: dict[int, list[int]], user_idx: np.ndarray, top_k: int) -> np.ndarray:
    """Convert {user: item_list} to [n_users, top_k] int32 array, pad with -1."""
    mat = np.full((len(user_idx), top_k), -1, dtype=np.int32)
    for i, u in enumerate(user_idx):
        items = cands_dict.get(int(u), [])
        n = min(len(items), top_k)
        if n > 0:
            mat[i, :n] = items[:n]
    return mat


def pairwise_jaccard(a_mat: np.ndarray, b_mat: np.ndarray, k: int) -> tuple[float, float]:
    """Compute mean per-user Jaccard@k and mean intersection count."""
    n = len(a_mat)
    jaccard_sum = 0.0
    inter_sum = 0.0
    for i in range(n):
        a_set = set(int(x) for x in a_mat[i, :k] if x >= 0)
        b_set = set(int(x) for x in b_mat[i, :k] if x >= 0)
        if not a_set and not b_set:
            continue
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        inter_sum += inter
        if union > 0:
            jaccard_sum += inter / union
    return jaccard_sum / n, inter_sum / n


def rebuild_rrf_and_attribute(
    icf_mat: np.ndarray,
    tt_mat: np.ndarray,
    text_mat: np.ndarray,
    pop_mat: np.ndarray,
    user_idx: np.ndarray,
    eval_targets: pd.DataFrame,
    test_seen: dict[int, set[int]],
    k_list: list[int],
) -> dict[str, Any]:
    """
    Rebuild final RRF top-50 for all test users.
    Returns aggregated verification metrics + attribution stats.
    """
    targets_map = dict(zip(eval_targets["user_idx"].tolist(), eval_targets["item_idx"].tolist()))
    n_users = len(user_idx)

    # Aggregation accumulators
    per_user_metrics: list[dict[str, float]] = []

    # Source channel distribution: item in how many channels' top-200
    source_count_dist: dict[int, int] = defaultdict(int)   # {n_channels: n_items}

    # RRF score contribution (sum over all final top-50 items × users)
    ch_score_total: dict[str, float] = {ch: 0.0 for ch in CHANNELS}
    grand_score_total = 0.0

    # Hit attribution
    multi_source_hits: dict[str, int] = {ch: 0 for ch in CHANNELS}
    frac_hits: dict[str, float] = {ch: 0.0 for ch in CHANNELS}
    excl_hits_top200: dict[str, int] = {ch: 0 for ch in CHANNELS}  # exclusive from top-200
    excl_hits_top50: dict[str, int] = {ch: 0 for ch in CHANNELS}   # exclusive from top-50
    total_hits = 0

    # Detailed sample rows for parquet
    sample_rows: list[dict[str, Any]] = []

    matrices = {"icf": icf_mat, "tt": tt_mat, "text": text_mat, "pop": pop_mat}

    for i, u in enumerate(user_idx):
        u_int = int(u)
        target = targets_map.get(u_int)
        if target is None:
            continue

        # Build per-channel candidate lists
        ch_cands: dict[str, list[int]] = {}
        ch_rank: dict[str, dict[int, int]] = {}  # ch -> {item: 1-indexed rank in top200}
        ch_rank50: dict[str, dict[int, int]] = {}  # ch -> {item: 1-indexed rank in top50}
        for ch, mat in matrices.items():
            row = mat[i]
            items = [int(x) for x in row if x >= 0]
            ch_cands[ch] = items
            ch_rank[ch] = {item: rank for rank, item in enumerate(items, 1)}
            ch_rank50[ch] = {item: rank for rank, item in enumerate(items[:50], 1)}

        # Weighted RRF: rebuild final top-50
        scores: defaultdict[int, float] = defaultdict(float)
        for ch, w in zip(CHANNELS, WEIGHTS):
            for rank, item in enumerate(ch_cands[ch], 1):
                scores[item] += w / (K_RRF + rank)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        final_top50 = [item for item, _ in ranked[:TOP_N]]

        # Recall / NDCG / MRR
        metrics = compute_metrics(final_top50, target, k_list)
        per_user_metrics.append(metrics)

        # Source channel attribution for each final top-50 item
        is_hit = target in set(final_top50)
        if is_hit:
            total_hits += 1

        # Determine which channels have each final top-50 item in their top-200
        target_in_ch_200 = {ch: target in ch_rank[ch] for ch in CHANNELS}
        target_in_ch_50 = {ch: target in ch_rank50[ch] for ch in CHANNELS}

        for rank_pos, item in enumerate(final_top50, 1):
            in_channels = [ch for ch in CHANNELS if item in ch_rank[ch]]
            nc = len(in_channels)
            source_count_dist[nc] += 1

            total_score = scores[item]
            grand_score_total += total_score
            for ch in CHANNELS:
                if item in ch_rank[ch]:
                    contrib = WEIGHTS[CHANNELS.index(ch)] / (K_RRF + ch_rank[ch][item])
                else:
                    contrib = 0.0
                ch_score_total[ch] += contrib

            # Sample parquet rows (first SAMPLE_USERS)
            if i < SAMPLE_USERS:
                row_data = {
                    "user_idx": u_int,
                    "target_item": target,
                    "final_rank": rank_pos,
                    "item_idx": item,
                    "is_target": int(item == target),
                    "total_score": float(total_score),
                    "icf_rank_200": ch_rank["icf"].get(item, 0),
                    "tt_rank_200": ch_rank["tt"].get(item, 0),
                    "text_rank_200": ch_rank["text"].get(item, 0),
                    "pop_rank_200": ch_rank["pop"].get(item, 0),
                }
                for ch, w in zip(CHANNELS, WEIGHTS):
                    rk = ch_rank[ch].get(item, 0)
                    row_data[f"{ch}_score"] = float(w / (K_RRF + rk)) if rk > 0 else 0.0
                row_data["n_source_channels"] = nc
                sample_rows.append(row_data)

        # Hit attribution
        if is_hit:
            # Which channels had target in top-200?
            ch_with_target_200 = [ch for ch in CHANNELS if target_in_ch_200[ch]]
            # Which channels had target in top-50?
            ch_with_target_50 = [ch for ch in CHANNELS if target_in_ch_50[ch]]
            n_ch_200 = len(ch_with_target_200)
            n_ch_50 = len(ch_with_target_50)

            # Multi-source: each channel with target in top-200 gets +1
            for ch in ch_with_target_200:
                multi_source_hits[ch] += 1
            # Fractional: split by number of channels with target in top-200
            if n_ch_200 > 0:
                frac = 1.0 / n_ch_200
                for ch in ch_with_target_200:
                    frac_hits[ch] += frac
            # Exclusive from top-200
            if n_ch_200 == 1:
                excl_hits_top200[ch_with_target_200[0]] += 1
            # Exclusive from top-50
            if n_ch_50 == 1:
                excl_hits_top50[ch_with_target_50[0]] += 1

        if (i + 1) % 50000 == 0:
            logging.info("Attribution progress: %d / %d", i + 1, n_users)

    # Aggregate recall
    agg = aggregate_metrics(per_user_metrics, k_list)
    agg["n_eval_users"] = n_users

    # Score contribution share
    score_share = {ch: ch_score_total[ch] / max(grand_score_total, 1e-12) for ch in CHANNELS}

    # Multi-source hit share
    multi_total = sum(multi_source_hits.values()) or 1
    multi_share = {ch: multi_source_hits[ch] / multi_total for ch in CHANNELS}

    # Fractional hit share
    frac_total = sum(frac_hits.values()) or 1
    frac_share = {ch: frac_hits[ch] / frac_total for ch in CHANNELS}

    return {
        "verification": {
            "recall@50": agg.get("recall@50", 0),
            "ndcg@50": agg.get("ndcg@50", 0),
            "mrr@50": agg.get("mrr@50", 0),
            "recall@50_expected": EXPECTED_R50,
            "recall@50_match": abs(agg.get("recall@50", 0) - EXPECTED_R50) < 1e-6,
            "n_eval_users": n_users,
        },
        "source_channel_dist": dict(source_count_dist),
        "rrf_score_share": score_share,
        "rrf_score_total": {ch: ch_score_total[ch] for ch in CHANNELS},
        "hit_attribution": {
            "total_hits": total_hits,
            "total_recall50": agg.get("recall@50", 0),
            "multi_source_hits": multi_source_hits,
            "multi_source_share": multi_share,
            "fractional_hits": {ch: round(frac_hits[ch], 4) for ch in CHANNELS},
            "fractional_share": frac_share,
            "exclusive_hits_top200": excl_hits_top200,
            "exclusive_hits_top50": excl_hits_top50,
        },
        "previous_exclusive_hits_top50_v2": PREV_UNIQUE_HITS_50,
        "sample_rows": sample_rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CANDS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)
    config["eval_max_users"] = None

    k_list = [int(k) for k in config["eval_k_list"]]

    # ── load shared resources ──────────────────────────────────────────────
    logging.info("=== Loading shared resources ===")
    shared = load_shared_resources(config)

    test_eval_df = shared["test_eval_df"]
    test_user_idx = test_eval_df["user_idx"].to_numpy(dtype=np.int64)
    n_users = len(test_user_idx)
    logging.info("Test eval users: %d", n_users)

    # ── generate test candidates ───────────────────────────────────────────
    logging.info("=== Generating test top-200 candidates ===")
    icf_cands, tt_cands, text_cands, pop_cands = generate_candidates(
        shared, test_eval_df,
        shared["test_seen"], shared["icf_test_seen"],
        shared["test_history_matrix"], config, "Test",
    )
    logging.info("Candidates generated: ICF=%d TT=%d Text=%d Pop=%d users",
                 len(icf_cands), len(tt_cands), len(text_cands), len(pop_cands))

    # ── persist to numpy arrays ────────────────────────────────────────────
    logging.info("=== Persisting candidates ===")
    top_k = int(config["candidates_per_channel"])
    icf_mat = dicts_to_matrix(icf_cands, test_user_idx, top_k)
    tt_mat = dicts_to_matrix(tt_cands, test_user_idx, top_k)
    text_mat = dicts_to_matrix(text_cands, test_user_idx, top_k)
    pop_mat = dicts_to_matrix(pop_cands, test_user_idx, top_k)

    np.save(str(CANDS_DIR / "test_user_idx.npy"), test_user_idx)
    np.save(str(CANDS_DIR / "candidates_icf.npy"), icf_mat)
    np.save(str(CANDS_DIR / "candidates_tt.npy"), tt_mat)
    np.save(str(CANDS_DIR / "candidates_text.npy"), text_mat)
    np.save(str(CANDS_DIR / "candidates_pop.npy"), pop_mat)

    icf_size = (CANDS_DIR / "candidates_icf.npy").stat().st_size
    logging.info("Candidates saved: shape=%s  icf_size=%.1f MB", icf_mat.shape, icf_size / 1e6)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "num_users": int(n_users),
        "topk": int(top_k),
        "channels": CHANNELS,
        "source_config": str(CONFIG_PATH),
        "frozen_config": FROZEN,
        "seen_mask_policy": "test masks train+valid seen items; target never masked",
        "history_policy": "test uses train+valid interactions, max_len=20",
        "icf_policy": "train interactions only, itemcf_max_user_history=100, sim_topk=100",
        "text_policy": "time_decay_rate=0.8, item_text_emb L2-normalized",
        "pop_policy": "train-count sorted, pop_buffer_size=1000, filtered by seen",
        "files": {
            "test_user_idx.npy": "int64 [n_users]",
            "candidates_icf.npy": "int32 [n_users, 200] padded with -1",
            "candidates_tt.npy": "int32 [n_users, 200] padded with -1",
            "candidates_text.npy": "int32 [n_users, 200] padded with -1",
            "candidates_pop.npy": "int32 [n_users, 200] padded with -1",
        },
    }
    write_json(CANDS_DIR / "metadata.json", metadata)
    logging.info("Metadata saved.")

    # ── compute overlaps ───────────────────────────────────────────────────
    logging.info("=== Computing pairwise overlap ===")
    channel_mats = {"icf": icf_mat, "tt": tt_mat, "text": text_mat, "pop": pop_mat}
    pairs = [
        ("icf", "tt"), ("icf", "text"), ("icf", "pop"),
        ("tt", "text"), ("tt", "pop"), ("text", "pop"),
    ]
    overlap_results: dict[str, Any] = {"num_users": int(n_users)}
    overlap_rows = []
    for k_eval in [50, 100]:
        for a_name, b_name in pairs:
            pkey = f"{a_name}-{b_name}"
            jaccard, avg_inter = pairwise_jaccard(
                channel_mats[a_name], channel_mats[b_name], k_eval
            )
            overlap_results[f"jaccard_{pkey}@{k_eval}"] = round(float(jaccard), 6)
            overlap_results[f"avg_intersection_{pkey}@{k_eval}"] = round(float(avg_inter), 4)
            logging.info("overlap@%d  %s vs %s: jaccard=%.6f  avg_inter=%.2f",
                         k_eval, a_name, b_name, jaccard, avg_inter)
            if k_eval == 50:
                prev = PREV_OVERLAP_50.get(pkey, None)
                delta_str = f"  prev={prev:.6f}  delta={jaccard - prev:+.6f}" if prev else ""
                logging.info("  @50 comparison with v2%s", delta_str)
            overlap_rows.append({
                "pair": pkey,
                "k": k_eval,
                "jaccard": round(float(jaccard), 6),
                "avg_intersection": round(float(avg_inter), 4),
            })

    write_json(AUDIT_DIR / "overlap_metrics.json", overlap_results)
    write_csv(AUDIT_DIR / "overlap_metrics.csv", overlap_rows,
              ["pair", "k", "jaccard", "avg_intersection"])
    logging.info("Overlap metrics saved.")

    # ── rebuild RRF + attribution ──────────────────────────────────────────
    logging.info("=== Rebuilding final RRF top-50 and computing attribution ===")
    attr = rebuild_rrf_and_attribute(
        icf_mat, tt_mat, text_mat, pop_mat,
        test_user_idx, test_eval_df, shared["test_seen"], k_list,
    )

    v = attr["verification"]
    logging.info("Verification: R@50=%.6f (expected=%.6f match=%s)  NDCG@50=%.6f  MRR@50=%.6f",
                 v["recall@50"], v["recall@50_expected"], v["recall@50_match"],
                 v["ndcg@50"], v["mrr@50"])
    if not v["recall@50_match"]:
        logging.warning("Recall@50 mismatch! Diff=%.2e", abs(v["recall@50"] - EXPECTED_R50))

    # Score share
    logging.info("RRF score share: %s", {ch: f"{s:.4f}" for ch, s in attr["rrf_score_share"].items()})
    # Hit attribution
    ha = attr["hit_attribution"]
    logging.info("Total hits: %d  Recall@50=%.6f", ha["total_hits"], ha["total_recall50"])
    logging.info("Multi-source hit share: %s", {ch: f"{ha['multi_source_share'][ch]:.4f}" for ch in CHANNELS})
    logging.info("Fractional hit share: %s", {ch: f"{ha['fractional_share'][ch]:.4f}" for ch in CHANNELS})
    logging.info("Exclusive hits (top200): %s", ha["exclusive_hits_top200"])
    logging.info("Exclusive hits (top50):  %s  [prev v2: %s]",
                 ha["exclusive_hits_top50"], PREV_UNIQUE_HITS_50)

    # Save attribution JSON (without sample_rows)
    attr_save = {k: v for k, v in attr.items() if k != "sample_rows"}
    write_json(AUDIT_DIR / "rrf_attribution.json", attr_save)

    # Attribution CSV summary
    attr_rows = []
    for ch in CHANNELS:
        attr_rows.append({
            "channel": ch,
            "weight": WEIGHTS[CHANNELS.index(ch)],
            "rrf_score_share": round(attr["rrf_score_share"][ch], 6),
            "multi_source_hits": ha["multi_source_hits"][ch],
            "multi_source_share": round(ha["multi_source_share"][ch], 6),
            "fractional_hits": round(ha["fractional_hits"][ch], 2),
            "fractional_share": round(ha["fractional_share"][ch], 6),
            "exclusive_hits_top200": ha["exclusive_hits_top200"][ch],
            "exclusive_hits_top50": ha["exclusive_hits_top50"][ch],
            "exclusive_hits_top50_prev_v2": PREV_UNIQUE_HITS_50.get(ch, ""),
        })
    write_csv(AUDIT_DIR / "rrf_attribution.csv", attr_rows, list(attr_rows[0].keys()))

    # Sample parquet
    if attr["sample_rows"]:
        df_sample = pd.DataFrame(attr["sample_rows"])
        df_sample.to_parquet(str(AUDIT_DIR / "final_top50_attribution_sample.parquet"), index=False)
        logging.info("Sample parquet saved: %d rows (%d users)",
                     len(df_sample), min(SAMPLE_USERS, n_users))

    # ── final summary ──────────────────────────────────────────────────────
    logging.info("")
    logging.info("=== AUDIT COMPLETE ===")
    logging.info("Candidates:   %s", CANDS_DIR)
    logging.info("Audit:        %s", AUDIT_DIR)
    logging.info("R@50 verify:  %.6f (expected %.6f)  PASS=%s",
                 v["recall@50"], EXPECTED_R50, v["recall@50_match"])
    sd = attr["source_channel_dist"]
    total_items = sum(sd.values())
    logging.info("Source dist:  1ch=%d(%.1f%%) 2ch=%d(%.1f%%) 3ch=%d(%.1f%%) 4ch=%d(%.1f%%)",
                 sd.get(1, 0), 100 * sd.get(1, 0) / total_items,
                 sd.get(2, 0), 100 * sd.get(2, 0) / total_items,
                 sd.get(3, 0), 100 * sd.get(3, 0) / total_items,
                 sd.get(4, 0), 100 * sd.get(4, 0) / total_items)
    logging.info("Score share:  %s", {ch: f"{attr['rrf_score_share'][ch]:.3f}" for ch in CHANNELS})
    logging.info("Frac share:   %s", {ch: f"{ha['fractional_share'][ch]:.3f}" for ch in CHANNELS})


if __name__ == "__main__":
    main()

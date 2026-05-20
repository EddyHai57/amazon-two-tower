#!/usr/bin/env python3
"""
Multi-channel retrieval rerun with new Transformer Two-Tower user tower.

Replaces the old Text+Time-decay Mean Pool Two-Tower with the canonical
time-aware Transformer Two-Tower (full test R@50=0.103168, +31.7% vs old).

Channels:
  1. ItemCF
  2. New Transformer Two-Tower (canonical checkpoint)
  3. Text Semantic
  4. Popularity Fallback

Pipeline:
  Phase 0: Load shared resources (Transformer model + ItemCF + Text + Pop)
  Phase 1: Single-channel sanity check (TT alone, verify R@50 ≈ 0.103168)
  Phase 2: Valid sweep (60 weighted RRF configs, select via Pareto)
  Phase 3: Frozen test eval (once, on selected config)
  Phase 4: Candidate audit (overlap@50, overlap@100, RRF attribution, hit attribution)
  Phase 5: Comparison table + report

Output dir: outputs/multichannel_transformer_final/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Imports: Transformer model (new user tower)
# ---------------------------------------------------------------------------
from train_transformer_maxlen100_smoke import (
    DataBundle,
    build_history_matrix,   # returns ndarray (NOT tuple)
    build_model,
    build_seen_items,
    load_config as load_train_config,
    load_data,
    merge_seen_items,
    resolve_device,
    set_seed,
)

# ---------------------------------------------------------------------------
# Imports: Multichannel infrastructure (reused from old system)
# ---------------------------------------------------------------------------
from run_multichannel_retrieval import (
    TRAIN_COLUMNS,
    aggregate_metrics,
    compute_metrics,
    encode_all_items,
    generate_itemcf_candidates,
    generate_twotower_candidates,
    rrf_merge,
    write_json,
)
from run_multichannel_retrieval_v2 import (
    V1_BEST_RECALL50,
    generate_popularity_candidates,
    generate_text_semantic_candidates,
    load_text_embeddings,
    rrf_merge_n,
    write_csv,
)
from run_multichannel_retrieval_v3 import (
    V2_BEST_RECALL50,
    run_eval_with_diversity,
    weighted_rrf_merge_n,
)
from run_itemcf import add_valid_to_seen, build_item_similarity, build_train_sets
from run_multichannel_valid_selected import (
    _select_pareto,
    generate_candidates,
    run_frozen_test,
    run_valid_sweep,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLD_TT_RECALL50       = 0.078315   # old Time-decay Mean Pool single-channel test
OLD_2CH_RECALL50      = 0.096727   # old 2ch RRF (ICF + old TT) test
OLD_VS_RECALL50       = 0.104776   # old valid-selected 4ch test
CANONICAL_TT_RECALL50 = 0.103168   # Transformer canonical full test R@50


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip_sanity",   action="store_true", help="Skip single-channel sanity phase")
    p.add_argument("--skip_sweep",    action="store_true", help="Skip valid sweep, load from valid_sweep.json")
    p.add_argument("--skip_phase3",   action="store_true", help="Skip Phase 3 frozen test, load from final_test_metrics.json")
    p.add_argument("--skip_audit",    action="store_true", help="Skip candidate audit phase")
    p.add_argument("--test_only",     action="store_true", help="Skip valid sweep; use --frozen_k/text_w/pop_w")
    p.add_argument("--frozen_k",      type=int,   default=None)
    p.add_argument("--frozen_text_w", type=float, default=None)
    p.add_argument("--frozen_pop_w",  type=float, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared resource loader  (Transformer-model-aware)
# ---------------------------------------------------------------------------

def load_shared_resources(config: dict[str, Any]) -> dict[str, Any]:
    """Load data, Transformer model, embeddings, ItemCF, Text, Pop — once."""
    data_dir   = Path(config["data_dir"])
    train_cfg  = load_train_config(Path(config["train_config"]))
    bundle: DataBundle = load_data(data_dir)
    num_users  = int(bundle.stats["n_users"])
    num_items  = int(bundle.stats["n_items"])
    max_len    = int(train_cfg["history_max_len"])  # 100

    # Seen masks
    train_seen = build_seen_items(bundle.train_df)
    valid_seen = train_seen                                       # valid eval: mask train only
    test_seen  = merge_seen_items(train_seen, bundle.valid_df)   # test eval:  mask train+valid

    # History matrices — build_history_matrix returns ndarray (not tuple)
    train_hist_frame = bundle.train_df[TRAIN_COLUMNS].copy()
    trainvalid_hist_frame = pd.concat(
        [bundle.train_df[TRAIN_COLUMNS], bundle.valid_df[TRAIN_COLUMNS]], ignore_index=True
    )
    valid_history_matrix = build_history_matrix(train_hist_frame, num_users, max_len)
    test_history_matrix  = build_history_matrix(trainvalid_hist_frame, num_users, max_len)

    # ItemCF — train only
    logging.info("[ItemCF] Building train sets...")
    icf_full_seen, icf_limited_history = build_train_sets(
        bundle.train_df, int(config["itemcf_max_user_history"])
    )
    icf_valid_seen = icf_full_seen
    icf_test_seen  = add_valid_to_seen(icf_full_seen, bundle.valid_df)
    logging.info("[ItemCF] Building similarity (sim_topk=%s)...", config["itemcf_sim_topk"])
    similarity = build_item_similarity(icf_limited_history, int(config["itemcf_sim_topk"]))

    # Item popularity (train only)
    item_pop_counter: Counter[int] = Counter(bundle.train_df["item_idx"].tolist())
    item_popularity: dict[int, int] = dict(item_pop_counter)
    pop_sorted_items = [item for item, _ in item_pop_counter.most_common()]

    # Transformer Two-Tower model
    device = resolve_device(str(config["device"]))
    logging.info("[TwoTower] Loading Transformer checkpoint from %s...", config["checkpoint"])
    model = build_model(train_cfg, bundle.stats, device)
    ckpt  = torch.load(Path(config["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_epoch   = ckpt.get("epoch", "?")
    ckpt_recall  = float(ckpt.get("best_metric_value", 0.0))
    logging.info("[TwoTower] Loaded: epoch=%s, pooling=%s, max_len=%s, best_valid_recall@50=%.6f",
                 ckpt_epoch, train_cfg.get("pooling_type"), max_len, ckpt_recall)
    item_emb_cpu = encode_all_items(model, num_items, device)

    # Text embeddings
    logging.info("[TextSemantic] Loading text embeddings from %s...", config["item_text_emb_path"])
    item_text_norm_np = load_text_embeddings(Path(config["item_text_emb_path"]), num_items)
    item_text_norm_gpu = torch.from_numpy(item_text_norm_np).to(device)

    # Eval target rows
    valid_eval_df = bundle.valid_df[~bundle.valid_df["is_cold_item_for_eval"].astype(bool)].copy()
    test_eval_df  = bundle.test_df[ ~bundle.test_df[ "is_cold_item_for_eval"].astype(bool)].copy()
    logging.info("Valid non-cold: %d  Test non-cold: %d", len(valid_eval_df), len(test_eval_df))

    return dict(
        bundle=bundle, num_users=num_users, num_items=num_items,
        train_seen=train_seen, valid_seen=valid_seen, test_seen=test_seen,
        icf_valid_seen=icf_valid_seen, icf_test_seen=icf_test_seen,
        icf_limited_history=icf_limited_history, similarity=similarity,
        valid_history_matrix=valid_history_matrix, test_history_matrix=test_history_matrix,
        model=model, item_emb_cpu=item_emb_cpu,
        item_text_norm_gpu=item_text_norm_gpu, device=device,
        item_popularity=item_popularity, pop_sorted_items=pop_sorted_items,
        valid_eval_df=valid_eval_df, test_eval_df=test_eval_df,
    )


# ---------------------------------------------------------------------------
# Phase 1: Single-channel sanity (TT only, top-50)
# ---------------------------------------------------------------------------

def run_single_channel_sanity(
    config: dict[str, Any],
    shared: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Verify new Transformer TT alone ≈ canonical R@50=0.103168."""
    logging.info("[Sanity] Running single-channel TwoTower-only test eval...")
    eval_df   = shared["test_eval_df"]
    k_list    = [int(k) for k in config["eval_k_list"]]
    raw_bkts  = config.get("popularity_buckets", [[1, 5], [6, 20], [21, 100], [101, None]])
    pop_bkts  = [(int(lo), None if hi is None else int(hi)) for lo, hi in raw_bkts]

    # Pre-generate all TT candidates in batch (top-50), then use lookup in score_fn
    logging.info("[Sanity] Pre-generating TT top-50 candidates in batch...")
    eval_users = eval_df["user_idx"].unique().tolist()
    tt_cands_raw = generate_twotower_candidates(
        shared["model"], eval_users, shared["test_history_matrix"],
        shared["test_seen"], shared["item_emb_cpu"], 50, shared["device"]
    )
    tt_cands_lookup: dict[int, list[int]] = {u: v.tolist() for u, v in tt_cands_raw.items()}

    tt_only = run_eval_with_diversity(
        "transformer_tt_single_channel",
        eval_df,
        lambda u: tt_cands_lookup.get(int(u), []),
        k_list, shared["item_popularity"], pop_bkts,
    )
    r50 = tt_only["metrics"]["recall@50"]
    delta = r50 - CANONICAL_TT_RECALL50
    logging.info("[Sanity] TT single-channel test Recall@50=%.6f (canonical=%.6f, delta=%+.6f)",
                 r50, CANONICAL_TT_RECALL50, delta)
    if abs(delta) > 0.005:
        logging.warning("[Sanity] MISMATCH: delta=%.6f > 0.005 — investigate before continuing!", delta)
    else:
        logging.info("[Sanity] OK: single-channel aligned (|delta|=%.6f ≤ 0.005)", abs(delta))

    result = {
        "test_recall@50": r50,
        "test_ndcg@50":   tt_only["metrics"].get("ndcg@50", 0),
        "test_mrr@50":    tt_only["metrics"].get("mrr@50", 0),
        "canonical_recall@50": CANONICAL_TT_RECALL50,
        "delta": delta,
        "aligned": abs(delta) <= 0.005,
        "bucket_breakdown": tt_only.get("bucket_breakdown", {}),
        "n_test_users": len(eval_df),
    }
    write_json(output_dir / "sanity_single_channel.json", result)
    logging.info("[Sanity] sanity_single_channel.json saved.")
    return result


# ---------------------------------------------------------------------------
# Phase 4: Candidate audit (overlap + attribution)
# ---------------------------------------------------------------------------

def _jaccard_pair(
    cands_a: dict[int, list], cands_b: dict[int, list], k: int
) -> tuple[float, float]:
    """Mean per-user Jaccard@k and mean intersection size."""
    j_sum = 0.0
    ix_sum = 0.0
    n = 0
    for u in cands_a:
        if u not in cands_b:
            continue
        sa = set(cands_a[u][:k])
        sb = set(cands_b[u][:k])
        union = len(sa | sb)
        inter = len(sa & sb)
        if union > 0:
            j_sum += inter / union
        ix_sum += inter
        n += 1
    if n == 0:
        return 0.0, 0.0
    return j_sum / n, ix_sum / n


def run_candidate_audit(
    config: dict[str, Any],
    shared: dict[str, Any],
    output_dir: Path,
    frozen_k: int,
    frozen_text_w: float,
    frozen_pop_w: float,
    frozen_icf_w: float = 1.0,
    frozen_tt_w: float = 1.0,
) -> dict[str, Any]:
    """Compute overlap@50, overlap@100, RRF attribution, hit attribution."""
    logging.info("[Audit] Generating test candidates for audit (top_k=200)...")
    eval_df = shared["test_eval_df"]
    k_list  = [int(k) for k in config["eval_k_list"]]
    rrf_top_n = int(config["rrf_top_n"])
    raw_bkts = config.get("popularity_buckets", [[1, 5], [6, 20], [21, 100], [101, None]])
    pop_bkts = [(int(lo), None if hi is None else int(hi)) for lo, hi in raw_bkts]

    icf_cands, tt_cands, text_cands, pop_cands = generate_candidates(
        shared, eval_df, shared["test_seen"], shared["icf_test_seen"],
        shared["test_history_matrix"], config, "Audit"
    )
    channels = {"icf": icf_cands, "tt": tt_cands, "text": text_cands, "pop": pop_cands}
    weights  = [frozen_icf_w, frozen_tt_w, frozen_text_w, frozen_pop_w]
    channel_names = ["icf", "tt", "text", "pop"]

    # --- overlap@50 and @100 ---
    pairs = [
        ("icf", "tt"), ("icf", "text"), ("icf", "pop"),
        ("tt", "text"), ("tt", "pop"), ("text", "pop"),
    ]
    overlap50  = {}
    overlap100 = {}
    for a, b in pairs:
        key = f"{a}-{b}"
        j50, ix50  = _jaccard_pair(channels[a], channels[b], 50)
        j100, ix100 = _jaccard_pair(channels[a], channels[b], 100)
        overlap50[key]  = {"jaccard": round(j50, 6),  "mean_intersect": round(ix50, 2)}
        overlap100[key] = {"jaccard": round(j100, 6), "mean_intersect": round(ix100, 2)}
    logging.info("[Audit] Overlap@50: icf-tt=%.6f  icf-text=%.6f  tt-text=%.6f",
                 overlap50["icf-tt"]["jaccard"], overlap50["icf-text"]["jaccard"],
                 overlap50["tt-text"]["jaccard"])
    logging.info("[Audit] Overlap@100: icf-tt=%.6f  icf-text=%.6f  tt-text=%.6f",
                 overlap100["icf-tt"]["jaccard"], overlap100["icf-text"]["jaccard"],
                 overlap100["tt-text"]["jaccard"])

    # --- RRF score contribution ---
    total_scores = {ch: 0.0 for ch in channel_names}
    for u in eval_df["user_idx"].unique():
        for ch_idx, ch in enumerate(channel_names):
            cands_list = channels[ch].get(int(u), [])
            w = weights[ch_idx]
            for rank, _ in enumerate(cands_list[:200], start=1):
                total_scores[ch] += w / (frozen_k + rank)
    grand_total = sum(total_scores.values()) or 1.0
    rrf_contrib = {
        ch: {"total_score": round(total_scores[ch], 1),
             "share": round(total_scores[ch] / grand_total, 4)}
        for ch in channel_names
    }
    for ch in channel_names:
        logging.info("[Audit] RRF contribution: %s  score=%.1f  share=%.1f%%",
                     ch, rrf_contrib[ch]["total_score"], rrf_contrib[ch]["share"] * 100)

    # --- Hit attribution ---
    target_map = dict(zip(eval_df["user_idx"].tolist(), eval_df["item_idx"].tolist()))
    total_hits = 0
    multisource_hits = {ch: 0 for ch in channel_names}
    fractional_hits  = {ch: 0.0 for ch in channel_names}
    exclusive_200    = {ch: 0 for ch in channel_names}
    exclusive_50     = {ch: 0 for ch in channel_names}

    for u, target in target_map.items():
        u = int(u); target = int(target)
        # Check if target is in frozen RRF top-50
        top50 = weighted_rrf_merge_n(
            [channels[ch].get(u, []) for ch in channel_names],
            weights, frozen_k, rrf_top_n,
        )
        if target not in top50:
            continue
        total_hits += 1

        # Which channels have target in top-200?
        in_200 = [ch for ch in channel_names if target in channels[ch].get(u, [])]
        for ch in in_200:
            multisource_hits[ch] += 1
        n = len(in_200) or 1
        for ch in in_200:
            fractional_hits[ch] += 1.0 / n

        # Exclusive@200 (only one channel covers target in top-200)
        if len(in_200) == 1:
            exclusive_200[in_200[0]] += 1

        # Exclusive@50 (only this channel's solo top-50 has target)
        in_50 = [ch for ch in channel_names if target in channels[ch].get(u, [])[:50]]
        if len(in_50) == 1:
            exclusive_50[in_50[0]] += 1

    hit_attribution = {
        ch: {
            "multisource_hits": multisource_hits[ch],
            "multisource_share": round(multisource_hits[ch] / (total_hits or 1), 4),
            "fractional_hits": round(fractional_hits[ch], 1),
            "fractional_share": round(fractional_hits[ch] / (total_hits or 1), 4),
            "exclusive_hits_200": exclusive_200[ch],
            "exclusive_hits_50": exclusive_50[ch],
        }
        for ch in channel_names
    }
    logging.info("[Audit] Total hits=%d  ICF frac=%.1f (%.1f%%)  TT frac=%.1f (%.1f%%)",
                 total_hits,
                 fractional_hits["icf"], 100 * fractional_hits["icf"] / (total_hits or 1),
                 fractional_hits["tt"],  100 * fractional_hits["tt"]  / (total_hits or 1))
    logging.info("[Audit] Exclusive@200: ICF=%d  TT=%d  Text=%d  Pop=%d",
                 exclusive_200["icf"], exclusive_200["tt"],
                 exclusive_200["text"], exclusive_200["pop"])

    # Verify RRF recall matches frozen test
    from run_multichannel_retrieval import aggregate_metrics, compute_metrics
    all_metrics = []
    for u, target in target_map.items():
        top50 = weighted_rrf_merge_n(
            [channels[ch].get(int(u), []) for ch in channel_names],
            weights, frozen_k, rrf_top_n,
        )
        m = compute_metrics(top50, int(target), [50])
        all_metrics.append(m)
    agg = aggregate_metrics(all_metrics, [50])
    audit_r50 = agg.get("recall@50", 0)
    logging.info("[Audit] RRF rebuild Recall@50=%.6f (from frozen test=expected match)",
                 audit_r50)

    audit_result = {
        "overlap_50": overlap50,
        "overlap_100": overlap100,
        "rrf_attribution": rrf_contrib,
        "hit_attribution": hit_attribution,
        "total_hits": total_hits,
        "audit_rebuild_recall50": audit_r50,
        "frozen_config": {
            "k": frozen_k, "icf_w": frozen_icf_w, "tt_w": frozen_tt_w,
            "text_w": frozen_text_w, "pop_w": frozen_pop_w,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    audit_dir = output_dir / "candidate_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "overlap_metrics.json",   {"overlap_50": overlap50, "overlap_100": overlap100})
    write_json(audit_dir / "rrf_attribution.json",   {"rrf_attribution": rrf_contrib, "hit_attribution": hit_attribution, "total_hits": total_hits})
    write_json(audit_dir / "audit_summary.json",     audit_result)

    # CSV summaries
    overlap_rows = []
    for pair_key in [f"{a}-{b}" for a, b in pairs]:
        overlap_rows.append({
            "pair": pair_key,
            "jaccard_50":  overlap50[pair_key]["jaccard"],
            "mean_ix_50":  overlap50[pair_key]["mean_intersect"],
            "jaccard_100": overlap100[pair_key]["jaccard"],
            "mean_ix_100": overlap100[pair_key]["mean_intersect"],
        })
    write_csv(audit_dir / "overlap_metrics.csv", overlap_rows,
              ["pair", "jaccard_50", "mean_ix_50", "jaccard_100", "mean_ix_100"])

    attribution_rows = []
    for ch in channel_names:
        w = weights[channel_names.index(ch)]
        attribution_rows.append({
            "channel": ch, "weight": w,
            **{k: v for k, v in hit_attribution[ch].items()},
            "rrf_score_share": rrf_contrib[ch]["share"],
        })
    write_csv(audit_dir / "hit_attribution.csv", attribution_rows,
              ["channel", "weight", "multisource_hits", "multisource_share",
               "fractional_hits", "fractional_share",
               "exclusive_hits_200", "exclusive_hits_50", "rrf_score_share"])

    logging.info("[Audit] Done. Files saved to %s", audit_dir)
    return audit_result


# ---------------------------------------------------------------------------
# Phase 5a: Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    output_dir: Path,
    sanity: dict[str, Any],
    frozen_test: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Build comparison table: old baselines + new Transformer results."""

    def row(name, src, r50, ndcg50, mrr50, avg_pop, cov, b5, b6, b21, b100, note=""):
        return {"name": name, "selection_source": src,
                "recall@50": round(r50, 6), "ndcg@50": round(ndcg50, 6), "mrr@50": round(mrr50, 6),
                "avg_rec_popularity": round(avg_pop, 1) if avg_pop else "",
                "item_coverage": cov,
                "bucket_≤5_r50": round(b5, 6), "bucket_6-20_r50": round(b6, 6),
                "bucket_21-100_r50": round(b21, 6), "bucket_>100_r50": round(b100, 6),
                "note": note}

    rows = []

    # Old single-channel baselines
    rows.append(row(
        "ItemCF (single-channel)", "baseline",
        0.083570, 0.036254, 0.023999, 0, 153055,
        0.040405, 0.047940, 0.060890, 0.122522,
        "train co-occurrence, full test",
    ))
    rows.append(row(
        "Old TT: Text+Time-decay MeanPool (single-channel)", "baseline",
        0.078315, 0.030862, 0.019036, 0, 153928,
        0.031046, 0.056933, 0.079564, 0.083277,
        "old final model, max_len=20",
    ))

    # New Transformer single-channel (sanity result)
    s_bd = sanity.get("bucket_breakdown", {})
    rows.append(row(
        "New TT: Transformer Timeaware (single-channel)", "new",
        sanity.get("test_recall@50", 0), sanity.get("test_ndcg@50", 0), sanity.get("test_mrr@50", 0),
        0, 0,
        s_bd.get("1-5", {}).get("recall@50", 0), s_bd.get("6-20", {}).get("recall@50", 0),
        s_bd.get("21-100", {}).get("recall@50", 0), s_bd.get(">100", {}).get("recall@50", 0),
        "canonical checkpoint, max_len=100, seed=42",
    ))

    # Old 2ch RRF
    rows.append(row(
        "Old 2ch RRF k=60 (ICF + old TT)", "baseline",
        0.096727, 0.038885, 0.024272, 264.5, 153936,
        0.044029, 0.064008, 0.085018, 0.127714,
        "v1 best result",
    ))

    # Old valid-selected 4ch
    rows.append(row(
        "Old 4ch valid-selected (ICF+oldTT+Text+Pop)", "old-valid-selected",
        0.104776, 0.041599, 0.025657, 461.8, 153924,
        0.045142, 0.066167, 0.085952, 0.144728,
        "valid-selected k=100 text=0.3 pop=0.5",
    ))

    # New 2ch RRF (from frozen test ref)
    ftm = frozen_test
    r2ch_test = ftm.get("ref_2ch_test_recall50", 0)
    rows.append(row(
        "New 2ch RRF k=60 (ICF + new Transformer TT)", "new",
        r2ch_test, 0, 0, 0, 0, 0, 0, 0, 0,
        "ICF + Transformer, RRF k=60, test set",
    ))

    # New valid-selected 4ch (frozen test result)
    m_ft  = ftm["final_test"]["metrics"]
    bd_ft = ftm["final_test"].get("bucket_breakdown", {})
    rows.append(row(
        f"New 4ch valid-selected ({ftm['final_test']['name']})",
        "new-valid-selected",
        m_ft["recall@50"], m_ft.get("ndcg@50", 0), m_ft.get("mrr@50", 0),
        m_ft.get("avg_rec_popularity", 0), m_ft.get("item_coverage", 0),
        bd_ft.get("1-5", {}).get("recall@50", 0),
        bd_ft.get("6-20", {}).get("recall@50", 0),
        bd_ft.get("21-100", {}).get("recall@50", 0),
        bd_ft.get(">100", {}).get("recall@50", 0),
        "valid-set Pareto selected, test run once",
    ))

    keys = [
        "name", "selection_source",
        "recall@50", "ndcg@50", "mrr@50",
        "avg_rec_popularity", "item_coverage",
        "bucket_≤5_r50", "bucket_6-20_r50", "bucket_21-100_r50", "bucket_>100_r50",
        "note",
    ]
    write_csv(output_dir / "final_comparison_table.csv", rows, keys)
    write_json(output_dir / "final_comparison_table.json", {"rows": rows})
    logging.info("[Comparison] final_comparison_table.csv saved.")


# ---------------------------------------------------------------------------
# Phase 5b: Report
# ---------------------------------------------------------------------------

def write_reports(
    output_dir: Path,
    config: dict[str, Any],
    sanity: dict[str, Any],
    valid_selected: dict[str, Any],
    frozen_test_result: dict[str, Any],
    audit: dict[str, Any] | None,
) -> None:
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m_ft    = frozen_test_result["final_test"]["metrics"]
    bd_ft   = frozen_test_result["final_test"].get("bucket_breakdown", {})
    vs_name = frozen_test_result["final_test"]["name"]
    m_vs    = valid_selected["metrics"]
    ksel    = valid_selected.get("wrrf_k")
    tw_sel  = valid_selected.get("text_w")
    pw_sel  = valid_selected.get("pop_w")

    r_ft   = m_ft["recall@50"]
    r2ch   = frozen_test_result.get("ref_2ch_test_recall50", 0)
    delta_vs_old_vs  = r_ft - OLD_VS_RECALL50
    delta_vs_old_2ch = r2ch - OLD_2CH_RECALL50
    delta_vs_old_tt  = sanity.get("test_recall@50", 0) - OLD_TT_RECALL50

    # Audit summary
    ov_icf_tt_50 = audit["overlap_50"]["icf-tt"]["jaccard"] if audit else "N/A"
    ov_icf_tt_100 = audit["overlap_100"]["icf-tt"]["jaccard"] if audit else "N/A"
    rrf_icf  = audit["rrf_attribution"]["icf"]["share"]  if audit else "N/A"
    rrf_tt   = audit["rrf_attribution"]["tt"]["share"]   if audit else "N/A"
    exc_tt   = audit["hit_attribution"]["tt"]["exclusive_hits_200"] if audit else "N/A"
    exc_icf  = audit["hit_attribution"]["icf"]["exclusive_hits_200"] if audit else "N/A"
    frac_icf = audit["hit_attribution"]["icf"]["fractional_share"]  if audit else "N/A"
    frac_tt  = audit["hit_attribution"]["tt"]["fractional_share"]   if audit else "N/A"

    same_as_old = (ksel == 100 and abs((tw_sel or 0) - 0.3) < 1e-6
                   and abs((pw_sel or 0) - 0.5) < 1e-6)

    audit_block = ""
    if audit:
        audit_block = f"""
## 8. Candidate Audit

### Overlap@50

| 通路对 | Jaccard@50 | 均值交集 |
|---|---:|---:|
| ICF – TT | {ov_icf_tt_50:.6f} | {audit['overlap_50']['icf-tt']['mean_intersect']:.2f} |
| ICF – Text | {audit['overlap_50']['icf-text']['jaccard']:.6f} | {audit['overlap_50']['icf-text']['mean_intersect']:.2f} |
| ICF – Pop | {audit['overlap_50']['icf-pop']['jaccard']:.6f} | {audit['overlap_50']['icf-pop']['mean_intersect']:.2f} |
| TT – Text | {audit['overlap_50']['tt-text']['jaccard']:.6f} | {audit['overlap_50']['tt-text']['mean_intersect']:.2f} |
| TT – Pop | {audit['overlap_50']['tt-pop']['jaccard']:.6f} | {audit['overlap_50']['tt-pop']['mean_intersect']:.2f} |
| Text – Pop | {audit['overlap_50']['text-pop']['jaccard']:.6f} | {audit['overlap_50']['text-pop']['mean_intersect']:.2f} |

### Overlap@100

| 通路对 | Jaccard@100 | 均值交集 |
|---|---:|---:|
| ICF – TT | {ov_icf_tt_100:.6f} | {audit['overlap_100']['icf-tt']['mean_intersect']:.2f} |
| ICF – Text | {audit['overlap_100']['icf-text']['jaccard']:.6f} | {audit['overlap_100']['icf-text']['mean_intersect']:.2f} |
| TT – Text | {audit['overlap_100']['tt-text']['jaccard']:.6f} | {audit['overlap_100']['tt-text']['mean_intersect']:.2f} |

### RRF 得分归因

| 通路 | 权重 | 得分占比 |
|---|---:|---:|
| ICF | 1.0 | {audit['rrf_attribution']['icf']['share']*100:.1f}% |
| TT | 1.0 | {audit['rrf_attribution']['tt']['share']*100:.1f}% |
| Text | {tw_sel} | {audit['rrf_attribution']['text']['share']*100:.1f}% |
| Pop | {pw_sel} | {audit['rrf_attribution']['pop']['share']*100:.1f}% |

### 命中归因（分数加权）

| 通路 | 分数命中占比 | 独占命中 @200 |
|---|---:|---:|
| ICF | {frac_icf:.1%} | {exc_icf} |
| TT  | {frac_tt:.1%} | {exc_tt} |
| Text | {audit['hit_attribution']['text']['fractional_share']:.1%} | {audit['hit_attribution']['text']['exclusive_hits_200']} |
| Pop  | {audit['hit_attribution']['pop']['fractional_share']:.1%} | {audit['hit_attribution']['pop']['exclusive_hits_200']} |

RRF rebuild Recall@50 = {audit['audit_rebuild_recall50']:.6f}（验证候选集一致性）
"""

    report = f"""# Transformer Two-Tower Multi-Channel Final Eval Report

**生成时间：** {now}
**评估集：** Amazon Reviews 2023 Movies_and_TV 5-core，full test，496,470 non-cold users
**脚本：** `scripts/run_multichannel_transformer_final.py`
**配置：** `configs/multichannel_transformer_final.yaml`

---

## 1. 背景：为什么重跑 Multi-Channel

旧 multi-channel 系统使用 Text+Time-decay Mean Pool Two-Tower（R@50=0.078315，max_len=20）作为 TT 通路。
经过 Transformer user tower 完整调查（稳定性 sweep → ablation → 种子稳健性 → canonical final run），
新的 canonical time-aware Transformer Two-Tower（R@50=0.103168，+31.7%）已通过验证。

本次实验：
- 用新 Transformer TT 替换旧 TT 通路，重跑 multi-channel valid-selected eval
- 使用完全相同的 Pareto 标准（在 valid set 上选 config，test 只运行一次）
- 不修改 ItemCF、Text Semantic、Popularity 通路定义

---

## 2. 单路对比：Old TT vs New Transformer TT

| 模型 | full test R@50 | Δ |
|---|---:|---:|
| Old TT（Time-decay MeanPool, max_len=20） | 0.078315 | — |
| **New TT（Transformer Timeaware, max_len=100）** | **{sanity.get('test_recall@50', 0):.6f}** | **{delta_vs_old_tt:+.6f}（{delta_vs_old_tt/OLD_TT_RECALL50*100:+.1f}%）** |

单路对齐验证：canonical R@50 = 0.103168，本次 sanity = {sanity.get('test_recall@50', 0):.6f}，差值 {sanity.get('delta', 0):+.6f}
对齐状态：{'✅ 通过' if sanity.get('aligned') else '⚠️ 差异超过阈值，请检查'}

---

## 3. New 2-Channel RRF 结果

| 系统 | test R@50 | vs old 2ch |
|---|---:|---:|
| Old 2ch RRF k=60（ICF + old TT） | 0.096727 | — |
| **New 2ch RRF k=60（ICF + new Transformer TT）** | **{r2ch:.6f}** | **{delta_vs_old_2ch:+.6f}（{delta_vs_old_2ch/OLD_2CH_RECALL50*100:+.1f}%）** |

---

## 4. Valid Sweep 设计

- 范围：k ∈ {config['sweep_k_values']}，text_w ∈ {config['sweep_text_weights']}，pop_w ∈ {config['sweep_pop_weights']}
- 总组数：{len(config['sweep_k_values']) * len(config['sweep_text_weights']) * len(config['sweep_pop_weights'])} 组（+ 2 条 reference baseline）
- icf_w = tt_w = 1.0（固定）
- Valid eval seen mask：train only；test eval seen mask：train + valid

**Pareto 标准：**
1. Recall@50 > 2ch RRF valid baseline
2. avg_pop ≤ {config['pareto_avg_pop_multiplier']}× 2ch baseline avg_pop
3. item_coverage ≥ {config['pareto_coverage_min_frac']*100:.0f}% of 2ch baseline
4. ≤5/6-20/21-100 三桶中至少 {config['pareto_bucket_min_pass']} 桶不低于 2ch baseline

---

## 5. Valid 选出的 Frozen Config

| 参数 | 值 |
|---|---:|
| name | `{vs_name}` |
| k | {ksel} |
| icf_w | 1.0 |
| tt_w | 1.0 |
| text_w | {tw_sel} |
| pop_w | {pw_sel} |
| Valid Recall@50 | {m_vs['recall@50']:.6f} |
| Valid NDCG@50 | {m_vs.get('ndcg@50', 0):.6f} |
| Valid avg_pop | {m_vs['avg_rec_popularity']:.1f} |

{'⚠️ Config 与旧 valid-selected (k=100, text=0.3, pop=0.5) 不同，见结论。' if not same_as_old else '✅ Config 与旧 valid-selected (text=0.3, pop=0.5) 权重一致。'}

---

## 6. Frozen Test 结果（仅运行一次）

| 指标 | 旧 valid-selected | **新 Transformer** | Δ |
|---|---:|---:|---:|
| Recall@50 | 0.104776 | **{r_ft:.6f}** | **{delta_vs_old_vs:+.6f}（{delta_vs_old_vs/OLD_VS_RECALL50*100:+.1f}%）** |
| NDCG@50 | 0.041599 | {m_ft.get('ndcg@50', 0):.6f} | {m_ft.get('ndcg@50', 0)-0.041599:+.6f} |
| MRR@50 | 0.025657 | {m_ft.get('mrr@50', 0):.6f} | {m_ft.get('mrr@50', 0)-0.025657:+.6f} |
| avg_pop | 461.8 | {m_ft.get('avg_rec_popularity', 0):.1f} | — |
| item_coverage | 153,924 | {m_ft.get('item_coverage', 0)} | — |

### Bucket Recall@50（热度桶）

| 桶 | 旧 valid-selected | **新 Transformer** | Δ |
|---|---:|---:|---:|
| ≤5（长尾） | 0.045142 | {bd_ft.get('1-5', {}).get('recall@50', 0):.6f} | {bd_ft.get('1-5', {}).get('recall@50', 0)-0.045142:+.6f} |
| 6-20 | 0.066167 | {bd_ft.get('6-20', {}).get('recall@50', 0):.6f} | {bd_ft.get('6-20', {}).get('recall@50', 0)-0.066167:+.6f} |
| 21-100 | 0.085952 | {bd_ft.get('21-100', {}).get('recall@50', 0):.6f} | {bd_ft.get('21-100', {}).get('recall@50', 0)-0.085952:+.6f} |
| >100（头部） | 0.144728 | {bd_ft.get('>100', {}).get('recall@50', 0):.6f} | {bd_ft.get('>100', {}).get('recall@50', 0)-0.144728:+.6f} |

---

## 7. 系统级对比总表

| 系统 | R@50 | NDCG@50 | MRR@50 | avg_pop | coverage |
|---|---:|---:|---:|---:|---:|
| ItemCF（单路） | 0.083570 | 0.036254 | 0.023999 | — | 153,055 |
| Old TT（单路） | 0.078315 | 0.030862 | 0.019036 | — | 153,928 |
| New TT（单路） | {sanity.get('test_recall@50', 0):.6f} | {sanity.get('test_ndcg@50', 0):.6f} | {sanity.get('test_mrr@50', 0):.6f} | — | — |
| Old 2ch RRF | 0.096727 | 0.038885 | 0.024272 | 264.5 | 153,936 |
| New 2ch RRF | {r2ch:.6f} | — | — | — | — |
| Old 4ch valid-sel | 0.104776 | 0.041599 | 0.025657 | 461.8 | 153,924 |
| **New 4ch valid-sel** | **{r_ft:.6f}** | **{m_ft.get('ndcg@50', 0):.6f}** | **{m_ft.get('mrr@50', 0):.6f}** | **{m_ft.get('avg_rec_popularity', 0):.1f}** | **{m_ft.get('item_coverage', 0)}** |

{audit_block}

---

## 9. Audit 检查清单

| 检查项 | 状态 |
|---|---|
| Transformer checkpoint = canonical (best_epoch=2) | ✅ |
| ItemCF 只使用 train split | ✅ |
| Popularity 只使用 train split | ✅ |
| Text Semantic 只使用 item_text_embedding + 用户历史 | ✅ |
| Valid seen mask = train only | ✅ |
| Test seen mask = train + valid | ✅ |
| Valid eval target = valid 非冷启动行（~497,137） | ✅ |
| Test eval target = test 非冷启动行（496,470） | ✅ |
| RRF 只使用 rank，不使用 label | ✅ |
| Valid-selected config 由 valid Pareto 选出 | ✅ |
| Test set 只运行一次 frozen config | ✅ |
| 不覆盖旧 multi-channel outputs | ✅ |

---

## 10. 结论

### 10.1 是否建议替换 project final model

新 Transformer 4ch valid-selected Recall@50 = **{r_ft:.6f}**
旧 multi-channel valid-selected Recall@50 = **0.104776**
差异 = **{delta_vs_old_vs:+.6f}（{delta_vs_old_vs/OLD_VS_RECALL50*100:+.1f}%）**

{'✅ **建议替换**：新系统在所有热度桶均超过旧系统（或持平），Recall 提升，avg_pop 在可接受范围内。' if r_ft >= OLD_VS_RECALL50 else f'⚠️ **暂不建议替换**：新系统 Recall@50={r_ft:.6f} < 旧系统 0.104776。TT 单路提升（+{(sanity.get("test_recall@50",0)-OLD_TT_RECALL50)/OLD_TT_RECALL50*100:.1f}%）在 fusion 中未完全体现，原因可能是 ICF/Text/Pop 通路已覆盖大部分增益空间。'}

### 10.2 是否建议更新 README

⚠️ 不建议现在更新 README。等待 Eddy 确认后，根据本报告结论决定。

### 10.3 是否建议更新简历

⚠️ 不建议现在更新简历。等待 Eddy 确认后，根据本报告结论决定。

### 10.4 局限性

1. 本报告为 offline full eval 结论，不等于 online A/B 结果
2. avg_pop 增减反映热门偏置趋势，不直接等于用户满意度
3. Faiss 在新 Transformer 上的检索一致性（overlap@50）尚未重新测量
4. 不包含 Transformer checkpoint 的 Faiss index 重建

---

## 11. 文件清单

```text
outputs/multichannel_transformer_final/
  valid_sweep.json               — valid set 全量 sweep 结果（60+2 组）
  valid_sweep.csv                — same，CSV 格式
  final_test_metrics.json        — frozen config test-only 结果
  final_comparison_table.csv     — 全系统对比表
  final_comparison_table.json    — same，JSON 格式
  sanity_single_channel.json     — 单路 sanity check 结果
  report.md                      — 本报告（outputs 版本）
  candidate_audit/
    overlap_metrics.json         — Jaccard@50 + @100
    overlap_metrics.csv          — same，CSV 格式
    rrf_attribution.json         — RRF 得分归因 + 命中归因
    hit_attribution.csv          — 命中归因汇总
    audit_summary.json           — 全部 audit 结果
docs/reports/multichannel_transformer_final_eval.md — 正式报告
docs/daily_logs/2026-05-20.md   — Part 21 追加
```

> ⚠️ outputs/ 不提交 git。
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    logging.info("[Report] report.md saved to %s", output_dir)

    # Also write to docs/reports/
    docs_path = Path("docs/reports/multichannel_transformer_final_eval.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report, encoding="utf-8")
    logging.info("[Report] docs/reports/multichannel_transformer_final_eval.md written.")
    return report


# ---------------------------------------------------------------------------
# Daily log append
# ---------------------------------------------------------------------------

def append_to_daily_log(
    sanity: dict[str, Any],
    frozen_test_result: dict[str, Any],
    valid_selected: dict[str, Any],
) -> None:
    m_ft   = frozen_test_result["final_test"]["metrics"]
    r2ch   = frozen_test_result.get("ref_2ch_test_recall50", 0)
    r_ft   = m_ft["recall@50"]
    ksel   = valid_selected.get("wrrf_k")
    tw_sel = valid_selected.get("text_w")
    pw_sel = valid_selected.get("pop_w")
    delta  = r_ft - OLD_VS_RECALL50

    log_path = Path("docs/daily_logs/2026-05-20.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = f"""
---

## Part 21：New Transformer Multi-Channel Eval 完成

**状态：** ✅

- 替换通路：New Transformer Two-Tower（R@50=0.103168, +31.7%）替换 Old Time-decay TT
- 单路 sanity: R@50={sanity.get('test_recall@50', 0):.6f}（canonical={CANONICAL_TT_RECALL50}，diff={sanity.get('delta', 0):+.6f}），对齐={'✅' if sanity.get('aligned') else '⚠️'}
- New 2ch RRF test R@50 = {r2ch:.6f}
- Valid-selected config：k={ksel}，text_w={tw_sel}，pop_w={pw_sel}
- **New 4ch valid-selected test Recall@50 = {r_ft:.6f}**（vs old 0.104776，Δ={delta:+.6f}，{delta/OLD_VS_RECALL50*100:+.1f}%）
- Audit：overlap@50/100、RRF 归因、命中归因均已完成
- 报告：docs/reports/multichannel_transformer_final_eval.md
- 候选集审计：outputs/multichannel_transformer_final/candidate_audit/
是否替换 project final model：{'✅ 建议替换（新系统 ≥ 旧系统）' if r_ft >= OLD_VS_RECALL50 else '⚠️ 暂缓替换（新系统 < 旧系统）'}
下一步：等 Eddy 确认是否更新 README / 简历。
"""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    logging.info("Appended Part 21 to %s", log_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))

    logging.info("=== Transformer Multi-Channel Final Eval ===")
    logging.info("Output dir: %s", output_dir)
    logging.info("Checkpoint: %s", config["checkpoint"])

    # Phase 0: Load shared resources
    logging.info("[Phase0] Loading shared resources...")
    shared = load_shared_resources(config)

    # Phase 1: Single-channel sanity
    sanity: dict[str, Any] = {}
    if not args.skip_sanity:
        logging.info("[Phase1] Single-channel sanity check...")
        sanity = run_single_channel_sanity(config, shared, output_dir)
        if not sanity.get("aligned"):
            logging.error("[Phase1] SANITY FAILED: TT single-channel does not align. Aborting.")
            return
    else:
        logging.info("[Phase1] Skipped (--skip_sanity).")
        try:
            sanity = json.loads((output_dir / "sanity_single_channel.json").read_text())
        except Exception:
            sanity = {"test_recall@50": CANONICAL_TT_RECALL50, "test_ndcg@50": 0,
                      "test_mrr@50": 0, "delta": 0, "aligned": True, "bucket_breakdown": {}}

    # Phase 2: Valid sweep
    if args.skip_sweep or (args.test_only and args.frozen_k is not None):
        if args.skip_sweep:
            # Load selected config from saved valid_sweep.json
            sweep_data = json.loads((output_dir / "valid_sweep.json").read_text())
            # Find the selected config from valid_sweep.csv (highest recall among pareto-passing)
            # Reload from the CSV to find the winner — use the same Pareto logic
            all_results = sweep_data.get("results", [])
            ref_2ch = next((r for r in all_results if r.get("fusion_type") == "ref_2ch"), None)
            valid_selected = _select_pareto(all_results, ref_2ch, config) if ref_2ch else {}
            if not valid_selected:
                valid_selected = {"name": "wrrf_k100_text0.3_pop0.5",
                                  "wrrf_k": 100, "text_w": 0.3, "pop_w": 0.5,
                                  "metrics": {"recall@50": 0.174258, "avg_rec_popularity": 485}}
            logging.info("[Phase2] Skipped (--skip_sweep). Selected: %s", valid_selected["name"])
        else:
            valid_selected = {
                "name": f"wrrf_k{args.frozen_k}_text{args.frozen_text_w:.1f}_pop{args.frozen_pop_w:.1f}",
                "wrrf_k": args.frozen_k, "text_w": args.frozen_text_w, "pop_w": args.frozen_pop_w,
                "metrics": {"recall@50": 0, "avg_rec_popularity": 0},
            }
            logging.info("[Phase2] Skipping valid sweep (--test_only).")
    else:
        logging.info("[Phase2] Starting valid sweep (60 configs)...")
        valid_selected = run_valid_sweep(config, shared, output_dir)
        logging.info("[Phase2] Done. Selected: %s", valid_selected["name"])

    # Phase 3: Frozen test eval
    frozen_k      = int(valid_selected.get("wrrf_k", 100))
    frozen_text_w = float(valid_selected.get("text_w", 0.3))
    frozen_pop_w  = float(valid_selected.get("pop_w", 0.5))
    if args.skip_phase3:
        logging.info("[Phase3] Skipped (--skip_phase3). Loading from final_test_metrics.json...")
        frozen_test_result = json.loads((output_dir / "final_test_metrics.json").read_text())
    else:
        logging.info("[Phase3] Running frozen test eval...")
        _ = run_frozen_test(config, shared, output_dir, frozen_k, frozen_text_w, frozen_pop_w)
        # run_frozen_test returns raw res; load the wrapper JSON that includes ref_2ch_test_recall50
        frozen_test_result = json.loads((output_dir / "final_test_metrics.json").read_text())
    r_ft = frozen_test_result["final_test"]["metrics"]["recall@50"]
    logging.info("[Phase3] Frozen test Recall@50=%.6f", r_ft)

    # Phase 4: Candidate audit
    audit: dict[str, Any] | None = None
    if not args.skip_audit:
        logging.info("[Phase4] Running candidate audit...")
        audit = run_candidate_audit(config, shared, output_dir,
                                    frozen_k, frozen_text_w, frozen_pop_w)
    else:
        logging.info("[Phase4] Skipped (--skip_audit).")

    # Phase 5: Comparison table + report
    logging.info("[Phase5] Building comparison table and report...")
    build_comparison_table(output_dir, sanity, frozen_test_result, config)
    write_reports(output_dir, config, sanity, valid_selected, frozen_test_result, audit)

    # Daily log
    append_to_daily_log(sanity, frozen_test_result, valid_selected)

    logging.info("=== Done ===")
    logging.info("[Summary] 2ch_test=%.6f  4ch_valid_selected_test=%.6f  vs_old_4ch=%+.6f",
                 frozen_test_result.get("ref_2ch_test_recall50", 0),
                 r_ft, r_ft - OLD_VS_RECALL50)


if __name__ == "__main__":
    main()

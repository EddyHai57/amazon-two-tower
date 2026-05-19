#!/usr/bin/env python3
"""
Multi-channel retrieval v2: ItemCF + Two-Tower + Text Semantic + Popularity Fallback.

Evaluates single-channel diagnostics for Text Semantic and Popularity, then
3-channel and 4-channel fusion, benchmarked against v1 two-channel best:
  ItemCF + TwoTower  RRF k=60  Recall@50 = 0.096727  (full test, 496,470 users)

Output dir: outputs/multichannel_v2/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# Reuse audited channel generators and eval helpers from v1
from run_multichannel_retrieval import (
    TRAIN_COLUMNS,
    aggregate_metrics,
    compute_metrics,
    encode_all_items,
    generate_itemcf_candidates,
    generate_twotower_candidates,
    write_json,
)
from run_itemcf import add_valid_to_seen, build_item_similarity, build_train_sets
from train_text_time_decay_mean_pool_two_tower_smoke import (
    DataBundle,
    build_history_matrix,
    build_model,
    build_seen_items,
    load_config as load_train_config,
    load_data,
    merge_seen_items,
    resolve_device,
    set_seed,
)

# v1 best baseline (full test, 496,470 users) — used as reference delta
V1_BEST_RECALL50 = 0.096727


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-channel retrieval v2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--full_only", action="store_true")
    return parser.parse_args()


def load_v2_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = [
        "data_dir", "train_config", "checkpoint", "item_text_emb_path",
        "output_dir", "seed", "device", "smoke_users", "candidates_per_channel",
        "text_decay_rate", "pop_buffer_size",
        "itemcf_sim_topk", "itemcf_max_user_history",
        "three_channel_quota_sweep", "three_channel_rrf_k_values", "three_channel_rrf_top_n",
        "four_channel_quota_sweep", "four_channel_rrf_k_values", "four_channel_rrf_top_n",
        "eval_k_list",
    ]
    for key in required:
        if key not in cfg:
            raise KeyError(f"config missing key: {key}")
    return cfg


def write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


# ---------------------------------------------------------------------------
# Text semantic channel
# ---------------------------------------------------------------------------

def load_text_embeddings(emb_path: Path, n_items: int) -> np.ndarray:
    """Load item text embeddings, L2-normalize (no-text items stay as zero vector)."""
    raw = np.load(emb_path)  # (n_items, 384) float32
    assert raw.shape[0] == n_items, f"text emb n_items mismatch: {raw.shape[0]} vs {n_items}"
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0  # avoid division by zero; no-text rows remain 0-vector
    return (raw / norms).astype(np.float32)


def generate_text_semantic_candidates(
    eval_users: list[int],
    test_history_matrix: np.ndarray,   # (n_users, max_len), -1 = padding
    test_seen: dict[int, set[int]],
    item_text_norm_gpu: torch.Tensor,  # (n_items, 384) on device, L2-normalized
    top_k: int,
    decay_rate: float = 0.8,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
) -> tuple[dict[int, list[int]], int]:
    """
    Returns (candidates dict, n_zero_query_users).
    Users with no text-covered history get empty candidate list.
    Decay weights mirror TwoTower: position i has weight decay_rate^(max_len-1-i).
    """
    seq_len = test_history_matrix.shape[1]
    n_items = item_text_norm_gpu.shape[0]

    # Decay weights: same formula as TextTimeDecayMeanPoolTwoTower
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    decay_w = decay_rate ** (seq_len - 1 - positions)  # (seq_len,), oldest=small, newest=1.0

    candidates: dict[int, list[int]] = {}
    n_zero_query = 0

    with torch.no_grad():
        for start in range(0, len(eval_users), batch_size):
            batch_users = eval_users[start: start + batch_size]
            users_np = np.asarray(batch_users, dtype=np.int64)

            hist_np = test_history_matrix[users_np]  # (B, seq_len), -1=padding
            hist_t = torch.as_tensor(hist_np, dtype=torch.long, device=device)

            valid_mask = (hist_t >= 0)                   # (B, seq_len)
            safe_hist = hist_t.clamp_min(0)              # (B, seq_len), no negatives for indexing

            # Text embeddings of history items: (B, seq_len, 384)
            hist_text = item_text_norm_gpu[safe_hist]    # (B, seq_len, 384)

            # Time-decay weighted mean (same as TwoTower's time_decay_mean_history_embedding)
            dw = decay_w.view(1, seq_len, 1)             # (1, seq_len, 1)
            mask_f = valid_mask.unsqueeze(-1).float()    # (B, seq_len, 1)
            weighted = hist_text * mask_f * dw           # (B, seq_len, 384)
            summed = weighted.sum(dim=1)                 # (B, 384)
            weight_sum = (mask_f * dw).sum(dim=1).clamp_min(1e-8)  # (B, 1)
            queries = summed / weight_sum               # (B, 384)

            # Check which users have a valid (non-zero) text query
            q_norms = torch.norm(queries, dim=1)        # (B,)
            has_text_q = (q_norms > 1e-7)              # (B,) bool

            # Normalize queries to unit vectors
            queries = queries / q_norms.unsqueeze(1).clamp_min(1e-8)  # (B, 384)

            # Cosine similarity: (B, n_items)
            scores_t = queries @ item_text_norm_gpu.T   # (B, n_items)
            scores_np = scores_t.cpu().numpy()

            for j, user_idx in enumerate(batch_users):
                if not has_text_q[j].item():
                    candidates[int(user_idx)] = []
                    n_zero_query += 1
                    continue

                s = scores_np[j].copy()
                seen = test_seen.get(int(user_idx), set())
                if seen:
                    seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen))
                    s[seen_arr] = -np.inf

                if top_k >= n_items:
                    top_idx = np.argsort(s)[::-1]
                else:
                    top_idx = np.argpartition(s, -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(s[top_idx])[::-1]]

                candidates[int(user_idx)] = top_idx.tolist()

    return candidates, n_zero_query


# ---------------------------------------------------------------------------
# Popularity fallback channel
# ---------------------------------------------------------------------------

def build_pop_sorted_items(train_df: pd.DataFrame) -> list[int]:
    """Items sorted by train interaction count, descending."""
    pop = Counter(train_df["item_idx"].tolist())
    return [item for item, _ in pop.most_common()]


def generate_popularity_candidates(
    eval_users: list[int],
    test_seen: dict[int, set[int]],
    pop_sorted_items: list[int],
    top_k: int,
    buffer_size: int = 1000,
) -> dict[int, list[int]]:
    """Return top-K globally popular items, excluding seen items per user."""
    buffer = pop_sorted_items[:buffer_size]
    candidates: dict[int, list[int]] = {}
    for user_idx in eval_users:
        seen = test_seen.get(int(user_idx), set())
        result = [item for item in buffer if item not in seen][:top_k]
        # If buffer exhausted (pathological case), extend
        if len(result) < top_k:
            extra = [item for item in pop_sorted_items[buffer_size:] if item not in seen]
            result.extend(extra[:top_k - len(result)])
        candidates[int(user_idx)] = result
    return candidates


# ---------------------------------------------------------------------------
# N-channel fusion
# ---------------------------------------------------------------------------

def quota_merge_n(channel_cands: list[tuple[list[int], int]]) -> list[int]:
    """N-channel quota merge. channel_cands = [(cands, quota), ...]. Dedup, preserve order."""
    result: list[int] = []
    seen: set[int] = set()
    for cands, quota in channel_cands:
        for item in cands[:quota]:
            if item not in seen:
                result.append(item)
                seen.add(item)
    return result


def rrf_merge_n(channel_cands: list[list[int]], k: int, top_n: int) -> list[int]:
    """N-channel Reciprocal Rank Fusion. score(item) = sum of 1/(k+rank) over channels."""
    scores: defaultdict[int, float] = defaultdict(float)
    for cands in channel_cands:
        for rank, item in enumerate(cands, start=1):
            scores[item] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [item for item, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Per-channel overlap analysis
# ---------------------------------------------------------------------------

def compute_channel_overlap_stats(
    eval_targets: pd.DataFrame,
    all_cands: dict[str, dict[int, list[int]]],
    channel_names: list[str],
    at_k: int = 50,
) -> dict[str, Any]:
    """
    For each channel: unique_hits (hit only by this channel), hit_rate.
    Pairwise Jaccard overlap between all channel pairs.
    Requires only rank information and test targets.
    """
    n = len(eval_targets)
    hit_counts: dict[str, int] = {ch: 0 for ch in channel_names}
    unique_hit_counts: dict[str, int] = {ch: 0 for ch in channel_names}
    overlap_hits_all: int = 0  # target in all channels

    pair_jaccard_sum: dict[tuple[str, str], float] = {}
    for i, a in enumerate(channel_names):
        for b in channel_names[i + 1:]:
            pair_jaccard_sum[(a, b)] = 0.0

    for row in eval_targets.itertuples(index=False):
        user_idx = int(row.user_idx)
        target = int(row.item_idx)

        sets: dict[str, set[int]] = {}
        for ch in channel_names:
            cands = all_cands[ch].get(user_idx, [])[:at_k]
            sets[ch] = set(cands)

        hits: dict[str, bool] = {ch: target in sets[ch] for ch in channel_names}

        for ch in channel_names:
            if hits[ch]:
                hit_counts[ch] += 1

        for ch in channel_names:
            if hits[ch] and all(not hits[other] for other in channel_names if other != ch):
                unique_hit_counts[ch] += 1

        if all(hits[ch] for ch in channel_names):
            overlap_hits_all += 1

        for i, a in enumerate(channel_names):
            for b in channel_names[i + 1:]:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                pair_jaccard_sum[(a, b)] += inter / max(union, 1)

    stats: dict[str, Any] = {
        "n_eval_users": n,
        "at_k": at_k,
        "overlap_hits_all_channels": overlap_hits_all,
    }
    for ch in channel_names:
        stats[f"{ch}_hits@{at_k}"] = hit_counts[ch]
        stats[f"{ch}_hit_rate@{at_k}"] = round(hit_counts[ch] / max(n, 1), 6)
        stats[f"{ch}_unique_hits@{at_k}"] = unique_hit_counts[ch]
        stats[f"{ch}_unique_hit_rate@{at_k}"] = round(unique_hit_counts[ch] / max(n, 1), 6)
    for (a, b), total in pair_jaccard_sum.items():
        stats[f"jaccard_{a}_{b}@{at_k}"] = round(total / max(n, 1), 6)

    return stats


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_single_eval(
    name: str,
    eval_targets: pd.DataFrame,
    get_merged: Any,
    k_list: list[int],
    item_popularity: dict[int, int],
    pop_buckets: list[tuple[int, int | None]],
) -> dict[str, Any]:
    """Evaluate a single fusion strategy (any number of channels)."""
    n = len(eval_targets)
    per_user: list[dict[str, float]] = []
    bucket_per_user: dict[str, list[dict[str, float]]] = {
        f"bucket_{i}": [] for i in range(len(pop_buckets))
    }
    for row in eval_targets.itertuples(index=False):
        user_idx = int(row.user_idx)
        target = int(row.item_idx)
        merged = get_merged(user_idx)
        m = compute_metrics(merged, target, k_list)
        per_user.append(m)
        pop = item_popularity.get(target, 0)
        for bidx, (lo, hi) in enumerate(pop_buckets):
            if hi is None:
                if pop >= lo:
                    bucket_per_user[f"bucket_{bidx}"].append(m)
                    break
            elif lo <= pop <= hi:
                bucket_per_user[f"bucket_{bidx}"].append(m)
                break
    agg = aggregate_metrics(per_user, k_list)
    agg["n_eval_users"] = n
    bucket_agg: dict[str, Any] = {}
    for bidx, (lo, hi) in enumerate(pop_buckets):
        bkey = f"bucket_{bidx}"
        blabel = f"{lo}-{hi}" if hi is not None else f">{lo - 1}"
        bm = aggregate_metrics(bucket_per_user[bkey], k_list)
        bm["n_users"] = len(bucket_per_user[bkey])
        bucket_agg[blabel] = bm
    return {"name": name, "metrics": agg, "bucket_breakdown": bucket_agg}


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(config: dict[str, Any], output_dir: Path, smoke: bool) -> None:
    data_dir = Path(config["data_dir"])
    train_config = load_train_config(Path(config["train_config"]))
    bundle: DataBundle = load_data(data_dir)
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_max_len = int(train_config["history_max_len"])
    run_type = "smoke" if smoke else "full"

    # --- Seen-item masks ---
    train_seen = build_seen_items(bundle.train_df)
    test_seen = merge_seen_items(train_seen, bundle.valid_df)  # train + valid

    # --- Test history matrix (train + valid) ---
    test_history_frame = pd.concat(
        [bundle.train_df[TRAIN_COLUMNS], bundle.valid_df[TRAIN_COLUMNS]], ignore_index=True
    )
    test_history_matrix, _ = build_history_matrix(test_history_frame, num_users, history_max_len)

    # --- ItemCF ---
    logging.info("[ItemCF] Building train sets...")
    icf_full_seen, icf_limited_history = build_train_sets(
        bundle.train_df, int(config["itemcf_max_user_history"])
    )
    icf_eval_seen = add_valid_to_seen(icf_full_seen, bundle.valid_df)
    logging.info("[ItemCF] Building item similarity (sim_topk=%s)...", config["itemcf_sim_topk"])
    similarity = build_item_similarity(icf_limited_history, int(config["itemcf_sim_topk"]))

    # --- Item popularity (train-only, same as v1) ---
    item_pop: Counter[int] = Counter(bundle.train_df["item_idx"].tolist())
    item_popularity: dict[int, int] = dict(item_pop)
    pop_sorted_items = [item for item, _ in item_pop.most_common()]

    # --- Two-Tower model + item embeddings ---
    device = resolve_device(str(config["device"]))
    logging.info("[TwoTower] Loading model from %s...", config["checkpoint"])
    from train_text_time_decay_mean_pool_two_tower_smoke import TextTimeDecayMeanPoolTwoTower
    model = build_model(train_config, bundle.stats, device)
    checkpoint = torch.load(Path(config["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(
        "[TwoTower] Loaded: epoch=%s recall@50=%.6f",
        checkpoint.get("epoch"),
        float(checkpoint.get("best_metric_value", 0.0)),
    )
    item_emb_cpu = encode_all_items(model, num_items, device)
    logging.info("[TwoTower] item_emb shape: %s", tuple(item_emb_cpu.shape))

    # --- Text embeddings: load and normalize ---
    logging.info("[TextSemantic] Loading text embeddings from %s...", config["item_text_emb_path"])
    item_text_norm_np = load_text_embeddings(Path(config["item_text_emb_path"]), num_items)
    item_text_norm_gpu = torch.from_numpy(item_text_norm_np).to(device)
    logging.info("[TextSemantic] item_text_norm shape: %s", tuple(item_text_norm_gpu.shape))

    # --- Select eval users ---
    cold_mask = bundle.test_df["is_cold_item_for_eval"].astype(bool)
    eval_targets_df = bundle.test_df[~cold_mask].copy()
    all_eval_users = eval_targets_df["user_idx"].unique().tolist()
    logging.info("Total non-cold test users: %d", len(all_eval_users))

    if smoke:
        rng = np.random.default_rng(int(config["seed"]))
        n_smoke = min(int(config["smoke_users"]), len(all_eval_users))
        sampled = set(rng.choice(all_eval_users, size=n_smoke, replace=False).tolist())
        eval_df = eval_targets_df[eval_targets_df["user_idx"].isin(sampled)].copy()
        logging.info("[smoke] Using %d users", len(eval_df))
    else:
        eval_df = eval_targets_df
        logging.info("[full] Using all %d users", len(eval_df))

    eval_users = eval_df["user_idx"].unique().tolist()
    cold_mask_eval = eval_df["is_cold_item_for_eval"].astype(bool)
    eval_targets = eval_df[~cold_mask_eval].copy()
    n_eval = len(eval_targets)

    # --- Parse config ---
    top_k = int(config["candidates_per_channel"])
    pop_buf = int(config["pop_buffer_size"])
    k_list = [int(k) for k in config["eval_k_list"]]
    raw_buckets = config.get("popularity_buckets", [[1, 5], [6, 20], [21, 100], [101, None]])
    pop_buckets = [(int(lo), None if hi is None else int(hi)) for lo, hi in raw_buckets]
    text_decay = float(config["text_decay_rate"])

    # --- Generate all 4 channel candidates ---
    logging.info("[ItemCF] Generating candidates (top_k=%d)...", top_k)
    icf_cands = generate_itemcf_candidates(
        eval_df, bundle.train_df, icf_eval_seen, icf_limited_history, similarity, top_k
    )
    logging.info("[TwoTower] Generating candidates (top_k=%d)...", top_k)
    tt_cands_raw = generate_twotower_candidates(
        model, eval_users, test_history_matrix, test_seen, item_emb_cpu, top_k, device
    )
    # Convert np.ndarray -> list[int] for uniform handling
    tt_cands: dict[int, list[int]] = {u: v.tolist() for u, v in tt_cands_raw.items()}

    logging.info("[TextSemantic] Generating candidates (top_k=%d, decay_rate=%.2f)...", top_k, text_decay)
    text_cands, n_zero_query = generate_text_semantic_candidates(
        eval_users, test_history_matrix, test_seen, item_text_norm_gpu,
        top_k, text_decay, device
    )
    logging.info("[TextSemantic] n_zero_query_users=%d / %d", n_zero_query, len(eval_users))

    logging.info("[Popularity] Generating candidates (top_k=%d, buffer=%d)...", top_k, pop_buf)
    pop_cands = generate_popularity_candidates(
        eval_users, test_seen, pop_sorted_items, top_k, pop_buf
    )

    all_channel_cands: dict[str, dict[int, list[int]]] = {
        "itemcf": icf_cands,
        "twotower": tt_cands,
        "text": text_cands,
        "pop": pop_cands,
    }

    # --- Sanity checks ---
    for ch, cands in all_channel_cands.items():
        n_covered = sum(1 for u in eval_users if u in cands and len(cands[u]) > 0)
        logging.info("[check] %s: %d/%d users have candidates", ch, n_covered, len(eval_users))

    # ----------------------------------------------------------------
    # Channel overlap / unique hit analysis (all 4 channels vs each)
    # ----------------------------------------------------------------
    logging.info("[Overlap] Computing 4-channel overlap stats...")
    overlap_4ch = compute_channel_overlap_stats(
        eval_targets, all_channel_cands, ["itemcf", "twotower", "text", "pop"], at_k=50
    )
    logging.info(
        "[Overlap] ICF hits=%.4f TT hits=%.4f text hits=%.4f pop hits=%.4f",
        overlap_4ch["itemcf_hit_rate@50"],
        overlap_4ch["twotower_hit_rate@50"],
        overlap_4ch["text_hit_rate@50"],
        overlap_4ch["pop_hit_rate@50"],
    )
    logging.info(
        "[Overlap] ICF unique=%.4f TT unique=%.4f text unique=%.4f pop unique=%.4f",
        overlap_4ch["itemcf_unique_hit_rate@50"],
        overlap_4ch["twotower_unique_hit_rate@50"],
        overlap_4ch["text_unique_hit_rate@50"],
        overlap_4ch["pop_unique_hit_rate@50"],
    )

    # ----------------------------------------------------------------
    # Single-channel evaluation: text_semantic and popularity
    # ----------------------------------------------------------------
    logging.info("[Eval] Single-channel: text_semantic...")
    text_single_result = run_single_eval(
        "text_semantic",
        eval_targets,
        lambda u: text_cands.get(u, [])[:50],
        k_list, item_popularity, pop_buckets,
    )
    logging.info(
        "[Eval] text_semantic Recall@50=%.6f n_zero_query=%d",
        text_single_result["metrics"].get("recall@50", 0),
        n_zero_query,
    )

    logging.info("[Eval] Single-channel: popularity...")
    pop_single_result = run_single_eval(
        "popularity",
        eval_targets,
        lambda u: pop_cands.get(u, [])[:50],
        k_list, item_popularity, pop_buckets,
    )
    logging.info(
        "[Eval] popularity Recall@50=%.6f",
        pop_single_result["metrics"].get("recall@50", 0),
    )

    # ----------------------------------------------------------------
    # 2-channel reference: ICF + TT RRF k=60 (replicate v1 best)
    # ----------------------------------------------------------------
    two_rrf_k_values = [int(k) for k in config.get("two_channel_rrf_k_values", [60])]
    two_rrf_top_n = int(config.get("two_channel_rrf_top_n", 50))
    two_rrf_results: list[dict[str, Any]] = []
    for k in two_rrf_k_values:
        name = f"2ch_rrf_k{k}"
        logging.info("[Eval] 2-channel ref: %s ...", name)

        def get_2ch_rrf(u: int, k: int = k) -> list[int]:
            from run_multichannel_retrieval import rrf_merge
            return rrf_merge(
                icf_cands.get(u, []), tt_cands.get(u, []), k, two_rrf_top_n
            )

        res = run_single_eval(name, eval_targets, get_2ch_rrf, k_list, item_popularity, pop_buckets)
        res["fusion_type"] = "2ch_rrf"
        res["rrf_k"] = k
        two_rrf_results.append(res)
        r50 = res["metrics"].get("recall@50", 0)
        logging.info("[Eval] %s Recall@50=%.6f (v1 ref=%.6f, delta=%+.6f)", name, r50, V1_BEST_RECALL50, r50 - V1_BEST_RECALL50)

    # ----------------------------------------------------------------
    # 3-channel: ICF + TT + Text
    # ----------------------------------------------------------------
    three_quota_sweep = config["three_channel_quota_sweep"]
    three_rrf_k_values = [int(k) for k in config["three_channel_rrf_k_values"]]
    three_rrf_top_n = int(config["three_channel_rrf_top_n"])

    three_quota_results: list[dict[str, Any]] = []
    for combo in three_quota_sweep:
        icf_q, tt_q, text_q = int(combo[0]), int(combo[1]), int(combo[2])
        name = f"3ch_icf{icf_q}_tt{tt_q}_text{text_q}"
        logging.info("[Eval] 3-channel quota: %s ...", name)

        def get_3ch_quota(u: int, icf_q: int = icf_q, tt_q: int = tt_q, text_q: int = text_q) -> list[int]:
            return quota_merge_n([
                (icf_cands.get(u, []), icf_q),
                (tt_cands.get(u, []), tt_q),
                (text_cands.get(u, []), text_q),
            ])

        res = run_single_eval(name, eval_targets, get_3ch_quota, k_list, item_popularity, pop_buckets)
        res["fusion_type"] = "3ch_quota"
        res["icf_quota"] = icf_q
        res["tt_quota"] = tt_q
        res["text_quota"] = text_q
        r50 = res["metrics"].get("recall@50", 0)
        res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
        three_quota_results.append(res)
        logging.info("[Eval] %s Recall@50=%.6f (delta vs v1 best: %+.6f)", name, r50, r50 - V1_BEST_RECALL50)

    three_rrf_results: list[dict[str, Any]] = []
    for k in three_rrf_k_values:
        name = f"3ch_rrf_k{k}"
        logging.info("[Eval] 3-channel RRF: %s ...", name)

        def get_3ch_rrf(u: int, k: int = k) -> list[int]:
            return rrf_merge_n([
                icf_cands.get(u, []),
                tt_cands.get(u, []),
                text_cands.get(u, []),
            ], k, three_rrf_top_n)

        res = run_single_eval(name, eval_targets, get_3ch_rrf, k_list, item_popularity, pop_buckets)
        res["fusion_type"] = "3ch_rrf"
        res["rrf_k"] = k
        r50 = res["metrics"].get("recall@50", 0)
        res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
        three_rrf_results.append(res)
        logging.info("[Eval] %s Recall@50=%.6f (delta vs v1 best: %+.6f)", name, r50, r50 - V1_BEST_RECALL50)

    # ----------------------------------------------------------------
    # 4-channel: ICF + TT + Text + Popularity
    # ----------------------------------------------------------------
    four_quota_sweep = config["four_channel_quota_sweep"]
    four_rrf_k_values = [int(k) for k in config["four_channel_rrf_k_values"]]
    four_rrf_top_n = int(config["four_channel_rrf_top_n"])

    four_quota_results: list[dict[str, Any]] = []
    for combo in four_quota_sweep:
        icf_q, tt_q, text_q, pop_q = int(combo[0]), int(combo[1]), int(combo[2]), int(combo[3])
        name = f"4ch_icf{icf_q}_tt{tt_q}_text{text_q}_pop{pop_q}"
        logging.info("[Eval] 4-channel quota: %s ...", name)

        def get_4ch_quota(u: int, icf_q: int = icf_q, tt_q: int = tt_q, text_q: int = text_q, pop_q: int = pop_q) -> list[int]:
            return quota_merge_n([
                (icf_cands.get(u, []), icf_q),
                (tt_cands.get(u, []), tt_q),
                (text_cands.get(u, []), text_q),
                (pop_cands.get(u, []), pop_q),
            ])

        res = run_single_eval(name, eval_targets, get_4ch_quota, k_list, item_popularity, pop_buckets)
        res["fusion_type"] = "4ch_quota"
        res["icf_quota"] = icf_q
        res["tt_quota"] = tt_q
        res["text_quota"] = text_q
        res["pop_quota"] = pop_q
        r50 = res["metrics"].get("recall@50", 0)
        res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
        four_quota_results.append(res)
        logging.info("[Eval] %s Recall@50=%.6f (delta vs v1 best: %+.6f)", name, r50, r50 - V1_BEST_RECALL50)

    four_rrf_results: list[dict[str, Any]] = []
    for k in four_rrf_k_values:
        name = f"4ch_rrf_k{k}"
        logging.info("[Eval] 4-channel RRF: %s ...", name)

        def get_4ch_rrf(u: int, k: int = k) -> list[int]:
            return rrf_merge_n([
                icf_cands.get(u, []),
                tt_cands.get(u, []),
                text_cands.get(u, []),
                pop_cands.get(u, []),
            ], k, four_rrf_top_n)

        res = run_single_eval(name, eval_targets, get_4ch_rrf, k_list, item_popularity, pop_buckets)
        res["fusion_type"] = "4ch_rrf"
        res["rrf_k"] = k
        r50 = res["metrics"].get("recall@50", 0)
        res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
        four_rrf_results.append(res)
        logging.info("[Eval] %s Recall@50=%.6f (delta vs v1 best: %+.6f)", name, r50, r50 - V1_BEST_RECALL50)

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    text_single_result["n_zero_query_users"] = n_zero_query
    text_single_result["text_decay_rate"] = text_decay
    text_single_result["run_type"] = run_type
    write_json(output_dir / f"text_semantic_single_{run_type}.json", text_single_result)

    pop_single_result["run_type"] = run_type
    write_json(output_dir / f"popularity_single_{run_type}.json", pop_single_result)

    write_json(output_dir / f"overlap_4ch_stats_{run_type}.json", overlap_4ch)

    # 2-channel ref JSON
    write_json(output_dir / f"two_channel_ref_{run_type}.json", {
        "run_type": run_type,
        "v1_best_recall50": V1_BEST_RECALL50,
        "results": two_rrf_results,
    })

    # 3-channel CSVs
    q3_keys = ["name", "fusion_type", "icf_quota", "tt_quota", "text_quota",
               "recall@50", "recall@100", "ndcg@50", "mrr@50", "delta_vs_v1"]
    q3_rows = []
    for r in three_quota_results:
        row = {k: r.get(k, r["metrics"].get(k, "")) for k in q3_keys}
        for mk in ["recall@50", "recall@100", "ndcg@50", "mrr@50"]:
            row[mk] = r["metrics"].get(mk, "")
        q3_rows.append(row)
    write_csv(output_dir / f"three_channel_quota_{run_type}.csv", q3_rows, q3_keys)

    rrf3_keys = ["name", "fusion_type", "rrf_k", "recall@50", "recall@100", "ndcg@50", "mrr@50", "delta_vs_v1"]
    rrf3_rows = []
    for r in three_rrf_results:
        row = {k: r.get(k, r["metrics"].get(k, "")) for k in rrf3_keys}
        for mk in ["recall@50", "recall@100", "ndcg@50", "mrr@50"]:
            row[mk] = r["metrics"].get(mk, "")
        rrf3_rows.append(row)
    write_csv(output_dir / f"three_channel_rrf_{run_type}.csv", rrf3_rows, rrf3_keys)

    # 4-channel CSVs
    q4_keys = ["name", "fusion_type", "icf_quota", "tt_quota", "text_quota", "pop_quota",
               "recall@50", "recall@100", "ndcg@50", "mrr@50", "delta_vs_v1"]
    q4_rows = []
    for r in four_quota_results:
        row = {k: r.get(k, r["metrics"].get(k, "")) for k in q4_keys}
        for mk in ["recall@50", "recall@100", "ndcg@50", "mrr@50"]:
            row[mk] = r["metrics"].get(mk, "")
        q4_rows.append(row)
    write_csv(output_dir / f"four_channel_quota_{run_type}.csv", q4_rows, q4_keys)

    rrf4_keys = ["name", "fusion_type", "rrf_k", "recall@50", "recall@100", "ndcg@50", "mrr@50", "delta_vs_v1"]
    rrf4_rows = []
    for r in four_rrf_results:
        row = {k: r.get(k, r["metrics"].get(k, "")) for k in rrf4_keys}
        for mk in ["recall@50", "recall@100", "ndcg@50", "mrr@50"]:
            row[mk] = r["metrics"].get(mk, "")
        rrf4_rows.append(row)
    write_csv(output_dir / f"four_channel_rrf_{run_type}.csv", rrf4_rows, rrf4_keys)

    # Combined metrics summary
    best_3q = max(three_quota_results, key=lambda r: r["metrics"].get("recall@50", 0))
    best_3rrf = max(three_rrf_results, key=lambda r: r["metrics"].get("recall@50", 0))
    best_4q = max(four_quota_results, key=lambda r: r["metrics"].get("recall@50", 0))
    best_4rrf = max(four_rrf_results, key=lambda r: r["metrics"].get("recall@50", 0))

    metrics_summary = {
        "run_type": run_type,
        "n_eval_users": n_eval,
        "v1_best_recall50": V1_BEST_RECALL50,
        "n_zero_text_query_users": n_zero_query,
        "text_single_recall50": text_single_result["metrics"].get("recall@50", 0),
        "pop_single_recall50": pop_single_result["metrics"].get("recall@50", 0),
        "two_ch_rrf_k60_recall50": next(
            (r["metrics"].get("recall@50", 0) for r in two_rrf_results if r.get("rrf_k") == 60), None
        ),
        "best_3ch_quota": {
            "name": best_3q["name"],
            "recall50": best_3q["metrics"].get("recall@50", 0),
            "delta_vs_v1": best_3q.get("delta_vs_v1"),
        },
        "best_3ch_rrf": {
            "name": best_3rrf["name"],
            "recall50": best_3rrf["metrics"].get("recall@50", 0),
            "delta_vs_v1": best_3rrf.get("delta_vs_v1"),
        },
        "best_4ch_quota": {
            "name": best_4q["name"],
            "recall50": best_4q["metrics"].get("recall@50", 0),
            "delta_vs_v1": best_4q.get("delta_vs_v1"),
        },
        "best_4ch_rrf": {
            "name": best_4rrf["name"],
            "recall50": best_4rrf["metrics"].get("recall@50", 0),
            "delta_vs_v1": best_4rrf.get("delta_vs_v1"),
        },
        "channel_overlap": overlap_4ch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / f"metrics_{run_type}.json", metrics_summary)

    # Human-readable report
    _write_report(
        output_dir / f"report_{run_type}.md",
        run_type, metrics_summary, overlap_4ch,
        text_single_result, pop_single_result,
        two_rrf_results,
        three_quota_results, three_rrf_results,
        four_quota_results, four_rrf_results,
        k_list,
    )

    logging.info("[Done] %s outputs saved to %s", run_type, output_dir)
    logging.info(
        "[Summary] text_single=%.6f pop_single=%.6f best_3ch_rrf=%.6f best_4ch_rrf=%.6f delta_best=%.6f",
        metrics_summary["text_single_recall50"],
        metrics_summary["pop_single_recall50"],
        metrics_summary["best_3ch_rrf"]["recall50"],
        metrics_summary["best_4ch_rrf"]["recall50"],
        max(
            metrics_summary["best_3ch_rrf"].get("delta_vs_v1") or -999,
            metrics_summary["best_4ch_rrf"].get("delta_vs_v1") or -999,
        ),
    )


def _write_report(
    path: Path,
    run_type: str,
    summary: dict[str, Any],
    overlap_4ch: dict[str, Any],
    text_single: dict[str, Any],
    pop_single: dict[str, Any],
    two_rrf_results: list[dict[str, Any]],
    three_quota: list[dict[str, Any]],
    three_rrf: list[dict[str, Any]],
    four_quota: list[dict[str, Any]],
    four_rrf: list[dict[str, Any]],
    k_list: list[int],
) -> None:
    def mrow(r: dict[str, Any]) -> str:
        m = r["metrics"]
        d = r.get("delta_vs_v1", "")
        delta_str = f"{d:+.6f}" if isinstance(d, float) else str(d)
        return (
            f"| {r['name']} | {m.get('recall@50',0):.6f} | {m.get('ndcg@50',0):.6f}"
            f" | {m.get('mrr@50',0):.6f} | {delta_str} |"
        )

    lines = [
        f"# Multi-channel Retrieval v2 Report ({run_type})",
        "",
        f"**Run time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**v1 best baseline:** ItemCF + TwoTower RRF k=60  Recall@50 = {V1_BEST_RECALL50}",
        f"**Eval users:** {summary['n_eval_users']:,}",
        "",
        "---",
        "",
        "## Channel Overlap & Unique Hits (all 4 channels, @50)",
        "",
        "| Channel | Hits@50 | Hit rate | Unique hits@50 | Unique hit rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ch in ["itemcf", "twotower", "text", "pop"]:
        lines.append(
            f"| {ch} | {overlap_4ch.get(f'{ch}_hits@50', '')} | {overlap_4ch.get(f'{ch}_hit_rate@50', ''):.4f}"
            f" | {overlap_4ch.get(f'{ch}_unique_hits@50', '')} | {overlap_4ch.get(f'{ch}_unique_hit_rate@50', ''):.4f} |"
        )
    lines += [
        "",
        "**Pairwise Jaccard overlap@50:**",
        "",
        "| Pair | Jaccard |",
        "| --- | ---: |",
    ]
    for key, val in overlap_4ch.items():
        if key.startswith("jaccard_"):
            pair = key.replace("jaccard_", "").replace("@50", "")
            lines.append(f"| {pair} | {val:.4f} |")
    lines += [
        "",
        "---",
        "",
        "## Single-Channel Diagnostics",
        "",
        f"**Text Semantic** (top-50, decay_rate={text_single.get('text_decay_rate', 0.8)}):",
        f"- Recall@50 = {text_single['metrics'].get('recall@50',0):.6f}",
        f"- NDCG@50  = {text_single['metrics'].get('ndcg@50',0):.6f}",
        f"- Zero-query users (no text history): {text_single.get('n_zero_query_users', 'n/a')}",
        "",
        f"**Popularity Fallback** (top-50 globally popular, seen-filtered):",
        f"- Recall@50 = {pop_single['metrics'].get('recall@50',0):.6f}",
        f"- NDCG@50  = {pop_single['metrics'].get('ndcg@50',0):.6f}",
        "",
        "---",
        "",
        "## 2-Channel Reference (sanity check, v1 replicate)",
        "",
        "| Config | Recall@50 | NDCG@50 | MRR@50 | delta vs v1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in two_rrf_results:
        lines.append(mrow(r))
    lines += [
        "",
        "---",
        "",
        "## 3-Channel: ItemCF + TwoTower + Text",
        "",
        "### Quota Sweep",
        "",
        "| Config | Recall@50 | NDCG@50 | MRR@50 | delta vs v1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in three_quota:
        lines.append(mrow(r))
    lines += [
        "",
        "### RRF Sweep",
        "",
        "| Config | Recall@50 | NDCG@50 | MRR@50 | delta vs v1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in three_rrf:
        lines.append(mrow(r))
    lines += [
        "",
        "---",
        "",
        "## 4-Channel: ItemCF + TwoTower + Text + Popularity",
        "",
        "### Quota Sweep",
        "",
        "| Config | Recall@50 | NDCG@50 | MRR@50 | delta vs v1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in four_quota:
        lines.append(mrow(r))
    lines += [
        "",
        "### RRF Sweep",
        "",
        "| Config | Recall@50 | NDCG@50 | MRR@50 | delta vs v1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in four_rrf:
        lines.append(mrow(r))
    lines += [
        "",
        "---",
        "",
        "## Evaluation Notes",
        "",
        "- All results: offline evaluation, test split, seen-item mask = train+valid",
        "- Not online A/B; not production latency",
        "- RRF uses rank information only; no test labels used",
        "- Text semantic: time-decay weighted mean of history item text embeddings (384-dim sentence-transformers, frozen)",
        "  mirroring TwoTower decay_rate=0.8; users with no text-covered history get empty candidates",
        "- Popularity fallback: globally sorted by train interaction count, seen-item filtered per user",
        "- Recall@100 = Recall@50 when top_n=50 (same as v1 limitation)",
        f"- v1 best two-channel baseline: RRF k=60  Recall@50 = {V1_BEST_RECALL50}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_v2_config(Path(args.config))
    set_seed(int(cfg["seed"]))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", cfg)

    if not args.full_only:
        logging.info("=== Smoke test (%d users) ===", cfg["smoke_users"])
        run(cfg, output_dir, smoke=True)
        logging.info("=== Smoke test PASSED ===")

    if not args.smoke_only:
        logging.info("=== Full eval ===")
        run(cfg, output_dir, smoke=False)
        logging.info("=== Full eval DONE ===")


if __name__ == "__main__":
    main()

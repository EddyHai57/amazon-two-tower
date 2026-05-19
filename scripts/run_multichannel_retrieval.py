#!/usr/bin/env python3
"""
Multi-channel retrieval: ItemCF + Text+Time-decay Two-Tower fusion.

Generates candidates from both channels, then fuses via quota merge and RRF.
Outputs Recall@K, NDCG@K, MRR@K, overlap, unique hit, and popularity-bucket breakdown.
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
from run_itemcf import build_train_sets, build_item_similarity, add_valid_to_seen
from train_text_time_decay_mean_pool_two_tower_smoke import (
    DataBundle,
    TextTimeDecayMeanPoolTwoTower,
    build_history_matrix,
    build_model,
    build_seen_items,
    load_config as load_train_config,
    load_data,
    merge_seen_items,
    resolve_device,
    set_seed,
)

TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-channel retrieval fusion benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_only", action="store_true", help="Run smoke test only, skip full eval.")
    parser.add_argument("--full_only", action="store_true", help="Skip smoke, run full eval only.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_mc_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = [
        "data_dir", "train_config", "checkpoint",
        "output_dir", "seed", "device",
        "smoke_users", "candidates_per_channel",
        "quota_sweep", "rrf_k_values", "rrf_top_n",
        "eval_k_list", "itemcf_sim_topk", "itemcf_max_user_history",
    ]
    for key in required:
        if key not in config:
            raise KeyError(f"config missing required key: {key}")
    return config


# ---------------------------------------------------------------------------
# ItemCF candidate generation
# ---------------------------------------------------------------------------

def generate_itemcf_candidates(
    test_eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    eval_seen: dict[int, set[int]],
    limited_history: dict[int, list[int]],
    similarity: dict[int, list[tuple[int, float]]],
    top_k: int,
) -> dict[int, list[int]]:
    """Returns {user_idx: [item_idx, ...]} with at most top_k candidates (seen-filtered)."""
    candidates: dict[int, list[int]] = {}
    cold_mask = test_eval_df["is_cold_item_for_eval"].astype(bool)
    eval_targets = test_eval_df[~cold_mask]
    for row in eval_targets.itertuples(index=False):
        user_idx = int(row.user_idx)
        if user_idx in candidates:
            continue
        target_item = int(row.item_idx)
        seen_items = eval_seen.get(user_idx, set())
        scores: defaultdict[int, float] = defaultdict(float)
        for history_item in limited_history.get(user_idx, []):
            for candidate_item, sim_score in similarity.get(history_item, []):
                if candidate_item in seen_items and candidate_item != target_item:
                    continue
                scores[candidate_item] += sim_score
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        candidates[user_idx] = [item_idx for item_idx, _ in ranked[:top_k]]
    return candidates


# ---------------------------------------------------------------------------
# Two-Tower candidate generation
# ---------------------------------------------------------------------------

def encode_all_items(
    model: TextTimeDecayMeanPoolTwoTower,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + num_items, num_items)
            end = min(start + 65536, num_items)
            item_idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(item_idx).detach().cpu())
    return torch.cat(chunks, dim=0)  # (n_items, dim)


def generate_twotower_candidates(
    model: TextTimeDecayMeanPoolTwoTower,
    test_eval_users: list[int],
    test_history_matrix: np.ndarray,
    test_seen: dict[int, set[int]],
    item_emb_cpu: torch.Tensor,
    top_k: int,
    device: torch.device,
    batch_size: int = 256,
) -> dict[int, np.ndarray]:
    """Returns {user_idx: int32 array of top_k item indices} (seen-filtered)."""
    candidates: dict[int, np.ndarray] = {}
    n_items = item_emb_cpu.shape[0]
    item_emb_gpu = item_emb_cpu.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_eval_users), batch_size):
            batch_users = test_eval_users[start: start + batch_size]
            users_np = np.asarray(batch_users, dtype=np.int64)
            user_tensor = torch.as_tensor(users_np, device=device)
            history_tensor = torch.as_tensor(
                test_history_matrix[users_np], dtype=torch.long, device=device
            )
            user_emb = model.encode_users(user_tensor, history_tensor)  # (B, dim)
            scores = (user_emb @ item_emb_gpu.T).cpu().numpy()  # (B, n_items)
            for j, user_idx in enumerate(batch_users):
                s = scores[j].copy()
                seen = test_seen.get(int(user_idx), set())
                if seen:
                    seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen))
                    s[seen_arr] = -np.inf
                if top_k >= n_items:
                    top_idx = np.argsort(s)[::-1]
                else:
                    top_idx = np.argpartition(s, -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(s[top_idx])[::-1]]
                candidates[int(user_idx)] = top_idx.astype(np.int32)
    return candidates


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def quota_merge(
    itemcf_cands: list[int],
    tt_cands: list[int],
    itemcf_quota: int,
    tt_quota: int,
) -> list[int]:
    """Pick top itemcf_quota from itemcf and top tt_quota from tt, dedup, preserve order."""
    result: list[int] = []
    seen_set: set[int] = set()
    for item in itemcf_cands[:itemcf_quota]:
        if item not in seen_set:
            result.append(item)
            seen_set.add(item)
    for item in tt_cands[:tt_quota]:
        if item not in seen_set:
            result.append(item)
            seen_set.add(item)
    return result


def rrf_merge(
    itemcf_cands: list[int],
    tt_cands: list[int],
    k: int,
    top_n: int,
) -> list[int]:
    """Reciprocal Rank Fusion. score(item) = sum of 1/(k + rank) across channels."""
    scores: defaultdict[int, float] = defaultdict(float)
    for rank, item in enumerate(itemcf_cands, start=1):
        scores[item] += 1.0 / (k + rank)
    for rank, item in enumerate(tt_cands, start=1):
        scores[item] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [item for item, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    recommended: list[int],
    target: int,
    k_list: list[int],
) -> dict[str, float]:
    if target not in recommended:
        return {f"recall@{k}": 0.0 for k in k_list} | {f"ndcg@{k}": 0.0 for k in k_list} | {f"mrr@{k}": 0.0 for k in k_list}
    rank = recommended.index(target) + 1
    result: dict[str, float] = {}
    for k in k_list:
        if rank <= k:
            result[f"recall@{k}"] = 1.0
            result[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1)
            result[f"mrr@{k}"] = 1.0 / rank
        else:
            result[f"recall@{k}"] = 0.0
            result[f"ndcg@{k}"] = 0.0
            result[f"mrr@{k}"] = 0.0
    return result


def aggregate_metrics(
    per_user_metrics: list[dict[str, float]],
    k_list: list[int],
) -> dict[str, float]:
    n = len(per_user_metrics)
    agg: dict[str, float] = {}
    for k in k_list:
        for metric in ["recall", "ndcg", "mrr"]:
            key = f"{metric}@{k}"
            agg[key] = sum(m[key] for m in per_user_metrics) / n if n else 0.0
    return agg


def evaluate_fusion(
    test_eval_df: pd.DataFrame,
    itemcf_candidates: dict[int, list[int]],
    tt_candidates: dict[int, np.ndarray],
    quota_sweep: list[list[int]],
    rrf_k_values: list[int],
    rrf_top_n: int,
    k_list: list[int],
    item_popularity: dict[int, int],
    pop_buckets: list[tuple[int, int | None]],
) -> dict[str, Any]:
    cold_mask = test_eval_df["is_cold_item_for_eval"].astype(bool)
    eval_targets = test_eval_df[~cold_mask]
    n_eval = len(eval_targets)

    quota_results: list[dict[str, Any]] = []
    rrf_results: list[dict[str, Any]] = []

    # Overlap and unique hit stats (computed once against the first quota combo)
    overlap_stats: dict[str, Any] = {}

    # Compute overlap and unique hit between ItemCF top-50 and TT top-50
    logging.info("Computing overlap/unique hit stats between ItemCF@50 and TwoTower@50...")
    overlap_hits = 0
    tt_unique_hits = 0
    icf_unique_hits = 0
    overlap_candidates = 0.0
    tt_cov_items: set[int] = set()
    icf_cov_items: set[int] = set()
    tt_unique_hit_users = 0
    icf_unique_hit_users = 0
    total_target_hits = 0

    for row in eval_targets.itertuples(index=False):
        user_idx = int(row.user_idx)
        target = int(row.item_idx)
        icf_set = set(itemcf_candidates.get(user_idx, [])[:50])
        tt_set = set((tt_candidates.get(user_idx, np.array([], dtype=np.int32))[:50]).tolist())
        icf_cov_items.update(icf_set)
        tt_cov_items.update(tt_set)
        inter = icf_set & tt_set
        union = icf_set | tt_set
        overlap_candidates += len(inter) / max(len(union), 1)
        icf_hit = target in icf_set
        tt_hit = target in tt_set
        total_target_hits += 1 if (icf_hit or tt_hit) else 0
        if icf_hit and tt_hit:
            overlap_hits += 1
        elif tt_hit and not icf_hit:
            tt_unique_hits += 1
            tt_unique_hit_users += 1
        elif icf_hit and not tt_hit:
            icf_unique_hits += 1
            icf_unique_hit_users += 1

    overlap_stats = {
        "n_eval_users": n_eval,
        "avg_candidate_overlap@50": float(overlap_candidates / max(n_eval, 1)),
        "overlap_hits@50": int(overlap_hits),
        "tt_unique_hits@50": int(tt_unique_hits),
        "icf_unique_hits@50": int(icf_unique_hits),
        "tt_unique_hit_rate@50": float(tt_unique_hits / max(n_eval, 1)),
        "icf_unique_hit_rate@50": float(icf_unique_hits / max(n_eval, 1)),
        "icf_item_coverage@50": int(len(icf_cov_items)),
        "tt_item_coverage@50": int(len(tt_cov_items)),
    }
    logging.info(
        "Overlap stats: avg_candidate_overlap=%.4f tt_unique_hits=%d icf_unique_hits=%d",
        overlap_stats["avg_candidate_overlap@50"],
        tt_unique_hits,
        icf_unique_hits,
    )

    def run_single(name: str, get_merged: Any) -> dict[str, Any]:
        per_user: list[dict[str, float]] = []
        bucket_per_user: dict[str, list[dict[str, float]]] = {f"bucket_{i}": [] for i in range(len(pop_buckets))}
        for row in eval_targets.itertuples(index=False):
            user_idx = int(row.user_idx)
            target = int(row.item_idx)
            merged = get_merged(user_idx)
            m = compute_metrics(merged, target, k_list)
            per_user.append(m)
            # Bucket assignment by target item popularity
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
        agg["n_eval_users"] = n_eval
        bucket_agg: dict[str, Any] = {}
        for bidx, (lo, hi) in enumerate(pop_buckets):
            bkey = f"bucket_{bidx}"
            blabel = f"{lo}-{hi}" if hi is not None else f">{lo - 1}"
            bm = aggregate_metrics(bucket_per_user[bkey], k_list)
            bm["n_users"] = len(bucket_per_user[bkey])
            bucket_agg[blabel] = bm
        return {"name": name, "metrics": agg, "bucket_breakdown": bucket_agg}

    # Quota sweep
    for combo in quota_sweep:
        icf_q, tt_q = int(combo[0]), int(combo[1])
        name = f"quota_icf{icf_q}_tt{tt_q}"
        logging.info("Running %s ...", name)

        def get_merged_quota(user_idx: int, icf_q: int = icf_q, tt_q: int = tt_q) -> list[int]:
            icf = itemcf_candidates.get(user_idx, [])
            tt = (tt_candidates.get(user_idx, np.array([], dtype=np.int32))).tolist()
            return quota_merge(icf, tt, icf_q, tt_q)

        result = run_single(name, get_merged_quota)
        result["fusion_type"] = "quota"
        result["itemcf_quota"] = icf_q
        result["tt_quota"] = tt_q
        quota_results.append(result)
        logging.info("%s Recall@50=%.6f", name, result["metrics"]["recall@50"])

    # RRF sweep
    for k in rrf_k_values:
        name = f"rrf_k{k}"
        logging.info("Running %s ...", name)

        def get_merged_rrf(user_idx: int, k: int = k) -> list[int]:
            icf = itemcf_candidates.get(user_idx, [])
            tt = (tt_candidates.get(user_idx, np.array([], dtype=np.int32))).tolist()
            return rrf_merge(icf, tt, k, rrf_top_n)

        result = run_single(name, get_merged_rrf)
        result["fusion_type"] = "rrf"
        result["rrf_k"] = k
        result["rrf_top_n"] = rrf_top_n
        rrf_results.append(result)
        logging.info("%s Recall@50=%.6f", name, result["metrics"]["recall@50"])

    return {
        "overlap_stats": overlap_stats,
        "quota_results": quota_results,
        "rrf_results": rrf_results,
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report_md(path: Path, config: dict[str, Any], eval_results: dict[str, Any], run_type: str) -> None:
    overlap = eval_results["overlap_stats"]
    lines = [
        f"# Multi-channel Retrieval Fusion Report ({run_type})",
        "",
        f"**Run time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Config:** {config.get('output_dir')}",
        "",
        "## Overlap / Unique Hit Analysis (ItemCF@50 vs TwoTower@50)",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| avg candidate overlap@50 | {overlap['avg_candidate_overlap@50']:.4f} |",
        f"| overlap hits@50 (both channels hit) | {overlap['overlap_hits@50']} |",
        f"| TwoTower unique hits@50 | {overlap['tt_unique_hits@50']} |",
        f"| ItemCF unique hits@50 | {overlap['icf_unique_hits@50']} |",
        f"| TwoTower unique hit rate@50 | {overlap['tt_unique_hit_rate@50']:.4f} |",
        f"| ItemCF unique hit rate@50 | {overlap['icf_unique_hit_rate@50']:.4f} |",
        f"| ItemCF item coverage@50 | {overlap['icf_item_coverage@50']} |",
        f"| TwoTower item coverage@50 | {overlap['tt_item_coverage@50']} |",
        "",
        "## Quota Sweep (Recall@50)",
        "",
        "| Combo | Recall@50 | Recall@100 | NDCG@50 | MRR@50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in eval_results["quota_results"]:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {m.get('recall@50', 0):.6f} | {m.get('recall@100', 0):.6f}"
            f" | {m.get('ndcg@50', 0):.6f} | {m.get('mrr@50', 0):.6f} |"
        )
    lines += [
        "",
        "## RRF Sweep (Recall@50)",
        "",
        "| Config | Recall@50 | Recall@100 | NDCG@50 | MRR@50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in eval_results["rrf_results"]:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {m.get('recall@50', 0):.6f} | {m.get('recall@100', 0):.6f}"
            f" | {m.get('ndcg@50', 0):.6f} | {m.get('mrr@50', 0):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: dict[str, Any], output_dir: Path, smoke: bool) -> dict[str, Any]:
    data_dir = Path(config["data_dir"])
    train_config = load_train_config(Path(config["train_config"]))
    bundle: DataBundle = load_data(data_dir)
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_max_len = int(train_config["history_max_len"])

    # Seen-item masks
    train_seen = build_seen_items(bundle.train_df)
    test_seen = merge_seen_items(train_seen, bundle.valid_df)

    # Test history includes train + valid
    test_history_frame = pd.concat(
        [bundle.train_df[TRAIN_COLUMNS], bundle.valid_df[TRAIN_COLUMNS]], ignore_index=True
    )
    test_history_matrix, _ = build_history_matrix(test_history_frame, num_users, history_max_len)

    # ItemCF
    logging.info("[ItemCF] Building train sets...")
    icf_full_seen, icf_limited_history = build_train_sets(
        bundle.train_df, int(config["itemcf_max_user_history"])
    )
    icf_eval_seen = add_valid_to_seen(icf_full_seen, bundle.valid_df)
    logging.info("[ItemCF] Building item similarity (sim_topk=%s)...", config["itemcf_sim_topk"])
    similarity = build_item_similarity(icf_limited_history, int(config["itemcf_sim_topk"]))

    # Item popularity for bucket analysis
    item_pop: Counter[int] = Counter(bundle.train_df["item_idx"].tolist())
    item_popularity: dict[int, int] = dict(item_pop)

    # Two-Tower model
    logging.info("[TwoTower] Loading model from %s...", config["checkpoint"])
    device = resolve_device(str(config["device"]))
    model = build_model(train_config, bundle.stats, device)
    checkpoint = torch.load(Path(config["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(
        "[TwoTower] checkpoint loaded: epoch=%s recall@50=%.6f",
        checkpoint.get("epoch"),
        float(checkpoint.get("best_metric_value", 0.0)),
    )
    logging.info("[TwoTower] Encoding all items...")
    item_emb_cpu = encode_all_items(model, num_items, device)
    logging.info("[TwoTower] item_emb shape: %s", tuple(item_emb_cpu.shape))

    # Test eval set
    cold_mask = bundle.test_df["is_cold_item_for_eval"].astype(bool)
    eval_targets_df = bundle.test_df[~cold_mask].copy()
    all_eval_users = eval_targets_df["user_idx"].unique().tolist()
    logging.info("Total non-cold test users: %d", len(all_eval_users))

    if smoke:
        rng = np.random.default_rng(int(config["seed"]))
        n_smoke = min(int(config["smoke_users"]), len(all_eval_users))
        sampled_users = set(rng.choice(all_eval_users, size=n_smoke, replace=False).tolist())
        eval_df = eval_targets_df[eval_targets_df["user_idx"].isin(sampled_users)].copy()
        logging.info("[smoke] Using %d users", len(eval_df))
    else:
        eval_df = eval_targets_df
        logging.info("[full] Using all %d users", len(eval_df))

    eval_users = eval_df["user_idx"].unique().tolist()

    # Generate candidates
    top_k = int(config["candidates_per_channel"])
    logging.info("[ItemCF] Generating candidates (top_k=%d)...", top_k)
    itemcf_cands = generate_itemcf_candidates(
        eval_df, bundle.train_df, icf_eval_seen, icf_limited_history, similarity, top_k
    )
    logging.info("[TwoTower] Generating candidates (top_k=%d)...", top_k)
    tt_cands = generate_twotower_candidates(
        model, eval_users, test_history_matrix, test_seen, item_emb_cpu, top_k, device
    )

    # Sanity checks
    n_icf = sum(1 for u in eval_users if u in itemcf_cands)
    n_tt = sum(1 for u in eval_users if u in tt_cands)
    logging.info("ItemCF candidates: %d/%d users", n_icf, len(eval_users))
    logging.info("TwoTower candidates: %d/%d users", n_tt, len(eval_users))

    # Parse popularity buckets
    raw_buckets = config.get("popularity_buckets", [[1, 5], [6, 20], [21, 100], [101, None]])
    pop_buckets = [(int(lo), None if hi is None else int(hi)) for lo, hi in raw_buckets]

    # Evaluate all fusion strategies
    logging.info("[Fusion] Running quota and RRF sweeps...")
    eval_results = evaluate_fusion(
        eval_df,
        itemcf_cands,
        tt_cands,
        config["quota_sweep"],
        [int(k) for k in config["rrf_k_values"]],
        int(config["rrf_top_n"]),
        [int(k) for k in config["eval_k_list"]],
        item_popularity,
        pop_buckets,
    )

    run_type = "smoke" if smoke else "full"
    eval_results["run_type"] = run_type
    eval_results["n_eval_users"] = len(eval_df)
    eval_results["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save outputs
    result_path = output_dir / f"results_{run_type}.json"
    write_json(result_path, eval_results)
    write_report_md(output_dir / f"report_{run_type}.md", config, eval_results, run_type)

    # Save quota CSV
    quota_keys = ["name", "fusion_type", "itemcf_quota", "tt_quota", "recall@50", "recall@100", "ndcg@50", "mrr@50"]
    quota_rows = []
    for r in eval_results["quota_results"]:
        row = {k: r.get(k, r["metrics"].get(k, "")) for k in quota_keys}
        for mk in ["recall@50", "recall@100", "ndcg@50", "mrr@50"]:
            row[mk] = r["metrics"].get(mk, "")
        quota_rows.append(row)
    write_csv(output_dir / f"quota_sweep_{run_type}.csv", quota_rows, quota_keys)

    # Save RRF CSV
    rrf_keys = ["name", "fusion_type", "rrf_k", "rrf_top_n", "recall@50", "recall@100", "ndcg@50", "mrr@50"]
    rrf_rows = []
    for r in eval_results["rrf_results"]:
        row = {k: r.get(k, "") for k in rrf_keys}
        for mk in ["recall@50", "recall@100", "ndcg@50", "mrr@50"]:
            row[mk] = r["metrics"].get(mk, "")
        rrf_rows.append(row)
    write_csv(output_dir / f"rrf_sweep_{run_type}.csv", rrf_rows, rrf_keys)

    logging.info("[Done] Results saved to %s", output_dir)
    return eval_results


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_mc_config(Path(args.config))
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    if not args.full_only:
        logging.info("=== Smoke test (%d users) ===", config["smoke_users"])
        run(config, output_dir, smoke=True)
        logging.info("=== Smoke test PASSED ===")

    if not args.smoke_only:
        logging.info("=== Full eval ===")
        run(config, output_dir, smoke=False)
        logging.info("=== Full eval DONE ===")


if __name__ == "__main__":
    main()

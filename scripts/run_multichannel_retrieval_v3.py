#!/usr/bin/env python3
"""
Multi-channel retrieval v3: balanced fusion with diversity-aware evaluation.

Problem: v2's 4ch_rrf_k60 achieves Recall@50=0.108766 (+12.4% vs v1) but at severe
diversity cost (avg_pop ×6.2, item_coverage -23.5%, tail/mid bucket regression).

Experiments:
  1. Pop-limited quota: ICF + TT + Pop with small pop quota (3-5 slots)
  2. Pop-limited quota: ICF + TT + Text + Pop (small text+pop quotas)
  3. Weighted RRF: icf_w=tt_w=1.0, sweep pop_w ∈ {0.1,0.2,0.3,0.5,1.0},
                  text_w ∈ {0.0,0.2,0.3}

Diversity metrics added to every eval result:
  avg_rec_popularity: mean over users of mean(train_count(recommended_items))
  item_coverage: unique items recommended across all eval users

Reference baselines included in same run:
  ref_v1_2ch_rrf_k60  → must reproduce Recall@50 = 0.096727
  ref_v2_4ch_rrf_k60  → must reproduce Recall@50 = 0.108766

Output dir: outputs/multichannel_v3_balanced/
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

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
    quota_merge_n,
    rrf_merge_n,
    write_csv,
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

V2_BEST_RECALL50 = 0.108766  # 4ch_rrf_k60 full eval (496,470 users)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-channel retrieval v3: balanced fusion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--full_only", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Weighted RRF
# ---------------------------------------------------------------------------

def weighted_rrf_merge_n(
    channel_cands: list[list[int]],
    weights: list[float],
    k: int,
    top_n: int,
) -> list[int]:
    """Weighted RRF: score(item) = sum of w_ch / (k + rank) over channels.

    Channels with weight <= 0 are skipped entirely.
    """
    scores: defaultdict[int, float] = defaultdict(float)
    for cands, w in zip(channel_cands, weights):
        if w <= 0:
            continue
        for rank, item in enumerate(cands, start=1):
            scores[item] += w / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [item for item, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Diversity-aware evaluation
# ---------------------------------------------------------------------------

def run_eval_with_diversity(
    name: str,
    eval_targets: pd.DataFrame,
    get_merged: Callable[[int], list[int]],
    k_list: list[int],
    item_popularity: dict[int, int],
    pop_buckets: list[tuple[int, int | None]],
) -> dict[str, Any]:
    """Evaluate a fusion strategy; returns recall/ndcg/mrr + diversity + bucket breakdown.

    Diversity metrics:
      avg_rec_popularity: mean over users of mean(item_pop(rec_list))
      item_coverage: unique items recommended across all eval users
    """
    n = len(eval_targets)
    per_user: list[dict[str, float]] = []
    bucket_per_user: dict[str, list[dict[str, float]]] = {
        f"bucket_{i}": [] for i in range(len(pop_buckets))
    }
    total_rec_pop: float = 0.0
    all_rec_items: set[int] = set()

    for row in eval_targets.itertuples(index=False):
        user_idx = int(row.user_idx)
        target = int(row.item_idx)
        merged = get_merged(user_idx)

        if merged:
            user_avg_pop = sum(item_popularity.get(i, 0) for i in merged) / len(merged)
            total_rec_pop += user_avg_pop
            all_rec_items.update(merged)

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
    agg["avg_rec_popularity"] = round(total_rec_pop / max(n, 1), 2)
    agg["item_coverage"] = len(all_rec_items)

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

    # Seen-item masks
    train_seen = build_seen_items(bundle.train_df)
    test_seen = merge_seen_items(train_seen, bundle.valid_df)

    # Test history matrix (train + valid)
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

    # Item popularity (train-only)
    item_pop_counter: Counter[int] = Counter(bundle.train_df["item_idx"].tolist())
    item_popularity: dict[int, int] = dict(item_pop_counter)
    pop_sorted_items = [item for item, _ in item_pop_counter.most_common()]

    # Two-Tower
    device = resolve_device(str(config["device"]))
    logging.info("[TwoTower] Loading model from %s...", config["checkpoint"])
    model = build_model(train_config, bundle.stats, device)
    ckpt = torch.load(Path(config["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info(
        "[TwoTower] Loaded: epoch=%s recall@50=%.6f",
        ckpt.get("epoch"),
        float(ckpt.get("best_metric_value", 0.0)),
    )
    item_emb_cpu = encode_all_items(model, num_items, device)

    # Text embeddings
    logging.info("[TextSemantic] Loading text embeddings from %s...", config["item_text_emb_path"])
    item_text_norm_np = load_text_embeddings(Path(config["item_text_emb_path"]), num_items)
    item_text_norm_gpu = torch.from_numpy(item_text_norm_np).to(device)

    # Select eval users
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
    eval_targets = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()

    # Parse config
    top_k = int(config["candidates_per_channel"])
    pop_buf = int(config["pop_buffer_size"])
    k_list = [int(k) for k in config["eval_k_list"]]
    raw_buckets = config.get("popularity_buckets", [[1, 5], [6, 20], [21, 100], [101, None]])
    pop_buckets = [(int(lo), None if hi is None else int(hi)) for lo, hi in raw_buckets]
    text_decay = float(config["text_decay_rate"])
    rrf_top_n = int(config.get("rrf_top_n", 50))

    # Generate all 4 channel candidates
    logging.info("[ItemCF] Generating candidates (top_k=%d)...", top_k)
    icf_cands = generate_itemcf_candidates(
        eval_df, bundle.train_df, icf_eval_seen, icf_limited_history, similarity, top_k
    )
    logging.info("[TwoTower] Generating candidates (top_k=%d)...", top_k)
    tt_cands_raw = generate_twotower_candidates(
        model, eval_users, test_history_matrix, test_seen, item_emb_cpu, top_k, device
    )
    tt_cands: dict[int, list[int]] = {u: v.tolist() for u, v in tt_cands_raw.items()}

    logging.info("[TextSemantic] Generating candidates (top_k=%d, decay_rate=%.2f)...", top_k, text_decay)
    text_cands, n_zero_query = generate_text_semantic_candidates(
        eval_users, test_history_matrix, test_seen, item_text_norm_gpu, top_k, text_decay, device
    )
    logging.info("[TextSemantic] n_zero_query_users=%d / %d", n_zero_query, len(eval_users))

    logging.info("[Popularity] Generating candidates (top_k=%d, buffer=%d)...", top_k, pop_buf)
    pop_cands = generate_popularity_candidates(
        eval_users, test_seen, pop_sorted_items, top_k, pop_buf
    )

    all_results: list[dict[str, Any]] = []

    # ----------------------------------------------------------------
    # Reference baselines
    # ----------------------------------------------------------------
    logging.info("[Ref] v1 2ch_rrf_k60 (expected Recall@50=0.096727)...")
    v1_ref = run_eval_with_diversity(
        "ref_v1_2ch_rrf_k60", eval_targets,
        lambda u: rrf_merge(icf_cands.get(u, []), tt_cands.get(u, []), 60, rrf_top_n),
        k_list, item_popularity, pop_buckets,
    )
    v1_ref["fusion_type"] = "ref_v1"
    r50 = v1_ref["metrics"]["recall@50"]
    v1_ref["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
    logging.info(
        "[Ref] v1 Recall@50=%.6f avg_pop=%.0f coverage=%d (expected 0.096727, delta=%+.6f)",
        r50, v1_ref["metrics"]["avg_rec_popularity"],
        v1_ref["metrics"]["item_coverage"], r50 - V1_BEST_RECALL50,
    )
    all_results.append(v1_ref)

    logging.info("[Ref] v2 4ch_rrf_k60 (expected Recall@50=0.108766)...")
    v2_ref = run_eval_with_diversity(
        "ref_v2_4ch_rrf_k60", eval_targets,
        lambda u: rrf_merge_n([
            icf_cands.get(u, []), tt_cands.get(u, []),
            text_cands.get(u, []), pop_cands.get(u, []),
        ], 60, rrf_top_n),
        k_list, item_popularity, pop_buckets,
    )
    v2_ref["fusion_type"] = "ref_v2"
    r50 = v2_ref["metrics"]["recall@50"]
    v2_ref["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
    logging.info(
        "[Ref] v2 Recall@50=%.6f avg_pop=%.0f coverage=%d (expected 0.108766, delta=%+.6f)",
        r50, v2_ref["metrics"]["avg_rec_popularity"],
        v2_ref["metrics"]["item_coverage"], r50 - V2_BEST_RECALL50,
    )
    all_results.append(v2_ref)

    # ----------------------------------------------------------------
    # Pop-limited quota combos: 3-channel (ICF + TT + Pop)
    # ----------------------------------------------------------------
    for combo in config.get("pop_quota_3ch", []):
        icf_q, tt_q, pop_q = int(combo[0]), int(combo[1]), int(combo[2])
        name = f"quota_icf{icf_q}_tt{tt_q}_pop{pop_q}"
        logging.info("[Quota-3ch] %s ...", name)
        res = run_eval_with_diversity(
            name, eval_targets,
            lambda u, iq=icf_q, tq=tt_q, pq=pop_q: quota_merge_n([
                (icf_cands.get(u, []), iq),
                (tt_cands.get(u, []), tq),
                (pop_cands.get(u, []), pq),
            ]),
            k_list, item_popularity, pop_buckets,
        )
        res["fusion_type"] = "3ch_quota_pop"
        res["icf_quota"] = icf_q
        res["tt_quota"] = tt_q
        res["pop_quota"] = pop_q
        r50 = res["metrics"]["recall@50"]
        res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
        logging.info(
            "[Quota-3ch] %s Recall@50=%.6f avg_pop=%.0f coverage=%d delta=%+.6f",
            name, r50, res["metrics"]["avg_rec_popularity"],
            res["metrics"]["item_coverage"], r50 - V1_BEST_RECALL50,
        )
        all_results.append(res)

    # ----------------------------------------------------------------
    # Pop-limited quota combos: 4-channel (ICF + TT + Text + Pop)
    # ----------------------------------------------------------------
    for combo in config.get("pop_quota_4ch", []):
        icf_q, tt_q, text_q, pop_q = int(combo[0]), int(combo[1]), int(combo[2]), int(combo[3])
        name = f"quota_icf{icf_q}_tt{tt_q}_text{text_q}_pop{pop_q}"
        logging.info("[Quota-4ch] %s ...", name)
        res = run_eval_with_diversity(
            name, eval_targets,
            lambda u, iq=icf_q, tq=tt_q, xq=text_q, pq=pop_q: quota_merge_n([
                (icf_cands.get(u, []), iq),
                (tt_cands.get(u, []), tq),
                (text_cands.get(u, []), xq),
                (pop_cands.get(u, []), pq),
            ]),
            k_list, item_popularity, pop_buckets,
        )
        res["fusion_type"] = "4ch_quota_pop"
        res["icf_quota"] = icf_q
        res["tt_quota"] = tt_q
        res["text_quota"] = text_q
        res["pop_quota"] = pop_q
        r50 = res["metrics"]["recall@50"]
        res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
        logging.info(
            "[Quota-4ch] %s Recall@50=%.6f avg_pop=%.0f coverage=%d delta=%+.6f",
            name, r50, res["metrics"]["avg_rec_popularity"],
            res["metrics"]["item_coverage"], r50 - V1_BEST_RECALL50,
        )
        all_results.append(res)

    # ----------------------------------------------------------------
    # Weighted RRF sweep
    # ----------------------------------------------------------------
    wrrf_k = int(config.get("wrrf_k", 60))
    wrrf_icf_w = float(config.get("wrrf_icf_w", 1.0))
    wrrf_tt_w = float(config.get("wrrf_tt_w", 1.0))
    wrrf_pop_weights = [float(w) for w in config.get("wrrf_pop_weights", [0.1, 0.2, 0.3, 0.5, 1.0])]
    wrrf_text_weights = [float(w) for w in config.get("wrrf_text_weights", [0.0, 0.2, 0.3])]

    for pop_w in wrrf_pop_weights:
        for text_w in wrrf_text_weights:
            name = f"wrrf_pop{pop_w:.1f}_text{text_w:.1f}"
            logging.info(
                "[wRRF] %s (k=%d icf=%.1f tt=%.1f text=%.1f pop=%.1f) ...",
                name, wrrf_k, wrrf_icf_w, wrrf_tt_w, text_w, pop_w,
            )
            weights = [wrrf_icf_w, wrrf_tt_w, text_w, pop_w]
            res = run_eval_with_diversity(
                name, eval_targets,
                lambda u, w=weights: weighted_rrf_merge_n([
                    icf_cands.get(u, []), tt_cands.get(u, []),
                    text_cands.get(u, []), pop_cands.get(u, []),
                ], w, wrrf_k, rrf_top_n),
                k_list, item_popularity, pop_buckets,
            )
            res["fusion_type"] = "weighted_rrf"
            res["wrrf_k"] = wrrf_k
            res["icf_w"] = wrrf_icf_w
            res["tt_w"] = wrrf_tt_w
            res["text_w"] = text_w
            res["pop_w"] = pop_w
            r50 = res["metrics"]["recall@50"]
            res["delta_vs_v1"] = round(r50 - V1_BEST_RECALL50, 6)
            logging.info(
                "[wRRF] %s Recall@50=%.6f avg_pop=%.0f coverage=%d delta=%+.6f",
                name, r50, res["metrics"]["avg_rec_popularity"],
                res["metrics"]["item_coverage"], r50 - V1_BEST_RECALL50,
            )
            all_results.append(res)

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    write_json(output_dir / f"all_results_{run_type}.json", {
        "run_type": run_type,
        "n_eval_users": len(eval_targets),
        "v1_best_recall50": V1_BEST_RECALL50,
        "v2_best_recall50": V2_BEST_RECALL50,
        "n_zero_text_query_users": n_zero_query,
        "results": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Pareto CSV: all combos sorted by recall@50 desc
    _write_pareto_csv(output_dir / f"pareto_{run_type}.csv", all_results)
    _write_report(output_dir / f"report_{run_type}.md", run_type, all_results, len(eval_targets))

    logging.info("[Done] %s outputs saved to %s", run_type, output_dir)

    # Summary log
    best = max(all_results, key=lambda r: r["metrics"].get("recall@50", 0))
    logging.info(
        "[Summary] best=%s recall@50=%.6f avg_pop=%.0f coverage=%d delta=%+.6f",
        best["name"], best["metrics"]["recall@50"],
        best["metrics"]["avg_rec_popularity"], best["metrics"]["item_coverage"],
        best["metrics"]["recall@50"] - V1_BEST_RECALL50,
    )


def _fmt_f(v: Any, fmt: str = ".6f") -> str:
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def _write_pareto_csv(path: Path, all_results: list[dict[str, Any]]) -> None:
    keys = [
        "name", "fusion_type",
        "recall@50", "ndcg@50", "mrr@50",
        "avg_rec_popularity", "item_coverage", "delta_vs_v1",
        "bucket_1-5_r50", "bucket_6-20_r50", "bucket_21-100_r50", "bucket_>100_r50",
    ]
    rows = []
    for r in sorted(all_results, key=lambda x: x["metrics"].get("recall@50", 0), reverse=True):
        m = r["metrics"]
        bd = r.get("bucket_breakdown", {})
        row: dict[str, Any] = {
            "name": r["name"],
            "fusion_type": r.get("fusion_type", ""),
            "recall@50": m.get("recall@50", ""),
            "ndcg@50": m.get("ndcg@50", ""),
            "mrr@50": m.get("mrr@50", ""),
            "avg_rec_popularity": m.get("avg_rec_popularity", ""),
            "item_coverage": m.get("item_coverage", ""),
            "delta_vs_v1": r.get("delta_vs_v1", ""),
            "bucket_1-5_r50": bd.get("1-5", {}).get("recall@50", ""),
            "bucket_6-20_r50": bd.get("6-20", {}).get("recall@50", ""),
            "bucket_21-100_r50": bd.get("21-100", {}).get("recall@50", ""),
            "bucket_>100_r50": bd.get(">100", {}).get("recall@50", ""),
        }
        rows.append(row)
    write_csv(path, rows, keys)


def _write_report(
    path: Path,
    run_type: str,
    all_results: list[dict[str, Any]],
    n_eval: int,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Multi-channel Retrieval v3 Balanced Fusion — {run_type}",
        "",
        f"**Run type:** {run_type}  ",
        f"**Eval users:** {n_eval:,}  ",
        f"**Generated:** {now}  ",
        "",
        "## Pareto Table（按 Recall@50 降序）",
        "",
        "| Name | Recall@50 | avg_pop | coverage | delta_v1 | ≤5 R@50 | 6-20 R@50 | 21-100 R@50 | >100 R@50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in sorted(all_results, key=lambda x: x["metrics"].get("recall@50", 0), reverse=True):
        m = r["metrics"]
        bd = r.get("bucket_breakdown", {})
        delta = r.get("delta_vs_v1", round(m.get("recall@50", 0) - V1_BEST_RECALL50, 6))
        delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else str(delta)
        b5 = bd.get("1-5", {}).get("recall@50", "")
        b6 = bd.get("6-20", {}).get("recall@50", "")
        b21 = bd.get("21-100", {}).get("recall@50", "")
        b100 = bd.get(">100", {}).get("recall@50", "")
        lines.append(
            f"| {r['name']} "
            f"| {_fmt_f(m.get('recall@50', ''), '.6f')} "
            f"| {_fmt_f(m.get('avg_rec_popularity', ''), '.0f')} "
            f"| {m.get('item_coverage', '')} "
            f"| {delta_str} "
            f"| {_fmt_f(b5, '.4f')} "
            f"| {_fmt_f(b6, '.4f')} "
            f"| {_fmt_f(b21, '.4f')} "
            f"| {_fmt_f(b100, '.4f')} |"
        )

    # Pareto frontier: Recall > v1 AND avg_pop < 3 × v1_avg_pop (800) AND coverage > 85% v1
    v1_r = next((r for r in all_results if r["name"] == "ref_v1_2ch_rrf_k60"), None)
    v1_avg_pop = v1_r["metrics"]["avg_rec_popularity"] if v1_r else 268
    v1_coverage = v1_r["metrics"]["item_coverage"] if v1_r else 0
    pareto = [
        r for r in all_results
        if r["metrics"].get("recall@50", 0) > V1_BEST_RECALL50
        and r["metrics"].get("avg_rec_popularity", 9999) < v1_avg_pop * 3
        and (v1_coverage == 0 or r["metrics"].get("item_coverage", 0) > v1_coverage * 0.85)
    ]

    lines += [
        "",
        "## Pareto-Optimal 候选（Recall > v1 AND avg_pop < 3× v1 AND coverage > 85% v1）",
        "",
    ]
    if pareto:
        lines.append("| Name | Recall@50 | avg_pop | coverage | delta_v1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for r in sorted(pareto, key=lambda x: x["metrics"]["recall@50"], reverse=True):
            m = r["metrics"]
            delta = r.get("delta_vs_v1", round(m["recall@50"] - V1_BEST_RECALL50, 6))
            lines.append(
                f"| {r['name']} | {m['recall@50']:.6f} "
                f"| {m['avg_rec_popularity']:.0f} "
                f"| {m['item_coverage']} "
                f"| {delta:+.4f} |"
            )
    else:
        lines.append("*No config simultaneously beats v1 Recall AND keeps avg_pop < 3× v1 AND coverage > 85% v1.*")

    lines += [
        "",
        "## Notes",
        "",
        "- **avg_rec_popularity**: mean over users of mean(train_count(recommended items)).",
        "  v1 baseline (ref_v1_2ch_rrf_k60) ≈ 268; v2 baseline (ref_v2_4ch_rrf_k60) ≈ 1,649.",
        "- **item_coverage**: unique items recommended across all eval users.",
        "  v1 ≈ 20,226 (500-user extrapolation from audit); v2 ≈ 15,469 (same).",
        "- **delta_v1**: Recall@50 minus v1 best 0.096727.",
        "- Bucket label = train interaction count of the **target** item.",
        "- Offline evaluation only (full test set, 496,470 users in full mode).",
        "- Recall@100 = Recall@50 because rrf_top_n=50.",
        "- Weighted RRF: icf_w=tt_w=1.0 fixed; pop_w and text_w swept.",
        "  text_w=0.0 → ICF+TT+Pop only (text channel excluded).",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))

    if not args.full_only:
        logging.info("=== Smoke test ===")
        run(config, output_dir, smoke=True)
        logging.info("=== Smoke test DONE ===")

    if not args.smoke_only:
        logging.info("=== Full eval ===")
        run(config, output_dir, smoke=False)
        logging.info("=== Full eval DONE ===")


if __name__ == "__main__":
    main()

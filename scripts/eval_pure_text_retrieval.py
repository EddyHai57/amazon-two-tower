"""
M5.5 Pure text retrieval smoke test.

Candidate generation reuses the 4-channel Text Semantic implementation:
time-decay weighted history query + per-row/query L2 normalization + cosine similarity.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_transformer_maxlen100_smoke import (
    TRAIN_COLUMNS,
    build_history_matrix,
    build_seen_items,
    merge_seen_items,
)
from run_multichannel_retrieval_v2 import (
    generate_text_semantic_candidates,
    load_text_embeddings,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="M5.5 Pure text retrieval eval")
    p.add_argument("--data_dir", default="data/processed/movies_tv_5core")
    p.add_argument(
        "--embedding_path",
        default="outputs/item_text_embeddings/movies_tv_5core/item_text_embedding.npy",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/pure_text_retrieval/movies_tv_5core",
    )
    p.add_argument("--max_users", type=int, default=None,
                   help="Limit number of eval users per split (None = all)")
    p.add_argument("--batch_size_users", type=int, default=256)
    p.add_argument("--topk", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--device", default="auto")
    p.add_argument("--history_max_len", type=int, default=100)
    p.add_argument("--text_decay_rate", type=float, default=0.8)
    p.add_argument("--hf_cache_dir", default="/workspace/.hf_home/datasets")
    p.add_argument(
        "--skip_has_text_mask",
        action="store_true",
        help="Skip building has_text mask; grouped metrics will be omitted",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# has_text mask
# ---------------------------------------------------------------------------

def build_has_text_mask(data_dir: str, hf_cache_dir: str, n_items: int) -> np.ndarray:
    """Returns bool[n_items]: True if item has real title or description."""
    with open(os.path.join(data_dir, "id2item.json")) as f:
        id2item: dict = json.load(f)  # {str(item_idx): parent_asin}

    print("  Loading HuggingFace metadata from cache ...")
    from datasets import load_dataset
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Movies_and_TV",
        split="full",
        cache_dir=hf_cache_dir,
        trust_remote_code=True,
    )

    asin_has_text: dict[str, bool] = {}
    for row in ds:
        asin = row.get("parent_asin")
        if asin and asin not in asin_has_text:
            title = str(row.get("title") or "").strip()
            desc = row.get("description")
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc).strip()
            elif desc is None:
                desc = ""
            else:
                desc = str(desc).strip()
            asin_has_text[asin] = bool(title or desc)

    mask = np.zeros(n_items, dtype=bool)
    for idx_str, asin in id2item.items():
        mask[int(idx_str)] = asin_has_text.get(asin, False)

    n_has = mask.sum()
    print(f"  has_text=1 items: {n_has} / {n_items}  ({100*n_has/n_items:.1f}%)")
    return mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ranks: list[int | None], topk_list: list[int]) -> dict:
    """
    ranks: list of 1-indexed rank of target for each user, or None if not found.
    Returns dict with Recall@K, MRR@50, NDCG@50.
    """
    n = len(ranks)
    result = {}
    for k in topk_list:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        result[f"Recall@{k}"] = hits / n if n > 0 else 0.0
    # MRR@50
    mrr50 = sum(1.0 / r for r in ranks if r is not None and r <= 50) / n if n > 0 else 0.0
    result["MRR@50"] = mrr50
    # NDCG@50  (single relevant item: DCG = 1/log2(rank+1), IDCG = 1)
    ndcg50 = sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= 50) / n if n > 0 else 0.0
    result["NDCG@50"] = ndcg50
    result["num_eval_users"] = n
    return result


# ---------------------------------------------------------------------------
# Eval loop (one split)
# ---------------------------------------------------------------------------

def eval_split(
    split_name: str,
    split_path: str,
    history_matrix: np.ndarray,
    seen_items: dict[int, set[int]],
    item_emb: torch.Tensor,          # (n_items, dim) on device, L2-normalized
    has_text_mask: np.ndarray | None,
    args,
    device: torch.device,
) -> dict:
    t0 = time.time()
    n_items = item_emb.shape[0]
    max_k = max(args.topk)

    df = pd.read_parquet(split_path)
    df = df[~df["is_cold_item_for_eval"].astype(bool)].copy()
    # One target per user
    target_map: dict[int, int] = dict(zip(df["user_idx"].tolist(), df["item_idx"].tolist()))

    eval_users = sorted(target_map.keys())
    if args.max_users is not None:
        eval_users = eval_users[: args.max_users]

    candidates, num_skipped = generate_text_semantic_candidates(
        eval_users=eval_users,
        test_history_matrix=history_matrix,
        test_seen=seen_items,
        item_text_norm_gpu=item_emb,
        top_k=max_k,
        decay_rate=args.text_decay_rate,
        device=device,
        batch_size=args.batch_size_users,
    )

    ranks: list[int | None] = []
    target_has_text_flags: list[bool] = []
    for uid in eval_users:
        target_idx = target_map[uid]
        top_k_list = candidates.get(uid, [])
        ranks.append(top_k_list.index(target_idx) + 1 if target_idx in top_k_list else None)
        if has_text_mask is not None:
            target_has_text_flags.append(bool(has_text_mask[target_idx]))

    metrics = compute_metrics(ranks, args.topk)
    metrics["num_skipped_users"] = num_skipped
    metrics["elapsed_sec"] = round(time.time() - t0, 1)

    # has_text grouped analysis
    if has_text_mask is not None and target_has_text_flags:
        ranks_arr = np.array([r if r is not None else 9999999 for r in ranks])
        flags_arr = np.array(target_has_text_flags, dtype=bool)

        n_has = flags_arr.sum()
        n_no = (~flags_arr).sum()
        cov_has = n_has / len(flags_arr) if flags_arr.size > 0 else 0.0
        cov_no = n_no / len(flags_arr) if flags_arr.size > 0 else 0.0

        recall50_has = (ranks_arr[flags_arr] <= 50).mean() if n_has > 0 else float("nan")
        recall50_no = (ranks_arr[~flags_arr] <= 50).mean() if n_no > 0 else float("nan")

        metrics["target_has_text_1_count"] = int(n_has)
        metrics["target_has_text_0_count"] = int(n_no)
        metrics["target_has_text_1_ratio"] = round(float(cov_has), 4)
        metrics["target_has_text_0_ratio"] = round(float(cov_no), 4)
        metrics["Recall@50_target_has_text_1"] = round(float(recall50_has), 6)
        metrics["Recall@50_target_has_text_0"] = round(float(recall50_no), 6)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"device: {device}")

    # --- Load item embeddings ---
    print(f"[1] Loading item embeddings from {args.embedding_path}")
    raw_shape = np.load(args.embedding_path, mmap_mode="r").shape
    emb_np = load_text_embeddings(Path(args.embedding_path), n_items=raw_shape[0])
    n_items, dim = emb_np.shape
    print(f"    shape={emb_np.shape}  dtype={emb_np.dtype}")
    item_emb = torch.from_numpy(emb_np).to(device)  # (n_items, dim)

    # --- has_text mask ---
    has_text_mask: np.ndarray | None = None
    if not args.skip_has_text_mask:
        print(f"[2] Building has_text mask ...")
        has_text_mask = build_has_text_mask(args.data_dir, args.hf_cache_dir, n_items)
    else:
        print("[2] Skipping has_text mask (--skip_has_text_mask)")

    # --- Build train history ---
    print(f"[3] Loading train history from {args.data_dir}/train.parquet")
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    with open(os.path.join(args.data_dir, "stats.json")) as f:
        stats = json.load(f)
    n_users = int(stats["n_users"])
    train_history_matrix = build_history_matrix(train_df, n_users, args.history_max_len)
    train_seen = build_seen_items(train_df)
    print(f"    {n_users} users in history matrix")

    valid_df = pd.read_parquet(os.path.join(args.data_dir, "valid.parquet"))
    test_frame = pd.concat(
        [train_df, valid_df[TRAIN_COLUMNS]], ignore_index=True
    )
    test_history_matrix = build_history_matrix(test_frame, n_users, args.history_max_len)
    test_seen = merge_seen_items(train_seen, valid_df)

    # --- Eval splits ---
    all_results = {}
    for split_name in ["valid", "test"]:
        print(f"\n[4] Evaluating {split_name} ...")
        split_path = os.path.join(args.data_dir, f"{split_name}.parquet")
        metrics = eval_split(
            split_name=split_name,
            split_path=split_path,
            history_matrix=train_history_matrix if split_name == "valid" else test_history_matrix,
            seen_items=train_seen if split_name == "valid" else test_seen,
            item_emb=item_emb,
            has_text_mask=has_text_mask,
            args=args,
            device=device,
        )
        all_results[split_name] = metrics

        print(f"\n  --- {split_name} results ---")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    # --- Save results ---
    result_path = os.path.join(args.output_dir, "metrics.json")
    run_config = {
        "max_users": args.max_users,
        "topk": args.topk,
        "embedding_path": args.embedding_path,
        "history_max_len": args.history_max_len,
        "text_decay_rate": args.text_decay_rate,
        "device": device_str,
        "skip_has_text_mask": args.skip_has_text_mask,
    }
    output = {"config": run_config, "results": all_results}
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[5] Saved metrics to {result_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name in ["valid", "test"]:
        m = all_results[split_name]
        print(f"\n  [{split_name}]")
        for kk in [f"Recall@{k}" for k in args.topk] + ["MRR@50", "NDCG@50",
                                                          "num_eval_users", "num_skipped_users"]:
            if kk in m:
                v = m[kk]
                fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
                print(f"    {kk}: {fmt}")
        if "target_has_text_1_ratio" in m:
            print(f"    target has_text=1 ratio : {m['target_has_text_1_ratio']:.4f}")
            print(f"    target has_text=0 ratio : {m['target_has_text_0_ratio']:.4f}")
            print(f"    Recall@50 (has_text=1)  : {m['Recall@50_target_has_text_1']:.6f}")
            print(f"    Recall@50 (has_text=0)  : {m['Recall@50_target_has_text_0']:.6f}")


if __name__ == "__main__":
    main()

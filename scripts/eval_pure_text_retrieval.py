"""
M5.5 Pure text retrieval smoke test.

User query = mean-pooling of history item text embeddings.
Item embeddings are unit-normalized (L2), so dot product = cosine similarity.

Modes:
  use_all_items          : all history items contribute to query (default)
  history_has_text_only  : only history items with real text contribute to query;
                           candidate pool still includes all items
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch


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
    p.add_argument("--batch_size_users", type=int, default=2000)
    p.add_argument("--topk", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--mode",
        choices=["use_all_items", "history_has_text_only"],
        default="use_all_items",
    )
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
    train_history: dict[int, list[int]],
    item_emb: torch.Tensor,          # (n_items, dim) on device, already normalized
    has_text_mask: np.ndarray | None,
    args,
    device: torch.device,
) -> dict:
    t0 = time.time()
    n_items = item_emb.shape[0]
    max_k = max(args.topk)

    df = pd.read_parquet(split_path)
    # One target per user
    target_map: dict[int, int] = dict(zip(df["user_idx"].tolist(), df["item_idx"].tolist()))

    eval_users = sorted(target_map.keys())
    if args.max_users is not None:
        eval_users = eval_users[: args.max_users]

    ranks: list[int | None] = []
    target_has_text_flags: list[bool] = []
    num_skipped = 0

    for batch_start in range(0, len(eval_users), args.batch_size_users):
        batch_users = eval_users[batch_start : batch_start + args.batch_size_users]
        queries = []
        histories = []
        targets_batch = []

        for uid in batch_users:
            hist = train_history.get(uid, [])
            target_idx = target_map[uid]

            if not hist:
                num_skipped += 1
                queries.append(None)
                histories.append([])
                targets_batch.append(target_idx)
                continue

            hist_arr = np.array(hist, dtype=np.int64)

            if args.mode == "history_has_text_only" and has_text_mask is not None:
                hist_text = hist_arr[has_text_mask[hist_arr]]
                if len(hist_text) == 0:
                    hist_text = hist_arr  # fallback: use all if none have text
                hist_arr = hist_text

            # Mean pool history embeddings
            hist_emb = item_emb[torch.from_numpy(hist_arr).long().to(device)]  # (h, dim)
            query = hist_emb.mean(dim=0)  # (dim,)
            queries.append(query)
            histories.append(hist)
            targets_batch.append(target_idx)

        # Batch similarity
        valid_indices = [i for i, q in enumerate(queries) if q is not None]
        if valid_indices:
            query_mat = torch.stack([queries[i] for i in valid_indices])  # (m, dim)
            # dot product = cosine (embeddings are unit-norm)
            scores_mat = query_mat @ item_emb.T  # (m, n_items)

            for local_i, global_i in enumerate(valid_indices):
                uid = batch_users[global_i]
                hist = histories[global_i]
                target_idx = targets_batch[global_i]
                scores = scores_mat[local_i]  # (n_items,)

                # Mask history
                if hist:
                    scores[torch.tensor(hist, dtype=torch.long, device=device)] = float("-inf")

                # Top-K (sorted)
                top_k_vals, top_k_idxs = torch.topk(scores, k=min(max_k, n_items), largest=True)
                top_k_list = top_k_idxs.cpu().tolist()

                if target_idx in top_k_list:
                    rank = top_k_list.index(target_idx) + 1
                else:
                    rank = None
                ranks.append(rank)

                if has_text_mask is not None:
                    target_has_text_flags.append(bool(has_text_mask[target_idx]))

        elapsed = time.time() - t0
        done = batch_start + len(batch_users)
        print(
            f"  [{split_name}] {done}/{len(eval_users)} users  "
            f"({100*done/len(eval_users):.1f}%)  {elapsed:.1f}s",
            end="\r",
        )

    print()

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
    emb_np = np.load(args.embedding_path).astype(np.float32)
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
    train_history: dict[int, list[int]] = (
        train_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    )
    print(f"    {len(train_history)} users with train history")

    # --- Eval splits ---
    all_results = {}
    for split_name in ["valid", "test"]:
        print(f"\n[4] Evaluating {split_name} ...")
        split_path = os.path.join(args.data_dir, f"{split_name}.parquet")
        metrics = eval_split(
            split_name=split_name,
            split_path=split_path,
            train_history=train_history,
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
        "mode": args.mode,
        "max_users": args.max_users,
        "topk": args.topk,
        "embedding_path": args.embedding_path,
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

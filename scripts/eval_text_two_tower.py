"""M6.3 full evaluation for text-enhanced Two-Tower (additive residual).

Loads best_model.pt and outputs:
  full_eval.{json,md}         — full valid + test main metrics
  popularity_buckets.{json,md} — test Recall@50 by train popularity bucket
  has_text_split.{json,md}    — test Recall by has_text flag
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model — must stay in sync with train_text_two_tower.py
# ---------------------------------------------------------------------------

class TextEnhancedTwoTower(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim,
                 text_emb, has_text, text_proj_dim, use_l2_norm, use_has_text_mask):
        super().__init__()
        assert text_proj_dim == embedding_dim, (
            f"additive fusion requires text_proj_dim == embedding_dim, "
            f"got {text_proj_dim} vs {embedding_dim}"
        )
        self.use_l2_norm = use_l2_norm
        self.use_has_text_mask = use_has_text_mask
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.text_proj = nn.Linear(text_emb.shape[1], embedding_dim, bias=False)
        self.register_buffer("_text_emb", text_emb.float(), persistent=False)
        self.register_buffer("_has_text", has_text.float(), persistent=False)

    def _item_prenorm(self, item_idx):
        id_emb = self.item_id_embedding(item_idx)
        txt_proj = self.text_proj(self._text_emb[item_idx])
        if self.use_has_text_mask:
            txt_proj = txt_proj * self._has_text[item_idx].unsqueeze(-1)
        return id_emb + txt_proj

    def encode_users(self, user_idx):
        u = self.user_embedding(user_idx)
        return F.normalize(u, p=2, dim=-1) if self.use_l2_norm else u

    def encode_items(self, item_idx):
        out = self._item_prenorm(item_idx)
        return F.normalize(out, p=2, dim=-1) if self.use_l2_norm else out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    train_df = pd.read_parquet(data_dir / "train.parquet", columns=["user_idx", "item_idx"])
    valid_df = pd.read_parquet(data_dir / "valid.parquet",
                               columns=["user_idx", "item_idx", "is_cold_item_for_eval"])
    test_df  = pd.read_parquet(data_dir / "test.parquet",
                               columns=["user_idx", "item_idx", "is_cold_item_for_eval"])
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    return train_df, valid_df, test_df, stats


def build_seen(base_df, extra_df=None):
    seen: dict[int, set[int]] = {}
    for uid, grp in base_df.groupby("user_idx", sort=False):
        seen[int(uid)] = set(int(x) for x in grp["item_idx"])
    if extra_df is not None:
        for uid, grp in extra_df.groupby("user_idx", sort=False):
            seen.setdefault(int(uid), set()).update(int(x) for x in grp["item_idx"])
    return seen


# ---------------------------------------------------------------------------
# Eval core
# ---------------------------------------------------------------------------

def encode_all_items(model: nn.Module, n_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        idx = torch.arange(n_items, device=device)
        return model.encode_items(idx).cpu()


def eval_split(
    model: nn.Module,
    eval_df: pd.DataFrame,
    seen_items: dict[int, set[int]],
    n_items: int,
    device: torch.device,
    eval_batch_size: int = 256,
) -> tuple[list[tuple], int, int]:
    """Returns (results, n_eval_users, n_skipped_cold).

    results: list of (rank_or_None, target_item_idx) for each non-cold user.
    rank is 1-indexed; None means target not in top-100.
    """
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    n_skipped = int(eval_df["is_cold_item_for_eval"].astype(bool).sum())
    max_k = 100

    item_emb_cpu = encode_all_items(model, n_items, device)
    results: list[tuple] = []

    model.eval()
    t0 = time.time()
    n_batches = math.ceil(len(non_cold) / eval_batch_size)

    with torch.no_grad():
        for batch_i, start in enumerate(range(0, len(non_cold), eval_batch_size)):
            batch = non_cold.iloc[start : start + eval_batch_size]
            u_t   = torch.tensor(batch["user_idx"].to_numpy(dtype=np.int64, copy=True), device=device)
            tgt_a = batch["item_idx"].to_numpy(dtype=np.int64, copy=True)
            tgt_t = torch.tensor(tgt_a, device=device)

            u_emb  = model.encode_users(u_t)        # (bs, D)
            i_emb  = item_emb_cpu.to(device)         # (n_items, D)
            scores = u_emb @ i_emb.T                 # (bs, n_items)

            # Preserve target score before masking seen items
            row_idx    = torch.arange(scores.shape[0], device=device)
            tgt_scores = scores[row_idx, tgt_t].clone()

            for rpos, (uid, tgt_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist())
            ):
                seen = seen_items.get(int(uid), set())
                if seen:
                    seen_t = torch.tensor(list(seen), dtype=torch.long, device=device)
                    scores[rpos, seen_t] = float("-inf")
                scores[rpos, int(tgt_item)] = tgt_scores[rpos]

            topk_idxs = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

            for tgt_item, recs in zip(tgt_a, topk_idxs):
                hit = np.where(recs == tgt_item)[0]
                rank = int(hit[0]) + 1 if hit.size else None
                results.append((rank, int(tgt_item)))

            if (batch_i + 1) % max(1, n_batches // 5) == 0:
                done = start + len(batch)
                print(f"    {done}/{len(non_cold)} ({100*done/len(non_cold):.1f}%)  {time.time()-t0:.0f}s",
                      flush=True)

    return results, len(non_cold), n_skipped


def compute_metrics(results: list[tuple], k_list=(20, 50, 100)) -> dict:
    ranks = [r for r, _ in results]
    n = len(ranks)
    m: dict = {}
    for k in k_list:
        m[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
        m[f"ndcg@{k}"]   = sum(1.0 / math.log2(r + 1)
                                for r in ranks if r is not None and r <= k) / n
        m[f"mrr@{k}"]    = sum(1.0 / r for r in ranks if r is not None and r <= k) / n
    return m


# ---------------------------------------------------------------------------
# Popularity bucket
# ---------------------------------------------------------------------------

BUCKETS = [
    ("<=5",    lambda c: c <= 5),
    ("6-20",   lambda c: 6 <= c <= 20),
    ("21-100", lambda c: 21 <= c <= 100),
    (">100",   lambda c: c > 100),
]


def popularity_buckets(results: list[tuple], item_counts: dict) -> list[dict]:
    bucket_ranks: dict[str, list] = {name: [] for name, _ in BUCKETS}
    total = len(results)

    for rank, tgt_idx in results:
        count = item_counts.get(tgt_idx, 0)
        for name, fn in BUCKETS:
            if fn(count):
                bucket_ranks[name].append(rank)
                break

    rows = []
    for name, _ in BUCKETS:
        ranks = bucket_ranks[name]
        n = len(ranks)
        rows.append({
            "bucket": name,
            "num_targets": n,
            "target_ratio": round(n / total, 4) if total else 0.0,
            "recall@50": round(
                sum(1 for r in ranks if r is not None and r <= 50) / n, 6
            ) if n else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# has_text split
# ---------------------------------------------------------------------------

def has_text_split(results: list[tuple], has_text_arr: np.ndarray) -> list[dict]:
    groups: dict[int, list] = {1: [], 0: []}
    for rank, tgt_idx in results:
        flag = int(bool(has_text_arr[tgt_idx]))
        groups[flag].append(rank)

    rows = []
    total = len(results)
    for flag in (1, 0):
        ranks = groups[flag]
        n = len(ranks)
        ratio = n / total if total else 0.0

        def rc(k, _ranks=ranks, _n=n):
            return round(
                sum(1 for r in _ranks if r is not None and r <= k) / _n, 6
            ) if _n else float("nan")

        rows.append({
            "has_text": flag,
            "num_targets": n,
            "target_ratio": round(ratio, 4),
            "recall@20":  rc(20),
            "recall@50":  rc(50),
            "recall@100": rc(100),
        })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_full_eval(out_dir: Path, valid_m, test_m, n_valid, n_skip_v, n_test, n_skip_t) -> dict:
    payload = {
        "valid": {**valid_m, "num_eval_users": n_valid, "num_skipped_cold": n_skip_v},
        "test":  {**test_m,  "num_eval_users": n_test,  "num_skipped_cold": n_skip_t},
    }
    write_json(out_dir / "full_eval.json", payload)

    v, t = payload["valid"], payload["test"]
    md = "\n".join([
        "# Full Eval — Additive Text-Enhanced Two-Tower (M6)",
        "",
        "## Valid（全量，非 cold item target）",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| num_eval_users | {v['num_eval_users']} |",
        f"| num_skipped_cold | {v['num_skipped_cold']} |",
        f"| Recall@20 | {v['recall@20']:.6f} |",
        f"| Recall@50 | {v['recall@50']:.6f} |",
        f"| Recall@100 | {v['recall@100']:.6f} |",
        f"| NDCG@50 | {v['ndcg@50']:.6f} |",
        f"| MRR@50 | {v['mrr@50']:.6f} |",
        "",
        "## Test（全量，非 cold item target）",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| num_eval_users | {t['num_eval_users']} |",
        f"| num_skipped_cold | {t['num_skipped_cold']} |",
        f"| Recall@20 | {t['recall@20']:.6f} |",
        f"| Recall@50 | {t['recall@50']:.6f} |",
        f"| Recall@100 | {t['recall@100']:.6f} |",
        f"| NDCG@50 | {t['ndcg@50']:.6f} |",
        f"| MRR@50 | {t['mrr@50']:.6f} |",
        "",
        "## 三方对比（Recall@50）",
        "",
        "| 方法 | valid Recall@50 | test Recall@50 |",
        "|---|---:|---:|",
        "| clean ItemCF | 0.140698 | 0.083570 |",
        f"| ID-only Two-Tower 20ep | 0.092144 | 0.053198 |",
        f"| Additive text-enhanced 20ep | {v['recall@50']:.6f} | {t['recall@50']:.6f} |",
    ]) + "\n"
    (out_dir / "full_eval.md").write_text(md)
    return payload


def write_pop_buckets(out_dir: Path, rows: list[dict]) -> None:
    write_json(out_dir / "popularity_buckets.json", rows)
    lines = [
        "# Test Popularity Bucket — Additive Text-Enhanced Two-Tower (M6)",
        "",
        "按 test target 在 train 中的出现次数分桶。",
        "",
        "| bucket | num_targets | target_ratio | Recall@50 |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bucket']} | {r['num_targets']} | {r['target_ratio']:.4f} | {r['recall@50']:.6f} |"
        )
    (out_dir / "popularity_buckets.md").write_text("\n".join(lines) + "\n")


def write_has_text(out_dir: Path, rows: list[dict]) -> None:
    write_json(out_dir / "has_text_split.json", rows)
    lines = [
        "# Test has_text Split — Additive Text-Enhanced Two-Tower (M6)",
        "",
        "按 test target item 是否有真实 title/description 分组。",
        "",
        "| has_text | num_targets | target_ratio | Recall@20 | Recall@50 | Recall@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['has_text']} | {r['num_targets']} | {r['target_ratio']:.4f} | "
            f"{r['recall@20']:.6f} | {r['recall@50']:.6f} | {r['recall@100']:.6f} |"
        )
    (out_dir / "has_text_split.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="M6.3 full eval for text-enhanced Two-Tower")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [1] Load checkpoint
    print(f"\n[1] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    stats  = ckpt["stats"]
    print(f"    best_epoch={ckpt['epoch']}  "
          f"{ckpt['best_metric_name']}={ckpt['best_metric_value']:.6f}")

    data_dir = Path(config["data_dir"])
    n_users  = int(stats["n_users"])
    n_items  = int(stats["n_items"])

    # [2] Load text artifacts
    print("\n[2] Loading text artifacts")
    emb_path = Path(config["item_text_embedding_path"])
    has_path = Path(config["item_has_text_path"])
    text_emb = torch.from_numpy(np.load(emb_path).astype(np.float32))
    has_text  = torch.from_numpy(np.load(has_path).astype(np.float32))
    print(f"    text_emb={tuple(text_emb.shape)}  "
          f"has_text=1: {int(has_text.sum())}/{len(has_text)} ({100*has_text.mean():.1f}%)")

    # [3] Recreate model and load weights
    print("\n[3] Recreating model")
    model = TextEnhancedTwoTower(
        num_users=n_users, num_items=n_items,
        embedding_dim=int(config["embedding_dim"]),
        text_emb=text_emb, has_text=has_text,
        text_proj_dim=int(config["text_proj_dim"]),
        use_l2_norm=bool(config["use_l2_norm"]),
        use_has_text_mask=bool(config["use_has_text_mask"]),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    # missing keys = persistent=False buffers (_text_emb, _has_text) — expected
    print(f"    loaded OK  missing_keys={missing}  unexpected_keys={unexpected}")
    print(f"    trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # [4] Load data
    print("\n[4] Loading data")
    train_df, valid_df, test_df, _ = load_data(data_dir)
    print(f"    train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    train_seen = build_seen(train_df)
    test_seen  = build_seen(train_df, valid_df)

    # [5] Item popularity counts (for bucket analysis)
    print("\n[5] Computing item popularity from train")
    item_counts: dict[int, int] = train_df["item_idx"].value_counts().to_dict()

    # [6] has_text array
    has_text_arr = np.load(has_path).astype(bool)

    # [7] Full valid eval
    print("\n[6] Full valid eval ...")
    t0 = time.time()
    valid_results, n_valid, n_skip_v = eval_split(
        model, valid_df, train_seen, n_items, device, args.eval_batch_size
    )
    valid_m = compute_metrics(valid_results)
    print(f"    done in {time.time()-t0:.1f}s  users={n_valid}  "
          f"Recall@50={valid_m['recall@50']:.6f}")

    # [8] Full test eval
    print("\n[7] Full test eval ...")
    t0 = time.time()
    test_results, n_test, n_skip_t = eval_split(
        model, test_df, test_seen, n_items, device, args.eval_batch_size
    )
    test_m = compute_metrics(test_results)
    print(f"    done in {time.time()-t0:.1f}s  users={n_test}  "
          f"Recall@50={test_m['recall@50']:.6f}")

    # [9] Popularity bucket
    print("\n[8] Popularity bucket analysis")
    pop_rows = popularity_buckets(test_results, item_counts)
    for r in pop_rows:
        print(f"    {r['bucket']:>6}: n={r['num_targets']:>6}  "
              f"ratio={r['target_ratio']:.4f}  Recall@50={r['recall@50']:.6f}")

    # [10] has_text split
    print("\n[9] has_text split analysis")
    ht_rows = has_text_split(test_results, has_text_arr)
    for r in ht_rows:
        print(f"    has_text={r['has_text']}: n={r['num_targets']:>6}  "
              f"ratio={r['target_ratio']:.4f}  Recall@50={r['recall@50']:.6f}")

    # [11] Write outputs
    print("\n[10] Writing output files")
    payload = write_full_eval(out_dir, valid_m, test_m, n_valid, n_skip_v, n_test, n_skip_t)
    write_pop_buckets(out_dir, pop_rows)
    write_has_text(out_dir, ht_rows)

    v, t = payload["valid"], payload["test"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  full valid Recall@50 = {v['recall@50']:.6f}")
    print(f"  full test  Recall@50 = {t['recall@50']:.6f}")
    print()
    print("  Popularity bucket (test):")
    for r in pop_rows:
        print(f"    {r['bucket']:>6}: Recall@50 = {r['recall@50']:.6f}  (n={r['num_targets']})")
    print()
    print("  has_text split (test):")
    for r in ht_rows:
        print(f"    has_text={r['has_text']}: Recall@50 = {r['recall@50']:.6f}  (n={r['num_targets']})")
    print()
    print("  Three-way comparison (Recall@50):")
    print(f"    clean ItemCF                    valid=0.140698  test=0.083570")
    print(f"    ID-only Two-Tower 20ep           valid=0.092144  test=0.053198")
    print(f"    Additive text-enhanced 20ep      valid={v['recall@50']:.6f}  test={t['recall@50']:.6f}")

    test_r50 = t["recall@50"]
    if test_r50 >= 0.075:
        branch = "α"
    elif test_r50 >= 0.055:
        branch = "β"
    else:
        branch = "γ"
    print(f"\n  判断分支：{branch}  (test Recall@50 = {test_r50:.6f})")
    print("=" * 60)

    print(f"\n[DONE] 输出目录: {out_dir}")
    for fname in ["full_eval.json", "full_eval.md", "popularity_buckets.json",
                  "popularity_buckets.md", "has_text_split.json", "has_text_split.md"]:
        p = out_dir / fname
        print(f"  {'OK' if p.exists() else 'MISSING':>7}  {p}")


if __name__ == "__main__":
    main()

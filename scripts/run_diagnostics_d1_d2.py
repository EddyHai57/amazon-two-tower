"""D1 + D2 diagnostics (eval-only, no training, no model modification).

D1: ID-only Two-Tower test has_text split
D2: Three-model popularity bucket matrix (ItemCF / ID-only / text-enhanced)

Outputs:
  outputs/diagnostics_movies_tv_5core/id_only_has_text_split.json
  outputs/diagnostics_movies_tv_5core/id_only_has_text_split.md
  outputs/diagnostics_movies_tv_5core/popularity_bucket_model_matrix.json
  outputs/diagnostics_movies_tv_5core/popularity_bucket_model_matrix.md
"""

import collections
import heapq
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
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data/processed/movies_tv_5core")
HAS_TEXT_NPY = Path("outputs/item_text_embeddings/movies_tv_5core/item_has_text.npy")
ID_ONLY_CKPT = Path("outputs/two_tower_movies_tv_5core_clean_20epoch/checkpoints/best_model.pt")
TEXT_POP_JSON = Path("outputs/text_two_tower_additive_movies_tv_5core_20epoch/popularity_buckets.json")
OUT_DIR    = Path("outputs/diagnostics_movies_tv_5core")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 256
K_LIST = (20, 50, 100)

BUCKETS = [
    ("<=5",    lambda c: c <= 5),
    ("6-20",   lambda c: 6 <= c <= 20),
    ("21-100", lambda c: 21 <= c <= 100),
    (">100",   lambda c: c > 100),
]

# ---------------------------------------------------------------------------
# ID-only model (mirrors IDOnlyTwoTower in train_two_tower.py)
# ---------------------------------------------------------------------------
class IDOnlyTwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, use_l2_norm: bool) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.use_l2_norm = use_l2_norm

    def encode_users(self, user_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_idx)
        return F.normalize(u, p=2, dim=-1) if self.use_l2_norm else u

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        v = self.item_embedding(item_idx)
        return F.normalize(v, p=2, dim=-1) if self.use_l2_norm else v


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print("读取数据...", flush=True)
    train_df = pd.read_parquet(DATA_DIR / "train.parquet",
                               columns=["user_idx", "item_idx", "timestamp"])
    valid_df = pd.read_parquet(DATA_DIR / "valid.parquet",
                               columns=["user_idx", "item_idx", "is_cold_item_for_eval"])
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet",
                               columns=["user_idx", "item_idx", "is_cold_item_for_eval"])
    with open(DATA_DIR / "stats.json") as f:
        stats = json.load(f)
    print(f"  train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}", flush=True)
    return train_df, valid_df, test_df, stats


def build_seen(base_df: pd.DataFrame, extra_df: pd.DataFrame | None = None) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for uid, grp in base_df.groupby("user_idx", sort=False):
        seen[int(uid)] = set(int(x) for x in grp["item_idx"])
    if extra_df is not None:
        for uid, grp in extra_df.groupby("user_idx", sort=False):
            seen.setdefault(int(uid), set()).update(int(x) for x in grp["item_idx"])
    return seen


def compute_item_train_counts(train_df: pd.DataFrame) -> dict[int, int]:
    return dict(train_df["item_idx"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_metrics(results: list[tuple], k_list=K_LIST) -> dict:
    n = len(results)
    m: dict = {}
    for k in k_list:
        m[f"recall@{k}"] = round(
            sum(1 for r, _ in results if r is not None and r <= k) / n, 6) if n else 0.0
        m[f"ndcg@{k}"]   = round(
            sum(1.0 / math.log2(r + 1)
                for r, _ in results if r is not None and r <= k) / n, 6) if n else 0.0
        m[f"mrr@{k}"]    = round(
            sum(1.0 / r for r, _ in results if r is not None and r <= k) / n, 6) if n else 0.0
    return m


def split_by_has_text(results: list[tuple], has_text_arr: np.ndarray) -> list[dict]:
    groups: dict[int, list] = {1: [], 0: []}
    for rank, tgt in results:
        groups[int(bool(has_text_arr[tgt]))].append(rank)
    total = len(results)
    rows = []
    for flag in (1, 0):
        ranks = groups[flag]
        n = len(ranks)
        ratio = n / total if total else 0.0
        rows.append({
            "has_text": flag,
            "num_targets": n,
            "target_ratio": round(ratio, 4),
            "recall@20":  round(sum(1 for r in ranks if r is not None and r <= 20) / n, 6) if n else float("nan"),
            "recall@50":  round(sum(1 for r in ranks if r is not None and r <= 50) / n, 6) if n else float("nan"),
            "recall@100": round(sum(1 for r in ranks if r is not None and r <= 100) / n, 6) if n else float("nan"),
        })
    return rows


def split_by_pop_bucket(results: list[tuple], item_counts: dict) -> list[dict]:
    bucket_ranks: dict[str, list] = {name: [] for name, _ in BUCKETS}
    total = len(results)
    for rank, tgt in results:
        count = item_counts.get(tgt, 0)
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
# D1: ID-only Two-Tower evaluation
# ---------------------------------------------------------------------------
def run_id_only_eval(test_df: pd.DataFrame, seen_items: dict[int, set[int]], n_items: int) -> list[tuple]:
    print("\n[D1] 加载 ID-only 20ep checkpoint...", flush=True)
    ckpt = torch.load(ID_ONLY_CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    s    = ckpt["stats"]

    model = IDOnlyTwoTower(
        num_users=int(s["n_users"]),
        num_items=int(s["n_items"]),
        embedding_dim=int(cfg["embedding_dim"]),
        use_l2_norm=bool(cfg["use_l2_norm"]),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"  checkpoint epoch={ckpt['epoch']}, best_metric={ckpt['best_metric_name']}={ckpt['best_metric_value']:.6f}", flush=True)

    # Encode all items once
    with torch.no_grad():
        item_idx_all = torch.arange(n_items, device=DEVICE)
        item_emb_cpu = model.encode_items(item_idx_all).cpu()

    non_cold = test_df[~test_df["is_cold_item_for_eval"].astype(bool)].copy()
    n_cold   = int(test_df["is_cold_item_for_eval"].astype(bool).sum())
    print(f"  non_cold test users={len(non_cold)}, skipped_cold={n_cold}", flush=True)

    results: list[tuple] = []
    t0 = time.time()
    total = len(non_cold)

    with torch.no_grad():
        for start in range(0, total, EVAL_BATCH_SIZE):
            batch = non_cold.iloc[start : start + EVAL_BATCH_SIZE]
            u_t   = torch.tensor(batch["user_idx"].to_numpy(dtype=np.int64, copy=True), device=DEVICE)
            tgt_a = batch["item_idx"].to_numpy(dtype=np.int64, copy=True)
            tgt_t = torch.tensor(tgt_a, device=DEVICE)

            u_emb  = model.encode_users(u_t)
            i_emb  = item_emb_cpu.to(DEVICE)
            scores = u_emb @ i_emb.T  # (bs, n_items); L2-norm so dot = cosine; monotone in temperature

            row_idx    = torch.arange(scores.shape[0], device=DEVICE)
            tgt_scores = scores[row_idx, tgt_t].clone()

            for rpos, (uid, tgt_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist())
            ):
                seen = seen_items.get(int(uid), set())
                if seen:
                    seen_t = torch.tensor(list(seen), dtype=torch.long, device=DEVICE)
                    scores[rpos, seen_t] = float("-inf")
                scores[rpos, int(tgt_item)] = tgt_scores[rpos]

            topk = torch.topk(scores, k=100, dim=1).indices.cpu().numpy()
            for tgt_item, recs in zip(tgt_a, topk):
                hit = np.where(recs == tgt_item)[0]
                rank = int(hit[0]) + 1 if hit.size else None
                results.append((rank, int(tgt_item)))

            done = start + len(batch)
            if done % 50000 < EVAL_BATCH_SIZE or done == total:
                print(f"  {done}/{total} ({100*done/total:.1f}%)  {time.time()-t0:.0f}s", flush=True)

    overall = compute_metrics(results)
    print(f"  ID-only test Recall@20={overall['recall@20']:.6f} @50={overall['recall@50']:.6f} @100={overall['recall@100']:.6f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# D2 part: ItemCF evaluation
# ---------------------------------------------------------------------------
def _build_itemcf_structures(train_df: pd.DataFrame, max_user_history: int = 100, sim_topk: int = 100):
    print("\n[D2-ItemCF] 构建 user history 和 item-item similarity...", flush=True)
    sorted_train = train_df.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")

    full_seen: dict[int, set[int]] = {}
    limited_history: dict[int, list[int]] = {}

    for uid, grp in sorted_train.groupby("user_idx", sort=False):
        items = [int(x) for x in grp["item_idx"].tolist()]
        full_seen[int(uid)] = set(items)
        recent: list[int] = []
        used: set[int] = set()
        for item in reversed(items):
            if item in used:
                continue
            recent.append(item)
            used.add(item)
            if len(recent) >= max_user_history:
                break
        limited_history[int(uid)] = list(reversed(recent))

    # Build item-item similarity
    item_counts: collections.Counter = collections.Counter()
    co_counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for items in limited_history.values():
        unique = sorted(set(items))
        for x in unique:
            item_counts[x] += 1
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                co_counts[a][b] += 1
                co_counts[b][a] += 1

    similarity: dict[int, list[tuple[int, float]]] = {}
    for item_i, related in co_counts.items():
        top: list[tuple[float, int]] = []
        ci = item_counts[item_i]
        for item_j, co in related.items():
            denom = math.sqrt(ci * item_counts[item_j])
            if denom <= 0:
                continue
            top.append((co / denom, item_j))
        best = heapq.nlargest(sim_topk, top, key=lambda p: (p[0], -p[1]))
        similarity[item_i] = [(j, s) for s, j in best]

    print(f"  similarity items={len(similarity)}", flush=True)
    return full_seen, limited_history, similarity


def run_itemcf_eval(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> list[tuple]:
    full_seen, limited_history, similarity = _build_itemcf_structures(train_df)

    # Add valid to seen for test mask (train_valid filter)
    seen_with_valid = {uid: set(items) for uid, items in full_seen.items()}
    for uid, grp in valid_df.groupby("user_idx", sort=False):
        seen_with_valid.setdefault(int(uid), set()).update(int(x) for x in grp["item_idx"])

    non_cold = test_df[~test_df["is_cold_item_for_eval"].astype(bool)].copy()
    n_cold   = int(test_df["is_cold_item_for_eval"].astype(bool).sum())
    print(f"  non_cold test users={len(non_cold)}, skipped_cold={n_cold}", flush=True)

    results: list[tuple] = []
    max_k = 100
    t0 = time.time()

    for i, row in enumerate(non_cold.itertuples(index=False)):
        uid = int(row.user_idx)
        tgt = int(row.item_idx)
        seen = seen_with_valid.get(uid, set())

        scores: dict[int, float] = collections.defaultdict(float)
        for hist_item in limited_history.get(uid, []):
            for cand, sim in similarity.get(hist_item, []):
                if cand in seen and cand != tgt:
                    continue
                scores[cand] += sim

        ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:max_k]
        rec_items = [item for item, _ in ranked]
        rank = next((pos + 1 for pos, item in enumerate(rec_items) if item == tgt), None)
        results.append((rank, tgt))

        if (i + 1) % 50000 == 0 or (i + 1) == len(non_cold):
            print(f"  {i+1}/{len(non_cold)} ({100*(i+1)/len(non_cold):.1f}%)  {time.time()-t0:.0f}s", flush=True)

    overall = compute_metrics(results)
    print(f"  ItemCF test Recall@20={overall['recall@20']:.6f} @50={overall['recall@50']:.6f} @100={overall['recall@100']:.6f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_d1(out_dir: Path, rows: list[dict]) -> None:
    write_json(out_dir / "id_only_has_text_split.json", rows)

    lines = [
        "# D1: ID-only Two-Tower 20ep — Test has_text Split",
        "",
        "checkpoint: outputs/two_tower_movies_tv_5core_clean_20epoch/checkpoints/best_model.pt (best_epoch=18)",
        "eval: test split, seen_mask=train+valid, exclude cold items",
        "",
        "| has_text | num_targets | target_ratio | Recall@20 | Recall@50 | Recall@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        label = "1（有真实文本）" if r["has_text"] == 1 else "0（无文本，fallback）"
        lines.append(
            f"| {label} | {r['num_targets']} | {r['target_ratio']:.4f} "
            f"| {r['recall@20']:.6f} | {r['recall@50']:.6f} | {r['recall@100']:.6f} |"
        )

    lines += [
        "",
        "## 与 text-enhanced has_text split 对比（Recall@50）",
        "",
        "| has_text | ID-only R@50 | Text-enhanced R@50 | delta |",
        "|---|---:|---:|---:|",
    ]
    # text-enhanced known values
    text_ht = {1: 0.070464, 0: 0.041407}
    for r in rows:
        flag = r["has_text"]
        delta = r["recall@50"] - text_ht[flag]
        lines.append(
            f"| {'1（有文本）' if flag == 1 else '0（无文本）'} "
            f"| {r['recall@50']:.6f} | {text_ht[flag]:.6f} | {delta:+.6f} |"
        )

    (out_dir / "id_only_has_text_split.md").write_text("\n".join(lines) + "\n")


def write_d2(out_dir: Path, id_only_rows: list[dict], itemcf_rows: list[dict], text_rows: list[dict]) -> None:
    # Build merged matrix
    matrix = []
    for i, bkt in enumerate(id_only_rows):
        name = bkt["bucket"]
        matrix.append({
            "bucket": name,
            "num_targets": bkt["num_targets"],
            "target_ratio": bkt["target_ratio"],
            "itemcf_recall@50":    itemcf_rows[i]["recall@50"],
            "id_only_recall@50":   bkt["recall@50"],
            "text_enh_recall@50":  text_rows[i]["recall@50"],
        })
    write_json(out_dir / "popularity_bucket_model_matrix.json", matrix)

    lines = [
        "# D2: Test Popularity Bucket × Model Matrix (Recall@50)",
        "",
        "train item count buckets; test split; seen_mask=train+valid; exclude cold items",
        "",
        "| bucket | num_targets | target_ratio | ItemCF R@50 | ID-only R@50 | Text-enh R@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in matrix:
        lines.append(
            f"| {r['bucket']} | {r['num_targets']} | {r['target_ratio']:.4f} "
            f"| {r['itemcf_recall@50']:.6f} | {r['id_only_recall@50']:.6f} | {r['text_enh_recall@50']:.6f} |"
        )

    lines += [
        "",
        "## 模型整体 test Recall@50（参考）",
        "",
        "| 方法 | test Recall@50 |",
        "|---|---:|",
        "| clean ItemCF | 0.083570 |",
        "| ID-only Two-Tower 20ep | 0.053198 |",
        "| Additive text-enhanced 20ep | 0.054561 |",
    ]
    (out_dir / "popularity_bucket_model_matrix.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"device={DEVICE}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, stats = load_data()
    n_items = int(stats["n_items"])

    # --- Item train counts (for popularity bucket) ---
    item_counts = compute_item_train_counts(train_df)
    print(f"item_counts: {len(item_counts)} unique items in train", flush=True)

    # --- has_text array ---
    has_text_arr = np.load(HAS_TEXT_NPY)
    print(f"has_text_arr shape={has_text_arr.shape}, has_text=1: {has_text_arr.sum()}", flush=True)

    # --- Test seen mask for Two-Tower (train + valid) ---
    seen_train_valid = build_seen(train_df[["user_idx", "item_idx"]], valid_df[["user_idx", "item_idx"]])

    # ===== D1: ID-only has_text split =====
    print("\n" + "="*60, flush=True)
    print("D1: ID-only Two-Tower has_text split", flush=True)
    print("="*60, flush=True)

    id_only_results = run_id_only_eval(test_df, seen_train_valid, n_items)

    d1_rows = split_by_has_text(id_only_results, has_text_arr)
    print("\n[D1] has_text split 结果：", flush=True)
    for r in d1_rows:
        print(f"  has_text={r['has_text']}: n={r['num_targets']}, ratio={r['target_ratio']:.4f}, "
              f"R@20={r['recall@20']:.6f}, R@50={r['recall@50']:.6f}, R@100={r['recall@100']:.6f}", flush=True)

    write_d1(OUT_DIR, d1_rows)

    # ===== D2: popularity bucket matrix =====
    print("\n" + "="*60, flush=True)
    print("D2: popularity bucket — ID-only", flush=True)
    print("="*60, flush=True)

    id_only_pop_rows = split_by_pop_bucket(id_only_results, item_counts)
    print("[D2-ID-only] bucket 结果：", flush=True)
    for r in id_only_pop_rows:
        print(f"  {r['bucket']}: n={r['num_targets']}, ratio={r['target_ratio']:.4f}, R@50={r['recall@50']:.6f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("D2: popularity bucket — ItemCF", flush=True)
    print("="*60, flush=True)

    itemcf_results = run_itemcf_eval(test_df, train_df, valid_df)
    itemcf_pop_rows = split_by_pop_bucket(itemcf_results, item_counts)
    print("[D2-ItemCF] bucket 结果：", flush=True)
    for r in itemcf_pop_rows:
        print(f"  {r['bucket']}: n={r['num_targets']}, ratio={r['target_ratio']:.4f}, R@50={r['recall@50']:.6f}", flush=True)

    print("\n[D2] 加载 text-enhanced popularity buckets from JSON...", flush=True)
    with open(TEXT_POP_JSON) as f:
        text_pop_rows = json.load(f)
    print("[D2-text-enh] bucket 结果：", flush=True)
    for r in text_pop_rows:
        print(f"  {r['bucket']}: n={r['num_targets']}, ratio={r['target_ratio']:.4f}, R@50={r['recall@50']:.6f}", flush=True)

    # Sanity check: same bucket order
    assert [r["bucket"] for r in id_only_pop_rows] == [r["bucket"] for r in text_pop_rows], \
        "bucket order mismatch between id_only and text_enh"
    assert [r["bucket"] for r in itemcf_pop_rows] == [r["bucket"] for r in text_pop_rows], \
        "bucket order mismatch between itemcf and text_enh"

    write_d2(OUT_DIR, id_only_pop_rows, itemcf_pop_rows, text_pop_rows)

    print("\n" + "="*60, flush=True)
    print("完成", flush=True)
    print(f"输出目录：{OUT_DIR}", flush=True)
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}", flush=True)


if __name__ == "__main__":
    main()

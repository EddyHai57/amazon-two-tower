#!/usr/bin/env python3
"""Faiss IVF offline retrieval benchmark for the Time-decay Text + Mean Pooling Two-Tower model.

Reuses pre-computed embeddings from the FlatIP benchmark run (if available) to build a
Faiss IndexIVFFlat index, runs top-K retrieval over all full-test non-cold users, and
computes Recall@20/50/100 + NDCG@50 + MRR@50 with seen-item filtering identical to the
full eval protocol. Measures index train/add and search latency.

Primary purpose:
  Compare IVF approximate retrieval with FlatIP exact retrieval for the final model.
  Assess recall-speed trade-off.

Eval protocol (must match FlatIP benchmark and full test eval exactly):
  - test user history: train + valid history (history_max_len=20)
  - seen-item filter:  train + valid items masked out per user
  - target item is never masked
  - K_SEARCH = 300 over-fetch before seen-item filtering
"""

from __future__ import annotations

try:
    import argparse
    import json
    import logging
    import platform
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import faiss
    import numpy as np
    import pandas as pd
    import torch
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    logging.error("Missing dependency: %s", exc.name or "unknown")
    raise SystemExit(1) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_text_time_decay_mean_pool_two_tower_smoke as td  # noqa: E402
import benchmark_faiss_time_decay_text_mean_pool as flat_bench  # noqa: E402

# Matching FlatIP benchmark constants
K_SEARCH = 300
K_LIST = [20, 50, 100]

# FlatIP reference results (for comparison table)
FLATIP_RECALL50 = 0.07831490321670997
FLATIP_SEARCH_SEC = 426.15
FLATIP_THROUGHPUT = 1165.0
FLATIP_AVG_LATENCY_MS = 0.858


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faiss IVF offline benchmark: Time-decay Text+MP Two-Tower."
    )
    parser.add_argument(
        "--config",
        default="configs/two_tower_movies_tv_5core_text_time_decay_mean_pool_20epoch.yaml",
        help="Training config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/text_time_decay_mean_pool_20ep/checkpoints/best_model.pt",
        help="Best model checkpoint path.",
    )
    parser.add_argument(
        "--flat_output_dir",
        default="outputs/faiss_time_decay_text_mean_pool_tau015",
        help="Directory with pre-computed FlatIP embeddings (.npy files). Used to skip re-encoding.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/faiss_ivf_time_decay_text_mean_pool_tau015",
        help="Output directory for IVF metrics and report.",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of IVF coarse clusters.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
        help="Number of IVF cells to probe at search time.",
    )
    parser.add_argument(
        "--nprobe_high",
        type=int,
        default=64,
        help="High-recall nprobe value for optional second sweep.",
    )
    parser.add_argument(
        "--run_high_nprobe",
        action="store_true",
        default=True,
        help="Also run search with nprobe_high for comparison.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=8192,
        help="Batch size for encoding items and users (used only if .npy not available).",
    )
    parser.add_argument(
        "--search_batch_size",
        type=int,
        default=8192,
        help="Number of query vectors per Faiss batch search call.",
    )
    parser.add_argument(
        "--faiss_num_threads",
        type=int,
        default=8,
        help="Number of OMP threads for Faiss.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_or_encode_item_embeddings(
    flat_output_dir: Path,
    model: td.TextTimeDecayMeanPoolTwoTower,
    num_items: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float, bool]:
    """Load pre-computed item embeddings or re-encode from model.

    Returns:
        item_emb (np.ndarray), encode_or_load_sec (float), loaded_from_disk (bool)
    """
    npy_path = flat_output_dir / "item_embeddings.npy"
    if npy_path.exists():
        logging.info("Loading pre-computed item embeddings from %s ...", npy_path)
        t0 = time.perf_counter()
        item_emb = np.load(str(npy_path))
        elapsed = time.perf_counter() - t0
        logging.info(
            "item embeddings loaded: shape=%s  time=%.3fs", item_emb.shape, elapsed
        )
        return np.ascontiguousarray(item_emb, dtype=np.float32), elapsed, True
    logging.info("item_embeddings.npy not found; re-encoding from model ...")
    t0 = time.perf_counter()
    item_emb = flat_bench.encode_all_items(model, num_items, device, batch_size)
    elapsed = time.perf_counter() - t0
    logging.info(
        "item embeddings encoded: shape=%s  time=%.2fs", item_emb.shape, elapsed
    )
    return item_emb, elapsed, False


def load_or_encode_user_embeddings(
    flat_output_dir: Path,
    model: td.TextTimeDecayMeanPoolTwoTower,
    user_indices: np.ndarray,
    history_matrix: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float, bool]:
    """Load pre-computed user embeddings or re-encode from model.

    Returns:
        user_emb (np.ndarray), encode_or_load_sec (float), loaded_from_disk (bool)
    """
    emb_path = flat_output_dir / "test_user_embeddings.npy"
    idx_path = flat_output_dir / "test_user_idx.npy"
    if emb_path.exists() and idx_path.exists():
        logging.info("Loading pre-computed user embeddings from %s ...", emb_path)
        t0 = time.perf_counter()
        saved_idx = np.load(str(idx_path))
        user_emb = np.load(str(emb_path))
        elapsed = time.perf_counter() - t0
        if not np.array_equal(saved_idx, user_indices):
            logging.warning(
                "test_user_idx mismatch (saved=%d, current=%d). Re-encoding ...",
                len(saved_idx),
                len(user_indices),
            )
            t0 = time.perf_counter()
            user_emb = flat_bench.encode_all_users(
                model, user_indices, history_matrix, device, batch_size
            )
            elapsed = time.perf_counter() - t0
            return (
                np.ascontiguousarray(user_emb, dtype=np.float32),
                elapsed,
                False,
            )
        logging.info(
            "user embeddings loaded: shape=%s  time=%.3fs  (consistent with FlatIP run)",
            user_emb.shape,
            elapsed,
        )
        return np.ascontiguousarray(user_emb, dtype=np.float32), elapsed, True
    logging.info("test_user_embeddings.npy not found; re-encoding from model ...")
    t0 = time.perf_counter()
    user_emb = flat_bench.encode_all_users(
        model, user_indices, history_matrix, device, batch_size
    )
    elapsed = time.perf_counter() - t0
    logging.info(
        "user embeddings encoded: shape=%s  time=%.2fs", user_emb.shape, elapsed
    )
    return user_emb, elapsed, False


def build_ivf_index(
    item_emb: np.ndarray,
    nlist: int,
    nprobe: int,
) -> tuple[faiss.IndexIVFFlat, float, float]:
    """Build IndexIVFFlat. Returns (index, train_sec, add_sec)."""
    d = item_emb.shape[1]
    n = item_emb.shape[0]
    min_required = 39 * nlist
    if n < min_required:
        logging.warning(
            "n_items=%d < 39*nlist=%d — Faiss recommends at least %d training points "
            "for %d centroids; index will still function but centroid quality may be reduced.",
            n, min_required, min_required, nlist,
        )
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    logging.info("Training IVF index: nlist=%d  n_train=%d ...", nlist, n)
    t0 = time.perf_counter()
    index.train(item_emb)
    train_sec = time.perf_counter() - t0
    logging.info("IVF training done: %.3f s", train_sec)

    logging.info("Adding %d item vectors to IVF index ...", n)
    t1 = time.perf_counter()
    index.add(item_emb)
    add_sec = time.perf_counter() - t1
    logging.info("IVF add done: ntotal=%d  add_time=%.3f s", index.ntotal, add_sec)

    index.nprobe = nprobe
    return index, train_sec, add_sec


def ivf_batch_search(
    index: faiss.IndexIVFFlat,
    user_emb: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Search all users in batches. Returns (indices [N,k], search_sec)."""
    n = user_emb.shape[0]
    all_indices = np.empty((n, k), dtype=np.int64)
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, idx = index.search(
            np.ascontiguousarray(user_emb[start:end], dtype=np.float32), k
        )
        all_indices[start:end] = idx
    return all_indices, time.perf_counter() - t0


def _fmt(val: float | None, fmt: str = ".6f") -> str:
    return f"{val:{fmt}}" if val is not None else "n/a"


def write_report(
    path: Path,
    checkpoint_path: Path,
    item_emb_shape: tuple[int, int],
    num_test_users: int,
    num_skipped_cold: int,
    nlist: int,
    nprobe: int,
    nprobe_high: int,
    metrics32: dict[str, Any],
    metrics64: dict[str, Any] | None,
    index_train_sec: float,
    index_add_sec: float,
    search_sec32: float,
    search_sec64: float | None,
    embedding_loaded: bool,
    env: dict[str, Any],
) -> None:
    tp32 = num_test_users / search_sec32 if search_sec32 > 0 else 0.0
    lat32 = (search_sec32 / max(num_test_users, 1)) * 1000.0
    r50_32 = metrics32.get("recall@50", 0.0)
    abs_delta = r50_32 - FLATIP_RECALL50
    rel_delta = abs_delta / FLATIP_RECALL50 if FLATIP_RECALL50 > 0 else 0.0
    speedup = FLATIP_SEARCH_SEC / search_sec32 if search_sec32 > 0 else 0.0

    r50_64 = metrics64.get("recall@50") if metrics64 else None
    tp64 = (num_test_users / search_sec64) if search_sec64 and search_sec64 > 0 else None
    lat64 = ((search_sec64 / max(num_test_users, 1)) * 1000.0) if search_sec64 else None
    speedup64 = (FLATIP_SEARCH_SEC / search_sec64) if search_sec64 and search_sec64 > 0 else None

    def m32(key: str) -> str:
        return _fmt(metrics32.get(key))

    def m64(key: str) -> str:
        return _fmt(metrics64.get(key) if metrics64 else None)

    if abs(rel_delta) <= 0.01:
        conclusion = (
            f"**Conclusion A**: IVF nprobe={nprobe} recall is within 1% relative of FlatIP "
            f"({rel_delta:+.4%}) and search is {speedup:.2f}× faster."
        )
    elif abs(rel_delta) <= 0.05:
        conclusion = (
            f"**Conclusion B**: IVF nprobe={nprobe} recall drops {abs(rel_delta):.2%} "
            f"relative vs FlatIP. Increasing nprobe may recover recall."
        )
    else:
        conclusion = (
            f"**Conclusion C**: IVF nprobe={nprobe} recall drops {abs(rel_delta):.2%} "
            f"relative vs FlatIP. nprobe={nprobe} is too low for this embedding space."
        )

    lines = [
        "# Time-decay Text+MP Two-Tower — Faiss IVF Offline Benchmark",
        "",
        "## 1. Benchmark Purpose",
        "",
        "Compare Faiss IndexIVFFlat (approximate nearest neighbour) with IndexFlatIP (exact)",
        "retrieval for the final model. Assess the recall-speed trade-off.",
        "Offline benchmark only — not representative of online serving latency.",
        "",
        "## 2. Model",
        "",
        "- **Model**: Time-decay Text + Mean Pooling Two-Tower, τ=0.15, decay_rate=0.8",
        f"- **Checkpoint**: `{checkpoint_path}`",
        "- **Best epoch**: 17 (limited valid Recall@50 = 0.121140)",
        "- **No retraining performed.**",
        "",
        "## 3. IVF Index Parameters",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| Index type | IndexIVFFlat |",
        f"| d (embedding dim) | {item_emb_shape[1]} |",
        f"| nlist | {nlist} |",
        f"| nprobe (default run) | {nprobe} |",
        f"| nprobe (high-recall run) | {nprobe_high if metrics64 else 'not run'} |",
        "| Metric | METRIC_INNER_PRODUCT |",
        "| Embeddings | L2 normalized → inner product = cosine similarity |",
        f"| K_SEARCH | {K_SEARCH} (over-fetch before seen-item filtering) |",
        "",
        "## 4. Embeddings",
        "",
        f"- Item embeddings: {item_emb_shape[0]:,} × {item_emb_shape[1]}",
        f"- Test users (non-cold): {num_test_users:,}",
        f"- Skipped cold users: {num_skipped_cold:,}",
        f"- Source: {'loaded from FlatIP run (outputs/faiss_time_decay_text_mean_pool_tau015/)' if embedding_loaded else 're-encoded from checkpoint'}",
        "",
        "## 5. Metrics",
        "",
        f"| Metric | FlatIP (exact) | IVF nprobe={nprobe} | IVF nprobe={nprobe_high} |",
        "| --- | ---: | ---: | ---: |",
        f"| Recall@20  | 0.052724 | {m32('recall@20')} | {m64('recall@20')} |",
        f"| **Recall@50**  | **0.078315** | **{m32('recall@50')}** | **{m64('recall@50')}** |",
        f"| Recall@100 | 0.104792 | {m32('recall@100')} | {m64('recall@100')} |",
        f"| NDCG@50    | 0.030862 | {m32('ndcg@50')} | {m64('ndcg@50')} |",
        f"| MRR@50     | 0.019036 | {m32('mrr@50')} | {m64('mrr@50')} |",
        "",
        "## 6. Recall vs FlatIP (nprobe=32)",
        "",
        "| | Value |",
        "| --- | ---: |",
        f"| FlatIP Recall@50 | {FLATIP_RECALL50:.6f} |",
        f"| IVF nprobe={nprobe} Recall@50 | {r50_32:.6f} |",
        f"| Absolute delta | {abs_delta:+.6f} |",
        f"| Relative delta | {rel_delta:+.4%} |",
        "",
        "## 7. Latency / Throughput",
        "",
        f"| Metric | FlatIP | IVF nprobe={nprobe} | IVF nprobe={nprobe_high} |",
        "| --- | ---: | ---: | ---: |",
        f"| Index train time | N/A | {index_train_sec:.3f} s | same index |",
        f"| Index add time | 0.034 s | {index_add_sec:.3f} s | same index |",
        f"| Search time ({num_test_users:,} users) | {FLATIP_SEARCH_SEC:.2f} s | {search_sec32:.2f} s | {_fmt(search_sec64, '.2f') + ' s' if search_sec64 else 'n/a'} |",
        f"| Throughput | {FLATIP_THROUGHPUT:,.1f} users/s | {tp32:,.1f} users/s | {_fmt(tp64, ',.1f') + ' users/s' if tp64 else 'n/a'} |",
        f"| Avg latency | {FLATIP_AVG_LATENCY_MS:.4f} ms | {lat32:.4f} ms | {_fmt(lat64, '.4f') + ' ms' if lat64 else 'n/a'} |",
        f"| Speedup vs FlatIP | 1.00× | {speedup:.2f}× | {_fmt(speedup64, '.2f') + '×' if speedup64 else 'n/a'} |",
        "",
        "## 8. Trade-off Analysis",
        "",
        f"- Absolute Recall@50 drop (nprobe={nprobe}): {abs_delta:+.6f}",
        f"- Relative Recall@50 drop (nprobe={nprobe}): {rel_delta:+.4%}",
        f"- Search speedup (nprobe={nprobe} vs FlatIP): {speedup:.2f}×",
        "",
        conclusion,
        "",
        "## 9. Notes",
        "",
        "- Offline benchmark only; not representative of online serving latency.",
        "- No retraining performed. Final best checkpoint (epoch 17) loaded.",
        "- Seen-item filtering matches full eval protocol exactly.",
        f"- n_items={item_emb_shape[0]:,} < 39×nlist={39*nlist:,}: faiss emits a warning about",
        "  centroid count but the index is still functional.",
        "- outputs/ and logs/ not committed.",
        "",
        "## 10. Environment",
        "",
        f"- python: {env['python']}",
        f"- torch: {env['torch']}",
        f"- faiss: {env['faiss']}",
        f"- platform: {env['platform']}",
        f"- cuda: {env.get('cuda_device', 'n/a')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    flat_output_dir = Path(args.flat_output_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = td.load_config(config_path)
    config["config_path"] = str(config_path)
    td.require_config(config)
    config["eval_max_users"] = None
    td.set_seed(42)
    faiss.omp_set_num_threads(args.faiss_num_threads)

    device = td.resolve_device(str(config["device"]))
    bundle = td.load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_max_len = int(config["history_max_len"])

    # Full eval protocol: test history = train + valid; seen = train + valid
    test_history_frame = pd.concat(
        [bundle.train_df, bundle.valid_df[td.TRAIN_COLUMNS]], ignore_index=True
    )
    test_history_matrix, _ = td.build_history_matrix(
        test_history_frame, num_users, history_max_len
    )
    train_seen = td.build_seen_items(bundle.train_df)
    test_seen = td.merge_seen_items(train_seen, bundle.valid_df)

    # Load model (needed if embeddings not available on disk)
    model = td.build_model(config, bundle.stats, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logging.info(
        "checkpoint loaded: epoch=%s  valid_recall@50=%.6f",
        ckpt.get("epoch"),
        float(ckpt.get("best_metric_value", 0.0)),
    )

    # Test eval targets (non-cold only)
    eval_targets = bundle.test_df[
        ~bundle.test_df["is_cold_item_for_eval"].astype(bool)
    ].copy()
    num_skipped_cold = int(bundle.test_df["is_cold_item_for_eval"].astype(bool).sum())
    unique_test_users = eval_targets["user_idx"].to_numpy(dtype=np.int64)
    logging.info(
        "test non-cold targets: %d  skipped cold: %d",
        len(eval_targets),
        num_skipped_cold,
    )

    # Load or encode embeddings
    item_emb, item_load_sec, item_from_disk = load_or_encode_item_embeddings(
        flat_output_dir, model, num_items, device, args.embedding_batch_size
    )
    user_emb, user_load_sec, user_from_disk = load_or_encode_user_embeddings(
        flat_output_dir, model, unique_test_users, test_history_matrix,
        device, args.embedding_batch_size,
    )
    embedding_loaded = item_from_disk and user_from_disk
    logging.info(
        "embeddings ready: item=%s  user=%s  (loaded_from_disk=%s)",
        item_emb.shape, user_emb.shape, embedding_loaded,
    )

    # Build IVF index
    nlist = args.nlist
    nprobe = args.nprobe
    index, index_train_sec, index_add_sec = build_ivf_index(item_emb, nlist, nprobe)
    logging.info("IndexIVFFlat ready: nlist=%d  nprobe=%d", nlist, nprobe)

    # Search — nprobe=32
    logging.info(
        "Faiss IVF search: K_SEARCH=%d  nprobe=%d  n_users=%d ...",
        K_SEARCH, nprobe, len(user_emb),
    )
    topk32, search_sec32 = ivf_batch_search(
        index, user_emb, K_SEARCH, args.search_batch_size
    )
    logging.info("IVF search (nprobe=%d) complete: %.2f s", nprobe, search_sec32)

    logging.info("Computing recall metrics (nprobe=%d) ...", nprobe)
    metrics32 = flat_bench.compute_recall_metrics(
        topk32, eval_targets, test_seen, K_LIST
    )
    logging.info(
        "nprobe=%d: Recall@20=%.6f  Recall@50=%.6f  Recall@100=%.6f  NDCG@50=%.6f  MRR@50=%.6f",
        nprobe,
        metrics32["recall@20"], metrics32["recall@50"], metrics32["recall@100"],
        metrics32["ndcg@50"], metrics32["mrr@50"],
    )

    # Search — nprobe_high (optional)
    metrics64: dict[str, Any] | None = None
    search_sec64: float | None = None
    if args.run_high_nprobe:
        nprobe_high = args.nprobe_high
        index.nprobe = nprobe_high
        logging.info(
            "Faiss IVF search (high nprobe=%d) ...", nprobe_high
        )
        topk64, search_sec64 = ivf_batch_search(
            index, user_emb, K_SEARCH, args.search_batch_size
        )
        logging.info("IVF search (nprobe=%d) complete: %.2f s", nprobe_high, search_sec64)
        metrics64 = flat_bench.compute_recall_metrics(
            topk64, eval_targets, test_seen, K_LIST
        )
        logging.info(
            "nprobe=%d: Recall@20=%.6f  Recall@50=%.6f  Recall@100=%.6f  NDCG@50=%.6f  MRR@50=%.6f",
            nprobe_high,
            metrics64["recall@20"], metrics64["recall@50"], metrics64["recall@100"],
            metrics64["ndcg@50"], metrics64["mrr@50"],
        )

    # Summary stats
    tp32 = metrics32["num_eval_users"] / search_sec32 if search_sec32 > 0 else 0.0
    lat32_ms = (search_sec32 / max(metrics32["num_eval_users"], 1)) * 1000.0
    speedup32 = FLATIP_SEARCH_SEC / search_sec32 if search_sec32 > 0 else 0.0
    abs_delta = metrics32["recall@50"] - FLATIP_RECALL50
    rel_delta = abs_delta / FLATIP_RECALL50

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "faiss": faiss.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "Time-decay Text+MP Two-Tower tau=0.15 decay_rate=0.8",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "item_embedding_shape": list(item_emb.shape),
        "embedding_dim": item_emb.shape[1],
        "num_test_eval_users": metrics32["num_eval_users"],
        "num_skipped_cold_users": num_skipped_cold,
        "candidate_items": num_items,
        "index_type": "IndexIVFFlat",
        "nlist": nlist,
        "nprobe": nprobe,
        "nprobe_high": args.nprobe_high if metrics64 else None,
        "k_search": K_SEARCH,
        "k_list": K_LIST,
        "seen_item_filtering": "train+valid per user, target never masked",
        "embeddings_loaded_from_flatip": embedding_loaded,
        "metrics_nprobe32": metrics32,
        "metrics_nprobe64": metrics64,
        "reference_flatip": {
            "recall50": FLATIP_RECALL50,
            "search_sec": FLATIP_SEARCH_SEC,
            "throughput_users_per_sec": FLATIP_THROUGHPUT,
            "avg_latency_ms_per_user": FLATIP_AVG_LATENCY_MS,
        },
        "delta_vs_flatip_recall50": float(abs_delta),
        "relative_delta_vs_flatip_recall50": float(rel_delta),
        "timing": {
            "item_load_or_encode_sec": float(item_load_sec),
            "user_load_or_encode_sec": float(user_load_sec),
            "index_train_sec": float(index_train_sec),
            "index_add_sec": float(index_add_sec),
            "search_sec_nprobe32": float(search_sec32),
            "search_sec_nprobe64": float(search_sec64) if search_sec64 is not None else None,
            "avg_latency_ms_per_user_nprobe32": float(lat32_ms),
            "throughput_users_per_sec_nprobe32": float(tp32),
            "speedup_vs_flatip_nprobe32": float(speedup32),
        },
        "search_batch_size": args.search_batch_size,
        "embedding_batch_size": args.embedding_batch_size,
        "faiss_num_threads": args.faiss_num_threads,
        "environment": env,
    }
    write_json(output_dir / "metrics.json", result)

    write_report(
        path=output_dir / "report.md",
        checkpoint_path=checkpoint_path,
        item_emb_shape=tuple(item_emb.shape),
        num_test_users=metrics32["num_eval_users"],
        num_skipped_cold=num_skipped_cold,
        nlist=nlist,
        nprobe=nprobe,
        nprobe_high=args.nprobe_high,
        metrics32=metrics32,
        metrics64=metrics64,
        index_train_sec=index_train_sec,
        index_add_sec=index_add_sec,
        search_sec32=search_sec32,
        search_sec64=search_sec64,
        embedding_loaded=embedding_loaded,
        env=env,
    )

    logging.info("IVF benchmark complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Faiss ANN offline benchmark — Transformer Two-Tower (Text+Time-aware Transformer, τ=0.15).

Benchmark target: standalone full test Recall@50 ≈ 0.103168.
NOT the four-channel weighted RRF value (0.125164).

Index types:
  1. FlatIP   — exact retrieval, correctness baseline
  2. IVFFlat  — nprobe = 16 / 32 / 64
  3. HNSW     — M=32, efSearch = 64 / 128  (skipped if unsupported)

Output: outputs/faiss_transformer_two_tower_benchmark/
"""

from __future__ import annotations

import csv
import json
import logging
import math
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_transformer_maxlen100_smoke as tr  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────
K_SEARCH = 300          # over-fetch before seen-item filtering
K_LIST = [20, 50, 100]

NLIST = 1024
NPROBE_VALUES = [16, 32, 64]

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 40
HNSW_EF_SEARCH_VALUES = [64, 128]

CONFIG_PATH = Path("configs/two_tower_movies_tv_5core_text_timeaware_transformer_max100_final.yaml")
CHECKPOINT_PATH = Path("outputs/text_timeaware_transformer_max100_final/checkpoints/best_model.pt")
OUTPUT_DIR = Path("outputs/faiss_transformer_two_tower_benchmark")

BATCH_SIZE_EMBED = 8192
BATCH_SIZE_SEARCH = 8192
FAISS_THREADS = 8

EXPECTED_R50 = 0.10312808   # canonical full test Recall@50, seed=42, best_epoch=2


# ── helpers ────────────────────────────────────────────────────────────────

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compute_recall_metrics(
    topk_indices: np.ndarray,
    eval_targets: pd.DataFrame,
    seen_items: dict[int, set[int]],
    k_list: list[int],
) -> dict[str, Any]:
    """Apply seen-item filtering (target never masked). Return Recall/NDCG/MRR@k."""
    max_k = max(k_list)
    recall: dict[int, int] = {k: 0 for k in k_list}
    ndcg: dict[int, float] = {k: 0.0 for k in k_list}
    mrr: dict[int, float] = {k: 0.0 for k in k_list}

    users = eval_targets["user_idx"].to_numpy(dtype=np.int64)
    targets = eval_targets["item_idx"].to_numpy(dtype=np.int64)
    n = len(users)

    for i in range(n):
        u = int(users[i])
        t = int(targets[i])
        seen = seen_items.get(u, set())
        cands = topk_indices[i].tolist()

        filtered: list[int] = []
        for c in cands:
            if c not in seen or c == t:
                filtered.append(c)
                if len(filtered) == max_k:
                    break

        for rank, item in enumerate(filtered, 1):
            if item == t:
                inv_log = 1.0 / math.log2(rank + 1)
                inv_rank = 1.0 / rank
                for k in k_list:
                    if rank <= k:
                        recall[k] += 1
                        ndcg[k] += inv_log
                        mrr[k] += inv_rank
                break

    return {
        **{f"recall@{k}": recall[k] / n for k in k_list},
        **{f"ndcg@{k}": ndcg[k] / n for k in k_list},
        **{f"mrr@{k}": mrr[k] / n for k in k_list},
        "num_eval_users": n,
    }


def encode_all_items(
    model: tr.TextTwoTowerTransformerSmoke,
    num_items: int,
    device: torch.device,
    batch_size: int = BATCH_SIZE_EMBED,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(idx).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def encode_all_users(
    model: tr.TextTwoTowerTransformerSmoke,
    user_indices: np.ndarray,
    history_matrix: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE_EMBED,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            end = min(start + batch_size, len(user_indices))
            u_batch = user_indices[start:end]
            u_t = torch.as_tensor(u_batch, dtype=torch.long, device=device)
            h_t = torch.as_tensor(history_matrix[u_batch], dtype=torch.long, device=device)
            chunks.append(model.encode_users(u_t, h_t).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def batch_search(
    index: Any,
    user_emb: np.ndarray,
    k: int,
    batch_size: int = BATCH_SIZE_SEARCH,
) -> tuple[np.ndarray, float]:
    n = len(user_emb)
    result = np.empty((n, k), dtype=np.int64)
    t0 = time.perf_counter()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        _, idx = index.search(np.ascontiguousarray(user_emb[s:e], dtype=np.float32), k)
        result[s:e] = idx
    return result, time.perf_counter() - t0


def index_size_bytes(n: int, d: int) -> int:
    return n * d * 4   # float32 vectors only (lower bound)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(FAISS_THREADS)

    # ── load model & data ──────────────────────────────────────────────────
    config = tr.load_config(CONFIG_PATH)
    config["config_path"] = str(CONFIG_PATH)
    tr.require_config(config)
    config["eval_max_users"] = None
    tr.set_seed(42)

    device = tr.resolve_device(str(config["device"]))
    logging.info("Device: %s", device)

    bundle = tr.load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    num_items = int(bundle.stats["n_items"])
    history_max_len = int(config["history_max_len"])

    # Test eval protocol: history = train+valid, seen = train+valid
    test_hist_frame = pd.concat(
        [bundle.train_df, bundle.valid_df[tr.TRAIN_COLUMNS]], ignore_index=True
    )
    # Transformer build_history_matrix returns plain ndarray (not tuple)
    test_history = tr.build_history_matrix(test_hist_frame, num_users, history_max_len)
    train_seen = tr.build_seen_items(bundle.train_df)
    test_seen = tr.merge_seen_items(train_seen, bundle.valid_df)

    model = tr.build_model(config, bundle.stats, device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logging.info(
        "Checkpoint loaded: epoch=%s  best_valid_recall@50=%.6f",
        ckpt.get("epoch"), float(ckpt.get("best_metric_value", 0)),
    )

    eval_targets = bundle.test_df[~bundle.test_df["is_cold_item_for_eval"].astype(bool)].copy()
    n_cold = int(bundle.test_df["is_cold_item_for_eval"].astype(bool).sum())
    user_indices = eval_targets["user_idx"].to_numpy(dtype=np.int64)
    logging.info("Test non-cold: %d  cold skipped: %d", len(eval_targets), n_cold)

    # ── encode embeddings ──────────────────────────────────────────────────
    logging.info("Encoding %d items ...", num_items)
    t0 = time.perf_counter()
    item_emb = encode_all_items(model, num_items, device)
    item_sec = time.perf_counter() - t0
    logging.info("Item emb: shape=%s  time=%.2fs", item_emb.shape, item_sec)

    logging.info("Encoding %d test users ...", len(user_indices))
    t0 = time.perf_counter()
    user_emb = encode_all_users(model, user_indices, test_history, device)
    user_sec = time.perf_counter() - t0
    logging.info("User emb: shape=%s  time=%.2fs", user_emb.shape, user_sec)

    np.save(str(OUTPUT_DIR / "item_embeddings.npy"), item_emb)
    np.save(str(OUTPUT_DIR / "test_user_embeddings.npy"), user_emb)
    np.save(str(OUTPUT_DIR / "test_user_idx.npy"), user_indices)
    logging.info("Embeddings saved to %s", OUTPUT_DIR)

    d = item_emb.shape[1]
    n_eval = len(user_emb)
    all_results: dict[str, Any] = {}

    # ── FlatIP ─────────────────────────────────────────────────────────────
    logging.info("=== FlatIP (exact) ===")
    flat_idx = faiss.IndexFlatIP(d)
    t0 = time.perf_counter()
    flat_idx.add(item_emb)
    flat_add = time.perf_counter() - t0

    topk_flat, flat_search = batch_search(flat_idx, user_emb, K_SEARCH)
    flat_tp = n_eval / flat_search
    flat_lat = flat_search / n_eval * 1000
    logging.info("FlatIP search: %.2fs  %.0f users/s  %.4fms/user", flat_search, flat_tp, flat_lat)

    t0 = time.perf_counter()
    flat_m = compute_recall_metrics(topk_flat, eval_targets, test_seen, K_LIST)
    logging.info("FlatIP metrics: R@50=%.6f  NDCG@50=%.6f  MRR@50=%.6f  (eval %.1fs)",
                 flat_m["recall@50"], flat_m["ndcg@50"], flat_m["mrr@50"], time.perf_counter() - t0)

    rel_diff = abs(flat_m["recall@50"] - EXPECTED_R50) / EXPECTED_R50
    alignment_pass = rel_diff < 0.001
    logging.info(
        "Alignment check: expected=%.6f  actual=%.6f  rel_diff=%.4f%%  %s",
        EXPECTED_R50, flat_m["recall@50"], rel_diff * 100,
        "PASS ✓" if alignment_pass else "WARN ⚠",
    )

    all_results["flatip"] = {
        "index_type": "FlatIP",
        "add_sec": float(flat_add),
        "search_sec": float(flat_search),
        "throughput_users_per_sec": float(flat_tp),
        "avg_latency_ms": float(flat_lat),
        "index_size_bytes": index_size_bytes(num_items, d),
        "speedup_vs_flatip": 1.0,
        "recall50_delta_vs_flatip": 0.0,
        "metrics": flat_m,
        "alignment_expected_r50": EXPECTED_R50,
        "alignment_actual_r50": float(flat_m["recall@50"]),
        "alignment_rel_diff": float(rel_diff),
        "alignment_pass": alignment_pass,
    }

    # ── IVF ────────────────────────────────────────────────────────────────
    logging.info("=== Building IVF index: nlist=%d ===", NLIST)
    quantizer = faiss.IndexFlatIP(d)
    ivf_idx = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)

    t0 = time.perf_counter()
    ivf_idx.train(item_emb)
    ivf_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    ivf_idx.add(item_emb)
    ivf_add = time.perf_counter() - t0
    logging.info("IVF built: nlist=%d  train=%.2fs  add=%.2fs", NLIST, ivf_train, ivf_add)

    for nprobe in NPROBE_VALUES:
        ivf_idx.nprobe = nprobe
        logging.info("IVF search: nprobe=%d ...", nprobe)
        topk_ivf, ivf_search = batch_search(ivf_idx, user_emb, K_SEARCH)
        t0 = time.perf_counter()
        ivf_m = compute_recall_metrics(topk_ivf, eval_targets, test_seen, K_LIST)
        eval_t = time.perf_counter() - t0

        ivf_tp = n_eval / ivf_search
        ivf_lat = ivf_search / n_eval * 1000
        speedup = flat_search / ivf_search
        r50_delta = ivf_m["recall@50"] - flat_m["recall@50"]

        logging.info(
            "IVF nprobe=%2d: R@50=%.6f  delta=%+.6f (%+.3f%%)  speedup=%.1f×  lat=%.4fms  (eval %.1fs)",
            nprobe, ivf_m["recall@50"], r50_delta, 100 * r50_delta / flat_m["recall@50"],
            speedup, ivf_lat, eval_t,
        )

        all_results[f"ivf_nprobe{nprobe}"] = {
            "index_type": "IVFFlat",
            "nlist": NLIST,
            "nprobe": nprobe,
            "train_sec": float(ivf_train),
            "add_sec": float(ivf_add),
            "search_sec": float(ivf_search),
            "throughput_users_per_sec": float(ivf_tp),
            "avg_latency_ms": float(ivf_lat),
            "speedup_vs_flatip": float(speedup),
            "recall50_delta_vs_flatip": float(r50_delta),
            "recall50_relative_delta_vs_flatip": float(r50_delta / flat_m["recall@50"]),
            "index_size_bytes": index_size_bytes(num_items, d),
            "metrics": ivf_m,
        }

    # ── HNSW (optional) ────────────────────────────────────────────────────
    logging.info("=== HNSW: M=%d efCons=%d ===", HNSW_M, HNSW_EF_CONSTRUCTION)
    hnsw_skipped = False
    hnsw_skip_reason = ""
    try:
        hnsw_idx = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        hnsw_idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        t0 = time.perf_counter()
        hnsw_idx.add(item_emb)
        hnsw_add = time.perf_counter() - t0
        logging.info("HNSW built: ntotal=%d  add=%.2fs", hnsw_idx.ntotal, hnsw_add)

        for ef in HNSW_EF_SEARCH_VALUES:
            hnsw_idx.hnsw.efSearch = ef
            topk_hnsw, hnsw_search = batch_search(hnsw_idx, user_emb, K_SEARCH)
            hnsw_m = compute_recall_metrics(topk_hnsw, eval_targets, test_seen, K_LIST)
            hnsw_tp = n_eval / hnsw_search
            hnsw_lat = hnsw_search / n_eval * 1000
            speedup = flat_search / hnsw_search
            r50_delta = hnsw_m["recall@50"] - flat_m["recall@50"]
            logging.info(
                "HNSW efSearch=%d: R@50=%.6f  delta=%+.6f  speedup=%.1f×  lat=%.4fms",
                ef, hnsw_m["recall@50"], r50_delta, speedup, hnsw_lat,
            )
            all_results[f"hnsw_ef{ef}"] = {
                "index_type": "HNSWFlat",
                "M": HNSW_M,
                "efConstruction": HNSW_EF_CONSTRUCTION,
                "efSearch": ef,
                "add_sec": float(hnsw_add),
                "search_sec": float(hnsw_search),
                "throughput_users_per_sec": float(hnsw_tp),
                "avg_latency_ms": float(hnsw_lat),
                "speedup_vs_flatip": float(speedup),
                "recall50_delta_vs_flatip": float(r50_delta),
                "recall50_relative_delta_vs_flatip": float(r50_delta / flat_m["recall@50"]),
                "index_size_bytes": index_size_bytes(num_items, d),
                "metrics": hnsw_m,
            }
    except Exception as exc:
        hnsw_skipped = True
        hnsw_skip_reason = str(exc)
        logging.warning("HNSW skipped: %s", exc)
        all_results["hnsw_skipped"] = {"reason": hnsw_skip_reason}

    # ── save outputs ───────────────────────────────────────────────────────
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "faiss": faiss.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
    }

    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "Text+Time-aware Transformer Two-Tower tau=0.15 max_len=100",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": ckpt.get("epoch"),
        "alignment_target_recall50": EXPECTED_R50,
        "flatip_alignment_pass": alignment_pass,
        "flatip_alignment_rel_diff": float(rel_diff),
        "n_items": num_items,
        "embedding_dim": d,
        "k_search": K_SEARCH,
        "k_list": K_LIST,
        "nlist": NLIST,
        "nprobe_values": NPROBE_VALUES,
        "hnsw_m": HNSW_M,
        "hnsw_skipped": hnsw_skipped,
        "hnsw_skip_reason": hnsw_skip_reason,
        "seen_item_filtering": "test: mask train+valid per user; target never masked",
        "eval_protocol": "temporal leave-one-out, test non-cold users only",
        "num_eval_users": flat_m["num_eval_users"],
        "num_cold_skipped": n_cold,
        "timing": {
            "item_encode_sec": float(item_sec),
            "user_encode_sec": float(user_sec),
        },
        "results": all_results,
        "environment": env,
    }
    write_json(OUTPUT_DIR / "faiss_benchmark_results.json", output)

    # CSV summary
    csv_rows: list[dict[str, Any]] = []
    for key, res in all_results.items():
        if key == "hnsw_skipped":
            continue
        m = res.get("metrics", {})
        csv_rows.append({
            "index": key,
            "recall@20": f"{m.get('recall@20', ''):.6f}",
            "recall@50": f"{m.get('recall@50', ''):.6f}",
            "recall@100": f"{m.get('recall@100', ''):.6f}",
            "ndcg@50": f"{m.get('ndcg@50', ''):.6f}",
            "mrr@50": f"{m.get('mrr@50', ''):.6f}",
            "speedup_vs_flatip": f"{res.get('speedup_vs_flatip', 1.0):.2f}",
            "avg_latency_ms": f"{res.get('avg_latency_ms', 0):.4f}",
            "recall50_delta_vs_flatip": f"{res.get('recall50_delta_vs_flatip', 0):.6f}",
        })
    with (OUTPUT_DIR / "faiss_benchmark_results.csv").open("w", newline="", encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    _write_inline_report(OUTPUT_DIR / "faiss_benchmark_report.md", output, all_results, flat_m, env)

    # ── summary ────────────────────────────────────────────────────────────
    logging.info("")
    logging.info("=== SUMMARY ===")
    logging.info("Alignment: expected=%.6f  actual=%.6f  %s",
                 EXPECTED_R50, flat_m["recall@50"], "PASS" if alignment_pass else "WARN")
    for key, res in all_results.items():
        if key == "hnsw_skipped":
            logging.info("%-24s SKIPPED (%s)", "HNSW", hnsw_skip_reason[:60])
            continue
        m = res.get("metrics", {})
        sp = res.get("speedup_vs_flatip", 1.0)
        lat = res.get("avg_latency_ms", 0.0)
        delta = res.get("recall50_delta_vs_flatip", 0.0)
        logging.info("%-24s R@50=%.6f  delta=%+.6f  speedup=%5.1f×  lat=%.4fms",
                     key, m.get("recall@50", 0), delta, sp, lat)
    logging.info("Outputs: %s", OUTPUT_DIR)


def _write_inline_report(
    path: Path,
    meta: dict[str, Any],
    all_results: dict[str, Any],
    flat_m: dict[str, Any],
    env: dict[str, Any],
) -> None:
    flat = all_results.get("flatip", {})
    flat_lat = flat.get("avg_latency_ms", 0)

    lines = [
        "# Faiss Transformer Two-Tower Offline Benchmark",
        "",
        f"Created: {meta['created_at']}",
        "",
        "## Model",
        f"- {meta['model']}",
        f"- Checkpoint: `{meta['checkpoint']}`  epoch={meta['checkpoint_epoch']}",
        "- No retraining performed.",
        "",
        "## FlatIP Alignment Check",
        "",
        "| | Value |",
        "| --- | ---: |",
        f"| Expected Recall@50 | {EXPECTED_R50:.6f} |",
        f"| Actual Recall@50   | {flat_m['recall@50']:.6f} |",
        f"| Relative diff      | {meta['flatip_alignment_rel_diff']:.4%} |",
        f"| Alignment pass     | {'✓ PASS' if meta['flatip_alignment_pass'] else '⚠ WARN'} |",
        "",
        "## Results Table",
        "",
        "| Index | Recall@50 | NDCG@50 | MRR@50 | Speedup vs FlatIP | Latency (ms/user) | R@50 delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, res in all_results.items():
        if key == "hnsw_skipped":
            lines.append("| HNSW | — | — | — | — | — | skipped |")
            continue
        m = res.get("metrics", {})
        lines.append(
            f"| {key} | {m.get('recall@50', 0):.6f} | {m.get('ndcg@50', 0):.6f} | "
            f"{m.get('mrr@50', 0):.6f} | {res.get('speedup_vs_flatip', 1.0):.2f}× | "
            f"{res.get('avg_latency_ms', 0):.4f} | {res.get('recall50_delta_vs_flatip', 0):+.6f} |"
        )

    lines += [
        "",
        "## Latency / Throughput",
        "",
        f"| Index | Search time ({meta['num_eval_users']:,} users) | Throughput (users/s) | Avg latency (ms/user) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, res in all_results.items():
        if key == "hnsw_skipped":
            continue
        lines.append(
            f"| {key} | {res.get('search_sec', 0):.2f} s | "
            f"{res.get('throughput_users_per_sec', 0):,.0f} | "
            f"{res.get('avg_latency_ms', 0):.4f} |"
        )

    hnsw_note = (
        "HNSW skipped: faiss.IndexHNSWFlat does not support "
        "METRIC_INNER_PRODUCT in faiss " + env.get("faiss", "") + "."
        if meta.get("hnsw_skipped")
        else f"HNSW M={meta['hnsw_m']} efConstruction={HNSW_EF_CONSTRUCTION} included."
    )

    lines += [
        "",
        "## Notes",
        f"- IndexIVFFlat  nlist={meta['nlist']}  nprobe values={meta['nprobe_values']}",
        f"- Index size (vectors only): {meta['n_items']:,} × {meta['embedding_dim']} × float32 = "
        f"{meta['n_items'] * meta['embedding_dim'] * 4 / 1e6:.1f} MB",
        f"- K_SEARCH={meta['k_search']} (over-fetch before seen-item filtering)",
        f"- Seen-item filter: {meta['seen_item_filtering']}",
        "- Offline benchmark only. Not representative of online serving latency.",
        f"- {hnsw_note}",
        "",
        "## Environment",
        f"- Python: {env['python']}",
        f"- PyTorch: {env['torch']}",
        f"- Faiss: {env['faiss']}",
        f"- Platform: {env['platform']}",
        f"- CUDA: {env['cuda']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

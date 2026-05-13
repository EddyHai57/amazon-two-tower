#!/usr/bin/env python3
"""Offline Faiss benchmark for the ID-only Two-Tower checkpoint."""

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
    import yaml

    from train_two_tower import IDOnlyTwoTower, load_checkpoint, load_config, resolve_device, set_seed
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    missing_name = exc.name or "未知依赖"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("缺少依赖：%s。请先在项目 .venv 中安装 package：%s", missing_name, package_hint)
    raise SystemExit(1) from exc


REQUIRED_KEYS = [
    "train_config",
    "data_dir",
    "checkpoint",
    "output_dir",
    "device",
    "top_k",
    "sample_users_per_split",
    "sample_seed",
    "embedding_batch_size",
    "faiss_num_threads",
    "ivf_nlist",
    "ivf_nprobe",
    "ivf_train_sample_size",
    "warmup_queries",
    "eval_splits",
]
EVAL_COLUMNS = ["user_idx", "item_idx", "is_cold_item_for_eval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ID-only Two-Tower offline Faiss benchmark。")
    parser.add_argument("--config", required=True, help="Faiss benchmark YAML 配置文件路径。")
    parser.add_argument(
        "--ivf_nprobe_sweep_only",
        action="store_true",
        help="只复用已导出的 item/user embeddings，补跑 IVF-Flat nprobe sweep。",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_benchmark_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效：{path}")
    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"配置缺少必需字段：{key}")
    if int(config["top_k"]) <= 0:
        raise ValueError("top_k 必须大于 0。")
    if int(config["sample_users_per_split"]) <= 0:
        raise ValueError("sample_users_per_split 必须大于 0。")
    if int(config["embedding_batch_size"]) <= 0:
        raise ValueError("embedding_batch_size 必须大于 0。")
    if int(config["ivf_nlist"]) <= 0 or int(config["ivf_nprobe"]) <= 0:
        raise ValueError("ivf_nlist 和 ivf_nprobe 必须大于 0。")
    for nprobe in config.get("ivf_nprobe_sweep", [config["ivf_nprobe"]]):
        if int(nprobe) <= 0:
            raise ValueError(f"ivf_nprobe_sweep 中存在无效 nprobe：{nprobe}")
    if int(config["warmup_queries"]) < 0:
        raise ValueError("warmup_queries 不能小于 0。")
    for split in config["eval_splits"]:
        if split not in {"valid", "test"}:
            raise ValueError(f"不支持的 eval split：{split}")
    return config


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_stats(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def build_model(
    train_config: dict[str, Any],
    stats: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> IDOnlyTwoTower:
    model = IDOnlyTwoTower(
        num_users=int(stats["n_users"]),
        num_items=int(stats["n_items"]),
        embedding_dim=int(train_config["embedding_dim"]),
        use_l2_norm=bool(train_config["use_l2_norm"]),
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.info(
        "checkpoint 加载完成：epoch=%s metric=%s value=%s",
        checkpoint.get("epoch"),
        checkpoint.get("best_metric_name"),
        checkpoint.get("best_metric_value"),
    )
    return model


def encode_items(
    model: IDOnlyTwoTower,
    num_items: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            item_idx = torch.arange(start, end, dtype=torch.long, device=device)
            item_emb = model.encode_items(item_idx).detach().cpu().numpy().astype(np.float32, copy=False)
            chunks.append(item_emb)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def load_eval_frame(data_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(data_dir / f"{split}.parquet", columns=EVAL_COLUMNS)
    return frame[~frame["is_cold_item_for_eval"].astype(bool)].copy()


def sample_eval_frame(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed).sort_values("user_idx").reset_index(drop=True)
    else:
        frame = frame.sort_values("user_idx").reset_index(drop=True)
    return frame


def encode_users(
    model: IDOnlyTwoTower,
    user_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            end = min(start + batch_size, len(user_indices))
            user_idx = torch.as_tensor(user_indices[start:end], dtype=torch.long, device=device)
            user_emb = model.encode_users(user_idx).detach().cpu().numpy().astype(np.float32, copy=False)
            chunks.append(user_emb)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def exact_topk(query: np.ndarray, item_embeddings: np.ndarray, top_k: int) -> np.ndarray:
    scores = item_embeddings @ query
    if top_k >= scores.shape[0]:
        return np.argsort(scores)[::-1].astype(np.int64, copy=False)
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
    return ordered.astype(np.int64, copy=False)


def latency_stats_ms(latencies: list[float]) -> dict[str, float]:
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "p50": float(np.percentile(values, 50)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "p99": float(np.percentile(values, 99)) if values.size else 0.0,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def mean_overlap_at_k(reference: np.ndarray, candidate: np.ndarray, top_k: int) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(f"topk shape 不一致：reference={reference.shape} candidate={candidate.shape}")
    overlaps = []
    for ref_row, cand_row in zip(reference, candidate, strict=True):
        overlaps.append(len(set(ref_row[:top_k]).intersection(cand_row[:top_k])) / float(top_k))
    return float(np.mean(overlaps)) if overlaps else 0.0


def benchmark_bruteforce(item_embeddings: np.ndarray, user_embeddings: np.ndarray, top_k: int) -> dict[str, Any]:
    latencies = []
    topk_rows = []
    for query in user_embeddings:
        start = time.perf_counter_ns()
        topk = exact_topk(query, item_embeddings, top_k)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        latencies.append(elapsed_ms)
        topk_rows.append(topk)
    topk_array = np.vstack(topk_rows).astype(np.int64, copy=False)
    return {
        "latencies_ms": latencies,
        "topk": topk_array,
    }


def benchmark_faiss(index: faiss.Index, user_embeddings: np.ndarray, top_k: int, warmup_queries: int) -> dict[str, Any]:
    latencies = []
    topk_rows = []
    warmup = min(warmup_queries, len(user_embeddings))
    for query in user_embeddings[:warmup]:
        index.search(np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), top_k)
    for query in user_embeddings:
        query_matrix = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        start = time.perf_counter_ns()
        _, topk = index.search(query_matrix, top_k)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        latencies.append(elapsed_ms)
        topk_rows.append(topk[0].astype(np.int64, copy=False))
    topk_array = np.vstack(topk_rows).astype(np.int64, copy=False)
    return {
        "latencies_ms": latencies,
        "topk": topk_array,
    }


def build_flat_index(item_embeddings: np.ndarray) -> tuple[faiss.IndexFlatIP, float]:
    start = time.perf_counter()
    index = faiss.IndexFlatIP(item_embeddings.shape[1])
    index.add(item_embeddings)
    return index, time.perf_counter() - start


def build_ivf_index(
    item_embeddings: np.ndarray,
    nlist: int,
    nprobe: int,
    train_sample_size: int,
    seed: int,
) -> tuple[faiss.IndexIVFFlat, dict[str, float | int]]:
    dim = item_embeddings.shape[1]
    train_size = min(int(train_sample_size), item_embeddings.shape[0])
    rng = np.random.default_rng(seed)
    if train_size < item_embeddings.shape[0]:
        train_idx = rng.choice(item_embeddings.shape[0], size=train_size, replace=False)
        train_vectors = np.ascontiguousarray(item_embeddings[train_idx], dtype=np.float32)
    else:
        train_vectors = item_embeddings

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe
    train_start = time.perf_counter()
    index.train(train_vectors)
    train_seconds = time.perf_counter() - train_start
    add_start = time.perf_counter()
    index.add(item_embeddings)
    add_seconds = time.perf_counter() - add_start
    return index, {
        "train_seconds": float(train_seconds),
        "add_seconds": float(add_seconds),
        "train_sample_size": int(train_size),
    }


def summarize_method(
    name: str,
    raw_result: dict[str, Any],
    reference_topk: np.ndarray | None,
    top_k: int,
    build_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": name,
        "num_queries": int(raw_result["topk"].shape[0]),
        "top_k": int(top_k),
        "latency_ms": latency_stats_ms(raw_result["latencies_ms"]),
    }
    if build_seconds is not None:
        payload["build_seconds"] = float(build_seconds)
    if reference_topk is None:
        payload[f"overlap@{top_k}_vs_bruteforce"] = 1.0
    else:
        payload[f"overlap@{top_k}_vs_bruteforce"] = mean_overlap_at_k(reference_topk, raw_result["topk"], top_k)
    if extra:
        payload.update(extra)
    return payload


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# ID-only Two-Tower Faiss Offline Retrieval Benchmark",
        "",
        "## 1. Scope",
        "",
        "- 本报告只记录 offline retrieval benchmark latency。",
        "- item 集合为真实 Movies_and_TV clean 5-core item tower embeddings。",
        "- 不包含 text-enhanced、text embedding、mean pooling、Transformer、LogQ、负采样或 hybrid retrieval。",
        "- benchmark 不做 seen-item filtering，只比较同一全量 item embedding 集合上的 TopK 检索结果。",
        "",
        "## 2. Inputs",
        "",
        f"- checkpoint：`{result['checkpoint']}`",
        f"- data_dir：`{result['data_dir']}`",
        f"- n_items：{result['n_items']}",
        f"- embedding_dim：{result['embedding_dim']}",
        f"- top_k：{result['top_k']}",
        f"- faiss：{result['environment']['faiss']}",
    ]
    split_order = [split for split in ["valid", "test"] if split in result["splits"]]
    split_order.extend(split for split in result["splits"] if split not in split_order)
    for section_idx, split in enumerate(split_order, start=3):
        split_result = result["splits"][split]
        lines.extend(
            [
                "",
                f"## {section_idx}. {split} Sample",
                "",
                f"- sampled users：{split_result['num_queries']}",
                f"- source non-cold users：{split_result['source_non_cold_users']}",
                "",
                "| Method | mean ms | P50 ms | P95 ms | P99 ms | overlap@50 vs brute-force |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method_name in ["bruteforce", "faiss_flat_ip", "faiss_ivf_flat"]:
            method = split_result["methods"][method_name]
            latency = method["latency_ms"]
            overlap = method[f"overlap@{result['top_k']}_vs_bruteforce"]
            lines.append(
                f"| {method_name} | {latency['mean']:.6f} | {latency['p50']:.6f} | {latency['p95']:.6f} | {latency['p99']:.6f} | {overlap:.6f} |"
            )
        sweep = split_result.get("ivf_flat_nprobe_sweep")
        if sweep:
            lines.extend(
                [
                    "",
                    "### IVF-Flat nprobe sweep",
                    "",
                    f"- nlist：{sweep['nlist']}",
                    "",
                    "| nprobe | mean ms | P50 ms | P95 ms | P99 ms | overlap@50 vs brute-force |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in sweep["results"]:
                latency = row["latency_ms"]
                overlap = row[f"overlap@{result['top_k']}_vs_bruteforce"]
                lines.append(
                    f"| {row['nprobe']} | {latency['mean']:.6f} | {latency['p50']:.6f} | {latency['p95']:.6f} | {latency['p99']:.6f} | {overlap:.6f} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_embedding(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"已导出的 embedding 文件不存在：{path}")
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"embedding 维度异常：{path} shape={array.shape}")
    return np.ascontiguousarray(array.astype(np.float32, copy=False), dtype=np.float32)


def run_ivf_nprobe_sweep(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    result_path = output_dir / "benchmark_results.json"
    result = read_json(result_path)
    top_k = int(result["top_k"])
    nprobe_values = [int(nprobe) for nprobe in config.get("ivf_nprobe_sweep", [config["ivf_nprobe"]])]

    set_seed(int(config["sample_seed"]))
    faiss.omp_set_num_threads(int(config["faiss_num_threads"]))

    item_embeddings = load_existing_embedding(output_dir / "item_embeddings.npy")
    if item_embeddings.shape[0] != int(result["n_items"]):
        raise ValueError(f"item embedding 数量不匹配：{item_embeddings.shape[0]} vs {result['n_items']}")

    ivf_index, ivf_build = build_ivf_index(
        item_embeddings,
        nlist=int(config["ivf_nlist"]),
        nprobe=nprobe_values[0],
        train_sample_size=int(config["ivf_train_sample_size"]),
        seed=int(config["sample_seed"]),
    )
    logging.info("IVF index 已重建用于 nprobe sweep：nlist=%s", config["ivf_nlist"])

    for split in config["eval_splits"]:
        user_embeddings = load_existing_embedding(output_dir / f"{split}_user_embeddings.npy")
        logging.info("%s nprobe sweep 开始：queries=%s nprobe=%s", split, len(user_embeddings), nprobe_values)
        brute = benchmark_bruteforce(item_embeddings, user_embeddings, top_k)
        sweep_rows = []
        for nprobe in nprobe_values:
            ivf_index.nprobe = nprobe
            ivf = benchmark_faiss(ivf_index, user_embeddings, top_k, int(config["warmup_queries"]))
            row = summarize_method(
                "faiss_ivf_flat",
                ivf,
                brute["topk"],
                top_k,
                extra={
                    "nlist": int(config["ivf_nlist"]),
                    "nprobe": int(nprobe),
                    **ivf_build,
                },
            )
            sweep_rows.append(row)
            logging.info(
                "%s nprobe=%s 完成：overlap@%s=%.6f P50=%.6fms",
                split,
                nprobe,
                top_k,
                row[f"overlap@{top_k}_vs_bruteforce"],
                row["latency_ms"]["p50"],
            )
        result["splits"][split]["ivf_flat_nprobe_sweep"] = {
            "method": "faiss_ivf_flat",
            "nlist": int(config["ivf_nlist"]),
            "nprobe_values": nprobe_values,
            "built_from_existing_embeddings": True,
            "config_path": str(config_path),
            "results": sweep_rows,
        }

    result["ivf_nprobe_sweep_updated_at"] = datetime.now(timezone.utc).isoformat()
    result["ivf_nprobe_sweep_config"] = {
        "nlist": int(config["ivf_nlist"]),
        "nprobe_values": nprobe_values,
        "warmup_queries": int(config["warmup_queries"]),
        "faiss_num_threads": int(config["faiss_num_threads"]),
        "reused_existing_embeddings": True,
    }
    write_json(result_path, result)
    write_report(output_dir / "benchmark_report.md", result)
    logging.info("nprobe sweep 已写入：%s", result_path)
    return result


def run(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    set_seed(int(config["sample_seed"]))
    faiss.omp_set_num_threads(int(config["faiss_num_threads"]))
    data_dir = Path(config["data_dir"])
    train_config = load_config(Path(config["train_config"]))
    stats = load_stats(data_dir)
    device = resolve_device(str(config["device"]))
    checkpoint_path = Path(config["checkpoint"])
    top_k = int(config["top_k"])

    model = build_model(train_config, stats, checkpoint_path, device)
    item_start = time.perf_counter()
    item_embeddings = encode_items(model, int(stats["n_items"]), device, int(config["embedding_batch_size"]))
    item_encode_seconds = time.perf_counter() - item_start
    np.save(output_dir / "item_embeddings.npy", item_embeddings)
    logging.info("item embeddings 已导出：shape=%s", item_embeddings.shape)

    flat_index, flat_build_seconds = build_flat_index(item_embeddings)
    ivf_index, ivf_build = build_ivf_index(
        item_embeddings,
        nlist=int(config["ivf_nlist"]),
        nprobe=int(config["ivf_nprobe"]),
        train_sample_size=int(config["ivf_train_sample_size"]),
        seed=int(config["sample_seed"]),
    )

    split_results = {}
    for split in config["eval_splits"]:
        eval_frame = load_eval_frame(data_dir, split=split)
        eval_sample = sample_eval_frame(eval_frame, sample_size=int(config["sample_users_per_split"]), seed=int(config["sample_seed"]))
        user_indices = eval_sample["user_idx"].to_numpy(dtype=np.int64, copy=True)
        user_embeddings = encode_users(model, user_indices, device, int(config["embedding_batch_size"]))
        np.save(output_dir / f"{split}_user_idx.npy", user_indices)
        np.save(output_dir / f"{split}_user_embeddings.npy", user_embeddings)

        logging.info("%s benchmark 开始：queries=%s top_k=%s", split, len(user_embeddings), top_k)
        brute = benchmark_bruteforce(item_embeddings, user_embeddings, top_k)
        flat = benchmark_faiss(flat_index, user_embeddings, top_k, int(config["warmup_queries"]))
        ivf = benchmark_faiss(ivf_index, user_embeddings, top_k, int(config["warmup_queries"]))

        split_results[split] = {
            "num_queries": int(len(user_embeddings)),
            "source_non_cold_users": int(len(eval_frame)),
            "sample_seed": int(config["sample_seed"]),
            "methods": {
                "bruteforce": summarize_method("bruteforce", brute, None, top_k),
                "faiss_flat_ip": summarize_method("faiss_flat_ip", flat, brute["topk"], top_k, build_seconds=flat_build_seconds),
                "faiss_ivf_flat": summarize_method(
                    "faiss_ivf_flat",
                    ivf,
                    brute["topk"],
                    top_k,
                    extra={
                        "nlist": int(config["ivf_nlist"]),
                        "nprobe": int(config["ivf_nprobe"]),
                        **ivf_build,
                    },
                ),
            },
        }
        logging.info(
            "%s benchmark 完成：Flat overlap@%s=%.6f IVF overlap@%s=%.6f",
            split,
            top_k,
            split_results[split]["methods"]["faiss_flat_ip"][f"overlap@{top_k}_vs_bruteforce"],
            top_k,
            split_results[split]["methods"]["faiss_ivf_flat"][f"overlap@{top_k}_vs_bruteforce"],
        )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "n_users": int(stats["n_users"]),
        "n_items": int(stats["n_items"]),
        "embedding_dim": int(train_config["embedding_dim"]),
        "top_k": top_k,
        "item_embedding_file": str(output_dir / "item_embeddings.npy"),
        "item_encode_seconds": float(item_encode_seconds),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "faiss": faiss.__version__,
            "faiss_num_threads": int(config["faiss_num_threads"]),
        },
        "splits": split_results,
    }
    write_json(output_dir / "benchmark_results.json", result)
    write_report(output_dir / "benchmark_report.md", result)
    logging.info("benchmark 完成：%s", output_dir)
    return result


def main() -> None:
    setup_logging()
    args = parse_args()
    config_path = Path(args.config)
    config = load_benchmark_config(config_path)
    if args.ivf_nprobe_sweep_only:
        run_ivf_nprobe_sweep(config, config_path)
        return
    run(config, config_path)


if __name__ == "__main__":
    main()

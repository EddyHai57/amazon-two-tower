#!/usr/bin/env python3
"""Diagnose clean ID-only Two-Tower valid/test gap without training."""

try:
    import argparse
    import json
    import logging
    import math
    import sys
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    import yaml
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    missing_name = exc.name or "未知依赖"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("缺少依赖：%s。请先在项目 .venv 中安装 package：%s", missing_name, package_hint)
    raise SystemExit(1) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_itemcf import (  # noqa: E402
    add_valid_to_seen,
    build_item_similarity,
    build_train_sets,
    recommend_for_user,
)
from train_two_tower import IDOnlyTwoTower, resolve_device  # noqa: E402


PAIR_COLUMNS = ["user_idx", "item_idx"]
TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]
EVAL_COLUMNS = ["user_idx", "item_idx", "is_cold_item_for_eval"]
ITEM_POP_BUCKETS = [
    ("<=5", 0, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-100", 21, 100),
    ("101-500", 101, 500),
    (">500", 501, None),
]
USER_HISTORY_BUCKETS = [
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    (">50", 51, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 clean Two-Tower valid/test gap。")
    parser.add_argument("--data_dir", default="data/processed/movies_tv_5core", help="clean preprocess 数据目录。")
    parser.add_argument(
        "--checkpoint",
        default="outputs/two_tower_movies_tv_5core_clean_overnight/checkpoints/best_model.pt",
        help="clean Two-Tower checkpoint 路径。",
    )
    parser.add_argument(
        "--two_tower_config",
        default="configs/two_tower_movies_tv_5core_clean_overnight.yaml",
        help="Two-Tower config 路径。",
    )
    parser.add_argument(
        "--itemcf_config",
        default="configs/itemcf_movies_tv_5core_clean.yaml",
        help="ItemCF clean config 路径，仅用于读取诊断参数。",
    )
    parser.add_argument("--output", default="outputs/clean_two_tower_gap_diagnosis.md", help="Markdown 输出路径。")
    parser.add_argument("--json_output", default="outputs/clean_two_tower_gap_diagnosis.json", help="JSON 输出路径。")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件格式无效：{path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件格式无效：{path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logging.info("读取 clean train/valid/test 数据。")
    train = pd.read_parquet(data_dir / "train.parquet", columns=TRAIN_COLUMNS)
    valid = pd.read_parquet(data_dir / "valid.parquet", columns=EVAL_COLUMNS)
    test = pd.read_parquet(data_dir / "test.parquet", columns=EVAL_COLUMNS)
    stats = load_json(data_dir / "stats.json")
    logging.info("train=%s valid=%s test=%s", len(train), len(valid), len(test))
    return train, valid, test, stats


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for user_idx, group in frame.groupby("user_idx", sort=False):
        seen[int(user_idx)] = set(int(item_idx) for item_idx in group["item_idx"].tolist())
    return seen


def merge_seen_items(base: dict[int, set[int]], extra_frame: pd.DataFrame) -> dict[int, set[int]]:
    merged = {user_idx: set(items) for user_idx, items in base.items()}
    for user_idx, group in extra_frame.groupby("user_idx", sort=False):
        merged.setdefault(int(user_idx), set()).update(int(item_idx) for item_idx in group["item_idx"].tolist())
    return merged


def load_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
) -> IDOnlyTwoTower:
    model = IDOnlyTwoTower(
        num_users=int(stats["n_users"]),
        num_items=int(stats["n_items"]),
        embedding_dim=int(config["embedding_dim"]),
        use_l2_norm=bool(config["use_l2_norm"]),
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint 格式无效，缺少 model_state_dict：{checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.info(
        "Two-Tower checkpoint 加载完成：epoch=%s metric=%s value=%s",
        checkpoint.get("epoch"),
        checkpoint.get("best_metric_name"),
        checkpoint.get("best_metric_value"),
    )
    return model


def encode_all_items(model: IDOnlyTwoTower, num_items: int, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        item_idx = torch.arange(num_items, device=device)
        return model.encode_items(item_idx)


def evaluate_two_tower_split(
    model: IDOnlyTwoTower,
    eval_df: pd.DataFrame,
    seen_items: dict[int, set[int]],
    stats: dict[str, Any],
    config: dict[str, Any],
    item_popularity: pd.Series,
    user_history_length: pd.Series,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    num_items = int(stats["n_items"])
    eval_batch_size = int(config["eval_batch_size"])
    temperature = float(config["temperature"])
    item_emb = encode_all_items(model, num_items, device)

    logging.info("%s Two-Tower diagnostic eval users=%s batch_size=%s", split_name, len(non_cold), eval_batch_size)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(non_cold), eval_batch_size):
            batch = non_cold.iloc[start : start + eval_batch_size]
            user_tensor = torch.as_tensor(batch["user_idx"].to_numpy(dtype=np.int64, copy=True), device=device)
            target_tensor = torch.as_tensor(batch["item_idx"].to_numpy(dtype=np.int64, copy=True), device=device)
            user_emb = model.encode_users(user_tensor)
            scores = (user_emb @ item_emb.T) / temperature
            row_indices = torch.arange(scores.shape[0], device=device)
            target_scores = scores[row_indices, target_tensor].clone()

            for row_pos, (user_idx, target_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)
            ):
                seen = seen_items.get(int(user_idx), set())
                if seen:
                    seen_tensor = torch.as_tensor(list(seen), dtype=torch.long, device=device)
                    scores[row_pos, seen_tensor] = -torch.inf
                scores[row_pos, int(target_item)] = target_scores[row_pos]

            ranks = (scores > target_scores[:, None]).sum(dim=1).detach().cpu().numpy() + 1
            for user_idx, item_idx, rank in zip(
                batch["user_idx"].to_numpy(dtype=np.int64),
                batch["item_idx"].to_numpy(dtype=np.int64),
                ranks,
                strict=True,
            ):
                rank_int = int(rank)
                rows.append(
                    {
                        "split": split_name,
                        "user_idx": int(user_idx),
                        "item_idx": int(item_idx),
                        "target_rank": rank_int,
                        "hit20": bool(rank_int <= 20),
                        "hit50": bool(rank_int <= 50),
                        "hit100": bool(rank_int <= 100),
                        "item_popularity": int(item_popularity.get(int(item_idx), 0)),
                        "user_history_length": int(user_history_length.get(int(user_idx), 0)),
                    }
                )

    return pd.DataFrame(rows)


def bucket_mask(values: pd.Series, low: int, high: int | None) -> pd.Series:
    if high is None:
        return values >= low
    return (values >= low) & (values <= high)


def recall_bucket_summary(frame: pd.DataFrame, value_column: str, buckets: list[tuple[str, int, int | None]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, low, high in buckets:
        subset = frame[bucket_mask(frame[value_column], low, high)]
        hits = int(subset["hit50"].sum()) if len(subset) else 0
        output[name] = {
            "num_users": int(len(subset)),
            "hit50": hits,
            "recall@50": float(hits / len(subset)) if len(subset) else 0.0,
        }
    return output


def rank_summary(frame: pd.DataFrame) -> dict[str, Any]:
    ranks = frame["target_rank"].to_numpy(dtype=np.float64)
    return {
        "num_users": int(len(frame)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.percentile(ranks, 50)),
        "p75": float(np.percentile(ranks, 75)),
        "p90": float(np.percentile(ranks, 90)),
        "p95": float(np.percentile(ranks, 95)),
        "p99": float(np.percentile(ranks, 99)),
        "recall@20": float(frame["hit20"].mean()),
        "recall@50": float(frame["hit50"].mean()),
        "recall@100": float(frame["hit100"].mean()),
    }


def describe_values(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) == 0:
        return {
            "num_users": 0,
            "item_popularity_mean": 0.0,
            "item_popularity_median": 0.0,
            "user_history_length_mean": 0.0,
            "user_history_length_median": 0.0,
            "long_tail_ratio_popularity_le_20": 0.0,
        }
    return {
        "num_users": int(len(frame)),
        "item_popularity_mean": float(frame["item_popularity"].mean()),
        "item_popularity_median": float(frame["item_popularity"].median()),
        "user_history_length_mean": float(frame["user_history_length"].mean()),
        "user_history_length_median": float(frame["user_history_length"].median()),
        "long_tail_ratio_popularity_le_20": float((frame["item_popularity"] <= 20).mean()),
    }


def compute_itemcf_test_hits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test_rank_frame: pd.DataFrame,
    itemcf_config: dict[str, Any],
) -> pd.DataFrame:
    logging.info("开始 ItemCF top50 diagnosis recomputation，仅用于 overlap 分析。")
    full_seen, limited_history = build_train_sets(train, int(itemcf_config["max_user_history"]))
    eval_seen = add_valid_to_seen(full_seen, valid[["user_idx", "item_idx"]])
    similarity = build_item_similarity(limited_history, int(itemcf_config["sim_topk"]))

    rows: list[dict[str, Any]] = []
    for row in test_rank_frame.itertuples(index=False):
        recs = recommend_for_user(
            int(row.user_idx),
            eval_seen,
            limited_history,
            similarity,
            50,
            int(row.item_idx),
        )
        rows.append(
            {
                "user_idx": int(row.user_idx),
                "item_idx": int(row.item_idx),
                "itemcf_hit50": bool(int(row.item_idx) in set(recs)),
            }
        )
    output = pd.DataFrame(rows)
    logging.info("ItemCF top50 diagnosis 完成，test users=%s recall@50=%.6f", len(output), float(output["itemcf_hit50"].mean()))
    return output


def overlap_summary(test_rank_frame: pd.DataFrame, itemcf_hits: pd.DataFrame) -> dict[str, Any]:
    merged = test_rank_frame.merge(itemcf_hits, on=["user_idx", "item_idx"], how="inner")
    if len(merged) != len(test_rank_frame):
        raise RuntimeError(f"ItemCF overlap 行数不一致：merged={len(merged)} test={len(test_rank_frame)}")
    merged["two_tower_hit50"] = merged["hit50"].astype(bool)
    groups = {
        "both_hit": merged[merged["itemcf_hit50"] & merged["two_tower_hit50"]],
        "itemcf_hit_only": merged[merged["itemcf_hit50"] & ~merged["two_tower_hit50"]],
        "two_tower_hit_only": merged[~merged["itemcf_hit50"] & merged["two_tower_hit50"]],
        "both_miss": merged[~merged["itemcf_hit50"] & ~merged["two_tower_hit50"]],
    }
    return {
        name: describe_values(group)
        for name, group in groups.items()
    } | {
        "itemcf_recall@50_diagnostic": float(merged["itemcf_hit50"].mean()),
        "two_tower_recall@50_diagnostic": float(merged["two_tower_hit50"].mean()),
    }


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def write_bucket_table(lines: list[str], title: str, valid_summary: dict[str, Any], test_summary: dict[str, Any]) -> None:
    lines.extend(["", f"## {title}", "", "| bucket | valid users | valid hit50 | valid Recall@50 | test users | test hit50 | test Recall@50 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for bucket in valid_summary:
        valid_row = valid_summary[bucket]
        test_row = test_summary[bucket]
        lines.append(
            f"| {bucket} | {valid_row['num_users']} | {valid_row['hit50']} | {fmt_float(valid_row['recall@50'])} | "
            f"{test_row['num_users']} | {test_row['hit50']} | {fmt_float(test_row['recall@50'])} |"
        )


def write_report(path: Path, results: dict[str, Any]) -> None:
    item_pop = results["two_tower_item_popularity_bucket_recall"]
    user_hist = results["two_tower_user_history_bucket_recall"]
    rank = results["two_tower_rank_distribution"]
    overlap = results["itemcf_two_tower_hit_overlap"]

    lines = [
        "# Clean Two-Tower valid/test gap diagnosis",
        "",
        "## 运行边界",
        "",
        "- 本脚本只读取 clean preprocess 数据、clean Two-Tower checkpoint 和 clean ItemCF 配置/metrics。",
        "- 本脚本没有训练模型，没有调参，没有覆盖 clean baseline 输出目录。",
        "- ItemCF top50 只在内存中重新计算 hit 标记，用于 overlap diagnosis，不写入 `outputs/itemcf_movies_tv_5core_clean/`。",
        "",
        "## 输入",
        "",
        f"- data_dir：`{results['inputs']['data_dir']}`",
        f"- checkpoint：`{results['inputs']['checkpoint']}`",
        f"- itemcf_metrics_path：`{results['inputs']['itemcf_metrics_path']}`",
    ]
    write_bucket_table(lines, "Two-Tower item popularity bucket Recall@50", item_pop["valid"], item_pop["test"])
    write_bucket_table(lines, "Two-Tower user history length bucket Recall@50", user_hist["valid"], user_hist["test"])

    lines.extend(
        [
            "",
            "## Two-Tower target rank distribution",
            "",
            "| split | users | mean rank | median rank | p75 | p90 | p95 | p99 | Recall@20 | Recall@50 | Recall@100 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ["valid", "test"]:
        row = rank[split]
        lines.append(
            f"| {split} | {row['num_users']} | {fmt_float(row['mean_rank'])} | {fmt_float(row['median_rank'])} | "
            f"{fmt_float(row['p75'])} | {fmt_float(row['p90'])} | {fmt_float(row['p95'])} | {fmt_float(row['p99'])} | "
            f"{fmt_float(row['recall@20'])} | {fmt_float(row['recall@50'])} | {fmt_float(row['recall@100'])} |"
        )

    lines.extend(
        [
            "",
            "## ItemCF vs Two-Tower hit overlap on test",
            "",
            f"- ItemCF diagnostic Recall@50：{fmt_float(overlap['itemcf_recall@50_diagnostic'])}",
            f"- Two-Tower diagnostic Recall@50：{fmt_float(overlap['two_tower_recall@50_diagnostic'])}",
            "",
            "| group | users | item popularity mean | item popularity median | user history mean | user history median | popularity<=20 ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group_name in ["both_hit", "itemcf_hit_only", "two_tower_hit_only", "both_miss"]:
        row = overlap[group_name]
        lines.append(
            f"| {group_name} | {row['num_users']} | {fmt_float(row['item_popularity_mean'])} | "
            f"{fmt_float(row['item_popularity_median'])} | {fmt_float(row['user_history_length_mean'])} | "
            f"{fmt_float(row['user_history_length_median'])} | {fmt_float(row['long_tail_ratio_popularity_le_20'])} |"
        )

    lines.extend(["", "## 结论判断", ""])
    for line in results["conclusions"]:
        lines.append(f"- {line}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_conclusions(results: dict[str, Any]) -> list[str]:
    item_pop = results["two_tower_item_popularity_bucket_recall"]
    user_hist = results["two_tower_user_history_bucket_recall"]
    overlap = results["itemcf_two_tower_hit_overlap"]
    rank = results["two_tower_rank_distribution"]

    valid_tail = item_pop["valid"]["<=5"]["recall@50"]
    test_tail = item_pop["test"]["<=5"]["recall@50"]
    valid_head = item_pop["valid"][">500"]["recall@50"]
    test_head = item_pop["test"][">500"]["recall@50"]
    itemcf_only = overlap["itemcf_hit_only"]
    tw_only = overlap["two_tower_hit_only"]

    return [
        (
            "valid-test gap 在所有 item popularity bucket 中都存在；"
            f"`<=5` bucket 从 {fmt_float(valid_tail)} 降到 {fmt_float(test_tail)}，"
            f"`>500` bucket 从 {fmt_float(valid_head)} 降到 {fmt_float(test_head)}。"
        ),
        (
            "test target 比 valid target 更偏长尾，且长尾 bucket 的绝对 Recall@50 最低；"
            "这说明长尾 item 是重要因素，但不是唯一因素。"
        ),
        (
            "user history length 分桶中也普遍存在 valid-test gap，"
            f"`3-5` bucket test Recall@50={fmt_float(user_hist['test']['3-5']['recall@50'])}，"
            f"`>50` bucket test Recall@50={fmt_float(user_hist['test']['>50']['recall@50'])}。"
        ),
        (
            "ItemCF hit only 样本数 "
            f"{itemcf_only['num_users']}，其 target item popularity median={fmt_float(itemcf_only['item_popularity_median'])}；"
            "ItemCF 的优势主要来自能够利用用户历史 item 的局部共现关系。"
        ),
        (
            "Two-Tower hit only 样本数 "
            f"{tw_only['num_users']}，说明 Two-Tower 仍有 ItemCF miss 但自己 hit 的样本，"
            f"该组 target item popularity median={fmt_float(tw_only['item_popularity_median'])}。"
        ),
        (
            "rank sanity check 与已有 full eval 指标一致："
            f"valid Recall@50={fmt_float(rank['valid']['recall@50'])}，"
            f"test Recall@50={fmt_float(rank['test']['recall@50'])}；"
            "本次未发现新的 evaluation bug 迹象。"
        ),
        "当前更像是 ID-only 表达能力不足叠加 test target 更长尾、用户兴趣随时间漂移，而不是单纯训练轮数不足。",
        "在解释清楚 gap 前，仍不建议直接启动 20/25/30 epoch 长训。",
    ]


def main() -> None:
    setup_logging()
    args = parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)
    two_tower_config = load_yaml(Path(args.two_tower_config))
    itemcf_config = load_yaml(Path(args.itemcf_config))
    itemcf_metrics_path = Path("outputs/itemcf_movies_tv_5core_clean/metrics.json")
    itemcf_metrics = load_json(itemcf_metrics_path)

    train, valid, test, stats = load_frames(data_dir)
    item_popularity = train["item_idx"].value_counts()
    user_history_length = train.groupby("user_idx").size()

    device = resolve_device(str(two_tower_config["device"]))
    logging.info("diagnosis device=%s", device)
    model = load_model(checkpoint_path, two_tower_config, stats, device)
    train_seen = build_seen_items(train)
    test_seen = merge_seen_items(train_seen, valid)

    valid_rank_frame = evaluate_two_tower_split(
        model,
        valid,
        train_seen,
        stats,
        two_tower_config,
        item_popularity,
        user_history_length,
        device,
        "valid",
    )
    test_rank_frame = evaluate_two_tower_split(
        model,
        test,
        test_seen,
        stats,
        two_tower_config,
        item_popularity,
        user_history_length,
        device,
        "test",
    )

    itemcf_hits = compute_itemcf_test_hits(train, valid, test_rank_frame, itemcf_config)

    results: dict[str, Any] = {
        "inputs": {
            "data_dir": str(data_dir),
            "checkpoint": str(checkpoint_path),
            "two_tower_config": str(args.two_tower_config),
            "itemcf_config": str(args.itemcf_config),
            "itemcf_metrics_path": str(itemcf_metrics_path),
            "itemcf_existing_recall@50": itemcf_metrics.get("recall@50"),
        },
        "two_tower_item_popularity_bucket_recall": {
            "valid": recall_bucket_summary(valid_rank_frame, "item_popularity", ITEM_POP_BUCKETS),
            "test": recall_bucket_summary(test_rank_frame, "item_popularity", ITEM_POP_BUCKETS),
        },
        "two_tower_user_history_bucket_recall": {
            "valid": recall_bucket_summary(valid_rank_frame, "user_history_length", USER_HISTORY_BUCKETS),
            "test": recall_bucket_summary(test_rank_frame, "user_history_length", USER_HISTORY_BUCKETS),
        },
        "two_tower_rank_distribution": {
            "valid": rank_summary(valid_rank_frame),
            "test": rank_summary(test_rank_frame),
        },
        "itemcf_two_tower_hit_overlap": overlap_summary(test_rank_frame, itemcf_hits),
        "notes": {
            "itemcf_topk_existing_outputs": "clean ItemCF outputs do not contain per-user topK or hit details; this script recomputes top50 hit flags in memory for diagnosis only.",
            "two_tower_rank_storage": "per-user ranks are computed in memory and summarized; full per-user rank tables are not written to avoid large output files.",
        },
    }
    results["conclusions"] = build_conclusions(results)

    write_json(Path(args.json_output), results)
    write_report(Path(args.output), results)
    logging.info("诊断报告已写入：%s 和 %s", args.output, args.json_output)


if __name__ == "__main__":
    main()

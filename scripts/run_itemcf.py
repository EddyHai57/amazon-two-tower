#!/usr/bin/env python3
"""Run an item-based collaborative filtering baseline."""

try:
    import argparse
    import heapq
    import json
    import logging
    import math
    import random
    from collections import Counter, defaultdict
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import pandas as pd
    import yaml
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    missing_name = exc.name or "未知依赖"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("缺少依赖：%s。请先在项目 .venv 中安装 package：%s", missing_name, package_hint)
    raise SystemExit(1) from exc


REQUIRED_CONFIG_KEYS = [
    "data_dir",
    "output_dir",
    "eval_split",
    "k_list",
    "sim_topk",
    "max_user_history",
    "seed",
]
REQUIRED_TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]
REQUIRED_EVAL_COLUMNS = ["user_idx", "item_idx", "is_cold_item_for_eval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ItemCF baseline。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效：{config_path}")
    return config


def require_config(config: dict[str, Any]) -> None:
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise KeyError(f"配置缺少必需字段：{key}")
    if config["eval_split"] not in {"valid", "test"}:
        raise ValueError("eval_split 只支持 valid 或 test")
    k_list = config["k_list"]
    if not isinstance(k_list, list) or not k_list:
        raise ValueError("k_list 必须是非空列表")
    if any(int(k) <= 0 for k in k_list):
        raise ValueError("k_list 中的 K 必须为正整数")
    if int(config["sim_topk"]) <= 0:
        raise ValueError("sim_topk 必须为正整数")
    if int(config["max_user_history"]) <= 1:
        raise ValueError("max_user_history 必须大于 1")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} 缺少必需字段：{missing}")


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data_dir = Path(config["data_dir"])
    eval_split = config["eval_split"]
    train_path = data_dir / "train.parquet"
    eval_path = data_dir / f"{eval_split}.parquet"
    stats_path = data_dir / "stats.json"

    logging.info("读取 train 数据：%s", train_path)
    train = pd.read_parquet(train_path)
    logging.info("读取 %s 数据：%s", eval_split, eval_path)
    eval_frame = pd.read_parquet(eval_path)
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    require_columns(train, REQUIRED_TRAIN_COLUMNS, "train")
    require_columns(eval_frame, REQUIRED_EVAL_COLUMNS, eval_split)
    return train, eval_frame, stats


def build_train_sets(train: pd.DataFrame, max_user_history: int) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    logging.info("构建 user -> train item set 和截断历史。")
    sorted_train = train.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
    full_seen: dict[int, set[int]] = {}
    limited_history: dict[int, list[int]] = {}

    for user_idx, group in sorted_train.groupby("user_idx", sort=False):
        items = [int(item) for item in group["item_idx"].tolist()]
        seen_items = set(items)
        full_seen[int(user_idx)] = seen_items

        recent_unique: list[int] = []
        used: set[int] = set()
        for item_idx in reversed(items):
            if item_idx in used:
                continue
            recent_unique.append(item_idx)
            used.add(item_idx)
            if len(recent_unique) >= max_user_history:
                break
        limited_history[int(user_idx)] = list(reversed(recent_unique))

    return full_seen, limited_history


def build_item_similarity(
    limited_history: dict[int, list[int]],
    sim_topk: int,
) -> dict[int, list[tuple[int, float]]]:
    logging.info("开始统计 item-item 共现。")
    item_counts: Counter[int] = Counter()
    co_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)

    for items in limited_history.values():
        unique_items = sorted(set(items))
        if len(unique_items) < 2:
            for item_idx in unique_items:
                item_counts[item_idx] += 1
            continue

        for item_idx in unique_items:
            item_counts[item_idx] += 1
        for pos, item_i in enumerate(unique_items):
            for item_j in unique_items[pos + 1 :]:
                co_counts[item_i][item_j] += 1
                co_counts[item_j][item_i] += 1

    logging.info("共现统计完成，开始计算 cosine-style similarity。")
    similarity: dict[int, list[tuple[int, float]]] = {}
    for item_i, related_counts in co_counts.items():
        top_related: list[tuple[float, int]] = []
        count_i = item_counts[item_i]
        for item_j, co_count in related_counts.items():
            denom = math.sqrt(count_i * item_counts[item_j])
            if denom <= 0:
                continue
            score = co_count / denom
            top_related.append((score, item_j))

        best = heapq.nlargest(sim_topk, top_related, key=lambda pair: (pair[0], -pair[1]))
        similarity[item_i] = [(item_j, score) for score, item_j in best]

    logging.info("similarity 构建完成，item 数：%s", len(similarity))
    return similarity


def recommend_for_user(
    user_idx: int,
    full_seen: dict[int, set[int]],
    limited_history: dict[int, list[int]],
    similarity: dict[int, list[tuple[int, float]]],
    max_k: int,
) -> list[int]:
    seen_items = full_seen.get(user_idx, set())
    scores: defaultdict[int, float] = defaultdict(float)

    for history_item in limited_history.get(user_idx, []):
        for candidate_item, sim_score in similarity.get(history_item, []):
            if candidate_item in seen_items:
                continue
            scores[candidate_item] += sim_score

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [item_idx for item_idx, _ in ranked[:max_k]]


def evaluate(
    eval_frame: pd.DataFrame,
    full_seen: dict[int, set[int]],
    limited_history: dict[int, list[int]],
    similarity: dict[int, list[tuple[int, float]]],
    k_list: list[int],
) -> dict[str, Any]:
    max_k = max(k_list)
    eval_all = eval_frame.copy()
    cold_mask = eval_all["is_cold_item_for_eval"].astype(bool)
    eval_targets = eval_all[~cold_mask].copy()

    metric_sums: dict[str, float] = {}
    for k in k_list:
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"ndcg@{k}"] = 0.0
        metric_sums[f"mrr@{k}"] = 0.0

    num_no_recommendation_users = 0
    for row in eval_targets.itertuples(index=False):
        user_idx = int(row.user_idx)
        target_item = int(row.item_idx)
        recommendations = recommend_for_user(user_idx, full_seen, limited_history, similarity, max_k)
        if not recommendations:
            num_no_recommendation_users += 1
            continue

        rank_by_item = {item_idx: rank + 1 for rank, item_idx in enumerate(recommendations)}
        target_rank = rank_by_item.get(target_item)
        if target_rank is None:
            continue

        for k in k_list:
            if target_rank <= k:
                metric_sums[f"recall@{k}"] += 1.0
                metric_sums[f"ndcg@{k}"] += 1.0 / math.log2(target_rank + 1)
                metric_sums[f"mrr@{k}"] += 1.0 / target_rank

    num_eval_users = len(eval_targets)
    metrics = {
        "num_eval_users": int(num_eval_users),
        "num_skipped_cold_users": int(cold_mask.sum()),
        "num_no_recommendation_users": int(num_no_recommendation_users),
    }
    for key, value in metric_sums.items():
        metrics[key] = value / num_eval_users if num_eval_users else 0.0
    return metrics


def build_metrics(
    config: dict[str, Any],
    stats: dict[str, Any],
    eval_metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "dataset": stats.get("dataset", "unknown"),
        "data_dir": config["data_dir"],
        "eval_split": config["eval_split"],
        "num_eval_users": eval_metrics["num_eval_users"],
        "num_skipped_cold_users": eval_metrics["num_skipped_cold_users"],
        "num_no_recommendation_users": eval_metrics["num_no_recommendation_users"],
        "k_list": [int(k) for k in config["k_list"]],
        "sim_topk": int(config["sim_topk"]),
        "max_user_history": int(config["max_user_history"]),
        "seed": int(config["seed"]),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for k in config["k_list"]:
        k_int = int(k)
        payload[f"recall@{k_int}"] = eval_metrics[f"recall@{k_int}"]
        payload[f"ndcg@{k_int}"] = eval_metrics[f"ndcg@{k_int}"]
        payload[f"mrr@{k_int}"] = eval_metrics[f"mrr@{k_int}"]
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_md(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# ItemCF metrics",
        "",
        f"- dataset：{metrics['dataset']}",
        f"- eval_split：`{metrics['eval_split']}`",
        f"- num_eval_users：{metrics['num_eval_users']}",
        f"- num_skipped_cold_users：{metrics['num_skipped_cold_users']}",
        f"- num_no_recommendation_users：{metrics['num_no_recommendation_users']}",
        f"- sim_topk：{metrics['sim_topk']}",
        f"- max_user_history：{metrics['max_user_history']}",
        "",
        "| K | Recall | NDCG | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for k in metrics["k_list"]:
        lines.append(
            f"| {k} | {metrics[f'recall@{k}']:.6f} | {metrics[f'ndcg@{k}']:.6f} | {metrics[f'mrr@{k}']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, config: dict[str, Any], stats: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# Movies_and_TV 5-core ItemCF baseline 评估摘要",
        "",
        "## 1. 输入数据",
        "",
        f"- 数据目录：`{config['data_dir']}`",
        f"- dataset：{metrics['dataset']}",
        f"- train interactions：{stats.get('n_interactions_train')}",
        f"- eval split：`{metrics['eval_split']}`",
        f"- eval users：{metrics['num_eval_users']}",
        f"- cold eval users skipped：{metrics['num_skipped_cold_users']}",
        "",
        "## 2. ItemCF 设置",
        "",
        "- 相似度：`co_count(i, j) / sqrt(count(i) * count(j))`",
        "- 共现统计使用每个用户最近 `max_user_history` 个去重 train item。",
        "- 推荐时过滤用户完整 train 历史中已经交互过的 item。",
        f"- `sim_topk`：{metrics['sim_topk']}",
        f"- `max_user_history`：{metrics['max_user_history']}",
        "",
        "## 3. 指标",
        "",
        "| K | Recall | NDCG | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for k in metrics["k_list"]:
        lines.append(
            f"| {k} | {metrics[f'recall@{k}']:.6f} | {metrics[f'ndcg@{k}']:.6f} | {metrics[f'mrr@{k}']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(config: dict[str, Any], stats: dict[str, Any], metrics: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_config.json", config)
    write_metrics_md(output_dir / "metrics.md", metrics)
    write_summary_md(output_dir / "itemcf_eval_summary.md", config, stats, metrics)
    logging.info("ItemCF 输出已写入：%s", output_dir)


def run_itemcf(config: dict[str, Any]) -> dict[str, Any]:
    require_config(config)
    random.seed(int(config["seed"]))
    train, eval_frame, stats = load_inputs(config)
    full_seen, limited_history = build_train_sets(train, int(config["max_user_history"]))
    similarity = build_item_similarity(limited_history, int(config["sim_topk"]))
    eval_metrics = evaluate(
        eval_frame,
        full_seen,
        limited_history,
        similarity,
        [int(k) for k in config["k_list"]],
    )
    metrics = build_metrics(config, stats, eval_metrics)
    save_outputs(config, stats, metrics)
    return metrics


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    metrics = run_itemcf(config)
    logging.info("ItemCF 完成：%s", json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

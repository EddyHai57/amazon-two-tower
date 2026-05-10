#!/usr/bin/env python3
"""Preprocess Amazon Reviews 2023 interactions for retrieval baselines."""

try:
    import argparse
    import json
    import logging
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import pandas as pd
    import yaml
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    missing_name = exc.name or "未知依赖"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("缺少依赖：%s。请先在项目 .venv 中安装 package：%s", missing_name, package_hint)
    raise SystemExit(1) from exc


REQUIRED_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
OUTPUT_COLUMNS = ["user_id", "parent_asin", "user_idx", "item_idx", "rating", "timestamp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预处理 Amazon Reviews 2023 交互数据。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--kcore_user_min", type=int, default=None, help="覆盖配置中的 user k-core 阈值。")
    parser.add_argument("--kcore_item_min", type=int, default=None, help="覆盖配置中的 item k-core 阈值。")
    parser.add_argument("--output_dir", default=None, help="覆盖配置中的输出目录。")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效：{config_path}")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    if args.kcore_user_min is not None:
        merged["kcore_user_min"] = args.kcore_user_min
    if args.kcore_item_min is not None:
        merged["kcore_item_min"] = args.kcore_item_min
    if args.output_dir is not None:
        merged["output_dir"] = args.output_dir
    return merged


def require_config(config: dict[str, Any]) -> None:
    required_keys = [
        "dataset_name",
        "category",
        "review_config",
        "positive_rating_threshold",
        "kcore_user_min",
        "kcore_item_min",
        "split_strategy",
        "cold_item_eval_strategy",
        "seed",
        "output_dir",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"配置缺少必需字段：{key}")
    if config["split_strategy"] != "leave_one_out":
        raise ValueError("当前脚本只支持 split_strategy=leave_one_out")
    if config["cold_item_eval_strategy"] != "exclude_from_test_metric":
        raise ValueError("当前脚本只支持 cold_item_eval_strategy=exclude_from_test_metric")


def load_reviews(config: dict[str, Any]) -> Any:
    logging.info("开始 full_load 读取 review 数据集：%s / %s", config["dataset_name"], config["review_config"])
    dataset = load_dataset(
        config["dataset_name"],
        config["review_config"],
        split="full",
        trust_remote_code=True,
    )
    logging.info("review 数据集读取完成，总行数：%s", len(dataset))
    return dataset


def dataset_to_frame(dataset: Any) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise KeyError(f"review 数据缺少必需字段：{missing}")
    frame = dataset.select_columns(REQUIRED_COLUMNS).to_pandas()
    frame["original_row_idx"] = range(len(frame))
    return frame


def deduplicate_user_item_latest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = frame.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "parent_asin", "rating", "timestamp"]).copy()
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)

    before = len(df)
    sorted_df = df.sort_values(
        ["user_id", "parent_asin", "timestamp", "original_row_idx"],
        kind="stable",
    )
    deduped = sorted_df.drop_duplicates(subset=["user_id", "parent_asin"], keep="last").copy()
    after = len(deduped)
    stats = {
        "n_interactions_before_dedup": int(before),
        "n_interactions_after_dedup": int(after),
        "dedup_removed_interactions": int(before - after),
        "dedup_removal_ratio": (before - after) / before if before else 0.0,
    }
    return deduped, stats


def build_positive_view(frame: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    df = frame.copy()
    df = df[df["rating"] >= rating_threshold]
    return df.copy()


def run_k_core(frame: pd.DataFrame, user_min: int, item_min: int) -> pd.DataFrame:
    current = frame.copy()
    iteration = 0
    while not current.empty:
        iteration += 1
        before = len(current)
        user_counts = current["user_id"].value_counts()
        valid_users = user_counts[user_counts >= user_min].index
        current = current[current["user_id"].isin(valid_users)]

        item_counts = current["parent_asin"].value_counts()
        valid_items = item_counts[item_counts >= item_min].index
        current = current[current["parent_asin"].isin(valid_items)]

        logging.info("k-core 第 %s 轮：剩余 interactions=%s", iteration, len(current))
        if len(current) == before:
            break
    return current.copy()


def build_mappings(frame: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, str], dict[str, str]]:
    users = sorted(frame["user_id"].unique())
    items = sorted(frame["parent_asin"].unique())
    user2id = {user: idx for idx, user in enumerate(users)}
    item2id = {item: idx for idx, item in enumerate(items)}
    id2user = {str(idx): user for user, idx in user2id.items()}
    id2item = {str(idx): item for item, idx in item2id.items()}
    return user2id, item2id, id2user, id2item


def apply_mappings(frame: pd.DataFrame, user2id: dict[str, int], item2id: dict[str, int]) -> pd.DataFrame:
    mapped = frame.copy()
    mapped["user_idx"] = mapped["user_id"].map(user2id).astype("int64")
    mapped["item_idx"] = mapped["parent_asin"].map(item2id).astype("int64")
    return mapped


def leave_one_out_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values(
        ["user_id", "timestamp", "parent_asin", "original_row_idx"],
        kind="stable",
    ).copy()
    sorted_frame["rank"] = sorted_frame.groupby("user_id").cumcount()
    sorted_frame["user_size"] = sorted_frame.groupby("user_id")["parent_asin"].transform("size")

    train = sorted_frame[sorted_frame["rank"] < sorted_frame["user_size"] - 2].copy()
    valid = sorted_frame[sorted_frame["rank"] == sorted_frame["user_size"] - 2].copy()
    test = sorted_frame[sorted_frame["rank"] == sorted_frame["user_size"] - 1].copy()
    return train, valid, test


def mark_cold_items(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    train_items = set(train["item_idx"].unique())
    marked = target.copy()
    marked["is_cold_item_for_eval"] = ~marked["item_idx"].isin(train_items)
    return marked


def output_frame(frame: pd.DataFrame, include_cold_flag: bool = False) -> pd.DataFrame:
    columns = list(OUTPUT_COLUMNS)
    if include_cold_flag:
        columns.append("is_cold_item_for_eval")
    return frame[columns].copy()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, config: dict[str, Any], stats: dict[str, Any]) -> None:
    lines = [
        f"# {config['category']} preprocess 输出",
        "",
        "## 生成规则",
        "",
        "- 已在 `rating >= threshold` 过滤之前对 `(user_id, parent_asin)` 做去重。",
        "- 去重策略：按 `user_id`, `parent_asin`, `timestamp`, `original_row_idx` 稳定排序后，每对只保留最新一条 interaction。",
        f"- 正样本：`rating >= {config['positive_rating_threshold']}`",
        "- `rating < threshold` 暂时不作为显式负样本。",
        "- `verified_purchase` 暂时不参与过滤。",
        f"- k-core：`user>={config['kcore_user_min']}, item>={config['kcore_item_min']}`",
        "- 切分：按 `user_id`, `timestamp`, `parent_asin`, `original_row_idx` 稳定排序后做 leave-one-out。",
        "- valid/test 保留 cold target item，并用 `is_cold_item_for_eval` 标记。",
        "- 评估策略：`exclude_from_test_metric`。",
        "",
        "## 规模",
        "",
        f"- interactions before dedup：{stats['n_interactions_before_dedup']}",
        f"- interactions after dedup：{stats['n_interactions_after_dedup']}",
        f"- dedup removed interactions：{stats['dedup_removed_interactions']}",
        f"- dedup removal ratio：{stats['dedup_removal_ratio']:.6f}",
        f"- users：{stats['n_users']}",
        f"- items：{stats['n_items']}",
        f"- total interactions：{stats['n_interactions_total']}",
        f"- train interactions：{stats['n_interactions_train']}",
        f"- valid interactions：{stats['n_interactions_valid']}",
        f"- test interactions：{stats['n_interactions_test']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stats(
    config: dict[str, Any],
    dedup_stats: dict[str, Any],
    full_frame: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    n_valid = len(valid)
    n_test = len(test)
    n_cold_valid = int(valid["is_cold_item_for_eval"].sum())
    n_cold_test = int(test["is_cold_item_for_eval"].sum())
    return {
        "dataset": config["category"],
        "kcore_user_min": int(config["kcore_user_min"]),
        "kcore_item_min": int(config["kcore_item_min"]),
        "rating_threshold": float(config["positive_rating_threshold"]),
        **dedup_stats,
        "n_users": int(full_frame["user_idx"].nunique()),
        "n_items": int(full_frame["item_idx"].nunique()),
        "n_interactions_total": int(len(full_frame)),
        "n_interactions_train": int(len(train)),
        "n_interactions_valid": int(n_valid),
        "n_interactions_test": int(n_test),
        "n_cold_items_in_valid": n_cold_valid,
        "n_cold_items_in_test": n_cold_test,
        "cold_item_ratio_valid": n_cold_valid / n_valid if n_valid else 0.0,
        "cold_item_ratio_test": n_cold_test / n_test if n_test else 0.0,
        "preprocess_seed": int(config["seed"]),
        "preprocess_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_outputs(
    output_dir: Path,
    config: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    id2user: dict[str, str],
    id2item: dict[str, str],
    stats: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_frame(train).to_parquet(output_dir / "train.parquet", index=False)
    output_frame(valid, include_cold_flag=True).to_parquet(output_dir / "valid.parquet", index=False)
    output_frame(test, include_cold_flag=True).to_parquet(output_dir / "test.parquet", index=False)
    write_json(output_dir / "user2id.json", user2id)
    write_json(output_dir / "item2id.json", item2id)
    write_json(output_dir / "id2user.json", id2user)
    write_json(output_dir / "id2item.json", id2item)
    write_json(output_dir / "stats.json", stats)
    write_readme(output_dir / "README.md", config, stats)


def preprocess(config: dict[str, Any]) -> dict[str, Any]:
    require_config(config)
    dataset = load_reviews(config)
    raw_frame = dataset_to_frame(dataset)
    deduped, dedup_stats = deduplicate_user_item_latest(raw_frame)
    logging.info(
        "user-item 去重完成：before=%s after=%s removed=%s ratio=%.6f",
        dedup_stats["n_interactions_before_dedup"],
        dedup_stats["n_interactions_after_dedup"],
        dedup_stats["dedup_removed_interactions"],
        dedup_stats["dedup_removal_ratio"],
    )
    logging.info("去重策略：在 rating>=threshold 前，对每个 (user_id, parent_asin) 保留 timestamp 最新的一条 interaction。")
    positive = build_positive_view(deduped, float(config["positive_rating_threshold"]))
    logging.info("正样本 interactions=%s", len(positive))

    filtered = run_k_core(
        positive,
        user_min=int(config["kcore_user_min"]),
        item_min=int(config["kcore_item_min"]),
    )
    logging.info("k-core 后 interactions=%s users=%s items=%s", len(filtered), filtered["user_id"].nunique(), filtered["parent_asin"].nunique())

    user2id, item2id, id2user, id2item = build_mappings(filtered)
    mapped = apply_mappings(filtered, user2id, item2id)
    train, valid, test = leave_one_out_split(mapped)
    valid = mark_cold_items(train, valid)
    test = mark_cold_items(train, test)
    stats = build_stats(config, dedup_stats, mapped, train, valid, test)

    output_dir = Path(config["output_dir"])
    save_outputs(output_dir, config, train, valid, test, user2id, item2id, id2user, id2item, stats)
    logging.info("预处理输出已写入：%s", output_dir)
    return stats


def main() -> None:
    setup_logging()
    args = parse_args()
    config = apply_overrides(load_config(Path(args.config)), args)
    stats = preprocess(config)
    logging.info("预处理完成：%s", json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

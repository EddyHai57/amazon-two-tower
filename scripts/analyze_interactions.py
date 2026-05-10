#!/usr/bin/env python3
"""分析 Amazon Reviews 2023 指定品类的用户-物品交互统计。"""

try:
    import argparse
    import logging
    import time
    from dataclasses import dataclass
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


REQUIRED_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp", "verified_purchase"]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
CATEGORY_COMPARISON_COLUMNS = [
    "品类",
    "原始review",
    "rating>=4正样本",
    "unique user",
    "unique item",
    "user均交互(p50/p90)",
    "k-core(3,3)剩余interaction",
    "k-core(3,3)剩余user",
    "k-core(3,3)剩余item",
    "k-core(5,5)剩余interaction",
    "leave-one-out可用user数",
    "加载用时",
    "加载策略(full/streaming)",
]


@dataclass
class KCoreResult:
    name: str
    user_min: int
    item_min: int
    frame: pd.DataFrame
    interactions: int
    users: int
    items: int
    interaction_retention: float
    user_retention: float
    item_retention: float
    leave_one_out_users: int
    leave_one_out_user_ratio: float
    recommendation: str
    cold_start: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 Amazon All_Beauty 交互分布和 k-core 可行性。")
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


def infer_category(config: dict[str, Any]) -> str:
    if config.get("category"):
        return str(config["category"])
    review_config = str(config.get("review_config", ""))
    prefix = "raw_review_"
    if review_config.startswith(prefix):
        return review_config[len(prefix):]
    raise KeyError("配置缺少必需字段：category")


def category_slug(category: str) -> str:
    return category.lower()


def output_dir(config: dict[str, Any]) -> Path:
    return Path(str(config["output_dir"]))


def interaction_report_path(config: dict[str, Any]) -> Path:
    return output_dir(config) / f"interaction_analysis_{category_slug(infer_category(config))}.md"


def comparison_path(config: dict[str, Any]) -> Path:
    return output_dir(config) / "category_comparison.md"


def normalize_k_core_thresholds(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "k_core_thresholds" in config:
        simulations = []
        for idx, pair in enumerate(config["k_core_thresholds"]):
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                raise ValueError(f"k_core_thresholds 第 {idx + 1} 项必须是 [user_min, item_min]")
            simulations.append(
                {
                    "name": chr(ord("A") + idx),
                    "user_min_interactions": int(pair[0]),
                    "item_min_interactions": int(pair[1]),
                }
            )
        return simulations

    legacy = config.get("interaction_analysis", {})
    if "k_core_simulations" in legacy:
        return list(legacy["k_core_simulations"])
    raise KeyError("配置缺少必需字段：k_core_thresholds")


def normalized_analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    legacy = config.get("interaction_analysis", {})
    if "leave_one_out_min_interactions" in config:
        leave_one_out_min_interactions = config["leave_one_out_min_interactions"]
    elif "leave_one_out_min_interactions" in legacy:
        leave_one_out_min_interactions = legacy["leave_one_out_min_interactions"]
    else:
        raise KeyError("配置缺少必需字段：leave_one_out_min_interactions")

    if "phase_switch_thresholds" in config:
        phase_switch_thresholds = config["phase_switch_thresholds"]
    elif "phase_switch_thresholds" in legacy:
        phase_switch_thresholds = legacy["phase_switch_thresholds"]
    else:
        raise KeyError("配置缺少必需字段：phase_switch_thresholds")

    return {
        "leave_one_out_min_interactions": int(leave_one_out_min_interactions),
        "k_core_simulations": normalize_k_core_thresholds(config),
        "phase_switch_thresholds": phase_switch_thresholds,
    }


def require_config(config: dict[str, Any]) -> None:
    required_top_keys = [
        "dataset_name",
        "review_config",
        "meta_config",
        "positive_rating_threshold",
        "output_dir",
    ]
    for key in required_top_keys:
        if key not in config:
            raise KeyError(f"配置缺少必需字段：{key}")

    infer_category(config)
    normalized_analysis_config(config)


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
    return dataset.select_columns(REQUIRED_COLUMNS).to_pandas()


def missing_count(series: pd.Series) -> int:
    return int(series.isna().sum())


def pct(part: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return float(part) / float(total)


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_float(value: int | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: int | float) -> str:
    return f"{float(value) * 100:.2f}%"


def readable_datetime(timestamp_ms: Any) -> str:
    if pd.isna(timestamp_ms):
        return "不可用"
    return pd.to_datetime(timestamp_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def rating_distribution(df: pd.DataFrame, threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rating = pd.to_numeric(df["rating"], errors="coerce")
    total = len(df)
    rows = []
    for value in [1, 2, 3, 4, 5]:
        count = int((rating == value).sum())
        rows.append({"rating": value, "count": count, "ratio": pct(count, total)})

    positive_count = int((rating >= threshold).sum())
    negative_count = int((rating < threshold).sum())
    summary = {
        "positive_count": positive_count,
        "positive_ratio": pct(positive_count, total),
        "negative_count": negative_count,
        "negative_ratio": pct(negative_count, total),
    }
    return rows, summary


def verified_purchase_distribution(df: pd.DataFrame) -> list[dict[str, Any]]:
    series = df["verified_purchase"]
    total = len(df)
    true_count = int((series == True).sum())
    false_count = int((series == False).sum())
    missing = missing_count(series)
    return [
        {"value": "True", "count": true_count, "ratio": pct(true_count, total)},
        {"value": "False", "count": false_count, "ratio": pct(false_count, total)},
        {"value": "缺失", "count": missing, "ratio": pct(missing, total)},
    ]


def positive_interactions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = df.copy()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame[frame["rating"] >= threshold]
    return frame.dropna(subset=["user_id", "parent_asin", "timestamp"]).copy()


def distribution_stats(counts: pd.Series) -> dict[str, float]:
    if counts.empty:
        return {
            "min": 0.0,
            "mean": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    result = {
        "min": float(counts.min()),
        "mean": float(counts.mean()),
        "max": float(counts.max()),
    }
    for quantile in QUANTILES:
        result[f"p{int(quantile * 100)}"] = float(counts.quantile(quantile))
    return result


def bucket_histogram(counts: pd.Series, label_name: str) -> list[dict[str, Any]]:
    total = len(counts)
    buckets = [
        ("1", counts == 1),
        ("2", counts == 2),
        ("3-5", (counts >= 3) & (counts <= 5)),
        ("6-10", (counts >= 6) & (counts <= 10)),
        ("11-20", (counts >= 11) & (counts <= 20)),
        ("21+", counts >= 21),
    ]
    return [
        {label_name: label, "count": int(mask.sum()), "ratio": pct(int(mask.sum()), total)}
        for label, mask in buckets
    ]


def run_k_core(df: pd.DataFrame, user_min: int, item_min: int) -> pd.DataFrame:
    current = df[["user_id", "parent_asin", "timestamp"]].copy()
    while not current.empty:
        before = len(current)
        user_counts = current["user_id"].value_counts()
        valid_users = user_counts[user_counts >= user_min].index
        current = current[current["user_id"].isin(valid_users)]
        item_counts = current["parent_asin"].value_counts()
        valid_items = item_counts[item_counts >= item_min].index
        current = current[current["parent_asin"].isin(valid_items)]
        if len(current) == before:
            break
    return current.copy()


def phase_recommendation(category: str, interactions: int, thresholds: dict[str, Any]) -> str:
    full_phase0_min = int(thresholds["full_phase0_min_interactions"])
    itemcf_only_min = int(thresholds["itemcf_only_min_interactions"])
    if interactions >= full_phase0_min:
        return f"{category} 可以走到 Phase 0 完整跑通；Phase 1 仍需与候选品类对比后决定。"
    if interactions >= itemcf_only_min:
        return f"{category} 适合做 ItemCF 管道验证；双塔是否使用它需要继续比较。"
    return f"{category} 过滤后规模偏小，不建议作为 Phase 1 主实验数据集。"


def leave_one_out_cold_start(df: pd.DataFrame, min_interactions: int) -> dict[str, Any]:
    if df.empty:
        return {
            "train_interactions": 0,
            "valid_interactions": 0,
            "test_interactions": 0,
            "valid_cold_interactions": 0,
            "valid_cold_unique_items": 0,
            "valid_cold_ratio": 0.0,
            "test_cold_interactions": 0,
            "test_cold_unique_items": 0,
            "test_cold_ratio": 0.0,
        }

    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    split_df = df[df["user_id"].isin(valid_users)].sort_values(["user_id", "timestamp", "parent_asin"]).copy()
    split_df["rank"] = split_df.groupby("user_id").cumcount()
    split_df["user_size"] = split_df.groupby("user_id")["parent_asin"].transform("size")

    train = split_df[split_df["rank"] < split_df["user_size"] - 2]
    valid = split_df[split_df["rank"] == split_df["user_size"] - 2]
    test = split_df[split_df["rank"] == split_df["user_size"] - 1]

    train_items = set(train["parent_asin"].dropna().unique())
    valid_cold = valid[~valid["parent_asin"].isin(train_items)]
    test_cold = test[~test["parent_asin"].isin(train_items)]

    return {
        "train_interactions": int(len(train)),
        "valid_interactions": int(len(valid)),
        "test_interactions": int(len(test)),
        "valid_cold_interactions": int(len(valid_cold)),
        "valid_cold_unique_items": int(valid_cold["parent_asin"].nunique()),
        "valid_cold_ratio": pct(len(valid_cold), len(valid)),
        "test_cold_interactions": int(len(test_cold)),
        "test_cold_unique_items": int(test_cold["parent_asin"].nunique()),
        "test_cold_ratio": pct(len(test_cold), len(test)),
    }


def analyze_k_core(
    category: str,
    positive_df: pd.DataFrame,
    analysis_config: dict[str, Any],
) -> list[KCoreResult]:
    base_interactions = len(positive_df)
    base_users = positive_df["user_id"].nunique()
    base_items = positive_df["parent_asin"].nunique()
    min_split_interactions = int(analysis_config["leave_one_out_min_interactions"])
    thresholds = analysis_config["phase_switch_thresholds"]
    results = []

    for item in analysis_config["k_core_simulations"]:
        name = str(item["name"])
        user_min = int(item["user_min_interactions"])
        item_min = int(item["item_min_interactions"])
        logging.info("开始 k-core 模拟：%s user>=%s item>=%s", name, user_min, item_min)
        filtered = run_k_core(positive_df, user_min, item_min)
        interactions = len(filtered)
        users = filtered["user_id"].nunique()
        items = filtered["parent_asin"].nunique()
        user_counts = filtered["user_id"].value_counts()
        split_users = int((user_counts >= min_split_interactions).sum())
        cold_start = leave_one_out_cold_start(filtered, min_split_interactions)
        results.append(
            KCoreResult(
                name=name,
                user_min=user_min,
                item_min=item_min,
                frame=filtered,
                interactions=interactions,
                users=users,
                items=items,
                interaction_retention=pct(interactions, base_interactions),
                user_retention=pct(users, base_users),
                item_retention=pct(items, base_items),
                leave_one_out_users=split_users,
                leave_one_out_user_ratio=pct(split_users, users),
                recommendation=phase_recommendation(category, interactions, thresholds),
                cold_start=cold_start,
            )
        )
    return results


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def stats_table(stats: dict[str, float]) -> list[str]:
    keys = ["min", "mean", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    return markdown_table(
        ["指标", "数值"],
        [[key, fmt_float(stats[key])] for key in keys],
    )


def k_core_by_pair(results: list[KCoreResult], user_min: int, item_min: int) -> KCoreResult | None:
    for result in results:
        if result.user_min == user_min and result.item_min == item_min:
            return result
    return None


def parse_existing_comparison(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "品类" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == len(CATEGORY_COMPARISON_COLUMNS):
            rows[parts[0]] = parts
    return rows


def update_category_comparison(
    config: dict[str, Any],
    category: str,
    raw_reviews: int,
    rating_summary: dict[str, Any],
    unique_users: int,
    unique_items: int,
    user_stats: dict[str, float],
    k_core_results: list[KCoreResult],
    load_seconds: float,
    loading_strategy: str,
) -> None:
    path = comparison_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = parse_existing_comparison(path)
    k33 = k_core_by_pair(k_core_results, 3, 3)
    k55 = k_core_by_pair(k_core_results, 5, 5)
    row = [
        category,
        fmt_int(raw_reviews),
        f"{fmt_int(rating_summary['positive_count'])} ({fmt_pct(rating_summary['positive_ratio'])})",
        fmt_int(unique_users),
        fmt_int(unique_items),
        f"{fmt_float(user_stats['p50'])}/{fmt_float(user_stats['p90'])}",
        fmt_int(k33.interactions) if k33 else "未计算",
        fmt_int(k33.users) if k33 else "未计算",
        fmt_int(k33.items) if k33 else "未计算",
        fmt_int(k55.interactions) if k55 else "未计算",
        fmt_int(k33.leave_one_out_users) if k33 else "未计算",
        f"{fmt_float(load_seconds)} 秒",
        loading_strategy,
    ]
    rows[category] = row

    lines = [
        "# Amazon Reviews 2023 候选品类对比",
        "",
        "| " + " | ".join(CATEGORY_COMPARISON_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(CATEGORY_COMPARISON_COLUMNS)) + " |",
    ]
    for key in sorted(rows):
        lines.append("| " + " | ".join(rows[key]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    dataset: Any,
    df: pd.DataFrame,
    positive_df: pd.DataFrame,
    config: dict[str, Any],
    rating_rows: list[dict[str, Any]],
    rating_summary: dict[str, Any],
    verified_rows: list[dict[str, Any]],
    user_stats: dict[str, float],
    user_buckets: list[dict[str, Any]],
    item_stats: dict[str, float],
    item_buckets: list[dict[str, Any]],
    k_core_results: list[KCoreResult],
    load_seconds: float,
    loading_strategy: str,
) -> Path:
    category = infer_category(config)
    threshold = config["positive_rating_threshold"]
    timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    min_timestamp = timestamp.min()
    max_timestamp = timestamp.max()

    lines = [
        f"# Amazon {category} 交互数据审查报告",
        "",
        "## 1. 本次分析目的",
        "",
        f"- {category} 当前用于候选品类交互质量分析。",
        f"- 本报告用于判断 {category} 是否适合作为 Phase 1 主实验数据集候选。",
        "- 本报告不生成正式训练数据，只做统计分析和模拟。",
        "",
        "## 2. 数据基础信息",
        "",
        f"- review 总行数：{fmt_int(len(df))}",
        f"- 字段列表：`{dataset.column_names}`",
        f"- 加载策略：`{loading_strategy}`",
        f"- 加载用时：{fmt_float(load_seconds)} 秒",
        "",
        "### 缺失值检查",
        "",
    ]
    lines.extend(
        markdown_table(
            ["字段", "缺失数量", "缺失占比"],
            [
                ["user_id", fmt_int(missing_count(df["user_id"])), fmt_pct(pct(missing_count(df["user_id"]), len(df)))],
                ["parent_asin", fmt_int(missing_count(df["parent_asin"])), fmt_pct(pct(missing_count(df["parent_asin"]), len(df)))],
                ["rating", fmt_int(missing_count(df["rating"])), fmt_pct(pct(missing_count(df["rating"]), len(df)))],
                ["timestamp", fmt_int(missing_count(df["timestamp"])), fmt_pct(pct(missing_count(df["timestamp"]), len(df)))],
            ],
        )
    )
    lines.extend(
        [
            "",
            "### 时间范围",
            "",
            f"- timestamp 最小值：{fmt_int(min_timestamp)}，可读日期：{readable_datetime(min_timestamp)}",
            f"- timestamp 最大值：{fmt_int(max_timestamp)}，可读日期：{readable_datetime(max_timestamp)}",
            "",
            "## 3. rating 分布与正样本规模",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["rating", "数量", "占比"],
            [[row["rating"], fmt_int(row["count"]), fmt_pct(row["ratio"])] for row in rating_rows],
        )
    )
    lines.extend(
        [
            "",
            f"- 临时正样本定义：`rating >= {threshold}`。",
            f"- 正样本数量：{fmt_int(rating_summary['positive_count'])}，占比：{fmt_pct(rating_summary['positive_ratio'])}",
            f"- `rating < {threshold}` 数量：{fmt_int(rating_summary['negative_count'])}，占比：{fmt_pct(rating_summary['negative_ratio'])}",
            "- 这里只做统计，不做最终决策；最终是否使用该阈值需要 Eddy 确认。",
            "",
            "## 4. verified_purchase 分布",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["verified_purchase", "数量", "占比"],
            [[row["value"], fmt_int(row["count"]), fmt_pct(row["ratio"])] for row in verified_rows],
        )
    )
    lines.extend(
        [
            "",
            "- 这里只做统计，不把 `verified_purchase` 加入过滤规则。",
            "",
            "## 5. 用户交互数分布",
            "",
            f"- 正样本视图 interaction 数量：{fmt_int(len(positive_df))}",
            f"- 正样本视图 unique user_id 数量：{fmt_int(positive_df['user_id'].nunique())}",
            f"- 正样本视图 unique parent_asin 数量：{fmt_int(positive_df['parent_asin'].nunique())}",
            "",
            "### 用户正向交互数分位数",
            "",
        ]
    )
    lines.extend(stats_table(user_stats))
    lines.extend(["", "### 用户正向交互数分桶", ""])
    lines.extend(
        markdown_table(
            ["交互数桶", "用户数量", "用户占比"],
            [[row["user_bucket"], fmt_int(row["count"]), fmt_pct(row["ratio"])] for row in user_buckets],
        )
    )
    lines.extend(["", "## 6. item 交互数分布", "", "### item 正向交互数分位数", ""])
    lines.extend(stats_table(item_stats))
    lines.extend(["", "### item 正向交互数分桶", ""])
    lines.extend(
        markdown_table(
            ["交互数桶", "item 数量", "item 占比"],
            [[row["item_bucket"], fmt_int(row["count"]), fmt_pct(row["ratio"])] for row in item_buckets],
        )
    )
    lines.extend(["", "## 7. 多组 k-core 过滤模拟", ""])
    lines.extend(
        markdown_table(
            [
                "组别",
                "user_min",
                "item_min",
                "剩余 interaction",
                "剩余 user",
                "剩余 item",
                "interaction 保留率",
                "user 保留率",
                "item 保留率",
                "可 leave-one-out user",
                "占剩余 user 比例",
            ],
            [
                [
                    result.name,
                    result.user_min,
                    result.item_min,
                    fmt_int(result.interactions),
                    fmt_int(result.users),
                    fmt_int(result.items),
                    fmt_pct(result.interaction_retention),
                    fmt_pct(result.user_retention),
                    fmt_pct(result.item_retention),
                    fmt_int(result.leave_one_out_users),
                    fmt_pct(result.leave_one_out_user_ratio),
                ]
                for result in k_core_results
            ],
        )
    )
    lines.extend(["", "## 8. leave-one-out 可行性与 cold-start 率", ""])
    lines.extend(
        markdown_table(
            [
                "组别",
                "train interaction",
                "valid interaction",
                "test interaction",
                "valid cold interaction",
                "valid cold unique item",
                "valid cold 比例",
                "test cold interaction",
                "test cold unique item",
                "test cold 比例",
            ],
            [
                [
                    result.name,
                    fmt_int(result.cold_start["train_interactions"]),
                    fmt_int(result.cold_start["valid_interactions"]),
                    fmt_int(result.cold_start["test_interactions"]),
                    fmt_int(result.cold_start["valid_cold_interactions"]),
                    fmt_int(result.cold_start["valid_cold_unique_items"]),
                    fmt_pct(result.cold_start["valid_cold_ratio"]),
                    fmt_int(result.cold_start["test_cold_interactions"]),
                    fmt_int(result.cold_start["test_cold_unique_items"]),
                    fmt_pct(result.cold_start["test_cold_ratio"]),
                ]
                for result in k_core_results
            ],
        )
    )
    lines.extend(
        [
            "",
            "cold item 比例会影响 ItemCF 可解释性。如果 test 里很多 item 在 train 没出现过，ItemCF 不可能推荐这些 item，会导致 ItemCF 指标假性偏低。",
            "",
            "## 9. Phase 0 / Phase 1 判断建议",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["组别", "过滤后 interaction", "建议"],
            [[result.name, fmt_int(result.interactions), result.recommendation] for result in k_core_results],
        )
    )
    best_result = k_core_results[0] if k_core_results else None
    thresholds = normalized_analysis_config(config)["phase_switch_thresholds"]
    if best_result and best_result.interactions >= int(thresholds["full_phase0_min_interactions"]):
        conclusion = f"{category} 过滤后规模较充足，可以作为候选品类继续比较。"
    elif best_result and best_result.interactions >= int(thresholds["itemcf_only_min_interactions"]):
        conclusion = f"{category} 可用于部分管道验证，但是否适合作为主数据集需要继续比较。"
    else:
        conclusion = f"{category} 过滤后规模偏小，不建议作为 Phase 1 主实验数据集。"
    lines.extend(
        [
            "",
            "以上是分析建议，不是最终设计决策；最终由 Eddy 确认。",
            "",
            "## 10. 当前结论",
            "",
            "- 原始数据是否稀疏，需要结合 unique user、用户交互分位数和 k-core 后规模一起判断。",
            f"- {conclusion}",
            "- 本报告只用于候选品类比较，不生成正式 train/valid/test、ID mapping、ItemCF 输出、模型 checkpoint 或 text embeddings。",
            "",
        ]
    )

    path = interaction_report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    require_config(config)
    category = infer_category(config)
    start = time.perf_counter()
    dataset = load_reviews(config)
    load_seconds = time.perf_counter() - start
    loading_strategy = "full"
    df = dataset_to_frame(dataset)
    threshold = float(config["positive_rating_threshold"])

    rating_rows, rating_summary = rating_distribution(df, threshold)
    verified_rows = verified_purchase_distribution(df)
    positive_df = positive_interactions(df, threshold)
    user_counts = positive_df["user_id"].value_counts()
    item_counts = positive_df["parent_asin"].value_counts()
    user_stats = distribution_stats(user_counts)
    item_stats = distribution_stats(item_counts)
    user_buckets = bucket_histogram(user_counts, "user_bucket")
    item_buckets = bucket_histogram(item_counts, "item_bucket")
    k_core_results = analyze_k_core(category, positive_df, normalized_analysis_config(config))

    path = write_report(
        dataset=dataset,
        df=df,
        positive_df=positive_df,
        config=config,
        rating_rows=rating_rows,
        rating_summary=rating_summary,
        verified_rows=verified_rows,
        user_stats=user_stats,
        user_buckets=user_buckets,
        item_stats=item_stats,
        item_buckets=item_buckets,
        k_core_results=k_core_results,
        load_seconds=load_seconds,
        loading_strategy=loading_strategy,
    )
    update_category_comparison(
        config=config,
        category=category,
        raw_reviews=len(df),
        rating_summary=rating_summary,
        unique_users=positive_df["user_id"].nunique(),
        unique_items=positive_df["parent_asin"].nunique(),
        user_stats=user_stats,
        k_core_results=k_core_results,
        load_seconds=load_seconds,
        loading_strategy=loading_strategy,
    )
    logging.info("交互数据审查报告已写入：%s", path)
    logging.info("候选品类对比表已更新：%s", comparison_path(config))


if __name__ == "__main__":
    main()

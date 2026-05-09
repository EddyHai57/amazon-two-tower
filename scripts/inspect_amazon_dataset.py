#!/usr/bin/env python3
"""检查 Amazon Reviews 2023 All_Beauty 数据集结构。"""

try:
    import argparse
    import itertools
    import json
    import logging
    import signal
    from pathlib import Path
    from typing import Any, Iterable

    import yaml
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    missing_name = exc.name or "未知依赖"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("缺少依赖：%s。请先安装 package：%s", missing_name, package_hint)
    raise SystemExit(1) from exc


REPORT_PATH = Path("outputs/inspection_all_beauty.md")
TRUNCATION_MARKER = "[...truncated]"
MAX_STRING_LENGTH = 500
FULL_LOAD_TIMEOUT_SECONDS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Amazon Reviews 2023 数据集结构。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效：{config_path}")
    return config


def load_full_dataset(dataset_name: str, config_name: str) -> Any:
    with timeout(FULL_LOAD_TIMEOUT_SECONDS):
        return load_dataset(
            dataset_name,
            config_name,
            split="full",
            trust_remote_code=True,
        )


def load_streaming_dataset(dataset_name: str, config_name: str) -> Any:
    return load_dataset(
        dataset_name,
        config_name,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )


class LoadTimeoutError(TimeoutError):
    pass


class DatasetLoadCompatibilityError(RuntimeError):
    pass


class timeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        if not hasattr(signal, "SIGALRM"):
            return
        self.previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if not hasattr(signal, "SIGALRM"):
            return
        signal.alarm(0)
        if self.previous_handler is not None:
            signal.signal(signal.SIGALRM, self.previous_handler)

    def _handle_timeout(self, signum: int, frame: Any) -> None:
        raise LoadTimeoutError(f"full_load 超过 {self.seconds} 秒，视为可能挂起")


def load_pair_with_strategy(config: dict[str, Any]) -> tuple[Any, Any, str]:
    dataset_name = config["dataset_name"]
    review_config = config["review_config"]
    meta_config = config["meta_config"]

    try:
        logging.info("尝试 full_load 读取 review 数据集：%s / %s", dataset_name, review_config)
        reviews = load_full_dataset(dataset_name, review_config)
        logging.info("尝试 full_load 读取 meta 数据集：%s / %s", dataset_name, meta_config)
        meta = load_full_dataset(dataset_name, meta_config)
        logging.info("数据加载策略：full_load")
        return reviews, meta, "full_load"
    except Exception as exc:
        error_text = str(exc)
        logging.warning("full_load 失败：%s", error_text)
        if is_version_or_remote_script_error(error_text):
            logging.error("疑似 datasets 版本、trust_remote_code 或远程数据集脚本兼容性错误。")
            logging.error("不自动升级或降级依赖。原始错误：%s", error_text)
            raise DatasetLoadCompatibilityError(error_text) from exc

        logging.info("切换到 streaming_fallback。")
        reviews = load_streaming_dataset(dataset_name, review_config)
        meta = load_streaming_dataset(dataset_name, meta_config)
        logging.info("数据加载策略：streaming_fallback")
        return reviews, meta, "streaming_fallback"


def is_version_or_remote_script_error(error_text: str) -> bool:
    lowered = error_text.lower()
    markers = (
        "trust_remote_code",
        "remote code",
        "dataset scripts",
        "dataset script",
        "datasets version",
        "please upgrade",
        "please update",
        "not supported",
    )
    return any(marker in lowered for marker in markers)


def dataset_columns(dataset: Any, strategy: str, sample_rows: list[dict[str, Any]] | None = None) -> list[str]:
    features = getattr(dataset, "features", None)
    if features is not None:
        return list(features.keys())
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        return list(column_names)
    if strategy == "streaming_fallback" and sample_rows:
        return list(sample_rows[0].keys())
    return []


def row_count(dataset: Any, strategy: str) -> int | None:
    if strategy == "streaming_fallback":
        return None
    try:
        return int(dataset.num_rows)
    except AttributeError:
        return len(dataset)


def collect_streaming_sample(dataset: Iterable[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    return list(itertools.islice(dataset, sample_size))


def first_rows_from_full(dataset: Any, limit: int = 3) -> list[dict[str, Any]]:
    rows = []
    for idx in range(min(limit, len(dataset))):
        rows.append(dict(dataset[idx]))
    return rows


def exact_unique_count(dataset: Any, column: str) -> int | None:
    if column not in dataset.column_names:
        return None
    return len(set(dataset[column]))


def approximate_unique_count(rows: list[dict[str, Any]], column: str) -> int | None:
    if not rows or column not in rows[0]:
        return None
    return len({row.get(column) for row in rows if row.get(column) is not None})


def make_json_safe(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + TRUNCATION_MARKER
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): make_json_safe(value) for key, value in dict(row).items()}


def format_count(value: int | None, streaming: bool = False) -> str:
    if streaming:
        return "not available in streaming mode"
    if value is None:
        return "not available"
    return str(value)


def format_unique_count(value: int | None, strategy: str, sample_size: int) -> str:
    if value is None:
        return "not available"
    if strategy == "streaming_fallback":
        return f"approximately {value} in first {sample_size} rows"
    return str(value)


def write_report(
    report_path: Path,
    strategy: str,
    review_columns: list[str],
    meta_columns: list[str],
    review_count: int | None,
    meta_count: int | None,
    review_samples: list[dict[str, Any]],
    meta_samples: list[dict[str, Any]],
    unique_users: int | None,
    unique_items: int | None,
    sample_size: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    streaming = strategy == "streaming_fallback"

    lines = [
        "# Amazon Reviews 2023 All_Beauty Inspection",
        "",
        f"- loading strategy used: {strategy}",
        f"- review columns: {json.dumps(review_columns, ensure_ascii=False)}",
        f"- meta columns: {json.dumps(meta_columns, ensure_ascii=False)}",
        f"- review row count: {format_count(review_count, streaming=streaming)}",
        f"- meta row count: {format_count(meta_count, streaming=streaming)}",
        f"- unique user_id count: {format_unique_count(unique_users, strategy, sample_size)}",
        f"- unique parent_asin count: {format_unique_count(unique_items, strategy, sample_size)}",
        "",
        "## Review Samples",
        "",
    ]

    lines.extend(format_sample_blocks(review_samples))
    lines.extend(["", "## Meta Samples", ""])
    lines.extend(format_sample_blocks(meta_samples))
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def format_sample_blocks(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows[:3]:
        normalized = normalize_row(row)
        lines.append("```json")
        lines.append(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return lines


def inspect_datasets(config: dict[str, Any]) -> dict[str, Any]:
    reviews, meta, strategy = load_pair_with_strategy(config)
    sample_size = int(config["inspection_sample_size"])

    if strategy == "streaming_fallback":
        review_stream_rows = collect_streaming_sample(reviews, sample_size)
        meta_stream_rows = collect_streaming_sample(meta, sample_size)
        review_samples = review_stream_rows[:3]
        meta_samples = meta_stream_rows[:3]
        review_columns = dataset_columns(reviews, strategy, review_stream_rows)
        meta_columns = dataset_columns(meta, strategy, meta_stream_rows)
        review_count = None
        meta_count = None
        unique_users = approximate_unique_count(review_stream_rows, "user_id")
        unique_items = approximate_unique_count(review_stream_rows, "parent_asin")
    else:
        review_samples = first_rows_from_full(reviews)
        meta_samples = first_rows_from_full(meta)
        review_columns = dataset_columns(reviews, strategy)
        meta_columns = dataset_columns(meta, strategy)
        review_count = row_count(reviews, strategy)
        meta_count = row_count(meta, strategy)
        unique_users = exact_unique_count(reviews, "user_id")
        unique_items = exact_unique_count(reviews, "parent_asin")

    logging.info("review columns: %s", review_columns)
    logging.info("meta columns: %s", meta_columns)
    logging.info("review row count: %s", format_count(review_count, streaming=strategy == "streaming_fallback"))
    logging.info("meta row count: %s", format_count(meta_count, streaming=strategy == "streaming_fallback"))
    logging.info("unique user_id count: %s", format_unique_count(unique_users, strategy, sample_size))
    logging.info("unique parent_asin count: %s", format_unique_count(unique_items, strategy, sample_size))

    write_report(
        REPORT_PATH,
        strategy,
        review_columns,
        meta_columns,
        review_count,
        meta_count,
        review_samples,
        meta_samples,
        unique_users,
        unique_items,
        sample_size,
    )
    logging.info("检查报告已写入：%s", REPORT_PATH)

    return {
        "strategy": strategy,
        "review_columns": review_columns,
        "meta_columns": meta_columns,
        "review_count": review_count,
        "meta_count": meta_count,
        "unique_users": unique_users,
        "unique_items": unique_items,
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    try:
        inspect_datasets(config)
    except DatasetLoadCompatibilityError:
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

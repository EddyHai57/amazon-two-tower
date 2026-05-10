#!/usr/bin/env python3
"""Diagnose the full valid/test gap for the ID-only two-tower model."""

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

from train_two_tower import IDOnlyTwoTower  # noqa: E402


PAIR_COLUMNS = ["user_idx", "item_idx"]
EVAL_COLUMNS = ["user_idx", "item_idx", "is_cold_item_for_eval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 Two-Tower valid/test gap。")
    parser.add_argument("--config", required=True, help="Two-Tower YAML 配置文件路径。")
    parser.add_argument("--checkpoint", required=True, help="需要诊断的 checkpoint 路径。")
    parser.add_argument("--sample_size", type=int, default=20, help="rank / mask 诊断抽样用户数。")
    parser.add_argument("--output", default="outputs/two_tower_eval_gap_diagnosis.md", help="Markdown 诊断报告输出路径。")
    parser.add_argument("--json_output", default="outputs/two_tower_eval_gap_diagnosis.json", help="JSON 诊断结果输出路径。")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效：{path}")
    return config


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logging.info("读取 train/valid/test 数据。")
    train_df = pd.read_parquet(data_dir / "train.parquet", columns=PAIR_COLUMNS)
    valid_df = pd.read_parquet(data_dir / "valid.parquet", columns=EVAL_COLUMNS)
    test_df = pd.read_parquet(data_dir / "test.parquet", columns=EVAL_COLUMNS)
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    logging.info("train=%s valid=%s test=%s", len(train_df), len(valid_df), len(test_df))
    return train_df, valid_df, test_df, stats


def count_pair_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    right_pairs = right[PAIR_COLUMNS].drop_duplicates()
    overlap = left[PAIR_COLUMNS].merge(right_pairs, on=PAIR_COLUMNS, how="inner")
    return int(len(overlap))


def split_leakage_diagnostics(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    valid_user_counts = valid_df["user_idx"].value_counts()
    test_user_counts = test_df["user_idx"].value_counts()
    valid_users = set(int(x) for x in valid_user_counts.index)
    test_users = set(int(x) for x in test_user_counts.index)
    return {
        "test_in_train": count_pair_overlap(test_df, train_df),
        "test_in_valid": count_pair_overlap(test_df, valid_df),
        "valid_in_train": count_pair_overlap(valid_df, train_df),
        "valid_users": int(len(valid_users)),
        "test_users": int(len(test_users)),
        "valid_users_with_one_row": int((valid_user_counts == 1).sum()),
        "test_users_with_one_row": int((test_user_counts == 1).sum()),
        "valid_users_not_in_test": int(len(valid_users - test_users)),
        "test_users_not_in_valid": int(len(test_users - valid_users)),
    }


def popularity_stats(values: pd.Series, cold_flags: pd.Series) -> dict[str, Any]:
    arr = values.to_numpy(dtype=np.float64, copy=True)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "ratio_le_5": float(np.mean(arr <= 5)),
        "ratio_le_10": float(np.mean(arr <= 10)),
        "ratio_le_20": float(np.mean(arr <= 20)),
        "cold_item_count": int(cold_flags.astype(bool).sum()),
    }


def target_popularity_diagnostics(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    item_popularity = train_df["item_idx"].value_counts()
    valid_popularity = valid_df["item_idx"].map(item_popularity).fillna(0)
    test_popularity = test_df["item_idx"].map(item_popularity).fillna(0)
    return {
        "valid": popularity_stats(valid_popularity, valid_df["is_cold_item_for_eval"]),
        "test": popularity_stats(test_popularity, test_df["is_cold_item_for_eval"]),
    }


def sample_users(valid_df: pd.DataFrame, test_df: pd.DataFrame, sample_size: int, seed: int) -> np.ndarray:
    valid_non_cold_users = set(int(x) for x in valid_df.loc[~valid_df["is_cold_item_for_eval"].astype(bool), "user_idx"])
    test_non_cold_users = np.array(
        sorted(valid_non_cold_users.intersection(int(x) for x in test_df.loc[~test_df["is_cold_item_for_eval"].astype(bool), "user_idx"])),
        dtype=np.int64,
    )
    if len(test_non_cold_users) < sample_size:
        return test_non_cold_users
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(test_non_cold_users, size=sample_size, replace=False))


def build_sample_seen(train_df: pd.DataFrame, valid_df: pd.DataFrame, users: np.ndarray) -> tuple[dict[int, set[int]], dict[int, int], dict[int, int]]:
    user_set = set(int(x) for x in users.tolist())
    train_seen: dict[int, set[int]] = {int(user_idx): set() for user_idx in users.tolist()}
    for user_idx, group in train_df[train_df["user_idx"].isin(user_set)].groupby("user_idx", sort=False):
        train_seen[int(user_idx)] = set(int(item_idx) for item_idx in group["item_idx"].tolist())
    valid_targets = {
        int(row.user_idx): int(row.item_idx)
        for row in valid_df[valid_df["user_idx"].isin(user_set)][["user_idx", "item_idx"]].itertuples(index=False)
    }
    return train_seen, valid_targets, {user_idx: valid_targets[user_idx] for user_idx in valid_targets}


def load_model(config: dict[str, Any], stats: dict[str, Any], checkpoint_path: Path, device: torch.device) -> IDOnlyTwoTower:
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.info("checkpoint 加载完成：%s", checkpoint_path)
    return model


def rank_and_mask_diagnostics(
    model: IDOnlyTwoTower,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: dict[str, Any],
    config: dict[str, Any],
    users: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    train_seen, valid_targets, _ = build_sample_seen(train_df, valid_df, users)
    test_targets = {
        int(row.user_idx): int(row.item_idx)
        for row in test_df[test_df["user_idx"].isin(set(int(x) for x in users.tolist()))][["user_idx", "item_idx"]].itertuples(index=False)
    }

    num_items = int(stats["n_items"])
    temperature = float(config["temperature"])
    with torch.no_grad():
        item_idx = torch.arange(num_items, device=device)
        item_emb = model.encode_items(item_idx)
        user_tensor = torch.as_tensor(users, dtype=torch.long, device=device)
        user_emb = model.encode_users(user_tensor)
        scores = (user_emb @ item_emb.T) / temperature

    rows = []
    mask_train_hits = 0
    mask_train_valid_hits = 0
    for row_pos, user_idx_value in enumerate(users.tolist()):
        user_idx = int(user_idx_value)
        valid_item = int(valid_targets[user_idx])
        test_item = int(test_targets[user_idx])
        train_items = train_seen.get(user_idx, set())
        valid_seen = {valid_item}
        test_target_in_train = test_item in train_items
        test_target_in_valid = test_item in valid_seen
        test_target_in_seen_mask = test_target_in_train or test_target_in_valid

        user_scores = scores[row_pos]
        valid_score = float(user_scores[valid_item].item())
        test_score = float(user_scores[test_item].item())
        valid_raw_rank = int((user_scores > user_scores[valid_item]).sum().item()) + 1
        test_raw_rank = int((user_scores > user_scores[test_item]).sum().item()) + 1

        train_mask_scores = user_scores.clone()
        if train_items:
            train_mask_scores[torch.as_tensor(list(train_items), dtype=torch.long, device=device)] = -torch.inf
        train_mask_scores[test_item] = user_scores[test_item]
        train_top50 = torch.topk(train_mask_scores, k=50).indices
        train_hit50 = bool((train_top50 == test_item).any().item())
        mask_train_hits += int(train_hit50)

        train_valid_mask_scores = user_scores.clone()
        train_valid_seen = set(train_items)
        train_valid_seen.update(valid_seen)
        if train_valid_seen:
            train_valid_mask_scores[torch.as_tensor(list(train_valid_seen), dtype=torch.long, device=device)] = -torch.inf
        target_after_mask_before_unmask = float(train_valid_mask_scores[test_item].item())
        train_valid_mask_scores[test_item] = user_scores[test_item]
        target_after_unmask = float(train_valid_mask_scores[test_item].item())
        target_candidate_after_unmask = math.isfinite(target_after_unmask)
        train_valid_top50 = torch.topk(train_valid_mask_scores, k=50).indices
        train_valid_hit50 = bool((train_valid_top50 == test_item).any().item())
        mask_train_valid_hits += int(train_valid_hit50)

        rows.append(
            {
                "user_idx": user_idx,
                "train_seen_count": int(len(train_items)),
                "valid_item_idx": valid_item,
                "test_target_item_idx": test_item,
                "test_target_in_train_seen": bool(test_target_in_train),
                "test_target_in_valid_seen": bool(test_target_in_valid),
                "test_target_in_seen_mask": bool(test_target_in_seen_mask),
                "target_after_mask_before_unmask": target_after_mask_before_unmask,
                "target_after_unmask": target_after_unmask,
                "target_candidate_after_unmask": target_candidate_after_unmask,
                "valid_target_score": valid_score,
                "test_target_score": test_score,
                "valid_raw_rank": valid_raw_rank,
                "test_raw_rank": test_raw_rank,
                "test_hit50_train_mask": train_hit50,
                "test_hit50_train_valid_mask": train_valid_hit50,
            }
        )

    return {
        "sample_users": [int(x) for x in users.tolist()],
        "rows": rows,
        "test_hit50_train_mask": int(mask_train_hits),
        "test_hit50_train_valid_mask": int(mask_train_valid_hits),
        "sample_size": int(len(users)),
    }


def write_outputs(path: Path, json_path: Path, results: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    leakage = results["split_leakage"]
    popularity = results["target_popularity"]
    sample = results["sample_rank_and_mask"]
    lines = [
        "# Two-Tower valid-test gap 诊断报告",
        "",
        "## 1. Split 泄露检查",
        "",
        f"- test pair 出现在 train 中：{leakage['test_in_train']}",
        f"- test pair 出现在 valid 中：{leakage['test_in_valid']}",
        f"- valid pair 出现在 train 中：{leakage['valid_in_train']}",
        f"- valid users：{leakage['valid_users']}，其中单条 valid 的 users：{leakage['valid_users_with_one_row']}",
        f"- test users：{leakage['test_users']}，其中单条 test 的 users：{leakage['test_users_with_one_row']}",
        f"- valid 中有但 test 中没有的 users：{leakage['valid_users_not_in_test']}",
        f"- test 中有但 valid 中没有的 users：{leakage['test_users_not_in_valid']}",
        "",
        "## 2. Target popularity 分布",
        "",
        "| split | mean | median | p25 | p75 | p90 | <=5 | <=10 | <=20 | cold count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["valid", "test"]:
        stats = popularity[split]
        lines.append(
            f"| {split} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['p25']:.2f} | {stats['p75']:.2f} | {stats['p90']:.2f} | {stats['ratio_le_5']:.4f} | {stats['ratio_le_10']:.4f} | {stats['ratio_le_20']:.4f} | {stats['cold_item_count']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 小样本 mask 与 raw rank 诊断",
            "",
            f"- sample_size：{sample['sample_size']}",
            f"- test mask=train seen 的 hit@50 数：{sample['test_hit50_train_mask']}",
            f"- test mask=train+valid seen 的 hit@50 数：{sample['test_hit50_train_valid_mask']}",
            "",
            "| user_idx | train_seen | valid_item | test_item | test_in_train | test_in_valid | target_after_unmask_finite | valid_rank | test_rank | hit50_train | hit50_train_valid |",
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in sample["rows"]:
        lines.append(
            f"| {row['user_idx']} | {row['train_seen_count']} | {row['valid_item_idx']} | {row['test_target_item_idx']} | {row['test_target_in_train_seen']} | {row['test_target_in_valid_seen']} | {row['target_candidate_after_unmask']} | {row['valid_raw_rank']} | {row['test_raw_rank']} | {row['test_hit50_train_mask']} | {row['test_hit50_train_valid_mask']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    train_df, valid_df, test_df, stats = load_frames(Path(config["data_dir"]))
    leakage = split_leakage_diagnostics(train_df, valid_df, test_df)
    popularity = target_popularity_diagnostics(train_df, valid_df, test_df)
    users = sample_users(valid_df, test_df, int(args.sample_size), int(config["seed"]))
    device = torch.device("cuda" if str(config["device"]) == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_model(config, stats, Path(args.checkpoint), device)
    sample_diag = rank_and_mask_diagnostics(model, train_df, valid_df, test_df, stats, config, users, device)
    results = {
        "split_leakage": leakage,
        "target_popularity": popularity,
        "sample_rank_and_mask": sample_diag,
    }
    write_outputs(Path(args.output), Path(args.json_output), results)
    logging.info("诊断完成：%s / %s", args.output, args.json_output)


if __name__ == "__main__":
    main()

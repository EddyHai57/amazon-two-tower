#!/usr/bin/env python3
"""Train a lightweight user-history mean-pooling Two-Tower baseline."""

from __future__ import annotations

try:
    import argparse
    import csv
    import json
    import logging
    import math
    import random
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yaml
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    missing_name = exc.name or "未知依赖"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("缺少依赖：%s。请先在项目 .venv 中安装 package：%s", missing_name, package_hint)
    raise SystemExit(1) from exc


REQUIRED_CONFIG_KEYS = [
    "data_dir",
    "output_dir",
    "embedding_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "temperature",
    "use_l2_norm",
    "seed",
    "eval_k_list",
    "eval_batch_size",
    "num_workers",
    "device",
    "save_best_by",
    "history_max_len",
    "history_weight",
]
TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]
EVAL_COLUMNS = ["user_idx", "item_idx", "timestamp", "is_cold_item_for_eval"]
TRAIN_LOG_FIELDS = [
    "epoch",
    "train_loss",
    "valid_recall@20",
    "valid_recall@50",
    "valid_recall@100",
    "valid_ndcg@20",
    "valid_ndcg@50",
    "valid_ndcg@100",
    "valid_mrr@20",
    "valid_mrr@50",
    "valid_mrr@100",
    "learning_rate",
    "batch_size",
    "embedding_dim",
    "temperature",
    "use_l2_norm",
    "history_max_len",
    "history_weight",
    "epoch_time_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练用户历史 mean pooling Two-Tower baseline。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--smoke_test", action="store_true", help="运行 smoke test。")
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
    if int(config["num_workers"]) != 0:
        raise ValueError("当前脚本要求 num_workers=0，避免重复持有 parquet 数据。")
    if int(config["batch_size"]) <= 1:
        raise ValueError("batch_size 必须大于 1。")
    if float(config["temperature"]) <= 0:
        raise ValueError("temperature 必须大于 0。")
    if int(config["history_max_len"]) <= 0:
        raise ValueError("history_max_len 必须大于 0。")
    if float(config["history_weight"]) < 0:
        raise ValueError("history_weight 不能小于 0。")
    if not config["eval_k_list"]:
        raise ValueError("eval_k_list 不能为空。")


def apply_smoke_overrides(config: dict[str, Any], smoke_test: bool) -> dict[str, Any]:
    merged = dict(config)
    if smoke_test:
        merged["output_dir"] = "outputs/two_tower_movies_tv_5core_mean_pool_smoke"
        merged["epochs"] = 1
        merged["eval_max_users"] = 1000
        merged["smoke_train_batches"] = 2
    merged["smoke_test"] = bool(smoke_test)
    return merged


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("配置请求 cuda，但当前 CUDA 不可用，回退到 cpu。")
        return torch.device("cpu")
    return torch.device(requested)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} 缺少必需字段：{missing}")


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: dict[str, Any]


def load_data(data_dir: Path) -> DataBundle:
    logging.info("读取 train 数据：%s", data_dir / "train.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet", columns=TRAIN_COLUMNS)
    logging.info("读取 valid 数据：%s", data_dir / "valid.parquet")
    valid_df = pd.read_parquet(data_dir / "valid.parquet", columns=EVAL_COLUMNS)
    logging.info("读取 test 数据：%s", data_dir / "test.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet", columns=EVAL_COLUMNS)
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    require_columns(train_df, TRAIN_COLUMNS, "train")
    require_columns(valid_df, EVAL_COLUMNS, "valid")
    require_columns(test_df, EVAL_COLUMNS, "test")
    logging.info("train interactions 数量：%s", len(train_df))
    logging.info("valid interactions 数量：%s", len(valid_df))
    logging.info("test interactions 数量：%s", len(test_df))
    logging.info("n_users=%s n_items=%s", stats["n_users"], stats["n_items"])
    return DataBundle(train_df=train_df, valid_df=valid_df, test_df=test_df, stats=stats)


def build_history_matrix(frame: pd.DataFrame, num_users: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    histories = np.full((num_users, max_len), -1, dtype=np.int64)
    lengths = np.zeros(num_users, dtype=np.int64)
    ordered = frame.sort_values(["user_idx", "timestamp"], kind="mergesort")
    for user_idx, group in ordered.groupby("user_idx", sort=False):
        items = group["item_idx"].to_numpy(dtype=np.int64, copy=True)
        if items.size > max_len:
            items = items[-max_len:]
        user = int(user_idx)
        histories[user, : items.size] = items
        lengths[user] = int(items.size)
    return histories, lengths


class InteractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, users: np.ndarray, items: np.ndarray) -> None:
        self.users = torch.from_numpy(users.astype(np.int64, copy=False))
        self.items = torch.from_numpy(items.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index]


class MeanPoolCollator:
    def __init__(self, history_matrix: np.ndarray) -> None:
        self.history_matrix = history_matrix

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users = torch.stack([row[0] for row in batch])
        items = torch.stack([row[1] for row in batch])
        histories = torch.from_numpy(self.history_matrix[users.numpy()].copy())
        return users, items, histories


def make_dataloader(
    train_df: pd.DataFrame,
    history_matrix: np.ndarray,
    config: dict[str, Any],
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if config.get("smoke_test"):
        limit = int(config["batch_size"]) * int(config["smoke_train_batches"])
        train_df = train_df.head(limit).copy()
        logging.info("smoke test 只使用 train 前 %s 行。", len(train_df))
    users = train_df["user_idx"].to_numpy(dtype=np.int64, copy=True)
    items = train_df["item_idx"].to_numpy(dtype=np.int64, copy=True)
    dataset = InteractionDataset(users, items)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
        collate_fn=MeanPoolCollator(history_matrix),
    )


class MeanPoolTwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        use_l2_norm: bool,
        history_weight: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.use_l2_norm = use_l2_norm
        self.history_weight = float(history_weight)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)

    def mean_history_embedding(self, history_item_idx: torch.Tensor, exclude_item_idx: torch.Tensor | None = None) -> torch.Tensor:
        valid_mask = history_item_idx >= 0
        if exclude_item_idx is not None:
            valid_mask = valid_mask & (history_item_idx != exclude_item_idx.unsqueeze(1))
        safe_history = history_item_idx.clamp_min(0)
        history_emb = self.item_embedding(safe_history)
        mask = valid_mask.unsqueeze(-1).to(history_emb.dtype)
        summed = (history_emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def raw_user(self, user_idx: torch.Tensor, history_item_idx: torch.Tensor, exclude_item_idx: torch.Tensor | None = None) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        history_emb = self.mean_history_embedding(history_item_idx, exclude_item_idx)
        return user_emb + self.history_weight * history_emb

    def encode_users(
        self,
        user_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_emb = self.raw_user(user_idx, history_item_idx, exclude_item_idx)
        if self.use_l2_norm:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
        return user_emb

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_idx)
        if self.use_l2_norm:
            item_emb = F.normalize(item_emb, p=2, dim=-1)
        return item_emb

    def raw_batch(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.raw_user(user_idx, history_item_idx, exclude_item_idx=item_idx), self.item_embedding(item_idx)


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


def compute_logits(
    model: MeanPoolTwoTower,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_user_emb, raw_item_emb = model.raw_batch(user_idx, item_idx, history_item_idx)
    user_emb = F.normalize(raw_user_emb, p=2, dim=-1) if model.use_l2_norm else raw_user_emb
    item_emb = F.normalize(raw_item_emb, p=2, dim=-1) if model.use_l2_norm else raw_item_emb
    logits = (user_emb @ item_emb.T) / temperature
    return logits, raw_user_emb, raw_item_emb


def log_nan_diagnostics(logits: torch.Tensor, raw_user_emb: torch.Tensor, raw_item_emb: torch.Tensor) -> None:
    user_norm = raw_user_emb.norm(p=2, dim=-1)
    item_norm = raw_item_emb.norm(p=2, dim=-1)
    logging.error("user_emb norm min/max：%.6f / %.6f", float(user_norm.min()), float(user_norm.max()))
    logging.error("item_emb norm min/max：%.6f / %.6f", float(item_norm.min()), float(item_norm.max()))
    logging.error("similarity min/max：%.6f / %.6f", float(logits.min()), float(logits.max()))
    logging.error("logits has inf：%s", bool(torch.isinf(logits).any().item()))
    logging.error("logits has nan：%s", bool(torch.isnan(logits).any().item()))


def train_one_step(
    model: MeanPoolTwoTower,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
) -> tuple[float, float, float, int]:
    optimizer.zero_grad(set_to_none=True)
    logits, raw_user_emb, raw_item_emb = compute_logits(model, user_idx, item_idx, history_item_idx, temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        log_nan_diagnostics(logits, raw_user_emb, raw_item_emb)
        raise FloatingPointError("loss 出现 nan 或 inf，已停止训练。")
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(logits.min().item()), float(logits.max().item()), int(logits.shape[0])


def run_smoke_gradient_check(
    model: MeanPoolTwoTower,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    device: torch.device,
) -> None:
    user_idx, item_idx, history_item_idx = next(iter(train_loader))
    user_idx = user_idx.to(device)
    item_idx = item_idx.to(device)
    history_item_idx = history_item_idx.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    losses = []
    for _ in range(3):
        loss, _, _, effective_batch_size = train_one_step(
            model,
            optimizer,
            user_idx,
            item_idx,
            history_item_idx,
            float(config["temperature"]),
        )
        losses.append(loss)
    expected = math.log(effective_batch_size)
    lower = expected * 0.8
    upper = expected * 1.2
    logging.info(
        "smoke mini-batch loss check：losses=%s expected_log_batch=%.4f range=[%.4f, %.4f]",
        [round(loss, 6) for loss in losses],
        expected,
        lower,
        upper,
    )
    if not (lower <= losses[0] <= upper):
        raise RuntimeError("smoke test 失败：第一个 batch loss 不在 log(effective_batch_size) 附近。")
    if losses[-1] > losses[0] + 1e-4:
        raise RuntimeError("smoke test 失败：同一 mini-batch 连续训练 3 step 后 loss 未下降。")


def train_epoch(
    model: MeanPoolTwoTower,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float, int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    first_min = 0.0
    first_max = 0.0
    first_batch_size = 0
    for batch_idx, (user_idx, item_idx, history_item_idx) in enumerate(train_loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        history_item_idx = history_item_idx.to(device)
        loss, sim_min, sim_max, effective_batch_size = train_one_step(
            model,
            optimizer,
            user_idx,
            item_idx,
            history_item_idx,
            float(config["temperature"]),
        )
        if batch_idx == 0:
            first_min = sim_min
            first_max = sim_max
            first_batch_size = effective_batch_size
            logging.info(
                "epoch %s 第一个 batch similarity min/max：%.6f / %.6f，effective_batch_size=%s",
                epoch,
                sim_min,
                sim_max,
                effective_batch_size,
            )
        total_loss += loss * effective_batch_size
        total_examples += effective_batch_size
    return total_loss / total_examples, first_min, first_max, first_batch_size


def prepare_eval_frame(eval_df: pd.DataFrame, eval_max_users: int | None) -> pd.DataFrame:
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    if eval_max_users is not None:
        non_cold = non_cold.head(int(eval_max_users)).copy()
    return non_cold


def encode_all_items_cpu(model: MeanPoolTwoTower, num_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + 65536, num_items)
            item_idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(item_idx).detach().cpu())
    return torch.cat(chunks, dim=0)


def evaluate_once(
    model: MeanPoolTwoTower,
    eval_df: pd.DataFrame,
    history_matrix: np.ndarray,
    seen_items: dict[int, set[int]],
    config: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    eval_max_users = config.get("eval_max_users")
    eval_targets = prepare_eval_frame(eval_df, int(eval_max_users) if eval_max_users is not None else None)
    k_list = [int(k) for k in config["eval_k_list"]]
    max_k = max(k_list)
    num_items = int(stats["n_items"])
    eval_batch_size = int(config["eval_batch_size"])
    item_emb_cpu = encode_all_items_cpu(model, num_items, device)

    logging.info(
        "%s eval users 数量：%s，eval_max_users=%s，eval_batch_size=%s",
        split_name,
        len(eval_targets),
        eval_max_users,
        eval_batch_size,
    )

    metric_sums = {f"recall@{k}": 0.0 for k in k_list}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in k_list})
    metric_sums.update({f"mrr@{k}": 0.0 for k in k_list})
    diagnostics: dict[str, Any] = {}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_targets), eval_batch_size):
            batch = eval_targets.iloc[start : start + eval_batch_size]
            users_np = batch["user_idx"].to_numpy(dtype=np.int64, copy=True)
            user_tensor = torch.as_tensor(users_np, device=device)
            target_tensor = torch.as_tensor(batch["item_idx"].to_numpy(dtype=np.int64, copy=True), device=device)
            history_tensor = torch.as_tensor(history_matrix[users_np], dtype=torch.long, device=device)
            user_emb = model.encode_users(user_tensor, history_tensor)
            item_emb = item_emb_cpu.to(device)
            scores = (user_emb @ item_emb.T) / float(config["temperature"])

            row_indices = torch.arange(scores.shape[0], device=device)
            target_scores = scores[row_indices, target_tensor].clone()
            candidate_counts = []
            for row_pos, (user_idx, target_item) in enumerate(zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)):
                seen = seen_items.get(int(user_idx), set())
                if seen:
                    seen_tensor = torch.as_tensor(list(seen), dtype=torch.long, device=device)
                    scores[row_pos, seen_tensor] = -torch.inf
                scores[row_pos, int(target_item)] = target_scores[row_pos]
                candidate_counts.append(num_items - len(seen) + (1 if int(target_item) in seen else 0))

            topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
            targets = batch["item_idx"].to_numpy(dtype=np.int64)
            if start == 0:
                diagnostics = {
                    "eval_users": int(len(eval_targets)),
                    "target_item_in_candidate_range": bool(((targets >= 0) & (targets < num_items)).all()),
                    "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
                    "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
                    "topk_shape": list(topk.shape),
                }

            for target_item, rec_items in zip(targets, topk, strict=True):
                matched = np.where(rec_items == target_item)[0]
                if matched.size == 0:
                    continue
                rank = int(matched[0]) + 1
                for k in k_list:
                    if rank <= k:
                        metric_sums[f"recall@{k}"] += 1.0
                        metric_sums[f"ndcg@{k}"] += 1.0 / math.log2(rank + 1)
                        metric_sums[f"mrr@{k}"] += 1.0 / rank

    denom = len(eval_targets)
    metrics: dict[str, Any] = {
        "split": split_name,
        "num_eval_users": int(denom),
        "num_skipped_cold_users": int(eval_df["is_cold_item_for_eval"].astype(bool).sum()),
        "eval_max_users": eval_max_users,
        "eval_batch_size": eval_batch_size,
        "diagnostics": diagnostics,
    }
    for key, value in metric_sums.items():
        metrics[key] = value / denom if denom else 0.0
    return metrics


def evaluate_with_oom_retry(
    model: MeanPoolTwoTower,
    eval_df: pd.DataFrame,
    history_matrix: np.ndarray,
    seen_items: dict[int, set[int]],
    config: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    try:
        return evaluate_once(model, eval_df, history_matrix, seen_items, config, stats, device, split_name)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message or int(config["eval_batch_size"]) <= 128:
            raise
        old_batch_size = int(config["eval_batch_size"])
        config["eval_batch_size"] = 128
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.warning("评估阶段 OOM，eval_batch_size 从 %s 降到 128 后重试一次。", old_batch_size)
        return evaluate_once(model, eval_df, history_matrix, seen_items, config, stats, device, split_name)


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith(("recall@", "ndcg@", "mrr@")):
            output[f"{prefix}_{key}"] = value
    return output


def init_train_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS)
        writer.writeheader()


def append_train_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in TRAIN_LOG_FIELDS})


def save_checkpoint(
    path: Path,
    model: MeanPoolTwoTower,
    config: dict[str, Any],
    stats: dict[str, Any],
    epoch: int,
    metric_name: str,
    metric_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stats": stats,
            "epoch": epoch,
            "best_metric_name": metric_name,
            "best_metric_value": metric_value,
        },
        path,
    )


def write_report(
    path: Path,
    config: dict[str, Any],
    stats: dict[str, Any],
    best_epoch: int,
    best_valid_recall: float,
    valid_metrics: dict[str, Any] | None,
    test_skipped_reason: str | None,
) -> None:
    lines = [
        "# Movies_and_TV 5-core Mean Pooling Two-Tower 运行报告",
        "",
        "## 1. 运行目的",
        "",
        "本次运行训练轻量用户历史 mean pooling Two-Tower：用户向量为 user_id embedding 加上训练历史 item embedding 均值。",
        "",
        "## 2. 数据与配置",
        "",
        f"- data_dir：`{config['data_dir']}`",
        f"- output_dir：`{config['output_dir']}`",
        f"- users：{stats['n_users']}",
        f"- items：{stats['n_items']}",
        f"- train interactions：{stats['n_interactions_train']}",
        f"- embedding_dim：{config['embedding_dim']}",
        f"- batch_size：{config['batch_size']}",
        f"- temperature：{config['temperature']}",
        f"- use_l2_norm：{config['use_l2_norm']}",
        f"- history_max_len：{config['history_max_len']}",
        f"- history_weight：{config['history_weight']}",
        "",
        "## 3. 训练口径",
        "",
        "- training history 来自 train split，并在每个训练样本中排除当前正样本 item。",
        "- valid history 来自 train split。",
        "- test history 将来自 train + valid split；本次 limited 训练不运行 test。",
        "",
        "## 4. 训练结果",
        "",
        f"- best_epoch：{best_epoch}",
        f"- best_valid_recall@50：{best_valid_recall:.6f}",
    ]
    if valid_metrics:
        lines.extend(
            [
                "",
                "## 5. Valid 指标",
                "",
                f"- num_eval_users：{valid_metrics['num_eval_users']}",
                "",
                "| K | Recall | NDCG | MRR |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for k in config["eval_k_list"]:
            lines.append(
                f"| {k} | {valid_metrics[f'recall@{k}']:.6f} | {valid_metrics[f'ndcg@{k}']:.6f} | {valid_metrics[f'mrr@{k}']:.6f} |"
            )
    if test_skipped_reason:
        lines.extend(["", "## 6. Test 指标", "", f"- 未生成 test metrics：{test_skipped_reason}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(config: dict[str, Any]) -> dict[str, Any]:
    require_config(config)
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    device = resolve_device(str(config["device"]))
    logging.info("device=%s", device)
    logging.info(
        "embedding_dim=%s batch_size=%s temperature=%s history_max_len=%s history_weight=%s",
        config["embedding_dim"],
        config["batch_size"],
        config["temperature"],
        config["history_max_len"],
        config["history_weight"],
    )

    bundle = load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    history_max_len = int(config["history_max_len"])
    train_history_matrix, train_history_lengths = build_history_matrix(bundle.train_df, num_users, history_max_len)
    logging.info(
        "history matrix 完成：train_non_empty_users=%s train_avg_len=%.4f max_len=%s",
        int((train_history_lengths > 0).sum()),
        float(train_history_lengths.mean()),
        history_max_len,
    )

    train_loader = make_dataloader(bundle.train_df, train_history_matrix, config)
    train_seen = build_seen_items(bundle.train_df)
    model = MeanPoolTwoTower(
        num_users=num_users,
        num_items=int(bundle.stats["n_items"]),
        embedding_dim=int(config["embedding_dim"]),
        use_l2_norm=bool(config["use_l2_norm"]),
        history_weight=float(config["history_weight"]),
    ).to(device)

    if config.get("smoke_test"):
        run_smoke_gradient_check(model, train_loader, config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_log_path = output_dir / "train_log.csv"
    init_train_log(train_log_path)

    best_epoch = 0
    best_valid_recall = -1.0
    best_valid_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(config["epochs"]) + 1):
        logging.info("epoch %s 开始训练。", epoch)
        start_time = time.time()
        train_loss, _, _, _ = train_epoch(model, train_loader, optimizer, config, device, epoch)
        valid_metrics = evaluate_with_oom_retry(
            model,
            bundle.valid_df,
            train_history_matrix,
            train_seen,
            config,
            bundle.stats,
            device,
            split_name="valid",
        )
        epoch_time = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(config["learning_rate"]),
            "batch_size": int(config["batch_size"]),
            "embedding_dim": int(config["embedding_dim"]),
            "temperature": float(config["temperature"]),
            "use_l2_norm": bool(config["use_l2_norm"]),
            "history_max_len": int(config["history_max_len"]),
            "history_weight": float(config["history_weight"]),
            "epoch_time_seconds": epoch_time,
            **prefixed_metrics(valid_metrics, "valid"),
        }
        append_train_log(train_log_path, row)
        logging.info(
            "epoch %s 完成：train_loss=%.6f valid_recall@50=%.6f valid_ndcg@50=%.6f valid_mrr@50=%.6f epoch_time=%.2fs",
            epoch,
            train_loss,
            valid_metrics["recall@50"],
            valid_metrics["ndcg@50"],
            valid_metrics["mrr@50"],
            epoch_time,
        )
        if valid_metrics["recall@50"] > best_valid_recall:
            best_epoch = epoch
            best_valid_recall = float(valid_metrics["recall@50"])
            best_valid_metrics = valid_metrics
            save_checkpoint(
                output_dir / "checkpoints" / "best_model.pt",
                model,
                config,
                bundle.stats,
                epoch,
                "valid_recall@50",
                best_valid_recall,
            )
            logging.info("更新 best checkpoint：epoch=%s valid_recall@50=%.6f", epoch, best_valid_recall)

    if best_valid_metrics is not None:
        best_valid_metrics = {
            **best_valid_metrics,
            "best_epoch": best_epoch,
            "best_valid_recall@50": best_valid_recall,
        }
        write_json(output_dir / "metrics_valid.json", best_valid_metrics)

    test_skipped_reason = "本轮只做 mean pooling 轻量训练和 limited valid eval，不运行 full test。"
    write_report(
        output_dir / "mean_pool_two_tower_run_report.md",
        config,
        bundle.stats,
        best_epoch,
        best_valid_recall,
        best_valid_metrics,
        test_skipped_reason,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_valid_recall@50": best_valid_recall,
        "output_dir": str(output_dir),
        "test_skipped_reason": test_skipped_reason,
    }
    logging.info("mean pooling 训练完成：%s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    train(apply_smoke_overrides(config, args.smoke_test))


if __name__ == "__main__":
    main()

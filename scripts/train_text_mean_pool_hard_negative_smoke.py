#!/usr/bin/env python3
"""Hard Negative Mining smoke on Text + Mean Pooling Two-Tower.

SMOKE ONLY — not the final model. Does not replace the current best model
(Text + Mean Pooling tau=0.15, full test Recall@50=0.076337).

HNM constructs hard negatives via item text-embedding cosine similarity.
An auxiliary HNM cross-entropy loss is added to the original in-batch loss.
Hard negatives are NOT used for true new-user cold start; they are intended
to improve fine-grained discrimination between similar items.
"""

from __future__ import annotations

try:
    import argparse
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
    missing_name = exc.name or "unknown"
    package_hint = "pyyaml" if missing_name == "yaml" else missing_name
    logging.error("Missing dependency: %s. Install package: %s", missing_name, package_hint)
    raise SystemExit(1) from exc


REQUIRED_CONFIG_KEYS = [
    "data_dir", "output_dir", "embedding_dim", "batch_size",
    "learning_rate", "weight_decay", "epochs", "temperature",
    "use_l2_norm", "seed", "eval_k_list", "eval_batch_size",
    "eval_max_users", "num_workers", "device", "save_best_by",
    "history_max_len", "history_weight", "item_text_embedding_path",
    "item_has_text_path", "text_proj_dim", "use_has_text_mask", "item_fusion",
    "lambda_hn", "hard_negatives_per_sample", "hn_top_k",
]
TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]
EVAL_COLUMNS = ["user_idx", "item_idx", "timestamp", "is_cold_item_for_eval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HNM smoke: Text + Mean Pooling Two-Tower.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {config_path}")
    return config


def require_config(config: dict[str, Any]) -> None:
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise KeyError(f"Config missing required field: {key}")
    if int(config["num_workers"]) != 0:
        raise ValueError("num_workers must be 0.")
    if int(config["batch_size"]) <= 1:
        raise ValueError("batch_size must be > 1.")
    if float(config["temperature"]) <= 0:
        raise ValueError("temperature must be > 0.")
    if int(config["history_max_len"]) <= 0:
        raise ValueError("history_max_len must be > 0.")
    if float(config["history_weight"]) < 0:
        raise ValueError("history_weight must be >= 0.")
    if int(config["text_proj_dim"]) != int(config["embedding_dim"]):
        raise ValueError("additive item fusion requires text_proj_dim == embedding_dim.")
    if str(config["item_fusion"]) != "additive":
        raise ValueError("Only additive item_fusion is supported.")
    if float(config["lambda_hn"]) < 0:
        raise ValueError("lambda_hn must be >= 0.")
    if int(config["hard_negatives_per_sample"]) < 1:
        raise ValueError("hard_negatives_per_sample must be >= 1.")
    if int(config["hn_top_k"]) < int(config["hard_negatives_per_sample"]):
        raise ValueError("hn_top_k must be >= hard_negatives_per_sample.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("cuda unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: dict[str, Any]


def load_data(data_dir: Path) -> DataBundle:
    train_df = pd.read_parquet(data_dir / "train.parquet", columns=TRAIN_COLUMNS)
    valid_df = pd.read_parquet(data_dir / "valid.parquet", columns=EVAL_COLUMNS)
    test_df = pd.read_parquet(data_dir / "test.parquet", columns=EVAL_COLUMNS)
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    require_columns(train_df, TRAIN_COLUMNS, "train")
    require_columns(valid_df, EVAL_COLUMNS, "valid")
    require_columns(test_df, EVAL_COLUMNS, "test")
    logging.info("n_users=%s n_items=%s train=%s", stats["n_users"], stats["n_items"], len(train_df))
    return DataBundle(train_df=train_df, valid_df=valid_df, test_df=test_df, stats=stats)


def build_history_matrix(frame: pd.DataFrame, num_users: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    histories = np.full((num_users, max_len), -1, dtype=np.int64)
    lengths = np.zeros(num_users, dtype=np.int64)
    ordered = frame.sort_values(["user_idx", "timestamp"], kind="mergesort")
    for user_idx, group in ordered.groupby("user_idx", sort=False):
        items = group["item_idx"].to_numpy(dtype=np.int64, copy=True)
        if items.size > max_len:
            items = items[-max_len:]
        u = int(user_idx)
        histories[u, : items.size] = items
        lengths[u] = int(items.size)
    return histories, lengths


class InteractionDataset(Dataset):
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


def make_dataloader(train_df: pd.DataFrame, history_matrix: np.ndarray, config: dict[str, Any]) -> DataLoader:
    users = train_df["user_idx"].to_numpy(dtype=np.int64, copy=True)
    items = train_df["item_idx"].to_numpy(dtype=np.int64, copy=True)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        InteractionDataset(users, items),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
        collate_fn=MeanPoolCollator(history_matrix),
    )


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for user_idx, group in frame.groupby("user_idx", sort=False):
        seen[int(user_idx)] = set(int(i) for i in group["item_idx"].tolist())
    return seen


def load_text_artifacts(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    emb_path = Path(config["item_text_embedding_path"])
    has_path = Path(config["item_has_text_path"])
    if not emb_path.exists():
        raise FileNotFoundError(f"item_text_embedding_path not found: {emb_path}")
    if not has_path.exists():
        raise FileNotFoundError(f"item_has_text_path not found: {has_path}")
    text_emb = torch.from_numpy(np.load(emb_path).astype(np.float32))
    has_text = torch.from_numpy(np.load(has_path).astype(np.float32))
    logging.info(
        "text_emb shape=%s has_text=1: %d/%d (%.1f%%)",
        tuple(text_emb.shape), int(has_text.sum()), len(has_text),
        100.0 * float(has_text.mean()),
    )
    return text_emb, has_text


class TextMeanPoolTwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        text_emb: torch.Tensor,
        has_text: torch.Tensor,
        text_proj_dim: int,
        use_l2_norm: bool,
        use_has_text_mask: bool,
        history_weight: float,
    ) -> None:
        super().__init__()
        if text_proj_dim != embedding_dim:
            raise ValueError("additive item fusion requires text_proj_dim == embedding_dim.")
        self.use_l2_norm = bool(use_l2_norm)
        self.use_has_text_mask = bool(use_has_text_mask)
        self.history_weight = float(history_weight)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.text_proj = nn.Linear(int(text_emb.shape[1]), embedding_dim, bias=False)
        self.register_buffer("_text_emb", text_emb.float(), persistent=False)
        self.register_buffer("_has_text", has_text.float(), persistent=False)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_id_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.text_proj.weight)

    def mean_history_embedding(self, history_item_idx: torch.Tensor, exclude_item_idx: torch.Tensor | None = None) -> torch.Tensor:
        valid_mask = history_item_idx >= 0
        if exclude_item_idx is not None:
            valid_mask = valid_mask & (history_item_idx != exclude_item_idx.unsqueeze(1))
        safe_history = history_item_idx.clamp_min(0)
        history_emb = self.item_id_embedding(safe_history)
        mask = valid_mask.unsqueeze(-1).to(history_emb.dtype)
        summed = (history_emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def raw_user(self, user_idx: torch.Tensor, history_item_idx: torch.Tensor, exclude_item_idx: torch.Tensor | None = None) -> torch.Tensor:
        id_emb = self.user_embedding(user_idx)
        hist_pool = self.mean_history_embedding(history_item_idx, exclude_item_idx)
        return id_emb + self.history_weight * hist_pool

    def _item_prenorm(self, item_idx: torch.Tensor) -> torch.Tensor:
        id_emb = self.item_id_embedding(item_idx)
        txt_proj = self.text_proj(self._text_emb[item_idx])
        if self.use_has_text_mask:
            txt_proj = txt_proj * self._has_text[item_idx].unsqueeze(-1)
        return id_emb + txt_proj

    def encode_users(self, user_idx: torch.Tensor, history_item_idx: torch.Tensor, exclude_item_idx: torch.Tensor | None = None) -> torch.Tensor:
        user_emb = self.raw_user(user_idx, history_item_idx, exclude_item_idx)
        return F.normalize(user_emb, p=2, dim=-1) if self.use_l2_norm else user_emb

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_prenorm(item_idx)
        return F.normalize(item_emb, p=2, dim=-1) if self.use_l2_norm else item_emb

    def raw_batch(self, user_idx: torch.Tensor, item_idx: torch.Tensor, history_item_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.raw_user(user_idx, history_item_idx, exclude_item_idx=item_idx), self._item_prenorm(item_idx)


def compute_logits(
    model: TextMeanPoolTwoTower,
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


def encode_all_items_cpu(model: TextMeanPoolTwoTower, num_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + 65536, num_items)
            item_idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(item_idx).detach().cpu())
    return torch.cat(chunks, dim=0)


def prepare_eval_frame(eval_df: pd.DataFrame, eval_max_users: int | None) -> pd.DataFrame:
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    if eval_max_users is not None:
        non_cold = non_cold.head(int(eval_max_users)).copy()
    return non_cold


def evaluate_once(
    model: TextMeanPoolTwoTower,
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
    logging.info("%s eval users=%s eval_batch_size=%s", split_name, len(eval_targets), eval_batch_size)

    metric_sums = {f"recall@{k}": 0.0 for k in k_list}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in k_list})
    metric_sums.update({f"mrr@{k}": 0.0 for k in k_list})
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
            for row_pos, (user_idx_val, target_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)
            ):
                seen = seen_items.get(int(user_idx_val), set())
                if seen:
                    scores[row_pos, torch.as_tensor(list(seen), dtype=torch.long, device=device)] = -torch.inf
                scores[row_pos, int(target_item)] = target_scores[row_pos]

            topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
            targets = batch["item_idx"].to_numpy(dtype=np.int64)
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
    }
    for key, value in metric_sums.items():
        metrics[key] = value / denom if denom else 0.0
    return metrics


def evaluate_with_oom_retry(
    model: TextMeanPoolTwoTower,
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
        if "out of memory" not in str(exc).lower() or int(config["eval_batch_size"]) <= 128:
            raise
        config["eval_batch_size"] = 128
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.warning("OOM; retrying with eval_batch_size=128.")
        return evaluate_once(model, eval_df, history_matrix, seen_items, config, stats, device, split_name)


def build_model(config: dict[str, Any], stats: dict[str, Any], device: torch.device) -> TextMeanPoolTwoTower:
    text_emb, has_text = load_text_artifacts(config)
    model = TextMeanPoolTwoTower(
        num_users=int(stats["n_users"]),
        num_items=int(stats["n_items"]),
        embedding_dim=int(config["embedding_dim"]),
        text_emb=text_emb,
        has_text=has_text,
        text_proj_dim=int(config["text_proj_dim"]),
        use_l2_norm=bool(config["use_l2_norm"]),
        use_has_text_mask=bool(config["use_has_text_mask"]),
        history_weight=float(config["history_weight"]),
    ).to(device)
    logging.info("TextMeanPoolTwoTower trainable params: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# ---- HNM-specific functions ----

def build_hard_negative_table(text_emb_np: np.ndarray, top_k: int, device: torch.device) -> np.ndarray:
    """Precompute top-K text-cosine-similar items per item, excluding self.

    Returns int32 ndarray of shape (n_items, top_k).
    Chunked computation to avoid GPU OOM.
    """
    n_items = text_emb_np.shape[0]
    text_t = F.normalize(
        torch.from_numpy(text_emb_np.astype(np.float32)).to(device), p=2, dim=-1
    )
    hn_table = np.zeros((n_items, top_k), dtype=np.int32)
    chunk_size = 512
    logging.info("Building HN table: n_items=%s top_k=%s chunk_size=%s", n_items, top_k, chunk_size)
    with torch.no_grad():
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            chunk = text_t[start:end]          # (C, D)
            sim = chunk @ text_t.T             # (C, n_items)
            for i in range(end - start):
                sim[i, start + i] = -1e9       # exclude self
            topk_idx = torch.topk(sim, k=top_k, dim=1).indices
            hn_table[start:end] = topk_idx.cpu().numpy().astype(np.int32)
            if (start // chunk_size) % 60 == 0:
                logging.info("HN table progress: %d/%d items", end, n_items)
    logging.info("HN table built: shape=%s dtype=%s", hn_table.shape, hn_table.dtype)
    return hn_table


def compute_hn_loss(
    model: TextMeanPoolTwoTower,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    hn_table: np.ndarray,
    hard_negatives_per_sample: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Cross entropy over [positive + hard_negatives], label = 0 (positive).

    Excludes self and user train history from hard negative candidates.
    Returns (loss_tensor, num_valid_samples).
    """
    B = user_idx.shape[0]
    device = user_idx.device
    item_idx_cpu = item_idx.cpu().numpy()
    history_np = history_item_idx.cpu().numpy()

    selected_hn = np.full((B, hard_negatives_per_sample), 0, dtype=np.int64)
    valid_mask = np.zeros(B, dtype=bool)

    for i in range(B):
        candidates = hn_table[item_idx_cpu[i]].tolist()
        hist_set = set(int(x) for x in history_np[i] if x >= 0)
        hist_set.add(int(item_idx_cpu[i]))     # also exclude positive item itself
        available = [c for c in candidates if c not in hist_set]
        if len(available) >= hard_negatives_per_sample:
            selected_hn[i, :hard_negatives_per_sample] = available[:hard_negatives_per_sample]
            valid_mask[i] = True

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return torch.tensor(0.0, device=device), 0

    vi = np.where(valid_mask)[0]
    valid_users = user_idx[vi]
    valid_items = item_idx[vi]
    valid_hist = history_item_idx[vi]
    valid_hn = torch.from_numpy(selected_hn[vi]).long().to(device)  # (V, HN)
    V = len(vi)

    user_emb = model.encode_users(valid_users, valid_hist, exclude_item_idx=valid_items)  # (V, D)
    pos_emb = model.encode_items(valid_items)                                              # (V, D)
    hn_flat = valid_hn.reshape(-1)                                                         # (V*HN,)
    hn_emb = model.encode_items(hn_flat).reshape(V, hard_negatives_per_sample, -1)        # (V, HN, D)

    pos_score = (user_emb * pos_emb).sum(-1, keepdim=True) / temperature                  # (V, 1)
    hn_scores = (hn_emb * user_emb.unsqueeze(1)).sum(-1) / temperature                    # (V, HN)

    all_scores = torch.cat([pos_score, hn_scores], dim=1)                                 # (V, 1+HN)
    labels = torch.zeros(V, dtype=torch.long, device=device)
    return F.cross_entropy(all_scores, labels), n_valid


def train_one_step_hnm(
    model: TextMeanPoolTwoTower,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
    lambda_hn: float,
    hn_table: np.ndarray,
    hard_negatives_per_sample: int,
) -> tuple[float, float, float, int]:
    optimizer.zero_grad(set_to_none=True)

    logits, raw_user_emb, raw_item_emb = compute_logits(
        model, user_idx, item_idx, history_item_idx, temperature
    )
    labels = torch.arange(logits.shape[0], device=logits.device)
    main_loss = F.cross_entropy(logits, labels)

    if not torch.isfinite(main_loss):
        raise FloatingPointError(f"main_loss nan/inf: {float(main_loss):.4f}")

    hn_loss_val, n_valid = compute_hn_loss(
        model, user_idx, item_idx, history_item_idx,
        hn_table, hard_negatives_per_sample, temperature
    )

    total_loss = main_loss + lambda_hn * hn_loss_val

    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"total_loss nan/inf: {float(total_loss):.4f}")

    total_loss.backward()
    optimizer.step()

    hn_scalar = float(hn_loss_val.item()) if isinstance(hn_loss_val, torch.Tensor) else float(hn_loss_val)
    return float(total_loss.item()), float(main_loss.item()), hn_scalar, n_valid


def train_epoch_hnm(
    model: TextMeanPoolTwoTower,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    hn_table: np.ndarray,
) -> tuple[float, float, float]:
    model.train()
    total_loss = total_main = total_hn = 0.0
    total_examples = 0
    total_valid_hn = 0
    temperature = float(config["temperature"])
    lambda_hn = float(config["lambda_hn"])
    hn_per = int(config["hard_negatives_per_sample"])

    for batch_idx, (user_idx, item_idx, history_item_idx) in enumerate(train_loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        history_item_idx = history_item_idx.to(device)

        t_loss, m_loss, h_loss, n_valid = train_one_step_hnm(
            model, optimizer, user_idx, item_idx, history_item_idx,
            temperature, lambda_hn, hn_table, hn_per
        )
        bs = int(user_idx.shape[0])
        total_loss += t_loss * bs
        total_main += m_loss * bs
        total_hn += h_loss * bs
        total_examples += bs
        total_valid_hn += n_valid

        if batch_idx == 0:
            logging.info(
                "epoch %s batch 0: total_loss=%.4f main_loss=%.4f hn_loss=%.4f hn_valid=%d/%d",
                epoch, t_loss, m_loss, h_loss, n_valid, bs
            )

    logging.info(
        "epoch %s summary: total_valid_hn=%d / total_examples=%d (ratio=%.3f)",
        epoch, total_valid_hn, total_examples,
        total_valid_hn / total_examples if total_examples > 0 else 0.0
    )
    return (
        total_loss / total_examples,
        total_main / total_examples,
        total_hn / total_examples,
    )


def train_hnm_smoke(config: dict[str, Any]) -> None:
    require_config(config)
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)
    device = resolve_device(str(config["device"]))

    logging.info(
        "HNM smoke: device=%s temperature=%.2f lambda_hn=%.2f "
        "hard_negatives_per_sample=%s hn_top_k=%s epochs=%s eval_max_users=%s",
        device, config["temperature"], config["lambda_hn"],
        config["hard_negatives_per_sample"], config["hn_top_k"],
        config["epochs"], config["eval_max_users"],
    )

    bundle = load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    history_max_len = int(config["history_max_len"])
    train_history_matrix, _ = build_history_matrix(bundle.train_df, num_users, history_max_len)
    train_loader = make_dataloader(bundle.train_df, train_history_matrix, config)
    train_seen = build_seen_items(bundle.train_df)
    model = build_model(config, bundle.stats, device)

    # Precompute hard negative table from text embeddings (one-time cost)
    text_emb_np = np.load(Path(config["item_text_embedding_path"])).astype(np.float32)
    hn_top_k = int(config["hn_top_k"])
    hn_table = build_hard_negative_table(text_emb_np, hn_top_k, device)
    del text_emb_np

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    results = []
    for epoch in range(1, int(config["epochs"]) + 1):
        logging.info("epoch %s start", epoch)
        t0 = time.time()
        train_loss, main_loss, hn_loss = train_epoch_hnm(
            model, train_loader, optimizer, config, device, epoch, hn_table
        )
        valid_metrics = evaluate_with_oom_retry(
            model, bundle.valid_df, train_history_matrix, train_seen,
            config, bundle.stats, device, split_name="valid"
        )
        epoch_time = time.time() - t0
        logging.info(
            "epoch %s done: total_loss=%.6f main_loss=%.6f hn_loss=%.6f "
            "valid_recall@20=%.6f valid_recall@50=%.6f valid_recall@100=%.6f "
            "valid_ndcg@50=%.6f valid_mrr@50=%.6f time=%.2fs",
            epoch, train_loss, main_loss, hn_loss,
            valid_metrics["recall@20"], valid_metrics["recall@50"],
            valid_metrics["recall@100"], valid_metrics["ndcg@50"],
            valid_metrics["mrr@50"], epoch_time,
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_main_loss": main_loss,
            "train_hn_loss": hn_loss,
            "epoch_time_seconds": epoch_time,
            **{k: v for k, v in valid_metrics.items() if k.startswith(("recall@", "ndcg@", "mrr@"))},
        }
        results.append(row)

        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
            ckpt_dir / f"epoch_{epoch}.pt",
        )

    write_json(output_dir / "hnm_smoke_results.json", results)

    final = results[-1]
    baseline_r50 = 0.107460  # Text+MP tau=0.15 epoch 1 limited valid Recall@50
    delta = final["recall@50"] - baseline_r50
    logging.info(
        "HNM smoke complete.\n"
        "  Recall@20=%.6f  Recall@50=%.6f  Recall@100=%.6f\n"
        "  NDCG@50=%.6f  MRR@50=%.6f\n"
        "  Baseline Text+MP tau=0.15 epoch1 Recall@50=%.6f\n"
        "  Delta Recall@50=%+.6f",
        final["recall@20"], final["recall@50"], final["recall@100"],
        final["ndcg@50"], final["mrr@50"],
        baseline_r50, delta,
    )
    summary = {
        "smoke_only": True,
        "baseline_text_mp_tau015_epoch1_recall50": baseline_r50,
        "hnm_epoch1_recall50": final["recall@50"],
        "delta_recall50": delta,
        "lambda_hn": config["lambda_hn"],
        "hard_negatives_per_sample": config["hard_negatives_per_sample"],
        "hn_top_k": hn_top_k,
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    config["config_path"] = args.config
    train_hnm_smoke(config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attention Pooling smoke — paired comparison with Time-decay Mean Pool.

Supports two pooling_type values (controlled by config):
  time_decay  — exponential decay weights (same as final main model)
  attention   — scaled dot-product attention, query = user_id_emb

All other components (item tower, optimizer, temperature, eval code, seed) are
identical between the two modes, enabling a clean paired comparison.

Usage
-----
Training (one model at a time):
  python scripts/train_attention_pooling_smoke.py --config configs/attention_pooling_smoke_td_baseline.yaml
  python scripts/train_attention_pooling_smoke.py --config configs/attention_pooling_smoke_attention.yaml

Comparison (after both training runs complete):
  python scripts/train_attention_pooling_smoke.py --compare \
    --td_dir outputs/attention_pooling_smoke/time_decay_baseline \
    --attn_dir outputs/attention_pooling_smoke/attention_pooling \
    --out_dir outputs/attention_pooling_smoke
"""

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
    pkg = "pyyaml" if (exc.name or "") == "yaml" else (exc.name or "unknown")
    logging.error("Missing dependency: %s.  Install in .venv: %s", exc.name, pkg)
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = [
    "data_dir", "output_dir", "embedding_dim", "batch_size", "learning_rate",
    "weight_decay", "epochs", "temperature", "use_l2_norm", "seed",
    "eval_k_list", "eval_batch_size", "eval_max_users", "num_workers", "device",
    "save_best_by", "history_max_len", "history_weight",
    "item_text_embedding_path", "item_has_text_path", "text_proj_dim",
    "use_has_text_mask", "item_fusion", "pooling_type", "decay_rate",
]
TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]
EVAL_COLUMNS  = ["user_idx", "item_idx", "timestamp", "is_cold_item_for_eval"]
TRAIN_LOG_FIELDS = [
    "epoch", "train_loss", "valid_recall@20", "valid_recall@50",
    "valid_recall@100", "valid_ndcg@50", "valid_mrr@50",
    "epoch_time_seconds", "pooling_type",
]
HISTORY_BUCKETS = [("le5", lambda l: l <= 5), ("6to20", lambda l: 6 <= l <= 20), ("gt20", lambda l: l > 20)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",    help="YAML config for training mode.")
    p.add_argument("--compare",   action="store_true", help="Comparison mode.")
    p.add_argument("--td_dir",    default="outputs/attention_pooling_smoke/time_decay_baseline")
    p.add_argument("--attn_dir",  default="outputs/attention_pooling_smoke/attention_pooling")
    p.add_argument("--out_dir",   default="outputs/attention_pooling_smoke")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def require_config(cfg: dict[str, Any]) -> None:
    for k in REQUIRED_CONFIG_KEYS:
        if k not in cfg:
            raise KeyError(f"Config missing: {k}")
    if int(cfg["num_workers"]) != 0:
        raise ValueError("num_workers must be 0.")
    if int(cfg["batch_size"]) <= 1:
        raise ValueError("batch_size must be > 1.")
    if float(cfg["temperature"]) <= 0:
        raise ValueError("temperature must be > 0.")
    if str(cfg["item_fusion"]) != "additive":
        raise ValueError("Only additive item_fusion supported.")
    if int(cfg["text_proj_dim"]) != int(cfg["embedding_dim"]):
        raise ValueError("text_proj_dim must equal embedding_dim for additive fusion.")
    ptype = str(cfg["pooling_type"])
    if ptype not in ("time_decay", "attention"):
        raise ValueError(f"pooling_type must be 'time_decay' or 'attention', got '{ptype}'.")
    if ptype == "time_decay":
        dr = float(cfg["decay_rate"])
        if not (0.0 < dr < 1.0):
            raise ValueError(f"decay_rate must be in (0,1), got {dr}.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df:  pd.DataFrame
    stats:    dict[str, Any]


def load_data(data_dir: Path) -> DataBundle:
    logging.info("Loading data from %s", data_dir)
    train_df = pd.read_parquet(data_dir / "train.parquet",  columns=TRAIN_COLUMNS)
    valid_df = pd.read_parquet(data_dir / "valid.parquet",  columns=EVAL_COLUMNS)
    test_df  = pd.read_parquet(data_dir / "test.parquet",   columns=EVAL_COLUMNS)
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    logging.info("n_users=%s n_items=%s train=%s", stats["n_users"], stats["n_items"], len(train_df))
    return DataBundle(train_df=train_df, valid_df=valid_df, test_df=test_df, stats=stats)


def build_history_matrix(
    frame: pd.DataFrame, num_users: int, max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    histories = np.full((num_users, max_len), -1, dtype=np.int64)
    lengths   = np.zeros(num_users, dtype=np.int64)
    ordered   = frame.sort_values(["user_idx", "timestamp"], kind="mergesort")
    for uid, group in ordered.groupby("user_idx", sort=False):
        items = group["item_idx"].to_numpy(dtype=np.int64)
        if items.size > max_len:
            items = items[-max_len:]
        u = int(uid)
        histories[u, :items.size] = items
        lengths[u] = items.size
    return histories, lengths


class InteractionDataset(Dataset):
    def __init__(self, users: np.ndarray, items: np.ndarray) -> None:
        self.users = torch.from_numpy(users.astype(np.int64))
        self.items = torch.from_numpy(items.astype(np.int64))

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx]


class HistoryCollator:
    def __init__(self, history_matrix: np.ndarray) -> None:
        self.history_matrix = history_matrix

    def __call__(self, batch):
        users = torch.stack([r[0] for r in batch])
        items = torch.stack([r[1] for r in batch])
        hists = torch.from_numpy(self.history_matrix[users.numpy()].copy())
        return users, items, hists


def make_dataloader(train_df: pd.DataFrame, history_matrix: np.ndarray,
                    cfg: dict[str, Any]) -> DataLoader:
    users = train_df["user_idx"].to_numpy(dtype=np.int64)
    items = train_df["item_idx"].to_numpy(dtype=np.int64)
    gen   = torch.Generator(); gen.manual_seed(int(cfg["seed"]))
    return DataLoader(
        InteractionDataset(users, items),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=gen,
        collate_fn=HistoryCollator(history_matrix),
    )


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for uid, grp in frame.groupby("user_idx", sort=False):
        seen[int(uid)] = set(int(x) for x in grp["item_idx"])
    return seen


def load_text_artifacts(cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    emb_path = Path(cfg["item_text_embedding_path"])
    has_path = Path(cfg["item_has_text_path"])
    text_emb = torch.from_numpy(np.load(emb_path).astype(np.float32))
    has_text = torch.from_numpy(np.load(has_path).astype(np.float32))
    logging.info("text_emb: shape=%s", tuple(text_emb.shape))
    logging.info("has_text: %d/%d (%.1f%%)", int(has_text.sum()), len(has_text),
                 100.0 * float(has_text.mean()))
    return text_emb, has_text


# ---------------------------------------------------------------------------
# Model — unified time_decay / attention user tower
# ---------------------------------------------------------------------------

class TextTwoTowerSmoke(nn.Module):
    """Two-Tower supporting time_decay or attention history pooling.

    Item tower is identical in both variants.
    User tower differs only in _pool_history().
    Zero additional parameters vs time_decay variant for attention mode
    (user_id_emb serves as the attention query).
    """

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
        pooling_type: str,
        decay_rate: float,
    ) -> None:
        super().__init__()
        self.use_l2_norm      = bool(use_l2_norm)
        self.use_has_text_mask = bool(use_has_text_mask)
        self.history_weight   = float(history_weight)
        self.pooling_type     = str(pooling_type)
        self.decay_rate       = float(decay_rate)

        self.user_embedding    = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.text_proj         = nn.Linear(int(text_emb.shape[1]), embedding_dim, bias=False)
        self.register_buffer("_text_emb", text_emb.float(), persistent=False)
        self.register_buffer("_has_text",  has_text.float(), persistent=False)

        nn.init.normal_(self.user_embedding.weight,    mean=0.0, std=0.02)
        nn.init.normal_(self.item_id_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.text_proj.weight)

    def _pool_history(
        self,
        history_item_idx: torch.Tensor,
        user_emb: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (pooled_history_emb, attn_weights_or_None)."""
        valid_mask = history_item_idx >= 0  # (B, L)
        if exclude_item_idx is not None:
            valid_mask = valid_mask & (history_item_idx != exclude_item_idx.unsqueeze(1))

        safe = history_item_idx.clamp_min(0)
        hist_emb = self.item_id_embedding(safe)  # (B, L, dim)

        if self.pooling_type == "time_decay":
            L = history_item_idx.shape[1]
            pos = torch.arange(L, device=hist_emb.device, dtype=hist_emb.dtype)
            w   = self.decay_rate ** (L - 1 - pos)            # (L,)
            w   = w.unsqueeze(0).unsqueeze(-1)                 # (1, L, 1)
            mask = valid_mask.unsqueeze(-1).to(hist_emb.dtype)
            pooled = (hist_emb * mask * w).sum(1) / (mask * w).sum(1).clamp_min(1e-8)
            return pooled, None

        # attention pooling
        # query = user_id embedding (B, dim)  →  (B, 1, dim)
        scale = math.sqrt(hist_emb.shape[-1])
        query = user_emb.unsqueeze(1)                          # (B, 1, dim)
        scores = torch.bmm(query, hist_emb.transpose(1, 2)).squeeze(1) / scale  # (B, L)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        attn_w = torch.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)            # empty-history rows → 0
        pooled = torch.bmm(attn_w.unsqueeze(1), hist_emb).squeeze(1)   # (B, dim)
        return pooled, attn_w

    def raw_user(
        self,
        user_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        id_emb = self.user_embedding(user_idx)
        pooled, _ = self._pool_history(history_item_idx, id_emb, exclude_item_idx)
        return id_emb + self.history_weight * pooled

    def _item_prenorm(self, item_idx: torch.Tensor) -> torch.Tensor:
        id_emb   = self.item_id_embedding(item_idx)
        txt_proj = self.text_proj(self._text_emb[item_idx])
        if self.use_has_text_mask:
            txt_proj = txt_proj * self._has_text[item_idx].unsqueeze(-1)
        return id_emb + txt_proj

    def encode_users(
        self,
        user_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.raw_user(user_idx, history_item_idx, exclude_item_idx)
        return F.normalize(emb, p=2, dim=-1) if self.use_l2_norm else emb

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        emb = self._item_prenorm(item_idx)
        return F.normalize(emb, p=2, dim=-1) if self.use_l2_norm else emb

    def raw_batch(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.raw_user(user_idx, history_item_idx, exclude_item_idx=item_idx),
            self._item_prenorm(item_idx),
        )


def build_model(cfg: dict[str, Any], stats: dict[str, Any], device: torch.device) -> TextTwoTowerSmoke:
    text_emb, has_text = load_text_artifacts(cfg)
    model = TextTwoTowerSmoke(
        num_users=int(stats["n_users"]),
        num_items=int(stats["n_items"]),
        embedding_dim=int(cfg["embedding_dim"]),
        text_emb=text_emb,
        has_text=has_text,
        text_proj_dim=int(cfg["text_proj_dim"]),
        use_l2_norm=bool(cfg["use_l2_norm"]),
        use_has_text_mask=bool(cfg["use_has_text_mask"]),
        history_weight=float(cfg["history_weight"]),
        pooling_type=str(cfg["pooling_type"]),
        decay_rate=float(cfg["decay_rate"]),
    ).to(device)
    logging.info(
        "TextTwoTowerSmoke trainable params: %s  pooling_type=%s  decay_rate=%s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        cfg["pooling_type"],
        cfg["decay_rate"],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_step(
    model: TextTwoTowerSmoke,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    temperature: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    raw_user, raw_item = model.raw_batch(user_idx, item_idx, history_item_idx)
    user_emb = F.normalize(raw_user, p=2, dim=-1) if model.use_l2_norm else raw_user
    item_emb = F.normalize(raw_item, p=2, dim=-1) if model.use_l2_norm else raw_item
    logits   = (user_emb @ item_emb.T) / temperature
    labels   = torch.arange(logits.shape[0], device=logits.device)
    loss     = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("loss has nan or inf; stopping.")
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_epoch(
    model: TextTwoTowerSmoke,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_n    = 0
    for bi, (user_idx, item_idx, hist_idx) in enumerate(loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        hist_idx = hist_idx.to(device)
        loss = train_one_step(model, optimizer, user_idx, item_idx, hist_idx, float(cfg["temperature"]))
        n = user_idx.shape[0]
        if bi == 0:
            logging.info("epoch %s batch 0: loss=%.4f bs=%s", epoch, loss, n)
        total_loss += loss * n
        total_n    += n
    return total_loss / total_n


# ---------------------------------------------------------------------------
# Evaluation with bucket analysis
# ---------------------------------------------------------------------------

def encode_all_items(model: TextTwoTowerSmoke, num_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + 65536, num_items)
            idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(idx).detach().cpu())
    return torch.cat(chunks, dim=0)


def check_attention_sanity(
    model: TextTwoTowerSmoke,
    eval_df: pd.DataFrame,
    history_matrix: np.ndarray,
    device: torch.device,
    n_sample: int = 256,
) -> dict[str, Any]:
    """Sample a batch and report attention weight statistics."""
    if model.pooling_type != "attention":
        return {}
    sample = eval_df.head(n_sample)
    users_np = sample["user_idx"].to_numpy(dtype=np.int64)
    hist_np  = history_matrix[users_np]
    user_t   = torch.as_tensor(users_np, device=device)
    hist_t   = torch.as_tensor(hist_np,  dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        id_emb = model.user_embedding(user_t)
        _, attn_w = model._pool_history(hist_t, id_emb)
    if attn_w is None:
        return {}
    valid_mask = hist_t >= 0
    # Per-user: sum of attn weights over valid positions should be ~1.0
    sum_per_user = (attn_w * valid_mask.float()).sum(dim=1)
    has_hist = valid_mask.any(dim=1)
    sums_with_hist = sum_per_user[has_hist]
    nan_count = int(torch.isnan(attn_w).sum().item())
    inf_count = int(torch.isinf(attn_w).sum().item())
    sanity: dict[str, Any] = {
        "sample_size":      n_sample,
        "nan_count":        nan_count,
        "inf_count":        inf_count,
        "attn_sum_mean":    float(sums_with_hist.mean().item()) if sums_with_hist.numel() > 0 else 0.0,
        "attn_sum_min":     float(sums_with_hist.min().item())  if sums_with_hist.numel() > 0 else 0.0,
        "attn_sum_max":     float(sums_with_hist.max().item())  if sums_with_hist.numel() > 0 else 0.0,
        "attn_weight_max":  float(attn_w.max().item()),
        "attn_weight_min":  float(attn_w[valid_mask].min().item()) if valid_mask.any() else 0.0,
    }
    logging.info("[AttnSanity] NaN=%s Inf=%s  sum@valid mean=%.4f min=%.4f max=%.4f  w_max=%.4f",
                 nan_count, inf_count,
                 sanity["attn_sum_mean"], sanity["attn_sum_min"], sanity["attn_sum_max"],
                 sanity["attn_weight_max"])
    if nan_count > 0 or inf_count > 0:
        logging.warning("[AttnSanity] NaN/Inf detected in attention weights!")
    return sanity


def evaluate_with_buckets(
    model: TextTwoTowerSmoke,
    eval_df: pd.DataFrame,
    history_matrix: np.ndarray,
    history_lengths: np.ndarray,
    seen_items: dict[int, set[int]],
    cfg: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> tuple[dict[str, Any], set[int], dict[str, Any]]:
    """
    Returns:
      overall_metrics  : dict with recall@K, ndcg@K, mrr@K
      hit_users_r50    : set of user_idx that had a hit @50
      bucket_result    : {bucket: {recall@50, count, hits}}
    """
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    max_users = cfg.get("eval_max_users")
    if max_users is not None:
        non_cold = non_cold.head(int(max_users)).copy()

    k_list   = [int(k) for k in cfg["eval_k_list"]]
    max_k    = max(k_list)
    num_items = int(stats["n_items"])
    bs       = int(cfg["eval_batch_size"])

    item_emb_cpu = encode_all_items(model, num_items, device)
    logging.info("%s eval users=%s (eval_max_users=%s)", split_name, len(non_cold), max_users)

    metric_sums: dict[str, float] = {}
    for k in k_list:
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"ndcg@{k}"]   = 0.0
        metric_sums[f"mrr@{k}"]    = 0.0

    hit_users_r50: set[int] = set()
    bucket_hits:   dict[str, int] = {b: 0 for b, _ in HISTORY_BUCKETS}
    bucket_counts: dict[str, int] = {b: 0 for b, _ in HISTORY_BUCKETS}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(non_cold), bs):
            batch = non_cold.iloc[start : start + bs]
            users_np   = batch["user_idx"].to_numpy(dtype=np.int64)
            targets_np = batch["item_idx"].to_numpy(dtype=np.int64)
            user_t     = torch.as_tensor(users_np, device=device)
            target_t   = torch.as_tensor(targets_np, device=device)
            hist_t     = torch.as_tensor(history_matrix[users_np], dtype=torch.long, device=device)
            user_emb   = model.encode_users(user_t, hist_t)
            item_emb   = item_emb_cpu.to(device)
            scores     = (user_emb @ item_emb.T) / float(cfg["temperature"])

            row_idx    = torch.arange(scores.shape[0], device=device)
            tgt_scores = scores[row_idx, target_t].clone()
            for rp, (uid, tgt) in enumerate(zip(batch["user_idx"].tolist(),
                                                 batch["item_idx"].tolist(), strict=True)):
                seen = seen_items.get(int(uid), set())
                if seen:
                    scores[rp, torch.as_tensor(list(seen), dtype=torch.long, device=device)] = -torch.inf
                scores[rp, int(tgt)] = tgt_scores[rp]

            topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

            for rp, (uid, tgt, rec) in enumerate(
                zip(batch["user_idx"].tolist(), targets_np, topk, strict=True)
            ):
                matched = np.where(rec == int(tgt))[0]
                hist_len = int(history_lengths[int(uid)])
                # Bucket assignment
                for bname, bpred in HISTORY_BUCKETS:
                    if bpred(hist_len):
                        bucket_counts[bname] += 1
                        break

                if matched.size == 0:
                    continue
                rank = int(matched[0]) + 1
                for k in k_list:
                    if rank <= k:
                        metric_sums[f"recall@{k}"] += 1.0
                        metric_sums[f"ndcg@{k}"]   += 1.0 / math.log2(rank + 1)
                        metric_sums[f"mrr@{k}"]    += 1.0 / rank
                if rank <= 50:
                    hit_users_r50.add(int(uid))
                    for bname, bpred in HISTORY_BUCKETS:
                        if bpred(hist_len):
                            bucket_hits[bname] += 1
                            break

    denom = len(non_cold)
    overall: dict[str, Any] = {
        "split":           split_name,
        "num_eval_users":  denom,
        "eval_max_users":  max_users,
    }
    for key, val in metric_sums.items():
        overall[key] = val / denom if denom else 0.0

    bucket_result: dict[str, Any] = {}
    for bname, _ in HISTORY_BUCKETS:
        cnt = bucket_counts[bname]
        hits = bucket_hits[bname]
        bucket_result[bname] = {
            "count":      cnt,
            "hits":       hits,
            "recall@50":  hits / cnt if cnt else 0.0,
        }

    return overall, hit_users_r50, bucket_result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def init_train_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def append_train_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS).writerow(
            {k: row.get(k, "") for k in TRAIN_LOG_FIELDS}
        )


def save_checkpoint(path: Path, model: TextTwoTowerSmoke, cfg: dict[str, Any],
                    stats: dict[str, Any], epoch: int, metric_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg, "stats": stats,
        "epoch": epoch, "best_metric_value": metric_val,
    }, path)


def train(cfg: dict[str, Any]) -> None:
    require_config(cfg)
    set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", cfg)
    device   = resolve_device(str(cfg["device"]))
    logging.info("device=%s pooling_type=%s epochs=%s", device, cfg["pooling_type"], cfg["epochs"])

    bundle   = load_data(Path(cfg["data_dir"]))
    n_users  = int(bundle.stats["n_users"])
    max_len  = int(cfg["history_max_len"])
    hist_mat, hist_lens = build_history_matrix(bundle.train_df, n_users, max_len)
    logging.info("history non-empty: %d  avg_len=%.2f",
                 int((hist_lens > 0).sum()), float(hist_lens.mean()))
    loader   = make_dataloader(bundle.train_df, hist_mat, cfg)
    seen     = build_seen_items(bundle.train_df)
    model    = build_model(cfg, bundle.stats, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    log_path = out_dir / "train_log.csv"
    init_train_log(log_path)

    best_r50    = -1.0
    best_metrics: dict[str, Any] = {}
    best_bucket:  dict[str, Any] = {}
    best_epoch  = 0
    total_train_sec = 0.0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, cfg, device, epoch)
        valid_metrics, hit_users, bucket = evaluate_with_buckets(
            model, bundle.valid_df, hist_mat, hist_lens, seen,
            cfg, bundle.stats, device, "valid"
        )
        epoch_sec = time.time() - t0
        total_train_sec += epoch_sec

        logging.info(
            "epoch %s: train_loss=%.6f  R@50=%.6f  NDCG@50=%.6f  MRR@50=%.6f  time=%.1fs",
            epoch, train_loss, valid_metrics["recall@50"],
            valid_metrics["ndcg@50"], valid_metrics["mrr@50"], epoch_sec,
        )
        for bname, _ in HISTORY_BUCKETS:
            br = bucket[bname]
            logging.info("  bucket %-6s  count=%d  hits=%d  R@50=%.6f",
                         bname, br["count"], br["hits"], br["recall@50"])

        row = {
            "epoch": epoch, "train_loss": train_loss,
            "valid_recall@50": valid_metrics["recall@50"],
            "valid_recall@20": valid_metrics["recall@20"],
            "valid_recall@100": valid_metrics["recall@100"],
            "valid_ndcg@50":   valid_metrics["ndcg@50"],
            "valid_mrr@50":    valid_metrics["mrr@50"],
            "epoch_time_seconds": epoch_sec,
            "pooling_type": cfg["pooling_type"],
        }
        append_train_log(log_path, row)

        if valid_metrics["recall@50"] > best_r50:
            best_r50     = float(valid_metrics["recall@50"])
            best_metrics = valid_metrics
            best_bucket  = bucket
            best_epoch   = epoch
            hit_users_arr = np.array(sorted(hit_users), dtype=np.int64)
            np.save(out_dir / "hit_users_valid_r50.npy", hit_users_arr)
            save_checkpoint(out_dir / "checkpoints" / "best_model.pt",
                            model, cfg, bundle.stats, epoch, best_r50)
            logging.info("  → new best: epoch=%s R@50=%.6f  hit_users=%d",
                         epoch, best_r50, len(hit_users))

    # Attention sanity check (on best epoch checkpoint if attention)
    if cfg["pooling_type"] == "attention":
        sanity = check_attention_sanity(model, bundle.valid_df, hist_mat, device)
        write_json(out_dir / "attention_sanity.json", sanity)

    # Determine top-level output dir (parent of model's output_dir)
    top_dir = Path(cfg["output_dir"]).parent
    top_dir.mkdir(parents=True, exist_ok=True)

    final_summary = {
        "pooling_type":     cfg["pooling_type"],
        "best_epoch":       best_epoch,
        "best_valid_recall@50": best_r50,
        "best_valid_recall@20": best_metrics.get("recall@20", 0.0),
        "best_valid_recall@100": best_metrics.get("recall@100", 0.0),
        "best_valid_ndcg@50":  best_metrics.get("ndcg@50", 0.0),
        "best_valid_mrr@50":   best_metrics.get("mrr@50", 0.0),
        "num_eval_users":   best_metrics.get("num_eval_users", 0),
        "total_train_sec":  total_train_sec,
        "bucket_recall@50": {b: best_bucket[b]["recall@50"] for b, _ in HISTORY_BUCKETS},
        "bucket_counts":    {b: best_bucket[b]["count"] for b, _ in HISTORY_BUCKETS},
        "output_dir":       str(out_dir),
    }
    write_json(out_dir / "metrics_valid_best.json", final_summary)

    # Copy to canonical top-level filenames
    fname_map = {"time_decay": "time_decay_smoke_metrics.json",
                 "attention":  "attention_smoke_metrics.json"}
    fname = fname_map.get(cfg["pooling_type"])
    if fname:
        write_json(top_dir / fname, final_summary)
        logging.info("Summary → %s", top_dir / fname)

    logging.info(
        "Training complete: pooling=%s  best_epoch=%s  R@50=%.6f  total_time=%.1fs",
        cfg["pooling_type"], best_epoch, best_r50, total_train_sec,
    )


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def compare(td_dir: Path, attn_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-run summaries
    td_summary   = json.loads((td_dir   / "metrics_valid_best.json").read_text(encoding="utf-8"))
    attn_summary = json.loads((attn_dir / "metrics_valid_best.json").read_text(encoding="utf-8"))

    # Load hit user arrays
    td_hits   = set(int(x) for x in np.load(td_dir   / "hit_users_valid_r50.npy").tolist())
    attn_hits = set(int(x) for x in np.load(attn_dir / "hit_users_valid_r50.npy").tolist())

    only_attn = sorted(attn_hits - td_hits)    # attention hit, time_decay missed
    only_td   = sorted(td_hits   - attn_hits)  # time_decay hit, attention missed
    both      = sorted(attn_hits & td_hits)

    comparison = {
        "td_recall@50":         td_summary["best_valid_recall@50"],
        "attn_recall@50":       attn_summary["best_valid_recall@50"],
        "delta_recall@50":      attn_summary["best_valid_recall@50"] - td_summary["best_valid_recall@50"],
        "td_ndcg@50":           td_summary["best_valid_ndcg@50"],
        "attn_ndcg@50":         attn_summary["best_valid_ndcg@50"],
        "td_mrr@50":            td_summary["best_valid_mrr@50"],
        "attn_mrr@50":          attn_summary["best_valid_mrr@50"],
        "td_best_epoch":        td_summary["best_epoch"],
        "attn_best_epoch":      attn_summary["best_epoch"],
        "num_eval_users":       td_summary["num_eval_users"],
        "td_total_hits":        len(td_hits),
        "attn_total_hits":      len(attn_hits),
        "both_hit":             len(both),
        "only_attn_hits":       len(only_attn),
        "only_td_hits":         len(only_td),
        "td_bucket_recall@50":   td_summary["bucket_recall@50"],
        "attn_bucket_recall@50": attn_summary["bucket_recall@50"],
        "bucket_delta": {
            b: attn_summary["bucket_recall@50"].get(b, 0.0) - td_summary["bucket_recall@50"].get(b, 0.0)
            for b in ("le5", "6to20", "gt20")
        },
        "td_total_train_sec":   td_summary["total_train_sec"],
        "attn_total_train_sec": attn_summary["total_train_sec"],
        "meets_threshold":      (attn_summary["best_valid_recall@50"] - td_summary["best_valid_recall@50"]) >= 0.001,
    }

    write_json(out_dir / "attention_vs_time_decay_unique_hit.json", comparison)
    logging.info("Comparison saved: %s", out_dir / "attention_vs_time_decay_unique_hit.json")

    # Log summary
    delta = comparison["delta_recall@50"]
    logging.info("===== COMPARISON SUMMARY =====")
    logging.info("Time-decay  R@50=%.6f  (epoch %s, time=%.0fs)",
                 comparison["td_recall@50"], comparison["td_best_epoch"], comparison["td_total_train_sec"])
    logging.info("Attention   R@50=%.6f  (epoch %s, time=%.0fs)",
                 comparison["attn_recall@50"], comparison["attn_best_epoch"], comparison["attn_total_train_sec"])
    logging.info("Delta       R@50=%+.6f  meets_threshold(>=0.001)=%s", delta, comparison["meets_threshold"])
    for b in ("le5", "6to20", "gt20"):
        logging.info("  bucket %-6s  td=%.6f  attn=%.6f  delta=%+.6f",
                     b, comparison["td_bucket_recall@50"][b],
                     comparison["attn_bucket_recall@50"][b], comparison["bucket_delta"][b])
    logging.info("Unique hits  only_attn=%d  only_td=%d  both=%d",
                 comparison["only_attn_hits"], comparison["only_td_hits"], comparison["both_hit"])
    logging.info("==============================")

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    if args.compare:
        td_dir   = Path(args.td_dir)
        attn_dir = Path(args.attn_dir)
        out_dir  = Path(args.out_dir)
        for d in (td_dir, attn_dir):
            if not (d / "metrics_valid_best.json").exists():
                raise FileNotFoundError(f"Missing metrics_valid_best.json in {d}. Run training first.")
        compare(td_dir, attn_dir, out_dir)
        return

    if not args.config:
        raise ValueError("Provide --config for training mode or --compare for comparison mode.")

    cfg = load_config(Path(args.config))
    cfg["config_path"] = args.config
    train(cfg)


if __name__ == "__main__":
    main()

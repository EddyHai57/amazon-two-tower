#!/usr/bin/env python3
"""Transformer user tower investigation — max_len=100 paired smoke.

Lightweight vanilla Transformer user tower:
  - 1 TransformerEncoderLayer (Pre-LN, batch_first)
  - 4 attention heads
  - FFN dim=256
  - Learnable positional embedding (max_len positions)
  - Output: mean pool over valid (non-padding) positions

Time-aware Transformer adds a recency bucket embedding to each valid history
position.  The item tower is intentionally unchanged from the final
Text+Time-decay Two-Tower.

History bucket analysis uses raw (pre-truncation) history lengths to correctly
populate the >20 bucket.

Usage
-----
Training:
  python scripts/train_transformer_maxlen100_smoke.py \\
    --config configs/transformer_max100_smoke_td.yaml
  python scripts/train_transformer_maxlen100_smoke.py \\
    --config configs/transformer_max100_smoke_vanilla.yaml
  python scripts/train_transformer_maxlen100_smoke.py \\
    --config configs/transformer_max100_smoke_timeaware.yaml

Comparison (after both runs):
  python scripts/train_transformer_maxlen100_smoke.py --compare \\
    --td_dir        outputs/transformer_user_tower_investigation/time_decay_max100 \\
    --vanilla_dir   outputs/transformer_user_tower_investigation/transformer_vanilla_max100 \\
    --timeaware_dir outputs/transformer_user_tower_investigation/transformer_timeaware_max100 \\
    --out_dir       outputs/transformer_user_tower_investigation
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
# Config / constants
# ---------------------------------------------------------------------------

VALID_POOLING = ("time_decay", "transformer_vanilla", "transformer_timeaware", "mean_pool_timeaware")

REQUIRED_CONFIG_KEYS = [
    "data_dir", "output_dir", "embedding_dim", "batch_size", "learning_rate",
    "weight_decay", "epochs", "temperature", "use_l2_norm", "seed",
    "eval_k_list", "eval_batch_size", "eval_max_users", "num_workers", "device",
    "save_best_by", "history_max_len", "history_weight",
    "item_text_embedding_path", "item_has_text_path", "text_proj_dim",
    "use_has_text_mask", "item_fusion", "pooling_type", "decay_rate",
]
TRAIN_COLUMNS    = ["user_idx", "item_idx", "timestamp"]
EVAL_COLUMNS     = ["user_idx", "item_idx", "timestamp", "is_cold_item_for_eval"]
TRAIN_LOG_FIELDS = [
    "epoch", "train_loss", "valid_recall@20", "valid_recall@50",
    "valid_recall@100", "valid_ndcg@50", "valid_mrr@50",
    "train_time_seconds", "eval_time_seconds", "epoch_time_seconds", "pooling_type",
]
HISTORY_BUCKETS = [
    ("le5",   lambda l: l <= 5),
    ("6to20", lambda l: 6 <= l <= 20),
    ("gt20",  lambda l: l > 20),
]
FNAME_MAP = {
    "time_decay":            "time_decay_max100_metrics.json",
    "transformer_vanilla":    "transformer_vanilla_max100_metrics.json",
    "transformer_timeaware":  "transformer_timeaware_max100_metrics.json",
    "mean_pool_timeaware":    "mean_pool_timeaware_max100_metrics.json",
}

# Transformer defaults (overridable via config)
DEFAULT_NUM_HEADS  = 4
DEFAULT_FFN_DIM    = 256
DEFAULT_DROPOUT    = 0.1
DEFAULT_NUM_LAYERS = 1
RECENCY_BUCKET_COUNT = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",    help="YAML config for training mode.")
    p.add_argument("--phase0_check", action="store_true", help="Run tiny forward/train checks for all variants.")
    p.add_argument("--compare",   action="store_true", help="Run 2-way comparison.")
    p.add_argument("--eval_only", action="store_true", help="Load checkpoint and run valid/test eval.")
    p.add_argument("--full_eval", action="store_true", help="Use all non-cold valid/test users in eval-only mode.")
    p.add_argument("--checkpoint", help="Checkpoint path for eval-only mode.")
    p.add_argument("--eval_output_dir", help="Output dir for eval-only mode.")
    p.add_argument("--td_dir", default="outputs/transformer_user_tower_investigation/time_decay_max100")
    p.add_argument("--vanilla_dir", default="outputs/transformer_user_tower_investigation/transformer_vanilla_max100")
    p.add_argument("--timeaware_dir", default="outputs/transformer_user_tower_investigation/transformer_timeaware_max100")
    p.add_argument("--out_dir", default="outputs/transformer_user_tower_investigation")
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
            raise KeyError(f"Config missing key: {k}")
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
    if ptype not in VALID_POOLING:
        raise ValueError(f"pooling_type must be one of {VALID_POOLING}, got '{ptype}'.")
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    train_df = pd.read_parquet(data_dir / "train.parquet", columns=TRAIN_COLUMNS)
    valid_df = pd.read_parquet(data_dir / "valid.parquet", columns=EVAL_COLUMNS)
    test_df  = pd.read_parquet(data_dir / "test.parquet",  columns=EVAL_COLUMNS)
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    logging.info("n_users=%s n_items=%s train=%s",
                 stats["n_users"], stats["n_items"], len(train_df))
    return DataBundle(train_df=train_df, valid_df=valid_df, test_df=test_df, stats=stats)


def build_history_matrix(
    frame: pd.DataFrame, num_users: int, max_len: int
) -> np.ndarray:
    """Return truncated history matrix [num_users, max_len] with -1 padding."""
    histories = np.full((num_users, max_len), -1, dtype=np.int64)
    ordered   = frame.sort_values(["user_idx", "timestamp"], kind="mergesort")
    for uid, group in ordered.groupby("user_idx", sort=False):
        items = group["item_idx"].to_numpy(dtype=np.int64)
        if items.size > max_len:
            items = items[-max_len:]
        u = int(uid)
        histories[u, :items.size] = items
    return histories


def compute_raw_history_lengths(train_df: pd.DataFrame, n_users: int) -> np.ndarray:
    """Per-user history length BEFORE max_len truncation (for correct bucket assignment)."""
    counts = train_df.groupby("user_idx")["item_idx"].count()
    raw    = np.zeros(n_users, dtype=np.int64)
    raw[counts.index.to_numpy(dtype=np.int64)] = counts.to_numpy(dtype=np.int64)
    return raw


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


def make_dataloader(
    train_df: pd.DataFrame, history_matrix: np.ndarray, cfg: dict[str, Any]
) -> DataLoader:
    users = train_df["user_idx"].to_numpy(dtype=np.int64)
    items = train_df["item_idx"].to_numpy(dtype=np.int64)
    gen   = torch.Generator()
    gen.manual_seed(int(cfg["seed"]))
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


def merge_seen_items(base: dict[int, set[int]], extra_frame: pd.DataFrame) -> dict[int, set[int]]:
    merged = {uid: set(items) for uid, items in base.items()}
    for uid, grp in extra_frame.groupby("user_idx", sort=False):
        merged.setdefault(int(uid), set()).update(int(x) for x in grp["item_idx"])
    return merged


def load_text_artifacts(cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    text_emb = torch.from_numpy(
        np.load(Path(cfg["item_text_embedding_path"])).astype(np.float32)
    )
    has_text = torch.from_numpy(
        np.load(Path(cfg["item_has_text_path"])).astype(np.float32)
    )
    logging.info("text_emb: shape=%s", tuple(text_emb.shape))
    logging.info("has_text: %d/%d (%.1f%%)", int(has_text.sum()), len(has_text),
                 100.0 * float(has_text.mean()))
    return text_emb, has_text


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransformerHistoryEncoder(nn.Module):
    """Lightweight 1-layer Transformer encoder for user history pooling.

    Architecture:
      item_id_emb + learnable_pos_emb (+ optional recency_bucket_emb)
        → TransformerEncoderLayer (Pre-LN, batch_first)
        → mean pool over valid (non-padding) positions
    """

    def __init__(
        self,
        embedding_dim: int,
        max_len: int,
        num_heads: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_recency_buckets: bool = False,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.use_recency_buckets = bool(use_recency_buckets)
        if self.use_recency_buckets:
            self.recency_embedding = nn.Embedding(RECENCY_BUCKET_COUNT, embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        if self.use_recency_buckets:
            nn.init.normal_(self.recency_embedding.weight, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def recency_bucket_ids(valid_mask: torch.Tensor) -> torch.Tensor:
        """Bucket positions by age from each user's newest valid history item."""
        L = valid_mask.shape[1]
        pos = torch.arange(L, device=valid_mask.device).unsqueeze(0)
        valid_len = valid_mask.long().sum(dim=1, keepdim=True)
        age = (valid_len - 1 - pos).clamp_min(0)
        buckets = torch.zeros_like(age)
        buckets = torch.where(age <= 0, torch.zeros_like(buckets), buckets)
        buckets = torch.where((age >= 1) & (age <= 1), torch.ones_like(buckets), buckets)
        buckets = torch.where((age >= 2) & (age <= 3), torch.full_like(buckets, 2), buckets)
        buckets = torch.where((age >= 4) & (age <= 7), torch.full_like(buckets, 3), buckets)
        buckets = torch.where((age >= 8) & (age <= 15), torch.full_like(buckets, 4), buckets)
        buckets = torch.where((age >= 16) & (age <= 31), torch.full_like(buckets, 5), buckets)
        buckets = torch.where((age >= 32) & (age <= 63), torch.full_like(buckets, 6), buckets)
        buckets = torch.where(age >= 64, torch.full_like(buckets, 7), buckets)
        return buckets

    def forward(
        self,
        hist_emb: torch.Tensor,   # (B, L, dim)
        valid_mask: torch.Tensor,  # (B, L) bool, True=valid
    ) -> torch.Tensor:
        L   = hist_emb.shape[1]
        pos = torch.arange(L, device=hist_emb.device)
        x   = hist_emb + self.pos_embedding(pos).unsqueeze(0)  # (B, L, dim)
        if self.use_recency_buckets:
            recency_ids = self.recency_bucket_ids(valid_mask)
            x = x + self.recency_embedding(recency_ids)

        # TransformerEncoder src_key_padding_mask: True = MASKED (ignore as key/value)
        padding_mask = ~valid_mask  # (B, L)
        out = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, L, dim)

        # Mean pool over valid positions only
        valid_f = valid_mask.unsqueeze(-1).to(out.dtype)          # (B, L, 1)
        pooled  = (out * valid_f).sum(1) / valid_f.sum(1).clamp_min(1e-8)  # (B, dim)
        pooled  = torch.nan_to_num(pooled, nan=0.0)               # empty-history guard
        return pooled


class MeanPoolTimeawareHistoryEncoder(nn.Module):
    """Mean pool over item, positional, and recency embeddings without attention."""

    def __init__(self, embedding_dim: int, max_len: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.recency_embedding = nn.Embedding(RECENCY_BUCKET_COUNT, embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.recency_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hist_emb: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        L = hist_emb.shape[1]
        pos = torch.arange(L, device=hist_emb.device)
        recency_ids = TransformerHistoryEncoder.recency_bucket_ids(valid_mask)
        x = (
            hist_emb
            + self.pos_embedding(pos).unsqueeze(0)
            + self.recency_embedding(recency_ids)
        )
        valid_f = valid_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * valid_f).sum(1) / valid_f.sum(1).clamp_min(1e-8)
        return torch.nan_to_num(pooled, nan=0.0)


class TextTwoTowerTransformerSmoke(nn.Module):
    """Two-Tower with time_decay or Transformer user history pooling.

    Item tower is identical in both variants.
    Transformer adds ~39K params (pos_emb 6,400 + encoder ~33K).
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
        max_len: int,
        num_heads: int = DEFAULT_NUM_HEADS,
        ffn_dim: int = DEFAULT_FFN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.use_l2_norm       = bool(use_l2_norm)
        self.use_has_text_mask = bool(use_has_text_mask)
        self.history_weight    = float(history_weight)
        self.pooling_type      = str(pooling_type)
        self.decay_rate        = float(decay_rate)

        self.user_embedding    = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.text_proj         = nn.Linear(int(text_emb.shape[1]), embedding_dim, bias=False)
        self.register_buffer("_text_emb", text_emb.float(), persistent=False)
        self.register_buffer("_has_text",  has_text.float(), persistent=False)

        nn.init.normal_(self.user_embedding.weight,    mean=0.0, std=0.02)
        nn.init.normal_(self.item_id_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.text_proj.weight)

        if self.pooling_type in {"transformer_vanilla", "transformer_timeaware"}:
            self.transformer_encoder = TransformerHistoryEncoder(
                embedding_dim=embedding_dim,
                max_len=max_len,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                num_layers=num_layers,
                use_recency_buckets=self.pooling_type == "transformer_timeaware",
            )
        elif self.pooling_type == "mean_pool_timeaware":
            self.mean_pool_timeaware_encoder = MeanPoolTimeawareHistoryEncoder(
                embedding_dim=embedding_dim,
                max_len=max_len,
            )

    # ── pooling ──────────────────────────────────────────────────────────────

    def _pool_history(
        self,
        history_item_idx: torch.Tensor,
        user_emb: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = history_item_idx >= 0
        if exclude_item_idx is not None:
            valid_mask = valid_mask & (history_item_idx != exclude_item_idx.unsqueeze(1))

        safe     = history_item_idx.clamp_min(0)
        hist_emb = self.item_id_embedding(safe)  # (B, L, dim)

        if self.pooling_type == "time_decay":
            L    = hist_emb.shape[1]
            pos  = torch.arange(L, device=hist_emb.device, dtype=hist_emb.dtype)
            w    = self.decay_rate ** (L - 1 - pos)
            w    = w.unsqueeze(0).unsqueeze(-1)
            mask = valid_mask.unsqueeze(-1).to(hist_emb.dtype)
            return (hist_emb * mask * w).sum(1) / (mask * w).sum(1).clamp_min(1e-8)

        if self.pooling_type == "mean_pool_timeaware":
            return self.mean_pool_timeaware_encoder(hist_emb, valid_mask)

        # transformer_vanilla / transformer_timeaware
        return self.transformer_encoder(hist_emb, valid_mask)

    # ── forward ──────────────────────────────────────────────────────────────

    def raw_user(
        self,
        user_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        id_emb = self.user_embedding(user_idx)
        pooled = self._pool_history(history_item_idx, id_emb, exclude_item_idx)
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


def build_model(
    cfg: dict[str, Any], stats: dict[str, Any], device: torch.device
) -> TextTwoTowerTransformerSmoke:
    text_emb, has_text = load_text_artifacts(cfg)
    model = TextTwoTowerTransformerSmoke(
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
        max_len=int(cfg["history_max_len"]),
        num_heads=int(cfg.get("num_heads",  DEFAULT_NUM_HEADS)),
        ffn_dim=int(cfg.get("ffn_dim",     DEFAULT_FFN_DIM)),
        dropout=float(cfg.get("dropout",   DEFAULT_DROPOUT)),
        num_layers=int(cfg.get("num_layers", DEFAULT_NUM_LAYERS)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("TextTwoTowerTransformerSmoke params=%s  pooling=%s  max_len=%s",
                 n_params, cfg["pooling_type"], cfg["history_max_len"])
    return model


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def tensor_health(t: torch.Tensor) -> dict[str, int]:
    return {
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
    }


def sanity_forward_check(
    model: TextTwoTowerTransformerSmoke,
    train_df: pd.DataFrame,
    history_matrix: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    sample = train_df.head(batch_size)
    users_np = sample["user_idx"].to_numpy(dtype=np.int64)
    items_np = sample["item_idx"].to_numpy(dtype=np.int64)
    user_t = torch.as_tensor(users_np, device=device)
    item_t = torch.as_tensor(items_np, device=device)
    hist_t = torch.as_tensor(history_matrix[users_np], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        raw_user, raw_item = model.raw_batch(user_t, item_t, hist_t)
        user_emb = F.normalize(raw_user, p=2, dim=-1) if model.use_l2_norm else raw_user
        item_emb = F.normalize(raw_item, p=2, dim=-1) if model.use_l2_norm else raw_item
        logits = (user_emb @ item_emb.T) / float(cfg["temperature"])
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)
    valid_counts = (hist_t >= 0).sum(dim=1)
    payload = {
        "batch_size": int(len(sample)),
        "loss": float(loss.item()),
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "raw_user": tensor_health(raw_user),
        "raw_item": tensor_health(raw_item),
        "logits": tensor_health(logits),
        "history_shape": list(history_matrix.shape),
        "history_valid_min": int(valid_counts.min().item()),
        "history_valid_max": int(valid_counts.max().item()),
    }
    payload["passed"] = (
        payload["loss_is_finite"]
        and payload["raw_user"]["nan_count"] == 0
        and payload["raw_user"]["inf_count"] == 0
        and payload["raw_item"]["nan_count"] == 0
        and payload["raw_item"]["inf_count"] == 0
        and payload["logits"]["nan_count"] == 0
        and payload["logits"]["inf_count"] == 0
    )
    return payload


def tiny_train_step_check(
    model: TextTwoTowerTransformerSmoke,
    train_df: pd.DataFrame,
    history_matrix: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    sample = train_df.head(batch_size)
    users_np = sample["user_idx"].to_numpy(dtype=np.int64)
    items_np = sample["item_idx"].to_numpy(dtype=np.int64)
    user_t = torch.as_tensor(users_np, device=device)
    item_t = torch.as_tensor(items_np, device=device)
    hist_t = torch.as_tensor(history_matrix[users_np], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    model.train()
    loss = train_one_step(model, optimizer, user_t, item_t, hist_t, float(cfg["temperature"]))
    return {"loss": float(loss), "loss_is_finite": bool(math.isfinite(loss)), "passed": bool(math.isfinite(loss))}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_step(
    model: TextTwoTowerTransformerSmoke,
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
        raise FloatingPointError("loss is nan/inf; stopping.")
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_epoch(
    model: TextTwoTowerTransformerSmoke,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for bi, (user_idx, item_idx, hist_idx) in enumerate(loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        hist_idx = hist_idx.to(device)
        loss = train_one_step(model, optimizer, user_idx, item_idx, hist_idx,
                              float(cfg["temperature"]))
        n = user_idx.shape[0]
        if bi == 0:
            logging.info("epoch %s batch 0: loss=%.4f bs=%s", epoch, loss, n)
        total_loss += loss * n
        total_n    += n
    return total_loss / total_n


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def encode_all_items(
    model: TextTwoTowerTransformerSmoke, num_items: int, device: torch.device
) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + 65536, num_items)
            idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(idx).detach().cpu())
    return torch.cat(chunks, dim=0)


def log_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserv = torch.cuda.memory_reserved(device) / 1024**3
        logging.info("[GPU] allocated=%.2fGB  reserved=%.2fGB", alloc, reserv)


def evaluate_with_buckets(
    model: TextTwoTowerTransformerSmoke,
    eval_df: pd.DataFrame,
    history_matrix: np.ndarray,
    raw_history_lengths: np.ndarray,
    seen_items: dict[int, set[int]],
    cfg: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> tuple[dict[str, Any], set[int], dict[str, Any]]:
    """Evaluate; bucket assignment uses raw_history_lengths (pre-truncation)."""
    non_cold  = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    max_users = cfg.get("eval_max_users")
    if max_users is not None:
        non_cold = non_cold.head(int(max_users)).copy()

    k_list    = [int(k) for k in cfg["eval_k_list"]]
    max_k     = max(k_list)
    num_items = int(stats["n_items"])
    bs        = int(cfg["eval_batch_size"])

    item_emb_cpu = encode_all_items(model, num_items, device)
    logging.info("%s eval users=%s (eval_max_users=%s)", split_name, len(non_cold), max_users)

    metric_sums: dict[str, float] = {
        f"{m}@{k}": 0.0 for k in k_list for m in ("recall", "ndcg", "mrr")
    }
    hit_users_r50: set[int]       = set()
    bucket_hits:   dict[str, int] = {b: 0 for b, _ in HISTORY_BUCKETS}
    bucket_counts: dict[str, int] = {b: 0 for b, _ in HISTORY_BUCKETS}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(non_cold), bs):
            batch      = non_cold.iloc[start:start + bs]
            users_np   = batch["user_idx"].to_numpy(dtype=np.int64)
            targets_np = batch["item_idx"].to_numpy(dtype=np.int64)
            user_t     = torch.as_tensor(users_np, device=device)
            target_t   = torch.as_tensor(targets_np, device=device)
            hist_t     = torch.as_tensor(
                history_matrix[users_np], dtype=torch.long, device=device
            )
            user_emb = model.encode_users(user_t, hist_t)
            item_emb = item_emb_cpu.to(device)
            scores   = (user_emb @ item_emb.T) / float(cfg["temperature"])

            row_idx    = torch.arange(scores.shape[0], device=device)
            tgt_scores = scores[row_idx, target_t].clone()
            for rp, (uid, tgt) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)
            ):
                seen = seen_items.get(int(uid), set())
                if seen:
                    scores[rp, torch.as_tensor(
                        list(seen), dtype=torch.long, device=device
                    )] = -torch.inf
                scores[rp, int(tgt)] = tgt_scores[rp]

            topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

            for rp, (uid, tgt, rec) in enumerate(
                zip(batch["user_idx"].tolist(), targets_np, topk, strict=True)
            ):
                raw_len = int(raw_history_lengths[int(uid)])
                for bname, bpred in HISTORY_BUCKETS:
                    if bpred(raw_len):
                        bucket_counts[bname] += 1
                        break

                matched = np.where(rec == int(tgt))[0]
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
                        if bpred(raw_len):
                            bucket_hits[bname] += 1
                            break

    denom   = len(non_cold)
    overall = {"split": split_name, "num_eval_users": denom, "eval_max_users": max_users}
    for key, val in metric_sums.items():
        overall[key] = val / denom if denom else 0.0

    bucket_result: dict[str, Any] = {}
    for bname, _ in HISTORY_BUCKETS:
        cnt  = bucket_counts[bname]
        hits = bucket_hits[bname]
        bucket_result[bname] = {"count": cnt, "hits": hits,
                                "recall@50": hits / cnt if cnt else 0.0}
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


def save_checkpoint(
    path: Path, model: TextTwoTowerTransformerSmoke,
    cfg: dict[str, Any], stats: dict[str, Any], epoch: int, metric_val: float,
) -> None:
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
    device  = resolve_device(str(cfg["device"]))
    logging.info("device=%s pooling_type=%s epochs=%s max_len=%s",
                 device, cfg["pooling_type"], cfg["epochs"], cfg["history_max_len"])

    bundle  = load_data(Path(cfg["data_dir"]))
    n_users = int(bundle.stats["n_users"])
    max_len = int(cfg["history_max_len"])

    hist_mat  = build_history_matrix(bundle.train_df, n_users, max_len)
    raw_lens  = compute_raw_history_lengths(bundle.train_df, n_users)
    n_gt20    = int((raw_lens > 20).sum())
    n_gt100   = int((raw_lens > 100).sum())
    logging.info("raw history: non-empty=%d avg=%.2f pct_gt20=%.1f%% pct_gt100=%.1f%%",
                 int((raw_lens > 0).sum()), float(raw_lens.mean()),
                 100.0 * n_gt20 / n_users, 100.0 * n_gt100 / n_users)

    loader    = make_dataloader(bundle.train_df, hist_mat, cfg)
    seen      = build_seen_items(bundle.train_df)
    model     = build_model(cfg, bundle.stats, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    log_path  = out_dir / "train_log.csv"
    init_train_log(log_path)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        log_gpu_memory(device)

    best_r50    = -1.0
    best_metrics: dict[str, Any] = {}
    best_bucket:  dict[str, Any] = {}
    best_epoch  = 0
    total_train_sec = 0.0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0         = time.time()
        t_train0   = time.time()
        train_loss = train_epoch(model, loader, optimizer, cfg, device, epoch)
        train_sec  = time.time() - t_train0
        t_eval0    = time.time()
        valid_metrics, hit_users, bucket = evaluate_with_buckets(
            model, bundle.valid_df, hist_mat, raw_lens, seen,
            cfg, bundle.stats, device, "valid",
        )
        eval_sec        = time.time() - t_eval0
        epoch_sec       = time.time() - t0
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

        append_train_log(log_path, {
            "epoch": epoch, "train_loss": train_loss,
            "valid_recall@50": valid_metrics["recall@50"],
            "valid_recall@20": valid_metrics["recall@20"],
            "valid_recall@100": valid_metrics["recall@100"],
            "valid_ndcg@50":   valid_metrics["ndcg@50"],
            "valid_mrr@50":    valid_metrics["mrr@50"],
            "train_time_seconds": train_sec,
            "eval_time_seconds": eval_sec,
            "epoch_time_seconds": epoch_sec,
            "pooling_type": cfg["pooling_type"],
        })

        if valid_metrics["recall@50"] > best_r50:
            best_r50     = float(valid_metrics["recall@50"])
            best_metrics = valid_metrics
            best_bucket  = bucket
            best_epoch   = epoch
            np.save(out_dir / "hit_users_valid_r50.npy",
                    np.array(sorted(hit_users), dtype=np.int64))
            save_checkpoint(out_dir / "checkpoints" / "best_model.pt",
                            model, cfg, bundle.stats, epoch, best_r50)
            logging.info("  → new best: epoch=%s R@50=%.6f  hit_users=%d",
                         epoch, best_r50, len(hit_users))

        if device.type == "cuda":
            log_gpu_memory(device)

    top_dir = Path(cfg["output_dir"]).parent
    top_dir.mkdir(parents=True, exist_ok=True)

    final_summary = {
        "pooling_type":          cfg["pooling_type"],
        "history_max_len":       int(cfg["history_max_len"]),
        "parameter_count":       count_trainable_params(model),
        "best_epoch":            best_epoch,
        "best_valid_recall@50":  best_r50,
        "best_valid_recall@20":  best_metrics.get("recall@20", 0.0),
        "best_valid_recall@100": best_metrics.get("recall@100", 0.0),
        "best_valid_ndcg@50":    best_metrics.get("ndcg@50", 0.0),
        "best_valid_mrr@50":     best_metrics.get("mrr@50", 0.0),
        "num_eval_users":        best_metrics.get("num_eval_users", 0),
        "total_train_sec":       total_train_sec,
        "bucket_recall@50":      {b: best_bucket[b]["recall@50"] for b, _ in HISTORY_BUCKETS},
        "bucket_counts":         {b: best_bucket[b]["count"]     for b, _ in HISTORY_BUCKETS},
        "output_dir":            str(out_dir),
        "gpu_peak_memory_gb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**3)
            if device.type == "cuda" else None
        ),
    }
    write_json(out_dir / "metrics_valid_best.json", final_summary)

    fname = FNAME_MAP.get(cfg["pooling_type"])
    if fname:
        write_json(top_dir / fname, final_summary)
        logging.info("Summary → %s", top_dir / fname)

    logging.info("Training complete: pooling=%s  best_epoch=%s  R@50=%.6f  total_time=%.1fs",
                 cfg["pooling_type"], best_epoch, best_r50, total_train_sec)


# ---------------------------------------------------------------------------
# Phase 0 and comparison
# ---------------------------------------------------------------------------

def phase0_check(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    require_config(cfg)
    set_seed(int(cfg["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg["device"]))

    bundle = load_data(Path(cfg["data_dir"]))
    n_users = int(bundle.stats["n_users"])
    max_len = int(cfg["history_max_len"])
    hist_mat = build_history_matrix(bundle.train_df, n_users, max_len)
    raw_lens = compute_raw_history_lengths(bundle.train_df, n_users)
    seen = build_seen_items(bundle.train_df)

    variants = ["time_decay", "transformer_vanilla", "transformer_timeaware"]
    checks: dict[str, Any] = {}
    for variant in variants:
        vcfg = dict(cfg)
        vcfg["pooling_type"] = variant
        vcfg["num_heads"] = int(vcfg.get("num_heads", DEFAULT_NUM_HEADS))
        vcfg["ffn_dim"] = int(vcfg.get("ffn_dim", DEFAULT_FFN_DIM))
        vcfg["dropout"] = float(vcfg.get("dropout", DEFAULT_DROPOUT))
        vcfg["num_layers"] = int(vcfg.get("num_layers", DEFAULT_NUM_LAYERS))
        set_seed(int(vcfg["seed"]))
        model = build_model(vcfg, bundle.stats, device)
        forward = sanity_forward_check(model, bundle.train_df, hist_mat, vcfg, device)
        tiny_train = tiny_train_step_check(model, bundle.train_df, hist_mat, vcfg, device)
        checks[variant] = {
            "parameter_count": count_trainable_params(model),
            "forward_check": forward,
            "tiny_train_step": tiny_train,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    valid_non_cold = int((~bundle.valid_df["is_cold_item_for_eval"].astype(bool)).sum())
    test_non_cold = int((~bundle.test_df["is_cold_item_for_eval"].astype(bool)).sum())
    payload: dict[str, Any] = {
        "status": "passed" if all(
            c["forward_check"]["passed"] and c["tiny_train_step"]["passed"]
            for c in checks.values()
        ) else "failed",
        "data_dir": str(cfg["data_dir"]),
        "history_max_len": max_len,
        "history_matrix_shape": list(hist_mat.shape),
        "history_matrix_padding_value": -1,
        "raw_history": {
            "non_empty_users": int((raw_lens > 0).sum()),
            "avg_len": float(raw_lens.mean()),
            "gt20_users": int((raw_lens > 20).sum()),
            "gt100_users": int((raw_lens > 100).sum()),
        },
        "split_check": {
            "train_rows": int(len(bundle.train_df)),
            "valid_rows": int(len(bundle.valid_df)),
            "test_rows": int(len(bundle.test_df)),
            "valid_non_cold_eval_users": valid_non_cold,
            "test_non_cold_eval_users": test_non_cold,
        },
        "seen_mask_check": {
            "valid_seen_source": "train only",
            "test_seen_source": "train + valid in full-eval scripts; this smoke uses valid train-only eval",
            "train_seen_users": int(len(seen)),
        },
        "variants": checks,
        "blocker": None,
    }
    write_json(out_dir / "phase0_check.json", payload)
    logging.info("Phase 0 check saved: %s", out_dir / "phase0_check.json")
    return payload

def load_run(run_dir: Path) -> tuple[dict[str, Any], set[int]]:
    summary = json.loads((run_dir / "metrics_valid_best.json").read_text(encoding="utf-8"))
    hits    = set(int(x) for x in np.load(run_dir / "hit_users_valid_r50.npy").tolist())
    return summary, hits


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_phase1_reports(result: dict[str, Any], out_dir: Path) -> None:
    td = result["models"][0]
    vanilla = result["models"][1]
    timeaware = result["models"][2]
    stop = "进入 Phase 2" if result["continue_to_phase2"] else "停止，不进入 Phase 2"

    def fmt(v: Any) -> str:
        return f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v)

    report = f"""# Transformer User Tower Investigation Report

## Phase 1 Paired Smoke

**状态：** {stop}

| 模型 | best_epoch | R@50 | NDCG@50 | MRR@50 | Δ vs td | gt20 Δ | 参数量 | 训练秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| time_decay_max100 | {td['best_epoch']} | {fmt(td['recall@50'])} | {fmt(td['ndcg@50'])} | {fmt(td['mrr@50'])} | 0.000000 | 0.000000 | {td['parameter_count']} | {fmt(td['total_train_sec'])} |
| transformer_vanilla_max100 | {vanilla['best_epoch']} | {fmt(vanilla['recall@50'])} | {fmt(vanilla['ndcg@50'])} | {fmt(vanilla['mrr@50'])} | {fmt(vanilla['delta_vs_td_recall@50'])} | {fmt(vanilla['bucket_delta_vs_td']['gt20'])} | {vanilla['parameter_count']} | {fmt(vanilla['total_train_sec'])} |
| transformer_timeaware_max100 | {timeaware['best_epoch']} | {fmt(timeaware['recall@50'])} | {fmt(timeaware['ndcg@50'])} | {fmt(timeaware['mrr@50'])} | {fmt(timeaware['delta_vs_td_recall@50'])} | {fmt(timeaware['bucket_delta_vs_td']['gt20'])} | {timeaware['parameter_count']} | {fmt(timeaware['total_train_sec'])} |

### History Bucket Recall@50

| 桶 | time_decay | vanilla | vanilla Δ | timeaware | timeaware Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=5 | {fmt(td['bucket_recall@50']['le5'])} | {fmt(vanilla['bucket_recall@50']['le5'])} | {fmt(vanilla['bucket_delta_vs_td']['le5'])} | {fmt(timeaware['bucket_recall@50']['le5'])} | {fmt(timeaware['bucket_delta_vs_td']['le5'])} |
| 6-20 | {fmt(td['bucket_recall@50']['6to20'])} | {fmt(vanilla['bucket_recall@50']['6to20'])} | {fmt(vanilla['bucket_delta_vs_td']['6to20'])} | {fmt(timeaware['bucket_recall@50']['6to20'])} | {fmt(timeaware['bucket_delta_vs_td']['6to20'])} |
| >20 | {fmt(td['bucket_recall@50']['gt20'])} | {fmt(vanilla['bucket_recall@50']['gt20'])} | {fmt(vanilla['bucket_delta_vs_td']['gt20'])} | {fmt(timeaware['bucket_recall@50']['gt20'])} | {fmt(timeaware['bucket_delta_vs_td']['gt20'])} |

### Unique Hit vs Time-decay

| 模型 | both_with_td | only_model | only_td |
| --- | ---: | ---: | ---: |
| transformer_vanilla_max100 | {vanilla['both_with_td']} | {vanilla['only_model']} | {vanilla['only_td']} |
| transformer_timeaware_max100 | {timeaware['both_with_td']} | {timeaware['only_model']} | {timeaware['only_td']} |

### Stop / Continue

- best Transformer: `{result['best_transformer']}`
- best overall delta vs time_decay: {fmt(result['best_transformer_delta_recall@50'])}
- best >20 bucket delta vs time_decay: {fmt(result['best_transformer_gt20_delta_recall@50'])}
- decision: **{stop}**
- reason: {result['stop_reason'] or 'Phase 1 threshold met'}

本结果是 50K limited valid paired smoke，不是 full test 结论，不改变 final model。
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    docs_append = "\n---\n\n## 6. Phase 1：max_len=100 Paired Smoke\n\n" + report.split("## Phase 1 Paired Smoke\n", 1)[1]
    append_text(Path("docs/reports/transformer_user_tower_investigation.md"), docs_append)

    daily_append = f"""

---

## Part 13：Transformer User Tower Investigation - Phase 1 Paired Smoke

**脚本：** `scripts/train_transformer_maxlen100_smoke.py`  
**输出：** `outputs/transformer_user_tower_investigation/`  
**状态：** {stop}

| 模型 | best_epoch | R@50 | NDCG@50 | MRR@50 | Δ vs td | gt20 Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| time_decay_max100 | {td['best_epoch']} | {fmt(td['recall@50'])} | {fmt(td['ndcg@50'])} | {fmt(td['mrr@50'])} | 0.000000 | 0.000000 |
| transformer_vanilla_max100 | {vanilla['best_epoch']} | {fmt(vanilla['recall@50'])} | {fmt(vanilla['ndcg@50'])} | {fmt(vanilla['mrr@50'])} | {fmt(vanilla['delta_vs_td_recall@50'])} | {fmt(vanilla['bucket_delta_vs_td']['gt20'])} |
| transformer_timeaware_max100 | {timeaware['best_epoch']} | {fmt(timeaware['recall@50'])} | {fmt(timeaware['ndcg@50'])} | {fmt(timeaware['mrr@50'])} | {fmt(timeaware['delta_vs_td_recall@50'])} | {fmt(timeaware['bucket_delta_vs_td']['gt20'])} |

### Unique Hit vs Time-decay

| 模型 | only_model | only_td |
| --- | ---: | ---: |
| transformer_vanilla_max100 | {vanilla['only_model']} | {vanilla['only_td']} |
| transformer_timeaware_max100 | {timeaware['only_model']} | {timeaware['only_td']} |

### Stop reason

{result['stop_reason'] or 'Phase 1 threshold met; enter Phase 2.'}

**说明：** 本阶段是 50K limited valid smoke，不是 full test 结论，不改变 final model，不更新 README。
"""
    append_text(Path("docs/daily_logs/2026-05-20.md"), daily_append)


def compare(td_dir: Path, vanilla_dir: Path, timeaware_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    td_s, td_hits = load_run(td_dir)
    vanilla_s, vanilla_hits = load_run(vanilla_dir)
    timeaware_s, timeaware_hits = load_run(timeaware_dir)

    def bdelta(a: dict, b: dict) -> dict[str, float]:
        return {bk: a["bucket_recall@50"].get(bk, 0.0) - b["bucket_recall@50"].get(bk, 0.0)
                for bk in ("le5", "6to20", "gt20")}

    def model_row(name: str, summary: dict[str, Any], hits: set[int]) -> dict[str, Any]:
        return {
            "name": name,
            "pooling_type": summary["pooling_type"],
            "best_epoch": summary["best_epoch"],
            "recall@50": summary["best_valid_recall@50"],
            "ndcg@50": summary["best_valid_ndcg@50"],
            "mrr@50": summary["best_valid_mrr@50"],
            "delta_vs_td_recall@50": summary["best_valid_recall@50"] - td_s["best_valid_recall@50"],
            "bucket_recall@50": summary["bucket_recall@50"],
            "bucket_delta_vs_td": bdelta(summary, td_s),
            "bucket_counts": summary["bucket_counts"],
            "total_hits": len(hits),
            "both_with_td": len(hits & td_hits),
            "only_model": len(hits - td_hits),
            "only_td": len(td_hits - hits),
            "total_train_sec": summary["total_train_sec"],
            "parameter_count": summary.get("parameter_count"),
            "gpu_peak_memory_gb": summary.get("gpu_peak_memory_gb"),
        }

    vanilla_delta = vanilla_s["best_valid_recall@50"] - td_s["best_valid_recall@50"]
    timeaware_delta = timeaware_s["best_valid_recall@50"] - td_s["best_valid_recall@50"]
    vanilla_bucket_delta = bdelta(vanilla_s, td_s)
    timeaware_bucket_delta = bdelta(timeaware_s, td_s)
    best_name, best_summary = max(
        [("transformer_vanilla", vanilla_s), ("transformer_timeaware", timeaware_s)],
        key=lambda x: x[1]["best_valid_recall@50"],
    )
    best_delta = best_summary["best_valid_recall@50"] - td_s["best_valid_recall@50"]
    best_gt20_delta = bdelta(best_summary, td_s)["gt20"]
    continue_to_phase2 = best_delta >= 0.001 or best_gt20_delta >= 0.002

    result: dict[str, Any] = {
        "td_recall@50":     td_s["best_valid_recall@50"],
        "vanilla_recall@50": vanilla_s["best_valid_recall@50"],
        "timeaware_recall@50": timeaware_s["best_valid_recall@50"],
        "vanilla_delta_recall@50": vanilla_delta,
        "timeaware_delta_recall@50": timeaware_delta,
        "vanilla_gt20_delta_recall@50": vanilla_bucket_delta["gt20"],
        "timeaware_gt20_delta_recall@50": timeaware_bucket_delta["gt20"],
        "best_transformer": best_name,
        "best_transformer_delta_recall@50": best_delta,
        "best_transformer_gt20_delta_recall@50": best_gt20_delta,
        "continue_to_phase2": continue_to_phase2,
        "stop_reason": None if continue_to_phase2 else (
            "B/C both failed thresholds: overall delta >= +0.001 or gt20 delta >= +0.002"
        ),
        "td_ndcg@50":       td_s["best_valid_ndcg@50"],
        "td_mrr@50":        td_s["best_valid_mrr@50"],
        "td_best_epoch":    td_s["best_epoch"],
        "td_total_train_sec":    td_s["total_train_sec"],
        "td_bucket_recall@50":    td_s["bucket_recall@50"],
        "td_bucket_counts":       td_s["bucket_counts"],
        "num_eval_users":    td_s["num_eval_users"],
        "td_total_hits":     len(td_hits),
        "models": [
            model_row("time_decay_max100", td_s, td_hits),
            model_row("transformer_vanilla_max100", vanilla_s, vanilla_hits),
            model_row("transformer_timeaware_max100", timeaware_s, timeaware_hits),
        ],
    }

    write_json(out_dir / "unique_hit_comparison.json", result)
    write_json(out_dir / "phase1_results.json", result)
    write_phase1_reports(result, out_dir)
    logging.info("Comparison saved: %s", out_dir / "unique_hit_comparison.json")
    logging.info("Phase 1 results saved: %s", out_dir / "phase1_results.json")
    logging.info("Phase 1 reports written: %s and docs/reports/transformer_user_tower_investigation.md",
                 out_dir / "report.md")

    logging.info("===== COMPARISON SUMMARY =====")
    logging.info("Time-decay (max_len=100)   R@50=%.6f  ep=%s  t=%.0fs",
                 result["td_recall@50"], result["td_best_epoch"], result["td_total_train_sec"])
    for m in result["models"][1:]:
        logging.info("%-30s R@50=%.6f ep=%s t=%.0fs Δ=%+.6f gt20Δ=%+.6f",
                     m["name"], m["recall@50"], m["best_epoch"], m["total_train_sec"],
                     m["delta_vs_td_recall@50"], m["bucket_delta_vs_td"]["gt20"])
        for bk in ("le5", "6to20", "gt20"):
            logging.info("  %-6s td=%.6f model=%.6f Δ%+.6f",
                         bk, td_s["bucket_recall@50"][bk],
                         m["bucket_recall@50"][bk], m["bucket_delta_vs_td"][bk])
        logging.info("  unique: both_with_td=%d only_model=%d only_td=%d",
                     m["both_with_td"], m["only_model"], m["only_td"])
    logging.info("Continue to Phase 2: %s  reason=%s",
                 result["continue_to_phase2"], result["stop_reason"])
    logging.info("==============================")
    return result



# ---------------------------------------------------------------------------
# Eval-only full valid/test
# ---------------------------------------------------------------------------

def eval_only(cfg: dict[str, Any], checkpoint_path: Path, eval_output_dir: Path, full_eval: bool) -> dict[str, Any]:
    require_config(cfg)
    set_seed(int(cfg["seed"]))
    if full_eval:
        cfg["eval_max_users"] = None
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_output_dir / "run_config.json", cfg)
    device = resolve_device(str(cfg["device"]))

    bundle = load_data(Path(cfg["data_dir"]))
    n_users = int(bundle.stats["n_users"])
    max_len = int(cfg["history_max_len"])

    train_history_matrix = build_history_matrix(bundle.train_df, n_users, max_len)
    train_raw_lens = compute_raw_history_lengths(bundle.train_df, n_users)
    train_seen = build_seen_items(bundle.train_df)

    test_history_frame = pd.concat([bundle.train_df, bundle.valid_df[TRAIN_COLUMNS]], ignore_index=True)
    test_history_matrix = build_history_matrix(test_history_frame, n_users, max_len)
    test_raw_lens = compute_raw_history_lengths(test_history_frame, n_users)
    test_seen = merge_seen_items(train_seen, bundle.valid_df)

    model = build_model(cfg, bundle.stats, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(
        "checkpoint loaded: epoch=%s best_metric=%.6f path=%s",
        checkpoint.get("epoch"), float(checkpoint.get("best_metric_value", 0.0)), checkpoint_path,
    )

    valid_metrics, valid_hits, valid_bucket = evaluate_with_buckets(
        model, bundle.valid_df, train_history_matrix, train_raw_lens, train_seen,
        cfg, bundle.stats, device, "valid_full" if full_eval else "valid",
    )
    write_json(eval_output_dir / ("metrics_valid_full.json" if full_eval else "metrics_valid.json"), valid_metrics)
    write_json(eval_output_dir / ("bucket_valid_full.json" if full_eval else "bucket_valid.json"), valid_bucket)
    np.save(eval_output_dir / ("hit_users_valid_full_r50.npy" if full_eval else "hit_users_valid_r50.npy"),
            np.array(sorted(valid_hits), dtype=np.int64))

    test_metrics, test_hits, test_bucket = evaluate_with_buckets(
        model, bundle.test_df, test_history_matrix, test_raw_lens, test_seen,
        cfg, bundle.stats, device, "test_full" if full_eval else "test",
    )
    write_json(eval_output_dir / ("metrics_test_full.json" if full_eval else "metrics_test.json"), test_metrics)
    write_json(eval_output_dir / ("bucket_test_full.json" if full_eval else "bucket_test.json"), test_bucket)
    np.save(eval_output_dir / ("hit_users_test_full_r50.npy" if full_eval else "hit_users_test_r50.npy"),
            np.array(sorted(test_hits), dtype=np.int64))

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(eval_output_dir),
        "pooling_type": cfg["pooling_type"],
        "history_max_len": int(cfg["history_max_len"]),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "valid_recall@50": valid_metrics["recall@50"],
        "test_recall@50": test_metrics["recall@50"],
        "valid_ndcg@50": valid_metrics["ndcg@50"],
        "test_ndcg@50": test_metrics["ndcg@50"],
        "valid_mrr@50": valid_metrics["mrr@50"],
        "test_mrr@50": test_metrics["mrr@50"],
        "valid_bucket_recall@50": {k: v["recall@50"] for k, v in valid_bucket.items()},
        "test_bucket_recall@50": {k: v["recall@50"] for k, v in test_bucket.items()},
    }
    write_json(eval_output_dir / "eval_summary.json", summary)
    logging.info(
        "eval-only complete: valid R@50=%.6f test R@50=%.6f",
        valid_metrics["recall@50"], test_metrics["recall@50"],
    )
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    if args.eval_only:
        if not args.config:
            raise ValueError("Provide --config for --eval_only.")
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint for --eval_only.")
        if not args.eval_output_dir:
            raise ValueError("Provide --eval_output_dir for --eval_only.")
        cfg = load_config(Path(args.config))
        cfg["config_path"] = args.config
        eval_only(cfg, Path(args.checkpoint), Path(args.eval_output_dir), args.full_eval)
        return

    if args.phase0_check:
        if not args.config:
            raise ValueError("Provide --config for --phase0_check.")
        cfg = load_config(Path(args.config))
        cfg["config_path"] = args.config
        phase0_check(cfg, Path(args.out_dir))
        return

    if args.compare:
        td_dir = Path(args.td_dir)
        vanilla_dir = Path(args.vanilla_dir)
        timeaware_dir = Path(args.timeaware_dir)
        out_dir = Path(args.out_dir)
        for d in (td_dir, vanilla_dir, timeaware_dir):
            if not (d / "metrics_valid_best.json").exists():
                raise FileNotFoundError(
                    f"Missing metrics_valid_best.json in {d}. Run training first."
                )
        compare(td_dir, vanilla_dir, timeaware_dir, out_dir)
        return

    if not args.config:
        raise ValueError("Provide --config for training or --compare for comparison.")

    cfg = load_config(Path(args.config))
    cfg["config_path"] = args.config
    train(cfg)


if __name__ == "__main__":
    main()

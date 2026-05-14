#!/usr/bin/env python3
"""Model-based Hard Negative Mining smoke on Text + Mean Pooling Two-Tower.

SMOKE ONLY — not the final model. Does not replace the current best model
(Text + Mean Pooling tau=0.15, full test Recall@50=0.076337).

Hard negatives are drawn from the CURRENT FINAL MODEL's item embedding space
(loaded from checkpoint), not frozen text embeddings. This yields candidates
the model genuinely retrieves as near-neighbors, covering all items regardless
of whether they have text metadata.

Steps:
  1. Load final-model checkpoint → export all L2-normalized item embeddings.
  2. Build Faiss IndexFlatIP nearest-neighbor table (top_k=50 per item).
  3. Train a FRESH model for 1 epoch with:
       total_loss = in-batch cross-entropy + lambda_hn * HN cross-entropy
  4. Evaluate limited valid (eval_max_users=50000).
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
    logging.error("Missing dependency: %s. Install via project .venv: %s", missing_name, package_hint)
    raise SystemExit(1) from exc


REQUIRED_CONFIG_KEYS = [
    "data_dir", "output_dir", "embedding_dim", "batch_size",
    "learning_rate", "weight_decay", "epochs", "temperature",
    "use_l2_norm", "seed", "eval_k_list", "eval_batch_size",
    "eval_max_users", "num_workers", "device", "save_best_by",
    "history_max_len", "history_weight", "item_text_embedding_path",
    "item_has_text_path", "text_proj_dim", "use_has_text_mask", "item_fusion",
    "model_hn_checkpoint", "lambda_hn", "hard_negatives_per_sample", "hn_top_k",
]
TRAIN_COLUMNS = ["user_idx", "item_idx", "timestamp"]
EVAL_COLUMNS = ["user_idx", "item_idx", "timestamp", "is_cold_item_for_eval"]


# ---------------------------------------------------------------------------
# Config / Setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-based HNM smoke: Text + Mean Pooling Two-Tower.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format: {config_path}")
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
        logging.warning("Config requested cuda but CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: dict[str, Any]


def load_data(data_dir: Path) -> DataBundle:
    logging.info("Loading data from %s", data_dir)
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
        num_workers=int(config["num_workers"]),
        generator=generator,
        collate_fn=MeanPoolCollator(history_matrix),
    )


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for user_idx, group in frame.groupby("user_idx", sort=False):
        seen[int(user_idx)] = set(int(i) for i in group["item_idx"].tolist())
    return seen


# ---------------------------------------------------------------------------
# Model (standalone copy — identical to TextMeanPoolTwoTower)
# ---------------------------------------------------------------------------

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

    def mean_history_embedding(
        self,
        history_item_idx: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = history_item_idx >= 0
        if exclude_item_idx is not None:
            valid_mask = valid_mask & (history_item_idx != exclude_item_idx.unsqueeze(1))
        safe_history = history_item_idx.clamp_min(0)
        history_emb = self.item_id_embedding(safe_history)
        mask = valid_mask.unsqueeze(-1).to(history_emb.dtype)
        summed = (history_emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def raw_user(
        self,
        user_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
        exclude_item_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        id_emb = self.user_embedding(user_idx)
        hist_pool = self.mean_history_embedding(history_item_idx, exclude_item_idx)
        return id_emb + self.history_weight * hist_pool

    def _item_prenorm(self, item_idx: torch.Tensor) -> torch.Tensor:
        id_emb = self.item_id_embedding(item_idx)
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
        user_emb = self.raw_user(user_idx, history_item_idx, exclude_item_idx)
        return F.normalize(user_emb, p=2, dim=-1) if self.use_l2_norm else user_emb

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_prenorm(item_idx)
        return F.normalize(item_emb, p=2, dim=-1) if self.use_l2_norm else item_emb

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


def load_text_artifacts(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    emb_path = Path(config["item_text_embedding_path"])
    has_path = Path(config["item_has_text_path"])
    if not emb_path.exists():
        raise FileNotFoundError(f"item_text_embedding_path does not exist: {emb_path}")
    if not has_path.exists():
        raise FileNotFoundError(f"item_has_text_path does not exist: {has_path}")
    text_emb = torch.from_numpy(np.load(emb_path).astype(np.float32))
    has_text = torch.from_numpy(np.load(has_path).astype(np.float32))
    logging.info("text_emb: shape=%s", tuple(text_emb.shape))
    logging.info(
        "has_text: %d/%d items have real text (%.1f%%)",
        int(has_text.sum()), len(has_text), 100.0 * float(has_text.mean()),
    )
    return text_emb, has_text


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


# ---------------------------------------------------------------------------
# Step 1: Export item embeddings from final-model checkpoint
# ---------------------------------------------------------------------------

def export_item_embeddings_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Load the final-model checkpoint and export all L2-normalized item embeddings."""
    logging.info("Loading reference checkpoint for HN table: %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ref_config = checkpoint["config"]
    ref_stats = checkpoint["stats"]
    n_items = int(ref_stats["n_items"])
    embedding_dim = int(ref_config["embedding_dim"])
    logging.info("Reference model: n_items=%d embedding_dim=%d epoch=%s", n_items, embedding_dim, checkpoint.get("epoch"))

    ref_text_emb = torch.from_numpy(np.load(ref_config["item_text_embedding_path"]).astype(np.float32))
    ref_has_text = torch.from_numpy(np.load(ref_config["item_has_text_path"]).astype(np.float32))
    ref_model = TextMeanPoolTwoTower(
        num_users=int(ref_stats["n_users"]),
        num_items=n_items,
        embedding_dim=embedding_dim,
        text_emb=ref_text_emb,
        has_text=ref_has_text,
        text_proj_dim=int(ref_config["text_proj_dim"]),
        use_l2_norm=bool(ref_config["use_l2_norm"]),
        use_has_text_mask=bool(ref_config["use_has_text_mask"]),
        history_weight=float(ref_config["history_weight"]),
    ).to(device)
    ref_model.load_state_dict(checkpoint["model_state_dict"])
    ref_model.eval()

    logging.info("Exporting item embeddings from reference model (chunk_size=65536)...")
    chunks = []
    with torch.no_grad():
        for start in range(0, n_items, 65536):
            end = min(start + 65536, n_items)
            item_idx = torch.arange(start, end, device=device)
            chunks.append(ref_model.encode_items(item_idx).detach().cpu().numpy())
    item_emb_np = np.concatenate(chunks, axis=0).astype(np.float32)
    logging.info("Exported item embeddings: shape=%s (L2 normalized)", item_emb_np.shape)

    del ref_model, ref_text_emb, ref_has_text, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return item_emb_np


# ---------------------------------------------------------------------------
# Step 2: Build model-based HN table with Faiss IndexFlatIP
# ---------------------------------------------------------------------------

def build_model_hard_negative_table(item_emb_np: np.ndarray, top_k: int) -> np.ndarray:
    """Find top_k model-similar neighbors for each item using Faiss IndexFlatIP.

    item_emb_np is L2-normalized, so inner product == cosine similarity.
    Self is excluded from each row's results.
    Returns hn_table of shape (n_items, top_k) with dtype int32.
    """
    import faiss  # local import: optional dependency, verified available

    n_items, embedding_dim = item_emb_np.shape
    logging.info(
        "Building model-based HN table: n_items=%d embedding_dim=%d top_k=%d (Faiss IndexFlatIP)",
        n_items, embedding_dim, top_k,
    )
    t0 = time.time()
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_emb_np)
    # Search top_k + 2 to have slack for excluding self
    _D, I = index.search(item_emb_np, top_k + 2)
    del index, _D

    # Exclude self from each row; self has IP=1.0 and is always retrieved
    hn_table = np.full((n_items, top_k), -1, dtype=np.int32)
    for i in range(n_items):
        non_self = [int(j) for j in I[i] if int(j) != i and int(j) >= 0][:top_k]
        hn_table[i, : len(non_self)] = non_self

    elapsed = time.time() - t0
    valid_entries = int((hn_table >= 0).sum())
    logging.info(
        "HN table built in %.2fs: shape=%s valid_entries=%d (%.1f%%)",
        elapsed, hn_table.shape, valid_entries, 100.0 * valid_entries / (n_items * top_k),
    )
    return hn_table


# ---------------------------------------------------------------------------
# Hard Negative loss
# ---------------------------------------------------------------------------

def compute_hn_loss(
    model: TextMeanPoolTwoTower,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    hn_table: np.ndarray,
    hard_negatives_per_sample: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Compute auxiliary HN loss for valid samples in the batch.

    For each sample:
      - Exclusion set = positive item + user train history items.
      - Take up to hard_negatives_per_sample candidates from hn_table[positive].
      - CE([pos_score, hn_scores] / temperature, label=0).

    Returns (loss, n_valid_samples).
    """
    device = user_idx.device
    user_np = user_idx.cpu().numpy()
    item_np = item_idx.cpu().numpy()
    hist_np = history_item_idx.cpu().numpy()

    valid_user_rows: list[int] = []
    valid_pos_items: list[int] = []
    valid_hist_rows: list[np.ndarray] = []
    valid_hn_items: list[int] = []

    for i in range(len(user_np)):
        pos_item = int(item_np[i])
        hist_row = hist_np[i]
        exclusion: set[int] = {pos_item}
        for h in hist_row:
            if int(h) >= 0:
                exclusion.add(int(h))
        candidates = hn_table[pos_item]
        available = [int(c) for c in candidates if int(c) >= 0 and int(c) not in exclusion]
        if len(available) < hard_negatives_per_sample:
            continue
        selected = available[:hard_negatives_per_sample]
        valid_user_rows.append(i)
        valid_pos_items.append(pos_item)
        valid_hist_rows.append(hist_row)
        valid_hn_items.extend(selected)

    n_valid = len(valid_user_rows)
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    vu = torch.tensor(user_np[valid_user_rows], dtype=torch.long, device=device)
    vp = torch.tensor(valid_pos_items, dtype=torch.long, device=device)
    vh = torch.tensor(np.stack(valid_hist_rows), dtype=torch.long, device=device)
    vhn = torch.tensor(valid_hn_items, dtype=torch.long, device=device)

    user_emb = model.encode_users(vu, vh, exclude_item_idx=vp)           # (n_valid, d)
    pos_emb = model.encode_items(vp)                                      # (n_valid, d)
    hn_emb = model.encode_items(vhn).view(n_valid, hard_negatives_per_sample, -1)  # (n_valid, k, d)

    pos_score = (user_emb * pos_emb).sum(-1, keepdim=True) / temperature  # (n_valid, 1)
    hn_scores = torch.bmm(hn_emb, user_emb.unsqueeze(-1)).squeeze(-1) / temperature  # (n_valid, k)
    logits = torch.cat([pos_score, hn_scores], dim=1)                    # (n_valid, 1+k)
    labels = torch.zeros(n_valid, dtype=torch.long, device=device)       # positive is index 0
    loss = F.cross_entropy(logits, labels)
    return loss, n_valid


# ---------------------------------------------------------------------------
# Train step / epoch
# ---------------------------------------------------------------------------

def train_one_step_hnm(
    model: TextMeanPoolTwoTower,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    history_item_idx: torch.Tensor,
    hn_table: np.ndarray,
    hard_negatives_per_sample: int,
    lambda_hn: float,
    temperature: float,
) -> tuple[float, float, float, int]:
    optimizer.zero_grad(set_to_none=True)

    raw_user_emb, raw_item_emb = model.raw_batch(user_idx, item_idx, history_item_idx)
    user_emb = F.normalize(raw_user_emb, p=2, dim=-1) if model.use_l2_norm else raw_user_emb
    item_emb = F.normalize(raw_item_emb, p=2, dim=-1) if model.use_l2_norm else raw_item_emb
    logits = (user_emb @ item_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    main_loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(main_loss):
        raise FloatingPointError("main_loss has nan or inf; stopping.")

    hn_loss, n_valid = compute_hn_loss(
        model, user_idx, item_idx, history_item_idx,
        hn_table, hard_negatives_per_sample, temperature,
    )
    total_loss = main_loss + lambda_hn * hn_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("total_loss has nan or inf; stopping.")
    total_loss.backward()
    optimizer.step()
    return float(total_loss.item()), float(main_loss.item()), float(hn_loss.item()), n_valid


def train_epoch_hnm(
    model: TextMeanPoolTwoTower,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    hn_table: np.ndarray,
) -> tuple[float, float, float, int]:
    model.train()
    acc_total = 0.0
    acc_main = 0.0
    acc_hn = 0.0
    total_examples = 0
    total_valid_hn = 0

    lambda_hn = float(config["lambda_hn"])
    hard_negatives_per_sample = int(config["hard_negatives_per_sample"])
    temperature = float(config["temperature"])

    for batch_idx, (user_idx, item_idx, history_item_idx) in enumerate(train_loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        history_item_idx = history_item_idx.to(device)
        total_loss, main_loss, hn_loss, n_valid = train_one_step_hnm(
            model, optimizer, user_idx, item_idx, history_item_idx,
            hn_table, hard_negatives_per_sample, lambda_hn, temperature,
        )
        batch_size = int(user_idx.shape[0])
        if batch_idx == 0:
            logging.info(
                "epoch %d batch 0: total_loss=%.4f main_loss=%.4f hn_loss=%.4f hn_valid=%d/%d",
                epoch, total_loss, main_loss, hn_loss, n_valid, batch_size,
            )
        acc_total += total_loss * batch_size
        acc_main += main_loss * batch_size
        acc_hn += hn_loss * batch_size
        total_examples += batch_size
        total_valid_hn += n_valid

    avg_total = acc_total / total_examples if total_examples else 0.0
    avg_main = acc_main / total_examples if total_examples else 0.0
    avg_hn = acc_hn / total_examples if total_examples else 0.0
    logging.info(
        "epoch %d summary: total_valid_hn=%d / total_examples=%d (ratio=%.3f)",
        epoch, total_valid_hn, total_examples, total_valid_hn / max(total_examples, 1),
    )
    logging.info(
        "epoch %d done: train_total_loss=%.6f  train_main_loss=%.6f  train_hn_loss=%.6f",
        epoch, avg_total, avg_main, avg_hn,
    )
    return avg_total, avg_main, avg_hn, total_valid_hn


# ---------------------------------------------------------------------------
# Evaluation (identical protocol to parent script)
# ---------------------------------------------------------------------------

def prepare_eval_frame(eval_df: pd.DataFrame, eval_max_users: int | None) -> pd.DataFrame:
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    if eval_max_users is not None:
        non_cold = non_cold.head(int(eval_max_users)).copy()
    return non_cold


def encode_all_items_cpu(model: TextMeanPoolTwoTower, num_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_items, 65536):
            end = min(start + 65536, num_items)
            item_idx = torch.arange(start, end, device=device)
            chunks.append(model.encode_items(item_idx).detach().cpu())
    return torch.cat(chunks, dim=0)


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
    logging.info("%s eval users=%s eval_max_users=%s", split_name, len(eval_targets), eval_max_users)

    metric_sums = {f"recall@{k}": 0.0 for k in k_list}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in k_list})
    metric_sums.update({f"mrr@{k}": 0.0 for k in k_list})
    diagnostics: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_targets), eval_batch_size):
            batch = eval_targets.iloc[start: start + eval_batch_size]
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
            for row_pos, (user_idx, target_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)
            ):
                seen = seen_items.get(int(user_idx), set())
                if seen:
                    scores[row_pos, torch.as_tensor(list(seen), dtype=torch.long, device=device)] = -torch.inf
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
        message = str(exc).lower()
        if "out of memory" not in message or int(config["eval_batch_size"]) <= 128:
            raise
        old_bs = int(config["eval_batch_size"])
        config["eval_batch_size"] = 128
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.warning("Eval OOM; retrying with eval_batch_size 128 (was %d).", old_bs)
        return evaluate_once(model, eval_df, history_matrix, seen_items, config, stats, device, split_name)


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------

def train_model_hnm_smoke(config: dict[str, Any]) -> None:
    require_config(config)
    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    logging.info(
        "device=%s embedding_dim=%s batch_size=%s lr=%s temperature=%s "
        "lambda_hn=%s hard_negatives_per_sample=%s hn_top_k=%s",
        device, config["embedding_dim"], config["batch_size"], config["learning_rate"],
        config["temperature"], config["lambda_hn"], config["hard_negatives_per_sample"],
        config["hn_top_k"],
    )

    # ── Data ──────────────────────────────────────────────────────────────
    bundle = load_data(Path(config["data_dir"]))
    num_users = int(bundle.stats["n_users"])
    history_max_len = int(config["history_max_len"])
    train_history_matrix, train_history_lengths = build_history_matrix(
        bundle.train_df, num_users, history_max_len,
    )
    logging.info(
        "train history: non_empty_users=%d avg_len=%.4f max_len=%d",
        int((train_history_lengths > 0).sum()),
        float(train_history_lengths.mean()),
        history_max_len,
    )
    train_seen = build_seen_items(bundle.train_df)

    # ── Step 1: Export item embeddings from final-model checkpoint ────────
    checkpoint_path = Path(config["model_hn_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"model_hn_checkpoint not found: {checkpoint_path}")
    item_emb_np = export_item_embeddings_from_checkpoint(checkpoint_path, device)

    # ── Step 2: Build model-based HN table ────────────────────────────────
    top_k = int(config["hn_top_k"])
    hn_table = build_model_hard_negative_table(item_emb_np, top_k)
    del item_emb_np

    # ── Step 3: Build FRESH training model ───────────────────────────────
    model = build_model(config, bundle.stats, device)
    train_loader = make_dataloader(bundle.train_df, train_history_matrix, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    # ── Step 4: Train ─────────────────────────────────────────────────────
    epoch_results: dict[int, dict[str, Any]] = {}
    for epoch in range(1, int(config["epochs"]) + 1):
        t0 = time.time()
        avg_total, avg_main, avg_hn, total_valid_hn = train_epoch_hnm(
            model, train_loader, optimizer, config, device, epoch, hn_table,
        )
        epoch_time = time.time() - t0
        logging.info("epoch %d time: %.2fs", epoch, epoch_time)

        valid_metrics = evaluate_with_oom_retry(
            model, bundle.valid_df, train_history_matrix, train_seen,
            config, bundle.stats, device, "valid",
        )
        logging.info(
            "epoch %d valid: Recall@20=%.6f Recall@50=%.6f Recall@100=%.6f NDCG@50=%.6f MRR@50=%.6f",
            epoch,
            valid_metrics["recall@20"],
            valid_metrics["recall@50"],
            valid_metrics["recall@100"],
            valid_metrics["ndcg@50"],
            valid_metrics["mrr@50"],
        )

        epoch_results[epoch] = {
            "train_total_loss": avg_total,
            "train_main_loss": avg_main,
            "train_hn_loss": avg_hn,
            "total_valid_hn": total_valid_hn,
            "total_examples": int(len(bundle.train_df)),
            "epoch_time_seconds": epoch_time,
            "valid_recall@20": valid_metrics["recall@20"],
            "valid_recall@50": valid_metrics["recall@50"],
            "valid_recall@100": valid_metrics["recall@100"],
            "valid_ndcg@50": valid_metrics["ndcg@50"],
            "valid_mrr@50": valid_metrics["mrr@50"],
        }

        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "stats": bundle.stats,
                "epoch": epoch,
                "valid_recall@50": valid_metrics["recall@50"],
            },
            ckpt_dir / f"epoch_{epoch}.pt",
        )

    write_json(output_dir / "model_hnm_smoke_results.json", epoch_results)
    summary = {
        "model_hn_checkpoint": str(checkpoint_path),
        "hn_top_k": top_k,
        "lambda_hn": float(config["lambda_hn"]),
        "hard_negatives_per_sample": int(config["hard_negatives_per_sample"]),
        "epoch_1_valid_recall@50": epoch_results.get(1, {}).get("valid_recall@50"),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    logging.info("Model-based HNM smoke complete: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    config["config_path"] = args.config
    train_model_hnm_smoke(config)


if __name__ == "__main__":
    main()

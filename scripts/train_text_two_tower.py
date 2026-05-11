"""Train text-enhanced Two-Tower (M6 v1).

Item tower: id_embedding + text_projection → item_vec  (additive residual fusion).
User tower: id_embedding only (same as ID-only baseline).

The existing train_two_tower.py (ID-only path) is NOT modified.
"""

try:
    import argparse
    import csv
    import json
    import logging
    import math
    import random
    import time
    from dataclasses import dataclass
    from datetime import datetime, timezone
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
    import logging as _logging
    _logging.basicConfig(level=_logging.ERROR)
    _logging.error("缺少依赖：%s", exc.name)
    raise SystemExit(1) from exc


REQUIRED_CONFIG_KEYS = [
    "data_dir", "output_dir", "embedding_dim", "batch_size",
    "learning_rate", "weight_decay", "epochs", "temperature",
    "use_l2_norm", "seed", "eval_k_list", "eval_batch_size",
    "num_workers", "device", "save_best_by",
    "item_text_embedding_path", "item_has_text_path",
    "text_proj_dim", "use_has_text_mask", "item_fusion",
]
TRAIN_COLUMNS = ["user_idx", "item_idx"]
EVAL_COLUMNS = ["user_idx", "item_idx", "is_cold_item_for_eval"]
TRAIN_LOG_FIELDS = [
    "epoch", "train_loss",
    "valid_recall@20", "valid_recall@50", "valid_recall@100",
    "valid_ndcg@20", "valid_ndcg@50", "valid_ndcg@100",
    "valid_mrr@20", "valid_mrr@50", "valid_mrr@100",
    "learning_rate", "batch_size", "embedding_dim",
    "temperature", "use_l2_norm", "text_proj_dim", "epoch_time_seconds",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextEnhancedTwoTower(nn.Module):
    """Two-Tower with text-enhanced item tower (additive residual fusion).

    User tower : Embedding(user_idx) → [L2 norm] → user_vec
    Item tower : Embedding(item_idx) + Proj(frozen_text_emb) → [L2 norm] → item_vec

    Fusion is additive: item_vec = id_emb + text_proj (no MLP).
    text_proj_dim must equal embedding_dim for additive fusion.

    text_emb and has_text are frozen buffers (persistent=False so they are NOT
    saved in the checkpoint—they are reloaded from disk on each run).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        text_emb: torch.Tensor,    # (n_items, text_dim), float32, frozen
        has_text: torch.Tensor,    # (n_items,), float32 {0.0,1.0}, frozen
        text_proj_dim: int,
        use_l2_norm: bool,
        use_has_text_mask: bool,
    ) -> None:
        super().__init__()
        if text_proj_dim != embedding_dim:
            raise ValueError(
                f"additive fusion requires text_proj_dim == embedding_dim, "
                f"got text_proj_dim={text_proj_dim}, embedding_dim={embedding_dim}"
            )
        self.use_l2_norm = use_l2_norm
        self.use_has_text_mask = use_has_text_mask

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)

        text_input_dim = text_emb.shape[1]   # 384
        self.text_proj = nn.Linear(text_input_dim, embedding_dim, bias=False)

        # persistent=False: excluded from state_dict, reloaded from file
        self.register_buffer("_text_emb", text_emb.float(), persistent=False)
        self.register_buffer("_has_text", has_text.float(), persistent=False)

        nn.init.normal_(self.user_embedding.weight, 0.0, 0.02)
        nn.init.normal_(self.item_id_embedding.weight, 0.0, 0.02)
        nn.init.xavier_uniform_(self.text_proj.weight)

    def _item_prenorm(self, item_idx: torch.Tensor) -> torch.Tensor:
        id_emb = self.item_id_embedding(item_idx)           # (B, D)
        txt_proj = self.text_proj(self._text_emb[item_idx]) # (B, D)
        if self.use_has_text_mask:
            txt_proj = txt_proj * self._has_text[item_idx].unsqueeze(-1)
        return id_emb + txt_proj                             # additive fusion

    def encode_users(self, user_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_idx)
        return F.normalize(u, p=2, dim=-1) if self.use_l2_norm else u

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        out = self._item_prenorm(item_idx)
        return F.normalize(out, p=2, dim=-1) if self.use_l2_norm else out

    def raw_batch(
        self, user_idx: torch.Tensor, item_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding(user_idx), self._item_prenorm(item_idx)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    def __init__(self, users: np.ndarray, items: np.ndarray) -> None:
        self.users = torch.from_numpy(users.astype(np.int64, copy=False))
        self.items = torch.from_numpy(items.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index]


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: dict[str, Any]


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{name} 缺少必需字段：{missing}")


def load_data(data_dir: Path) -> DataBundle:
    logging.info("读取 train：%s", data_dir / "train.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet", columns=TRAIN_COLUMNS)
    valid_df = pd.read_parquet(data_dir / "valid.parquet", columns=EVAL_COLUMNS)
    test_df  = pd.read_parquet(data_dir / "test.parquet",  columns=EVAL_COLUMNS)
    with (data_dir / "stats.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)
    require_columns(train_df, TRAIN_COLUMNS, "train")
    require_columns(valid_df, EVAL_COLUMNS, "valid")
    require_columns(test_df,  EVAL_COLUMNS, "test")
    logging.info("n_users=%s  n_items=%s  train_interactions=%s",
                 stats["n_users"], stats["n_items"], len(train_df))
    return DataBundle(train_df=train_df, valid_df=valid_df, test_df=test_df, stats=stats)


def make_dataloader(
    train_df: pd.DataFrame, config: dict[str, Any]
) -> DataLoader:
    if config.get("smoke_test"):
        limit = int(config["batch_size"]) * int(config["smoke_train_batches"])
        train_df = train_df.head(limit).copy()
        logging.info("smoke test: 只使用 train 前 %s 行。", len(train_df))
    users = train_df["user_idx"].to_numpy(dtype=np.int64, copy=True)
    items = train_df["item_idx"].to_numpy(dtype=np.int64, copy=True)
    gen = torch.Generator()
    gen.manual_seed(int(config["seed"]))
    return DataLoader(
        InteractionDataset(users, items),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=gen,
    )


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for uid, grp in frame.groupby("user_idx", sort=False):
        seen[int(uid)] = set(int(x) for x in grp["item_idx"].tolist())
    return seen


def merge_seen_items(base: dict[int, set[int]], extra: pd.DataFrame) -> dict[int, set[int]]:
    merged = {u: set(s) for u, s in base.items()}
    for uid, grp in extra.groupby("user_idx", sort=False):
        merged.setdefault(int(uid), set()).update(int(x) for x in grp["item_idx"].tolist())
    return merged


# ---------------------------------------------------------------------------
# Text artifact loading
# ---------------------------------------------------------------------------

def load_text_artifacts(
    config: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (text_emb, has_text) as float32 tensors on CPU."""
    emb_path = Path(config["item_text_embedding_path"])
    has_path = Path(config["item_has_text_path"])

    if not emb_path.exists():
        raise FileNotFoundError(
            f"item_text_embedding_path 不存在: {emb_path}\n"
            "请先运行 scripts/build_item_text_embeddings.py --run_full"
        )
    if not has_path.exists():
        raise FileNotFoundError(
            f"item_has_text_path 不存在: {has_path}\n"
            "请先运行生成 item_has_text.npy 的脚本。"
        )

    text_emb = torch.from_numpy(np.load(emb_path).astype(np.float32))
    has_text  = torch.from_numpy(np.load(has_path).astype(np.float32))

    logging.info("text_emb loaded: shape=%s  dtype=%s", tuple(text_emb.shape), text_emb.dtype)
    logging.info("has_text loaded: shape=%s  dtype=%s  has_text=1: %d/%d (%.1f%%)",
                 tuple(has_text.shape), has_text.dtype,
                 int(has_text.sum()), len(has_text),
                 100.0 * has_text.mean().item())
    return text_emb, has_text


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_logits(
    model: nn.Module,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_u, raw_i = model.raw_batch(user_idx, item_idx)
    u = F.normalize(raw_u, p=2, dim=-1) if model.use_l2_norm else raw_u
    i = F.normalize(raw_i, p=2, dim=-1) if model.use_l2_norm else raw_i
    logits = (u @ i.T) / temperature
    return logits, raw_u, raw_i


def log_nan_diagnostics(logits: torch.Tensor, raw_u: torch.Tensor, raw_i: torch.Tensor) -> None:
    logging.error("user_emb norm min/max: %.4f / %.4f",
                  float(raw_u.norm(p=2, dim=-1).min()), float(raw_u.norm(p=2, dim=-1).max()))
    logging.error("item_emb norm min/max: %.4f / %.4f",
                  float(raw_i.norm(p=2, dim=-1).min()), float(raw_i.norm(p=2, dim=-1).max()))
    logging.error("logits has_nan=%s  has_inf=%s",
                  bool(torch.isnan(logits).any()), bool(torch.isinf(logits).any()))


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    temperature: float,
) -> tuple[float, float, float, int]:
    optimizer.zero_grad(set_to_none=True)
    logits, raw_u, raw_i = compute_logits(model, user_idx, item_idx, temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        log_nan_diagnostics(logits, raw_u, raw_i)
        raise FloatingPointError("loss 出现 nan 或 inf，已停止。")
    loss.backward()
    optimizer.step()
    return float(loss.item()), logits.detach().min().item(), logits.detach().max().item(), int(logits.shape[0])


def run_smoke_checks(
    model: nn.Module,
    train_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Forward pass shape + gradient descent sanity check."""
    # --- shape check ---
    user_idx, item_idx = next(iter(train_loader))
    user_idx = user_idx[:8].to(device)
    item_idx = item_idx[:8].to(device)
    with torch.no_grad():
        u = model.encode_users(user_idx)
        v = model.encode_items(item_idx)
    logging.info("SMOKE forward: user_idx=%s  item_idx=%s  user_vec=%s  item_vec=%s",
                 tuple(user_idx.shape), tuple(item_idx.shape),
                 tuple(u.shape), tuple(v.shape))
    assert u.shape == (8, config["embedding_dim"]), f"user_vec shape mismatch: {u.shape}"
    assert v.shape == (8, config["embedding_dim"]), f"item_vec shape mismatch: {v.shape}"
    assert torch.isfinite(u).all(), "user_vec has nan/inf"
    assert torch.isfinite(v).all(), "item_vec has nan/inf"

    # --- gradient descent check on fixed mini-batch ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    user_idx, item_idx = next(iter(train_loader))
    user_idx = user_idx.to(device)
    item_idx = item_idx.to(device)
    losses = []
    for _ in range(3):
        loss, _, _, bs = train_one_step(model, optimizer, user_idx, item_idx, float(config["temperature"]))
        losses.append(loss)

    expected = math.log(bs)
    lower, upper = expected * 0.8, expected * 1.2
    logging.info("SMOKE gradient check: losses=%s  expected_log_bs=%.4f  range=[%.4f, %.4f]",
                 [round(x, 5) for x in losses], expected, lower, upper)
    if not (lower <= losses[0] <= upper):
        raise RuntimeError(f"SMOKE FAIL: 初始 loss {losses[0]:.4f} 不在 log(bs) 附近 [{lower:.4f}, {upper:.4f}]。")
    if losses[-1] > losses[0] + 1e-4:
        raise RuntimeError("SMOKE FAIL: 同一 mini-batch 训练 3 step 后 loss 未下降。")
    logging.info("SMOKE 通过：forward shape OK，gradient 正常下降，loss 无 nan/inf。")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float, int]:
    model.train()
    total_loss, total_n = 0.0, 0
    first_min = first_max = first_bs = 0
    for bidx, (user_idx, item_idx) in enumerate(train_loader):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        loss, sim_min, sim_max, bs = train_one_step(
            model, optimizer, user_idx, item_idx, float(config["temperature"])
        )
        if bidx == 0:
            first_min, first_max, first_bs = sim_min, sim_max, bs
            logging.info("epoch %s first_batch sim min/max: %.4f / %.4f  bs=%s",
                         epoch, sim_min, sim_max, bs)
        total_loss += loss * bs
        total_n    += bs
    return total_loss / total_n, first_min, first_max, first_bs


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def encode_all_items_cpu(model: nn.Module, num_items: int, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        item_idx = torch.arange(num_items, device=device)
        return model.encode_items(item_idx).detach().cpu()


def evaluate_once(
    model: nn.Module,
    eval_df: pd.DataFrame,
    seen_items: dict[int, set[int]],
    config: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    eval_max = config.get("eval_max_users")
    non_cold = eval_df[~eval_df["is_cold_item_for_eval"].astype(bool)].copy()
    if eval_max:
        non_cold = non_cold.head(int(eval_max)).copy()

    k_list = [int(k) for k in config["eval_k_list"]]
    max_k  = max(k_list)
    num_items = int(stats["n_items"])
    eval_bs   = int(config["eval_batch_size"])

    item_emb_cpu = encode_all_items_cpu(model, num_items, device)
    logging.info("%s eval users: %s", split_name, len(non_cold))

    metric_sums: dict[str, float] = {}
    for k in k_list:
        for m in ("recall", "ndcg", "mrr"):
            metric_sums[f"{m}@{k}"] = 0.0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(non_cold), eval_bs):
            batch = non_cold.iloc[start : start + eval_bs]
            u_t = torch.as_tensor(batch["user_idx"].to_numpy(dtype=np.int64, copy=True), device=device)
            tgt = torch.as_tensor(batch["item_idx"].to_numpy(dtype=np.int64), device=device)
            u_emb = model.encode_users(u_t)
            i_emb = item_emb_cpu.to(device)
            scores = (u_emb @ i_emb.T) / float(config["temperature"])

            row_idx = torch.arange(scores.shape[0], device=device)
            tgt_scores = scores[row_idx, tgt].clone()
            for rpos, (uid, tgt_item) in enumerate(
                zip(batch["user_idx"].tolist(), batch["item_idx"].tolist(), strict=True)
            ):
                seen = seen_items.get(int(uid), set())
                if seen:
                    scores[rpos, torch.as_tensor(list(seen), dtype=torch.long, device=device)] = -torch.inf
                scores[rpos, int(tgt_item)] = tgt_scores[rpos]

            topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
            targets = batch["item_idx"].to_numpy(dtype=np.int64)
            for tgt_item, recs in zip(targets, topk, strict=True):
                hit = np.where(recs == tgt_item)[0]
                if not hit.size:
                    continue
                rank = int(hit[0]) + 1
                for k in k_list:
                    if rank <= k:
                        metric_sums[f"recall@{k}"] += 1.0
                        metric_sums[f"ndcg@{k}"]   += 1.0 / math.log2(rank + 1)
                        metric_sums[f"mrr@{k}"]    += 1.0 / rank

    denom = len(non_cold)
    metrics: dict[str, Any] = {
        "split": split_name,
        "num_eval_users": denom,
        "num_skipped_cold": int(eval_df["is_cold_item_for_eval"].astype(bool).sum()),
        "eval_max_users": eval_max,
    }
    for k, v in metric_sums.items():
        metrics[k] = v / denom if denom else 0.0
    return metrics


def evaluate_with_oom_retry(
    model: nn.Module,
    eval_df: pd.DataFrame,
    seen_items: dict[int, set[int]],
    config: dict[str, Any],
    stats: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    try:
        return evaluate_once(model, eval_df, seen_items, config, stats, device, split_name)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or int(config["eval_batch_size"]) <= 128:
            raise
        config["eval_batch_size"] = 128
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.warning("eval OOM，降 eval_batch_size 到 128 后重试。")
        return evaluate_once(model, eval_df, seen_items, config, stats, device, split_name)


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_{k}": v for k, v in metrics.items()
        if k.startswith(("recall@", "ndcg@", "mrr@"))
    }


# ---------------------------------------------------------------------------
# Checkpoint / logging
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def init_train_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def append_train_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS).writerow(
            {field: row.get(field, "") for field in TRAIN_LOG_FIELDS}
        )


def save_checkpoint(
    path: Path, model: nn.Module, config: dict, stats: dict,
    epoch: int, metric_name: str, metric_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "stats": stats,
        "epoch": epoch,
        "best_metric_name": metric_name,
        "best_metric_value": metric_value,
    }, path)


# ---------------------------------------------------------------------------
# Config / setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 text-enhanced Two-Tower (M6 v1)")
    p.add_argument("--config", required=True, help="YAML 配置文件路径")
    p.add_argument("--smoke_test", action="store_true", help="smoke test 模式")
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置格式无效: {path}")
    return cfg


def require_config(cfg: dict) -> None:
    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg:
            raise KeyError(f"配置缺少必需字段: {key}")
    if int(cfg["num_workers"]) != 0:
        raise ValueError("num_workers 必须为 0。")
    if int(cfg["batch_size"]) <= 1:
        raise ValueError("batch_size 必须 > 1。")


def apply_smoke_overrides(cfg: dict, smoke: bool) -> dict:
    out = dict(cfg)
    if smoke:
        if "smoke" not in str(out["output_dir"]):
            out["output_dir"] = "outputs/text_two_tower_movies_tv_5core_smoke"
        out["epochs"] = 1
        out["smoke_train_batches"] = 50
        out["eval_max_users"] = 0   # 0 = skip eval in smoke
    out["smoke_test"] = bool(smoke)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(req: str) -> torch.device:
    if req == "cuda" and not torch.cuda.is_available():
        logging.warning("cuda 不可用，回退到 cpu。")
        return torch.device("cpu")
    return torch.device(req)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict[str, Any]) -> dict[str, Any]:
    require_config(config)
    set_seed(int(config["seed"]))
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", config)

    device = resolve_device(str(config["device"]))
    logging.info("device=%s  embedding_dim=%s  text_proj_dim=%s  batch_size=%s  temperature=%s",
                 device, config["embedding_dim"], config["text_proj_dim"],
                 config["batch_size"], config["temperature"])

    bundle = load_data(Path(config["data_dir"]))
    train_loader = make_dataloader(bundle.train_df, config)
    train_seen = build_seen_items(bundle.train_df)
    test_seen  = merge_seen_items(train_seen, bundle.valid_df)

    text_emb, has_text = load_text_artifacts(config, device)

    model = TextEnhancedTwoTower(
        num_users=int(bundle.stats["n_users"]),
        num_items=int(bundle.stats["n_items"]),
        embedding_dim=int(config["embedding_dim"]),
        text_emb=text_emb,
        has_text=has_text,
        text_proj_dim=int(config["text_proj_dim"]),
        use_l2_norm=bool(config["use_l2_norm"]),
        use_has_text_mask=bool(config["use_has_text_mask"]),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("TextEnhancedTwoTower 可训练参数: %d", n_params)

    if config.get("smoke_test"):
        run_smoke_checks(model, train_loader, config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    log_path = out_dir / "train_log.csv"
    init_train_log(log_path)

    best_epoch, best_recall, best_valid_metrics = 0, -1.0, None
    skip_eval = (int(config.get("eval_max_users", 1)) == 0)

    for epoch in range(1, int(config["epochs"]) + 1):
        logging.info("epoch %s 开始。", epoch)
        t0 = time.time()
        train_loss, _, _, _ = train_epoch(model, train_loader, optimizer, config, device, epoch)

        if skip_eval:
            logging.info("epoch %s train_loss=%.6f  (eval skipped)", epoch, train_loss)
            row = {
                "epoch": epoch, "train_loss": train_loss,
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
                "embedding_dim": int(config["embedding_dim"]),
                "temperature": float(config["temperature"]),
                "use_l2_norm": bool(config["use_l2_norm"]),
                "text_proj_dim": int(config["text_proj_dim"]),
                "epoch_time_seconds": round(time.time() - t0, 2),
            }
            append_train_log(log_path, row)
            continue

        valid_metrics = evaluate_with_oom_retry(
            model, bundle.valid_df, train_seen, config, bundle.stats, device, "valid"
        )
        epoch_time = time.time() - t0
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "learning_rate": float(config["learning_rate"]),
            "batch_size": int(config["batch_size"]),
            "embedding_dim": int(config["embedding_dim"]),
            "temperature": float(config["temperature"]),
            "use_l2_norm": bool(config["use_l2_norm"]),
            "text_proj_dim": int(config["text_proj_dim"]),
            "epoch_time_seconds": round(epoch_time, 2),
            **prefixed_metrics(valid_metrics, "valid"),
        }
        append_train_log(log_path, row)
        logging.info(
            "epoch %s done: train_loss=%.6f  valid_recall@50=%.6f  time=%.1fs",
            epoch, train_loss, valid_metrics["recall@50"], epoch_time,
        )

        if valid_metrics["recall@50"] > best_recall:
            best_epoch, best_recall = epoch, float(valid_metrics["recall@50"])
            best_valid_metrics = valid_metrics
            save_checkpoint(
                out_dir / "checkpoints" / "best_model.pt",
                model, config, bundle.stats, epoch,
                "valid_recall@50", best_recall,
            )
            logging.info("新 best checkpoint: epoch=%s  valid_recall@50=%.6f", epoch, best_recall)

    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_valid_recall@50": best_recall,
        "output_dir": str(out_dir),
        "skip_eval": skip_eval,
    }
    if best_valid_metrics:
        write_json(out_dir / "metrics_valid.json", best_valid_metrics)
    write_json(out_dir / "summary.json", summary)
    logging.info("训练完成: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(Path(args.config))
    train(apply_smoke_overrides(config, args.smoke_test))


if __name__ == "__main__":
    main()

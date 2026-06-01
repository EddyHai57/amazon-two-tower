#!/usr/bin/env python3
"""Shared sampling-bias loss helpers for isolated Transformer experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


LOSS_VARIANTS = {
    "infonce",
    "old_logq",
    "uber_batchq",
    "refined_logq",
    "mns",
    "mns_refined_logq",
}
Q_ESTIMATORS = {"empirical_frequency", "batch_appearance"}


def validate_loss_variant(loss_variant: str) -> None:
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"Unsupported loss_variant: {loss_variant}")


def validate_q_estimator(q_estimator: str) -> None:
    if q_estimator not in Q_ESTIMATORS:
        raise ValueError(f"Unsupported q_estimator: {q_estimator}")


def validate_logq_alpha(logq_alpha: float) -> None:
    if not 0.0 <= logq_alpha <= 1.0:
        raise ValueError(f"logq_alpha must be within [0.0, 1.0], got {logq_alpha}")


def validate_mns_uniform_fraction(uniform_fraction: float) -> None:
    if not 0.0 <= uniform_fraction <= 1.0:
        raise ValueError(
            f"mns_uniform_fraction must be within [0.0, 1.0], got {uniform_fraction}"
        )


def build_item_log_q(
    train_item_idx: torch.Tensor,
    num_items: int,
    *,
    q_estimator: str,
    batch_size: int,
) -> torch.Tensor:
    """Build train-only item proposal values for LogQ-style corrections."""
    validate_q_estimator(q_estimator)
    counts = torch.bincount(train_item_idx.to(dtype=torch.long).cpu(), minlength=num_items)
    frequency = counts.to(dtype=torch.float32).clamp_min(1.0)
    frequency = frequency / frequency.sum()
    if q_estimator == "empirical_frequency":
        return frequency.log()
    # Numerically stable form of Q = 1 - (1 - w) ** batch_size.
    return (-torch.expm1(float(batch_size) * torch.log1p(-frequency))).log()


def apply_old_logq_and_duplicate_mask(
    logits: torch.Tensor,
    batch_item_idx: torch.Tensor,
    log_q: torch.Tensor,
    *,
    use_logq: bool,
    mask_duplicate_items: bool,
    logq_alpha: float = 1.0,
) -> torch.Tensor:
    validate_logq_alpha(logq_alpha)
    corrected = logits
    if use_logq:
        candidate_log_q = log_q.to(device=logits.device, dtype=logits.dtype)[batch_item_idx]
        corrected = corrected - logq_alpha * candidate_log_q.unsqueeze(0)
    if mask_duplicate_items:
        same_item = batch_item_idx.unsqueeze(0) == batch_item_idx.unsqueeze(1)
        diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        corrected = corrected.masked_fill(same_item & ~diagonal, torch.finfo(logits.dtype).min)
    return corrected


def sample_mns_candidate_ids(
    batch_item_idx: torch.Tensor,
    *,
    num_items: int,
    uniform_fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace a fixed fraction of in-batch candidates with uniform catalog samples."""
    validate_mns_uniform_fraction(uniform_fraction)
    num_uniform = int(round(batch_item_idx.numel() * uniform_fraction))
    num_inbatch = batch_item_idx.numel() - num_uniform
    if num_uniform == 0:
        return batch_item_idx
    uniform = torch.randint(
        low=0,
        high=num_items,
        size=(num_uniform,),
        generator=generator,
        device=batch_item_idx.device,
    )
    return torch.cat([batch_item_idx[:num_inbatch], uniform], dim=0)


def mask_target_collisions(
    scores: torch.Tensor,
    positive_item_idx: torch.Tensor,
    candidate_item_idx: torch.Tensor,
) -> torch.Tensor:
    collisions = positive_item_idx.unsqueeze(1) == candidate_item_idx.unsqueeze(0)
    return scores.masked_fill(collisions, -torch.inf)


def compute_mns_softmax_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    positive_item_idx: torch.Tensor,
    negative_item_idx: torch.Tensor,
) -> torch.Tensor:
    masked_negative = mask_target_collisions(
        negative_scores,
        positive_item_idx,
        negative_item_idx,
    )
    logits = torch.cat([positive_scores.unsqueeze(1), masked_negative], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def compute_refined_logq_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    positive_item_idx: torch.Tensor,
    negative_item_idx: torch.Tensor,
    log_q: torch.Tensor,
) -> torch.Tensor:
    """Implement equation 12 of arXiv:2507.09331 with stop-gradient weighting."""
    masked_negative = mask_target_collisions(
        negative_scores,
        positive_item_idx,
        negative_item_idx,
    )
    candidate_log_q = log_q.to(device=negative_scores.device, dtype=negative_scores.dtype)[
        negative_item_idx
    ]
    q_positive = log_q.to(device=negative_scores.device, dtype=negative_scores.dtype)[
        positive_item_idx
    ].exp()
    q_prime = candidate_log_q.exp().unsqueeze(0) / (1.0 - q_positive).unsqueeze(1)
    corrected_negative = masked_negative - q_prime.clamp_min(torch.finfo(q_prime.dtype).tiny).log()
    negative_logsumexp = torch.logsumexp(corrected_negative, dim=1)
    sample_weight = torch.sigmoid(negative_logsumexp - math.log(negative_scores.shape[1]) - positive_scores)
    original_loss = negative_logsumexp - positive_scores
    return (sample_weight.detach() * original_loss).mean()

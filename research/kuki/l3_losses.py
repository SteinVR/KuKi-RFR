"""Explicit hard-negative alternative for the pair classifier only."""

import torch
from torch.nn import functional as F


def hard_negative_bce(logits, target, ratio=8, minimum=64, negative_weight=1.0):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask = target.bool()
    positive = loss[mask]
    negative = loss[~mask]
    count = min(len(negative), max(minimum, ratio * len(positive)))
    # Stable ties and detached ranking: selection has no surrogate gradient.
    ranked = torch.argsort(negative.detach(), descending=True, stable=True)[:count]
    parts, weights = [], []
    if len(positive):
        parts.append(positive.mean())
        weights.append(1.0)
    if count:
        parts.append(negative[ranked].mean() * negative_weight)
        weights.append(negative_weight)
    objective = sum(parts) / sum(weights) if parts else logits.sum() * 0
    diagnostics = dict(
        selected_negative_count=count,
        selected_negative_bce=float(negative[ranked].detach().mean()) if count else None,
    )
    return objective, diagnostics

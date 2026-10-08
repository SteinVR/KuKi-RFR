"""Training-only regularization; the original L1 model and scoring remain unchanged."""

import torch
from torch.nn import functional as F

from research.kuki.l1 import RoleClassifier, conditional_loss, role_pairs


def smooth_loss(logits, pairs, annotations, epsilon):
    if not 0 <= epsilon < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if epsilon == 0:
        return conditional_loss(logits, pairs, annotations)
    index = {p: i for i, p in enumerate(pairs)}
    losses = []
    for answer in annotations:
        ap, labels = role_pairs(answer)
        if ap:
            losses.append(F.cross_entropy(
                logits[[index[p] for p in ap]],
                torch.tensor(labels, device=logits.device), label_smoothing=epsilon,
            ))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


class TunedRoleClassifier(RoleClassifier):
    def training_loss(self, item, annotations):
        epsilon = self.cfg["label_smoothing"]
        if epsilon == 0:
            return super().training_loss(item, annotations)
        pairs = sorted({p for a in annotations for p in role_pairs(a)[0]})
        if not pairs:
            raise ValueError("No observed role: mask this article before standalone training")
        loss = smooth_loss(self.role_logits(item, pairs), pairs, annotations, epsilon)
        return loss, dict(loss=float(loss.detach()))

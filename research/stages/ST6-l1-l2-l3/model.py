"""Matched independent L1/L2 heads; no messages and no L3 supervision."""

import torch
from torch import nn

from research.kuki.joint import JointModel
from research.kuki.l1 import RoleClassifier, conditional_loss, role_pairs
from research.kuki.l2 import article_loss


class JointRoles(RoleClassifier):
    # Reuse the already checked full-document pooling and character geometry.
    encode_shared = JointModel.encode_shared

    def __init__(self, encoder, cfg):
        super().__init__(encoder, cfg)
        # Preserve the selected ST5 L1 initialization and RNG position.
        with torch.random.fork_rng(devices=[]):
            self.head["l2"] = nn.Linear(self.head["token"].in_features, 14)
        self.l1_weight = cfg["l1_weight"]
        self.l2_weight = cfg["l2_weight"]
        self.l1_normalization = cfg["l1_normalization"]
        self.head["l2"].requires_grad_(self.l2_weight > 0)
        for name in ["token", "char", "boundary_features", "span"]:
            self.head[name].requires_grad_(self.l1_weight > 0)

    def roles_from_features(self, item, features, prefix, pairs):
        bounds = torch.tensor(pairs, device=features.device, dtype=torch.long).reshape(-1, 2)
        start, end = bounds.T
        length = (end - start).float()
        mean = (prefix[end] - prefix[start]) / length[:, None]
        context = (prefix[-1] / item["length"]).expand(len(pairs), -1)
        return self.head["span"](torch.cat([
            features[start], features[end], mean, context, torch.log1p(length[:, None]) / 10,
        ], dim=1))

    def training_loss(self, item, annotations):
        pooled, features, prefix = self.encode_shared(item, self.l1_weight > 0)
        l2 = (article_loss(self.head["l2"](pooled), [a["l2"] for a in annotations])
              if self.l2_weight else pooled.sum() * 0)
        l1 = pooled.sum() * 0
        if self.l1_weight:
            pairs = sorted({p for a in annotations for p in role_pairs(a)[0]})
            devices = [pooled.device.index] if pooled.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                if pairs:
                    logits = self.roles_from_features(item, features, prefix, pairs)
                    l1 = conditional_loss(logits, pairs, annotations)
                # Empty L1 articles never invent labels, including a wholly empty batch.
                l1 = l1 + sum(p.sum() * 0 for n, p in self.head.named_parameters()
                              if p.requires_grad and not n.startswith("l2."))
        loss = self.l1_weight * self.l1_normalization * l1 + self.l2_weight * l2
        return loss, dict(loss=float(loss.detach()), l1=float(l1.detach()), l2=float(l2.detach()))


def joint_gains(metrics, refs):
    return ((metrics["l1"]["macro_f1"] - refs["l1_f1"]) / 0.01,
            (refs["l2_brier"] - metrics["l2"]["brier"]) / 0.003)


def selection_key(metrics, refs=None):
    if metrics["l2"] is None:
        return [metrics["l1"]["macro_f1"], -metrics["l1"]["cross_entropy"]]
    if metrics["l1"] is None:
        return [-metrics["l2"]["brier"], metrics["l2"]["macro_f1"]]
    a, b = joint_gains(metrics, refs)
    return [min(a, b), a + b, -metrics["l1"]["cross_entropy"]]


def accepted(metrics, refs):
    a, b = joint_gains(metrics, refs)
    return min(a, b) >= -1 and max(a, b) >= 1

"""ST3 shared text encoder, independent L2 / conditional L3 heads, no prediction exchange."""

import torch
from torch import nn
from torch.nn import functional as F

from .l2 import article_loss
from .span_classifier import SpanClassifier, answer_pairs


def conditional_loss(model, features, prefix, annotations):
    losses = []
    for answer in annotations:
        pairs, target = answer_pairs(answer)
        if pairs:
            logits = model.span_logits(features, prefix, pairs)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits, torch.as_tensor(target, device=logits.device)
                )
            )
    # Empty observed layers provide no labels for conditional family classification.
    return torch.stack(losses).mean() if losses else features.sum() * 0


class JointModel(SpanClassifier):
    def __init__(self, encoder, cfg):
        super().__init__(encoder, cfg)
        self.head["l2"] = nn.Linear(self.head["token"].in_features, 14)
        self.l2_weight = cfg["l2_weight"]
        self.l3_weight = cfg["l3_weight"]
        # Always initialize the same modules before freezing inactive task heads.
        self.head["l2"].requires_grad_(self.l2_weight > 0)
        for key in ["token", "char", "boundary_features", "span"]:
            self.head[key].requires_grad_(self.l3_weight > 0)

    def encode_shared(self, item, need_l3=True):
        device = next(self.parameters()).device
        enc = item["encoded"]
        pooled = None
        tokens = None
        for window in enc["windows"]:
            ids = torch.tensor([window["input_ids"]], device=device)
            hidden = self.encoder(
                input_ids=ids, attention_mask=torch.ones_like(ids)
            ).last_hidden_state[0]
            weights = torch.tensor(window["weights"], device=device)
            value = (hidden.float() * weights[:, None]).sum(0) / enc["tokens"]
            pooled = value if pooled is None else pooled + value
            if need_l3:
                valid = weights > 0
                projected = self.head["token"](hidden[valid].float()) * weights[valid, None]
                padded = F.pad(projected, (0, 0, window["start"], enc["tokens"] - window["end"]))
                tokens = padded if tokens is None else tokens + padded
        if not need_l3:
            return pooled, None, None
        left = torch.tensor(item["left"], device=device)
        right = torch.tensor(item["right"], device=device)
        features = self.head["boundary_features"](
            torch.cat(
                [
                    tokens[left],
                    tokens[right],
                    self.head["char"](torch.tensor(item["before"], device=device)),
                    self.head["char"](torch.tensor(item["after"], device=device)),
                    torch.tensor(item["geometry"], device=device),
                ],
                dim=1,
            )
        )
        prefix = F.pad(tokens[right[:-1]].cumsum(0), (0, 0, 1, 0))
        return pooled, features, prefix

    def training_loss(self, item, annotations):
        pooled, features, prefix = self.encode_shared(item, self.l3_weight > 0)
        l2 = (
            article_loss(self.head["l2"](pooled), [a["l2"] for a in annotations])
            if self.l2_weight
            else pooled.sum() * 0
        )
        if self.l3_weight:
            # Isolate span-head dropout so enabling L3 does not change subsequent
            # encoder dropout RNG in the matched L2 control.
            devices = [pooled.device.index] if pooled.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                l3 = conditional_loss(self, features, prefix, annotations)
                # The final accumulation block can consist entirely of empty L3
                # articles. Explicit zero gradients keep AdamW/momentum steps
                # defined for the span head without inventing negative labels.
                l3 = l3 + sum(p.sum() * 0 for p in self.head["span"].parameters())
        else:
            l3 = pooled.sum() * 0
        return self.l2_weight * l2 + self.l3_weight * l3, dict(
            l2=float(l2.detach()), l3=float(l3.detach())
        )

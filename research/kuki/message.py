"""ST4 residual transfer from predicted spans; annotations never enter message forward."""

import math

import torch
from torch import nn

from .joint import JointModel, conditional_loss
from .l2 import article_loss

MODES = ("continue", "self", "hidden", "uniform", "confidence", "contextual")


class MessageModel(JointModel):
    def __init__(self, encoder, cfg):
        super().__init__(encoder, cfg)
        self.mode = cfg["condition"]
        if self.mode not in MODES:
            raise ValueError(self.mode)
        dim = cfg["span_dimension"]
        width = dim + 9
        hidden = self.head["l2"].in_features
        self.head["refine_self"] = nn.Sequential(nn.Linear(hidden, width), nn.GELU())
        self.head["refine_value"] = nn.Sequential(nn.Linear(width, dim), nn.GELU())
        self.head["refine_query"] = nn.Linear(hidden, dim)
        self.head["refine_key"] = nn.Linear(width, dim)
        self.head["refine_out"] = nn.Linear(dim, 14)
        nn.init.zeros_(self.head["refine_out"].weight)
        nn.init.zeros_(self.head["refine_out"].bias)
        for key in ["refine_self", "refine_value", "refine_query", "refine_key", "refine_out"]:
            active = self.mode != "continue"
            if key == "refine_self":
                active = self.mode == "self"
            if key in ["refine_query", "refine_key"]:
                active = self.mode == "contextual"
            self.head[key].requires_grad_(active)

    def message_inputs(self, features, prefix, candidates, length):
        pairs = [(s, e) for s, e, _ in candidates]
        indices = torch.tensor(pairs, device=features.device, dtype=torch.long).reshape(-1, 2)
        start, end = indices.T
        if not len(pairs):
            return features.new_zeros((0, self.cfg["span_dimension"] + 9))
        if not bool(((start >= 0) & (end <= length) & (end > start)).all()):
            raise ValueError("Invalid automatic candidate coordinates")
        mean = (prefix[end] - prefix[start]) / (end - start)[:, None]
        if self.mode == "hidden":
            extra = mean.new_zeros((len(pairs), 9))
        else:
            # Prediction-as-message has no classifier dropout and no L2 gradient.
            training = self.head["span"].training
            self.head["span"].eval()
            try:
                with torch.no_grad():
                    probabilities = self.span_logits(features, prefix, pairs).sigmoid()
            finally:
                self.head["span"].train(training)
            geometry = indices.float() / length
            detector = mean.new_tensor([v for _, _, v in candidates]).sigmoid()[:, None]
            extra = torch.cat([probabilities, geometry, detector], 1)
        return torch.cat([mean, extra], 1)

    def l2_outputs(self, pooled, features, prefix, candidates, length, intervention=None):
        base = self.head["l2"](pooled)
        if self.mode == "continue" or intervention == "off":
            return base, base
        if self.mode == "self":
            message = self.head["refine_self"](pooled)[None]
        else:
            message = self.message_inputs(features, prefix, candidates, length)
            if intervention == "shuffle" and self.mode != "hidden" and len(message):
                # Fixed cyclic permutation, independent of labels and random state.
                message = torch.cat([message[:, :64], message[:, 64:].roll(1, 0)], 1)
        if not len(message):
            # Preserve defined zero gradients for all active residual parameters.
            delta = sum(
                p.sum() * 0
                for n, p in self.head.named_parameters()
                if n.startswith("refine_") and p.requires_grad
            )
            return base, base + delta
        value = self.head["refine_value"](message)
        if self.mode == "confidence":
            w = message[:, -1] * message[:, 64:70].max(1).values
            weights = w / w.sum().clamp_min(1e-12)
        elif self.mode == "contextual":
            query = self.head["refine_query"](pooled)
            weights = (
                self.head["refine_key"](message) @ query / math.sqrt(value.shape[1])
            ).softmax(0)
        else:
            weights = value.new_full((len(value),), 1 / len(value))
        delta = self.head["refine_out"]((value * weights[:, None]).sum(0))
        return base, base + delta

    def training_loss(self, item, annotations):
        pooled, features, prefix = self.encode_shared(item)
        base, final = self.l2_outputs(pooled, features, prefix, item["candidates"], item["length"])
        targets = [a["l2"] for a in annotations]
        l2 = (article_loss(base, targets) + article_loss(final, targets)) / 2
        devices = [pooled.device.index] if pooled.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            l3 = conditional_loss(self, features, prefix, annotations)
            l3 = l3 + sum(p.sum() * 0 for p in self.head["span"].parameters())
        return l2 + self.l3_weight * l3, dict(l2=float(l2.detach()), l3=float(l3.detach()))

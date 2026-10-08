"""Matched residual attention over immutable automatic ST3 features."""

import math

import torch
from torch import nn


class FrozenResidual(nn.Module):
    def __init__(self, pooled_dim=768, span_dim=64, families=True):
        super().__init__()
        self.families = families
        self.span_dim = span_dim
        self.value = nn.Sequential(nn.Linear(span_dim + 9, span_dim), nn.GELU())
        self.query = nn.Linear(pooled_dim, span_dim)
        self.key = nn.Linear(span_dim + 9, span_dim)
        self.output = nn.Linear(span_dim, 14)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def message(self, item):
        message = item["message"].detach()
        if not self.families:
            message = torch.cat(
                [message[:, : self.span_dim],
                 torch.zeros_like(message[:, self.span_dim : self.span_dim + 6]),
                 message[:, self.span_dim + 6 :]], dim=1,
            )
        return message

    def forward(self, item):
        message = self.message(item)
        base = item["base_logits"].detach()
        if not len(message):
            return base + sum(p.sum() * 0 for p in self.parameters())
        query = self.query(item["pooled"].detach())
        weights = (self.key(message) @ query / math.sqrt(self.span_dim)).softmax(0)
        return base + self.output((self.value(message) * weights[:, None]).sum(0))

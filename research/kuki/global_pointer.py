"""Character-level single-type GlobalPointer with exact streamed multilabel loss.

Method: Su et al., arXiv:2208.03054. The character feature adapter is KuKi-specific.
Streaming bounds storage, not quadratic arithmetic. No proposal/gold augmentation.
"""

import math

import numpy as np
import torch
from torch import nn

from .l3 import L3Model


def rope(x):
    positions = torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
    inv = 10000.0 ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1]
    )
    angle = positions[:, None] * inv[None, :]
    even, odd = x[:, 0::2], x[:, 1::2]
    return torch.stack(
        [even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()], dim=-1
    ).flatten(-2)


class _StreamingLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, gold, block_size):
        n, dim = q.shape
        scale = math.sqrt(dim)
        positive = []
        negative = [q.new_zeros(()) for _ in gold]
        for g in gold:
            vals = (q[g[:, 0]] * k[g[:, 1]]).sum(-1) / scale if len(g) else q.new_empty(0)
            positive.append(torch.logsumexp(torch.cat([q.new_zeros(1), -vals]), 0))
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            scores = q[start:stop] @ k.T / scale
            valid = (
                torch.arange(start, stop, device=q.device)[:, None]
                < torch.arange(n, device=q.device)[None, :]
            )
            scores = scores.masked_fill(~valid, -torch.inf)
            for a, g in enumerate(gold):
                neg = scores.clone()
                local = g[(g[:, 0] >= start) & (g[:, 0] < stop)]
                if len(local):
                    neg[local[:, 0] - start, local[:, 1]] = -torch.inf
                negative[a] = torch.logaddexp(negative[a], torch.logsumexp(neg.flatten(), 0))
        ctx.save_for_backward(q, k, torch.stack(positive), torch.stack(negative), *gold)
        ctx.block_size = block_size
        return (torch.stack(positive) + torch.stack(negative)).mean()

    @staticmethod
    def backward(ctx, upstream):
        q, k, pos_z, neg_z, *gold = ctx.saved_tensors
        n, dim = q.shape
        scale = math.sqrt(dim)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        for start in range(0, n, ctx.block_size):
            stop = min(n, start + ctx.block_size)
            scores = q[start:stop] @ k.T / scale
            valid = (
                torch.arange(start, stop, device=q.device)[:, None]
                < torch.arange(n, device=q.device)[None, :]
            )
            grad = torch.zeros_like(scores)
            for a, g in enumerate(gold):
                coeff = torch.exp(scores - neg_z[a]).masked_fill(~valid, 0)
                local = g[(g[:, 0] >= start) & (g[:, 0] < stop)]
                if len(local):
                    rows, cols = local[:, 0] - start, local[:, 1]
                    coeff[rows, cols] = -torch.exp(-scores[rows, cols] - pos_z[a])
                grad += coeff / len(gold)
            dq[start:stop] = grad @ k / scale
            dk += grad.T @ q[start:stop] / scale
        return dq * upstream, dk * upstream, None, None


def streaming_multilabel_loss(q, k, annotations, block_size=256):
    gold = [
        torch.tensor(sorted(set(a)), dtype=torch.long, device=q.device).reshape(-1, 2)
        for a in annotations
    ]
    if not gold:
        raise ValueError("At least one individual answer required")
    return _StreamingLoss.apply(q, k, gold, block_size)


def top_pairs(q, k, limit=500, block_size=256):
    """Exact global top-k, with lower linear pair index as deterministic tie breaker."""
    n, dim = q.shape
    shortlist = []
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        scores = q[start:stop] @ k.T / math.sqrt(dim)
        valid = (
            torch.arange(start, stop, device=q.device)[:, None]
            < torch.arange(n, device=q.device)[None, :]
        )
        scores = scores.masked_fill(~valid, -torch.inf)
        flat = scores.flatten()
        take = min(limit, flat.numel())
        cutoff = torch.topk(flat, take).values[-1]
        indices = torch.nonzero((flat >= cutoff) & torch.isfinite(flat), as_tuple=False).flatten()
        vals = flat[indices].detach().cpu().numpy()
        ids = indices.cpu().numpy() + start * n
        order = np.lexsort((ids, -vals))[:limit]
        shortlist.extend((float(vals[i]), int(ids[i])) for i in order)
    best = sorted(shortlist, key=lambda x: (-x[0], x[1]))[:limit]
    return [(i // n, i % n, value) for value, i in best]


class GlobalPointer(L3Model):
    def __init__(self, encoder, cfg):
        super().__init__(encoder, cfg)
        for key in ["span", "boundary"]:
            self.head[key].requires_grad_(False)
        self.head["qk"] = nn.Linear(cfg["span_dimension"], 2 * cfg.get("pointer_dimension", 64))

    def pointer(self, item):
        features, _, _ = self.encode(item)
        q, k = self.head["qk"](features).chunk(2, dim=-1)
        return rope(q), rope(k)

    def training_loss(self, item, annotations):
        q, k = self.pointer(item)
        gold = [[(s["start"], s["end"]) for s in a["l3_persuasion"]] for a in annotations]
        loss = streaming_multilabel_loss(q, k, gold, self.cfg.get("pair_block_size", 256))
        return loss, dict(
            loss=float(loss.detach()), pair_universe=item["length"] * (item["length"] + 1) // 2
        )

    def infer_pairs(self, item, limit=500):
        q, k = self.pointer(item)
        return top_pairs(q, k, limit, self.cfg.get("pair_block_size", 256))

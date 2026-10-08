import math

import numpy as np
import pytest
import torch

from research.kuki.global_pointer import rope, streaming_multilabel_loss, top_pairs


def dense(q, k, answers):
    n = len(q)
    scores = q @ k.T / math.sqrt(q.shape[1])
    valid = torch.ones((n, n), dtype=torch.bool).triu(1)
    losses = []
    for answer in answers:
        target = torch.zeros((n, n), dtype=torch.bool)
        for s, e in answer:
            target[s, e] = True
        losses.append(
            torch.logsumexp(torch.cat([scores.new_zeros(1), -scores[target]]), 0)
            + torch.logsumexp(torch.cat([scores.new_zeros(1), scores[valid & ~target]]), 0)
        )
    return torch.stack(losses).mean()


@pytest.mark.parametrize("block", [1, 3, 20])
def test_streaming_loss_and_gradients_equal_dense_for_overlaps_and_empty(block):
    torch.manual_seed(4)
    q = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    answers = [[(0, 4), (2, 5), (2, 6)], [], [(1, 3), (2, 5)]]
    expected = dense(q, k, answers)
    eg = torch.autograd.grad(expected, (q, k))
    actual = streaming_multilabel_loss(q, k, answers, block)
    ag = torch.autograd.grad(actual, (q, k))
    assert torch.allclose(expected, actual, atol=1e-12)
    for a, b in zip(eg, ag, strict=True):
        assert torch.allclose(a, b, atol=1e-12)


def test_top_pairs_matches_dense_order_including_ties_and_long_spans():
    for tied in [False, True]:
        torch.manual_seed(7)
        q = torch.randn(8, 4)
        k = torch.randn(8, 4)
        if tied:
            q.zero_()
        scores = (q @ k.T / 2).numpy()
        expected = sorted(
            [(s, e, float(scores[s, e])) for s in range(8) for e in range(s + 1, 8)],
            key=lambda p: (-p[2], p[0], p[1]),
        )[:15]
        actual = top_pairs(q, k, 15, 3)
        assert [(a, b) for a, b, _ in actual] == [(a, b) for a, b, _ in expected]
        assert np.allclose([v for _, _, v in actual], [v for _, _, v in expected], atol=1e-6)


def test_rope_preserves_norm_and_relative_dot_products():
    torch.manual_seed(8)
    v = torch.randn(1, 8)
    x = v.repeat(10, 1)
    y = rope(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-6)
    assert torch.allclose((y[0] * y[3]).sum(), (y[4] * y[7]).sum(), atol=1e-6)

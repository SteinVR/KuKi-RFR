import torch
from torch.nn import functional as F

from research.kuki.data import FAMILIES
from research.kuki.joint import conditional_loss


class PairLogits:
    def span_logits(self, features, prefix, pairs):
        return torch.stack([features[s] for s, _ in pairs])


def answer(spans):
    return {"l3_persuasion": [dict(start=s, end=e, label=FAMILIES[f]) for s, e, f in spans]}


def test_conditional_loss_equal_answers_not_pooled_spans_and_masks_empty():
    x = torch.tensor(
        [[2.0, -1.0, 0.0, 0.0, 0.0, 0.0], [-2.0, 1.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    a = answer([(0, 1, 0)])
    b = answer([(0, 1, 1), (1, 2, 0)])
    result = conditional_loss(PairLogits(), x, None, [a, b, answer([])])
    ya = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    yb = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    expected = (
        F.binary_cross_entropy_with_logits(x[:1], ya) + F.binary_cross_entropy_with_logits(x, yb)
    ) / 2
    assert torch.equal(result, expected)
    assert torch.allclose(
        result, conditional_loss(PairLogits(), x, None, [a, b, a, b]), rtol=1e-6, atol=1e-7
    )
    pooled = (
        F.binary_cross_entropy_with_logits(x[:1], ya)
        + 2 * F.binary_cross_entropy_with_logits(x, yb)
    ) / 3
    assert not torch.isclose(result, pooled)


def test_empty_conditional_supervision_has_zero_gradient():
    x = torch.randn(3, 6, requires_grad=True)
    loss = conditional_loss(PairLogits(), x, None, [answer([]), answer([])])
    loss.backward()
    assert loss.item() == 0 and torch.equal(x.grad, torch.zeros_like(x))


def test_partial_accumulation_averages_article_losses():
    x = torch.randn(3, 6, requires_grad=True)
    aa = [answer([(0, 1, 0)])]
    bb = [answer([(0, 1, 1), (1, 2, 0)]), answer([(2, 3, 2)])]
    grad = torch.autograd.grad(
        (conditional_loss(PairLogits(), x, None, aa) + conditional_loss(PairLogits(), x, None, bb))
        / 2,
        x,
    )[0]
    ga = torch.autograd.grad(conditional_loss(PairLogits(), x, None, aa), x)[0]
    gb = torch.autograd.grad(conditional_loss(PairLogits(), x, None, bb), x)[0]
    assert torch.allclose(grad, (ga + gb) / 2)


def test_wholly_empty_final_batch_keeps_explicit_zero_span_gradients():
    from torch import nn

    from research.kuki.joint import JointModel

    model = JointModel.__new__(JointModel)
    nn.Module.__init__(model)
    model.head = nn.ModuleDict({"span": nn.Linear(2, 6)})
    model.l2_weight = 0
    model.l3_weight = 1
    features = torch.randn(3, 2, requires_grad=True)
    model.encode_shared = lambda item, need_l3: (features.mean(0), features, None)
    loss, _ = model.training_loss({}, [answer([])])
    loss.backward()
    assert loss.item() == 0
    assert all(
        p.grad is not None and torch.equal(p.grad, torch.zeros_like(p))
        for p in model.head.parameters()
    )

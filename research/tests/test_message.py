import importlib

import pytest
import torch
from torch import nn

from research.kuki.message import MessageModel


def lightweight(mode):
    m = MessageModel.__new__(MessageModel)
    nn.Module.__init__(m)
    m.mode = mode
    m.cfg = {"span_dimension": 64}
    m.head = nn.ModuleDict(
        {
            "span": nn.Sequential(nn.Dropout(0.9), nn.Linear(64, 6)),
            "l2": nn.Linear(64, 14),
            "refine_value": nn.Sequential(nn.Linear(73, 64), nn.GELU()),
            "refine_query": nn.Linear(64, 64),
            "refine_key": nn.Linear(73, 64),
            "refine_out": nn.Linear(64, 14),
        }
    )
    m.span_logits = lambda f, p, pairs: m.head["span"](torch.stack([f[s] for s, e in pairs]))
    return m


def test_message_classifier_is_detached_deterministic_and_restores_dropout():
    model = lightweight("uniform").train()
    features = torch.randn(5, 64, requires_grad=True)
    prefix = torch.randn(5, 64, requires_grad=True)
    candidates = [(0, 2, 0.1), (1, 4, 0.2)]
    a = model.message_inputs(features, prefix, candidates, 4)
    b = model.message_inputs(features, prefix, candidates, 4)
    assert torch.equal(a, b) and model.head["span"].training
    a.sum().backward()
    assert prefix.grad.abs().sum() > 0
    assert features.grad is None
    assert all(p.grad is None for p in model.head["span"].parameters())


@pytest.mark.parametrize("mode", ["hidden", "uniform", "confidence", "contextual"])
def test_zero_residual_exact_baseline_and_off(mode):
    model = lightweight(mode)
    nn.init.zeros_(model.head["refine_out"].weight)
    nn.init.zeros_(model.head["refine_out"].bias)
    args = (torch.randn(64), torch.randn(5, 64), torch.randn(5, 64), [(0, 2, 0.1), (1, 4, 0.2)], 4)
    base, final = model.l2_outputs(*args)
    assert torch.equal(base, final)
    with torch.no_grad():
        model.head["refine_out"].weight.fill_(0.1)
    assert torch.equal(model.l2_outputs(*args, intervention="off")[1], base)


def test_group_excluded_folds_cover_each_document_once():
    module = importlib.import_module("research.stages.ST4-l3-to-l2.method")
    docs = [dict(doc_id=f"d{i}", group_id=f"g{i // 2}") for i in range(12)]
    assignment = module.folds(docs)
    assert assignment == module.folds(list(reversed(docs)))
    assert set(assignment) == {d["doc_id"] for d in docs}
    for i in range(3):
        train = {d["group_id"] for d in docs if assignment[d["doc_id"]] != i}
        held = {d["group_id"] for d in docs if assignment[d["doc_id"]] == i}
        assert train and held and not train & held


@pytest.mark.parametrize("mode", ["hidden", "uniform", "confidence", "contextual"])
def test_message_aggregation_invariant_to_candidate_order(mode):
    model = lightweight(mode).eval()
    pooled = torch.randn(64)
    features = torch.randn(5, 64)
    prefix = torch.randn(5, 64)
    candidates = [(0, 2, 0.1), (1, 4, 0.2), (2, 3, -0.4)]
    _, a = model.l2_outputs(pooled, features, prefix, candidates, 4)
    _, b = model.l2_outputs(pooled, features, prefix, list(reversed(candidates)), 4)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_candidate_coordinates_cannot_address_outside_article():
    model = lightweight("uniform")
    with pytest.raises(ValueError, match="coordinates"):
        model.message_inputs(torch.randn(5, 64), torch.randn(5, 64), [(0, 6, 0.1)], 4)

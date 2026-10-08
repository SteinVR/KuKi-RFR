import importlib

import pytest
import torch

from research.kuki.l1 import conditional_loss

M = importlib.import_module("research.stages.ST5-l1-roles.search.model")


def annotations():
    return [dict(l1_roles=[dict(start=0,end=1,label="PROTAGONIST"),
                           dict(start=2,end=3,label="INNOCENT")]),
            dict(l1_roles=[dict(start=0,end=1,label="ANTAGONIST")]),dict(l1_roles=[])]


def test_smoothing_matches_explicit_uniform_mixture_and_answer_weights():
    logits = torch.tensor([[2.,-1.,0.],[0.,1.,2.]],requires_grad=True)
    pairs = [(0,1),(2,3)]
    q = .9 * torch.eye(3)[[0,2,1]] + .1/3
    log_p = logits[[0,1,0]].log_softmax(-1)
    expected = (-(q*log_p).sum(1)*torch.tensor([.25,.25,.5])).sum()
    actual = M.smooth_loss(logits,pairs,annotations(),.1)
    assert torch.allclose(actual,expected,atol=1e-7)
    a = torch.autograd.grad(actual,logits,retain_graph=True)[0]
    b = torch.autograd.grad(expected,logits)[0]
    assert torch.allclose(a,b,atol=1e-7)


def test_zero_smoothing_is_exact_original_and_empty_answers_masked():
    logits = torch.tensor([[2.,-1.,0.],[0.,1.,2.]],requires_grad=True)
    pairs = [(0,1),(2,3)]
    assert torch.equal(M.smooth_loss(logits,pairs,annotations(),0),
                       conditional_loss(logits,pairs,annotations()))
    assert M.smooth_loss(logits,pairs,[dict(l1_roles=[])],.1) == 0
    for value in [-.1,1.1]:
        with pytest.raises(ValueError):
            M.smooth_loss(logits,pairs,annotations(),value)

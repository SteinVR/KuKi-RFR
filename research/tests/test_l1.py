import copy

import numpy as np
import pytest
import torch
from torch import nn

from research.kuki.l1 import RoleClassifier, conditional_loss, examples, role_pairs, score
from research.kuki.l3 import prepare_l3


def span(s, e, label="PROTAGONIST"):
    return dict(start=s, end=e, label=label)


def doc(answers):
    return dict(doc_id="d", group_id="d", source="test", text="abcdefghi",
                annotations=[dict(annotator_id=str(i), l1_roles=a) for i,a in enumerate(answers)])


def test_article_and_answer_normalization_and_empty_mask():
    d = doc([[span(0,1),span(2,3)], [span(0,1,"ANTAGONIST")], []])
    rows = examples([d, dict(doc([[]]),doc_id="empty")])
    assert [r["weight"] for r in rows] == [.25,.25,.5]
    logits = torch.tensor([[2.,-1.,0.], [0.,1.,2.]], requires_grad=True)
    loss = conditional_loss(logits, [(0,1),(2,3)], d["annotations"])
    raw = nn.functional.cross_entropy(logits[[0,1,0]], torch.tensor([0,0,1]), reduction="none")
    assert torch.allclose(loss, (raw*torch.tensor([.25,.25,.5])).sum())
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_conflicts_rejected_but_cross_answer_disagreement_retained():
    with pytest.raises(ValueError,match="Conflicting"):
        role_pairs(dict(l1_roles=[span(0,1),span(0,1,"INNOCENT")]))
    rows = examples([doc([[span(0,1)], [span(0,1,"INNOCENT")]])])
    assert len(rows) == 2 and rows[0]["target"] != rows[1]["target"]


def test_metrics_duplicate_answer_invariance_and_fixed_class_order():
    d = doc([[span(0,1),span(2,3,"ANTAGONIST"),span(4,5,"INNOCENT")]])
    p = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.8,.1]])
    result = score(examples([d]),p)
    assert result["macro_f1"] == pytest.approx((1 + 2/3)/3)
    assert result["article_accuracy"] == pytest.approx(2/3)
    doubled = copy.deepcopy(d)
    doubled["annotations"] *= 2
    again = score(examples([doubled]),np.vstack([p,p]))
    for key in ["macro_f1","article_accuracy","cross_entropy"]:
        assert result[key] == pytest.approx(again[key])


def test_metrics_invalid_probabilities_and_absent_classes():
    rows = examples([doc([[span(0,1)]])])
    assert score(rows,[[1,0,0]])["macro_f1"] == pytest.approx(1/3)
    with pytest.raises(ValueError,match="Invalid"):
        score(rows,[[2,0,0]])


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20,4)

    def forward(self,input_ids,attention_mask):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def tiny():
    torch.manual_seed(42)
    model = RoleClassifier.__new__(RoleClassifier)
    nn.Module.__init__(model)
    model.encoder = FakeEncoder()
    model.head = nn.ModuleDict(dict(token=nn.Linear(4,3), char=nn.Embedding(2049,8),
        boundary_features=nn.Sequential(nn.Linear(26,3),nn.GELU()), boundary=nn.Linear(3,2),
        span=nn.Linear(13,3)))
    model.head["boundary"].requires_grad_(False)
    return model


def test_character_bounds_cross_window_overlap_and_prediction_identity():
    single = dict(tokens=3,offsets=[(0,3),(3,6),(6,9)],windows=[
        dict(input_ids=[1,2,3],weights=[1.,1.,1.],start=0,end=3)])
    overlap = dict(tokens=3,offsets=single["offsets"],windows=[
        dict(input_ids=[1,2],weights=[1.,.5],start=0,end=2),
        dict(input_ids=[2,3],weights=[.5,1.],start=1,end=3)])
    model = tiny()
    pairs = [(1,8),(2,3),(1,8),(0,9)]
    a = model.role_logits(prepare_l3(single,"abcdefghi"),pairs)
    b = model.role_logits(prepare_l3(overlap,"abcdefghi"),pairs)
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.allclose(a[0],a[2],atol=1e-7)
    assert a.shape == (4,3) and torch.isfinite(a).all()
    assert model.role_logits(prepare_l3(single,"abcdefghi"),[]).shape == (0,3)
    with pytest.raises(ValueError,match="No observed role"):
        model.training_loss(prepare_l3(single,"abcdefghi"),doc([[]])["annotations"])

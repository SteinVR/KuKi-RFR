import importlib

import pytest
import torch
from torch import nn

J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def fake_model(l1=1, l2=1):
    model = J.JointRoles.__new__(J.JointRoles)
    nn.Module.__init__(model)
    model.head = nn.ModuleDict({'span': nn.Linear(9, 3), 'l2': nn.Linear(2, 14)})
    model.l1_weight, model.l2_weight, model.l1_normalization = l1, l2, 145/135
    features = torch.randn(4, 2, requires_grad=True)
    prefix = torch.cat([torch.zeros(1, 2), features[:3].cumsum(0)])
    pooled = features.mean(0)
    model.encode_shared = lambda item, need_l3: (pooled, features, prefix)
    return model


def answer(spans):
    return dict(l1_roles=[dict(start=s, end=e, label=y) for s, e, y in spans], l2=[1] + [0]*13)


def test_empty_role_labels_have_zero_head_gradients_without_masking_l2():
    m = fake_model()
    loss, detail = m.training_loss({'length': 3}, [answer([])])
    loss.backward()
    assert detail['l1'] == 0 and detail['l2'] > 0
    assert all(torch.count_nonzero(p.grad) == 0 for p in m.head['span'].parameters())
    assert all(torch.count_nonzero(p.grad) > 0 for p in m.head['l2'].parameters())


def test_l2_loss_and_gradient_do_not_depend_on_role_targets_or_positions():
    m = fake_model()
    aa = [answer([(0, 1, 'PROTAGONIST')])]
    bb = [answer([(1, 3, 'ANTAGONIST'), (0, 2, 'INNOCENT')])]
    loss_a, detail_a = m.training_loss({'length': 3}, aa)
    grad_a = torch.autograd.grad(loss_a, tuple(m.head['l2'].parameters()), retain_graph=True)
    loss_b, detail_b = m.training_loss({'length': 3}, bb)
    grad_b = torch.autograd.grad(loss_b, tuple(m.head['l2'].parameters()))
    assert detail_a['l2'] == detail_b['l2']
    assert all(torch.equal(a, b) for a, b in zip(grad_a, grad_b, strict=True))


def test_total_objective_has_declared_active_article_normalization():
    m = fake_model()
    loss, detail = m.training_loss({'length': 3}, [answer([(0, 1, 'PROTAGONIST')])])
    assert loss.item() == pytest.approx(detail['l1'] * 145/135 + detail['l2'])


def test_joint_selection_protects_both_tasks_and_does_not_splice_checkpoints():
    refs = dict(l1_f1=0.76, l2_brier=0.15)
    def metric(f, b):
        return dict(l1=dict(macro_f1=f, cross_entropy=1), l2=dict(brier=b))
    sacrificed = metric(0.79, 0.159)
    balanced = metric(0.775, 0.151)
    assert J.selection_key(balanced, refs) > J.selection_key(sacrificed, refs)
    assert J.accepted(balanced, refs) and not J.accepted(sacrificed, refs)
    assert not J.accepted(metric(0.76, 0.15), refs)


def test_head_dropout_does_not_advance_encoder_rng():
    m = fake_model()
    m.head['span'] = nn.Sequential(nn.Dropout(0.3), m.head['span'])
    before = torch.get_rng_state().clone()
    m.training_loss({'length': 3}, [answer([(0, 1, 'PROTAGONIST')])])
    assert torch.equal(before, torch.get_rng_state())

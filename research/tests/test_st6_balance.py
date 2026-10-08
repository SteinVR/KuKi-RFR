import importlib

import pytest
import torch

from research.tests.test_st6 import answer, fake_model

M = importlib.import_module('research.stages.ST6-l1-l2-l3.balance.method')


@pytest.mark.parametrize('weights', [(1.6, .4), (.4, 1.6)])
def test_weighted_objective_and_shared_gradient(weights):
    model = fake_model()
    model.l1_weight, model.l2_weight = weights
    item = {'length': 3}
    annotations = [answer([(0, 1, 'PROTAGONIST'), (1, 3, 'ANTAGONIST')])]
    pooled, features, prefix = model.encode_shared(item, True)
    from research.kuki.l1 import conditional_loss
    from research.kuki.l2 import article_loss
    logits = model.roles_from_features(item, features, prefix, [(0, 1), (1, 3)])
    loss1 = conditional_loss(logits, [(0, 1), (1, 3)], annotations)
    loss2 = article_loss(model.head['l2'](pooled), [a['l2'] for a in annotations])
    gradient1 = torch.autograd.grad(loss1, features, retain_graph=True)[0]
    gradient2 = torch.autograd.grad(loss2, features, retain_graph=True)[0]
    actual, detail = model.training_loss(item, annotations)
    actual_gradient = torch.autograd.grad(actual, features)[0]
    expected_gradient = weights[0] * (145/135) * gradient1 + weights[1] * gradient2
    assert actual.item() == pytest.approx(weights[0] * (145/135) * loss1.item() + weights[1] * loss2.item())
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert detail['l1'] == pytest.approx(loss1.item())


@pytest.mark.parametrize('condition', ['roles', 'frames'])
def test_only_declared_config_changes_are_permitted(condition):
    cfg = M.read(M.ROOT / 'configs' / f'{condition}.json')
    M.validate_config(cfg)
    with pytest.raises(AssertionError):
        M.validate_config(dict(cfg, head_dropout=.5))
    with pytest.raises(AssertionError):
        M.validate_config(dict(cfg, l1_weight=4, l2_weight=1))


@pytest.mark.parametrize('weights', [(1.6, .4), (.4, 1.6)])
def test_weighting_keeps_empty_l1_masked(weights):
    model = fake_model()
    model.l1_weight, model.l2_weight = weights
    loss, detail = model.training_loss({'length': 3}, [answer([])])
    loss.backward()
    assert loss.item() == pytest.approx(weights[1] * detail['l2'])
    assert detail['l1'] == 0
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.head['span'].parameters())

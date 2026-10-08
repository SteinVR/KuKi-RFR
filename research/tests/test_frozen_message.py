import copy

import pytest
import torch

from research.kuki.frozen_message import FrozenResidual


def fixture():
    torch.manual_seed(42)
    return dict(pooled=torch.randn(768, requires_grad=True),
                message=torch.randn(5, 73, requires_grad=True),
                base_logits=torch.randn(14, requires_grad=True))


def test_control_removes_only_families_and_never_mutates_cache():
    item = fixture()
    original = item['message'].detach().clone()
    model = FrozenResidual(families=False)
    message = model.message(item)
    assert torch.equal(message[:, :64], original[:, :64])
    assert torch.equal(message[:, 70:], original[:, 70:])
    assert torch.count_nonzero(message[:, 64:70]) == 0
    assert torch.equal(item['message'], original)


@pytest.mark.parametrize('families', [False, True])
def test_zero_init_exact_and_no_gradients_to_frozen_inputs(families):
    item = fixture()
    model = FrozenResidual(families=families)
    assert torch.equal(model(item), item['base_logits'])
    model(item).square().mean().backward()
    assert all(v.grad is None for v in item.values())
    assert model.output.weight.grad.abs().sum() > 0


def test_control_invariant_to_family_probabilities_but_treatment_can_use_them():
    item = fixture()
    model = FrozenResidual()
    with torch.no_grad():
        model.output.weight.normal_()
    other = {k: v.detach().clone() for k, v in item.items()}
    other['message'][:, 64:70] += 10
    assert not torch.allclose(model(item), model(other))
    model.families = False
    assert torch.equal(model(item), model(other))


def test_identical_weights_without_semantic_signal_and_candidate_order_invariance():
    item = fixture()
    with torch.no_grad():
        item['message'][:, 64:70] = 0
    model = FrozenResidual()
    with torch.no_grad():
        model.output.weight.normal_()
    control = copy.deepcopy(model)
    control.families = False
    assert torch.equal(model(item), control(item))
    reverse = dict(item, message=item['message'].flip(0))
    assert torch.allclose(model(item), model(reverse), atol=1e-5)


def test_empty_candidates_preserve_baseline_and_defined_gradients():
    item = dict(fixture(), message=torch.empty(0, 73))
    model = FrozenResidual()
    model(item).sum().backward()
    assert torch.equal(model(item), item['base_logits'])
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.parameters())

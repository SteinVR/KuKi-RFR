import numpy as np
import pytest
import torch

from research.kuki.l2 import article_loss, load_checkpoint, save_checkpoint


def test_individual_bce_matches_soft_votes_without_annotator_weighting_artifact():
    logits = torch.linspace(-2, 2, 14, requires_grad=True)
    answers = [[int((i + j) % 3 == 0) for i in range(14)] for j in range(3)]
    loss = article_loss(logits, answers)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor(answers).float().mean(0)
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(loss, article_loss(logits, answers * 2))


def test_checkpoint_guards_and_cpu_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0)
    path = tmp_path / "state.pt"
    model(torch.ones(2)).sum().backward()
    with pytest.raises(ValueError, match="boundary"):
        save_checkpoint(path, model, opt, sched, {}, {})
    opt.step()
    opt.zero_grad(set_to_none=True)
    save_checkpoint(path, model, opt, sched, {"next_position": 3}, {"split": "a"})
    expected = (torch.rand(3), np.random.random(3))
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(path, model, opt, sched, {"split": "b"})
    assert load_checkpoint(path, model, opt, sched, {"split": "a"}) == {"next_position": 3}
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.random(3), expected[1])

import numpy as np
import pytest
import torch

from research.kuki.data import FAMILIES
from research.kuki.l3 import balanced_bce, candidate_pairs, prepare_l3, propose, span_key
from research.kuki.l3_evaluation import empty_outputs, evaluate, f1, matched


def span(start, end, label=FAMILIES[0], score=1.0):
    return dict(start=start, end=end, label=label, score=score)


def test_all_character_boundaries_include_spaces_inside_tokens_and_window_crossings():
    text = " абвг д\nе🙂 "
    enc = {"offsets": [(0, 5), (5, 7), (7, 10), (10, 11)]}
    item = prepare_l3(enc, text)
    assert len(item["left"]) == len(text) + 1
    assert len(item["before"]) == len(item["after"]) == len(text) + 1
    full = set(candidate_pairs(range(len(text)), range(1, len(text) + 1)))
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            encoded = span_key(span(start, end))
            assert (encoded[0], encoded[1]) in full
            assert text[encoded[0] : encoded[1]] == text[start:end]
    assert (0, len(text)) in full


def test_proposals_use_logits_only_and_stable_ties():
    logits = torch.zeros((10, 2))
    assert propose(logits, 3, 9) == ([0, 1, 2], [1, 2, 3])
    logits[4, 0] = 2
    logits[7, 1] = 2
    starts, ends = propose(logits, 1, 9)
    assert candidate_pairs(starts, ends) == [(4, 7)]


def test_overlap_one_to_one_family_and_nested_spans():
    prediction = [span(0, 10)]
    gold = [span(0, 8), span(1, 10)]
    assert matched(prediction, gold, overlap=True) == (1, 1, 2)
    assert f1(*matched(prediction, gold, overlap=True)) == pytest.approx(2 / 3)
    assert matched(prediction, [span(0, 10, FAMILIES[1])], overlap=True)[0] == 0
    mixed = [span(0, 10), span(0, 10, FAMILIES[1]), span(3, 8)]
    assert matched(mixed, mixed) == (3, 3, 3)
    assert matched(mixed, mixed, typed=False) == (2, 2, 2)
    assert matched([], []) == (0, 0, 0)
    assert f1(0, 0, 0) == 0


def test_individual_metrics_article_weights_and_empty_errors():
    def answer(ss):
        return {"annotator_id": "a", "l3_persuasion": ss}

    docs = [
        dict(doc_id="a", group_id="a", annotations=[answer([span(0, 2)])] * 3),
        dict(doc_id="b", group_id="b", annotations=[answer([])]),
    ]
    output = [
        dict(
            doc_id=d["doc_id"],
            spans=[span(0, 2)],
            starts=[0],
            ends=[2],
            candidate_count=1,
            eligible_before_cap=1,
        )
        for d in docs
    ]
    metrics, _ = evaluate(docs, output, 0.5)
    assert metrics["exact_f1"] == 0.5  # not 3/4
    assert metrics["nonempty_exact_f1"] == 1
    assert metrics["proposal_recall"] == 1
    assert metrics["false_nonempty_given_empty"] == 1
    empty, _ = evaluate(docs, empty_outputs(docs), 0.5)
    assert empty["exact_f1"] == 0
    assert empty["correctly_empty_answers"] == 1
    assert empty["false_nonempty_given_empty"] == 0


def test_balanced_loss_negative_multiplier_and_empty_normalization():
    logits = torch.tensor([1.0, -2.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    positive, negative = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    assert balanced_bce(logits, targets).item() == pytest.approx(((positive + negative) / 2).item())
    assert balanced_bce(logits, targets, 0.25).item() == pytest.approx(
        ((positive + negative * 0.25) / 1.25).item()
    )
    repeated = torch.cat([logits[:1], logits[1:].repeat(100)])
    labels = torch.cat([targets[:1], targets[1:].repeat(100)])
    assert torch.allclose(balanced_bce(repeated, labels), balanced_bce(logits, targets))
    assert torch.allclose(
        balanced_bce(logits, torch.zeros_like(logits)),
        balanced_bce(logits, torch.zeros_like(logits), 0.25),
    )
    balanced_bce(logits, targets).backward()
    assert np.isfinite(logits.grad.numpy()).all()


def test_hard_negative_risk_does_not_hide_confident_false_positives():
    from research.kuki.l3_losses import hard_negative_bce

    logits = torch.tensor([2.0, 5.0, -10.0, -10.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    value, detail = hard_negative_bce(logits, target, ratio=1, minimum=1)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:2], target[:2])
    assert torch.equal(value, expected)
    assert detail["selected_negative_count"] == 1
    assert value > balanced_bce(logits, target)
    value.backward()
    assert logits.grad[1] > 0 and logits.grad[2] == logits.grad[3] == 0
    empty, detail = hard_negative_bce(logits, torch.zeros_like(target), ratio=1, minimum=1)
    assert detail["selected_negative_count"] == 1
    assert empty.item() == pytest.approx(torch.nn.functional.softplus(logits[1]).item())

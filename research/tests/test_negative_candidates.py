import importlib

import numpy as np
import torch

from research.kuki.data import FAMILIES

m = importlib.import_module("research.stages.ST2-l3-rhetoric.negative_fit.method")


def test_sampling_is_per_answer_and_deterministic():
    spans = [dict(start=0, end=1, label=FAMILIES[0])]
    d = dict(doc_id="d", annotations=[dict(l3_persuasion=spans), dict(l3_persuasion=[])])
    p = dict(
        pairs=[(i, i + 1, 0.0) for i in range(100)],
        probabilities=[[1 - i / 101] * 6 for i in range(100)],
    )
    a = m.samples(d, p, 20)
    b = m.samples(d, p, 20)
    c = m.samples(d, p, 21)
    assert a[0]["negative"] == b[0]["negative"]
    assert (0, 1) not in a[0]["negative"]
    assert (0, 1) in a[1]["negative"]  # Other annotator's positive is not a union-gold exclusion.
    assert len(set(a[0]["negative"])) == 64
    assert a[0]["negative"][:32] == [(i, i + 1) for i in range(1, 33)]
    assert a[0]["negative"][32:] != c[0]["negative"][32:]


def test_balanced_loss_and_empty_answer_have_finite_gradients():
    positive = torch.zeros((2, 6), requires_grad=True)
    negative = torch.zeros((64, 6), requires_grad=True)
    value = m.answer_loss(positive, torch.ones_like(positive), negative)
    assert np.isclose(float(value.detach()), np.log(2))
    value.backward()
    assert positive.grad is not None and negative.grad is not None
    empty = m.answer_loss(torch.empty(0, 6), torch.empty(0, 6), torch.zeros(64, 6))
    assert np.isclose(float(empty), np.log(2))


def test_typed_ap_excludes_zero_positive_answers_with_same_eligibility():
    docs = [
        dict(
            doc_id="d",
            annotations=[
                dict(annotator_id="a", l3_persuasion=[dict(start=0, end=1, label=FAMILIES[0])]),
                dict(annotator_id="b", l3_persuasion=[]),
            ],
        )
    ]
    base = [
        dict(
            doc_id="d",
            pairs=[(0, 1, 0.0), (2, 3, 0.0)],
            probabilities=[[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1] * 6],
        )
    ]
    other = [
        dict(
            doc_id="d",
            pairs=base[0]["pairs"],
            probabilities=[[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.95, 0.1, 0.1, 0.1, 0.1, 0.1]],
        )
    ]
    a = m.candidate_diagnostics(docs, base, 2, 0)
    b = m.candidate_diagnostics(docs, other, 2, 0)
    assert a["ap"] == 1 and b["ap"] == 0.5
    assert a["ap_eligible_answers"] == 1 and len(a["ap_excluded_answers"]) == 1
    assert a["ap_excluded_answers"] == b["ap_excluded_answers"]
    assert b["negative_pair_acceptance"] > a["negative_pair_acceptance"]

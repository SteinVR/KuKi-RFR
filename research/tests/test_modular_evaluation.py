import importlib

import numpy as np

from research.kuki.data import FAMILIES
from research.kuki.l3_evaluation import evaluate

chain = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.chain")
training = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.train")
audit = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.proposal_audit")


def span(s, e, f=0):
    return dict(start=s, end=e, label=FAMILIES[f])


def doc(answers):
    return dict(
        doc_id="d",
        group_id="g",
        text="А б, в! Г д.",
        annotations=[dict(annotator_id=str(i), l3_persuasion=a) for i, a in enumerate(answers)],
    )


def test_exact_threshold_matches_brute_ties_multiannotator_and_empty():
    docs = [doc([[span(0, 1), span(0, 1, 1)], [span(3, 4)], []])]
    outputs = [
        dict(
            doc_id="d",
            spans=[
                dict(**span(0, 1), score=0.5),
                dict(**span(0, 1, 1), score=0.5),
                dict(**span(3, 4), score=0.2),
                dict(**span(4, 5), score=0.6),
            ],
            starts=[],
            ends=[],
            candidate_count=4,
            eligible_before_cap=0,
        )
    ]
    threshold, score = chain.exact_threshold(docs, outputs)
    brute = [(evaluate(docs, outputs, t)[0]["exact_f1"], t) for t in [0.2, 0.5, 0.6, 1.6]]
    best = max(brute)
    assert np.isclose(score, best[0])
    assert threshold == best[1]


def test_detector_threshold_matches_full_score_sweep():
    docs = [doc([[span(0, 1), span(0, 1, 1)], [span(3, 4)], []])]
    outputs = [dict(doc_id="d", pairs=[(4, 5, 0.6), (0, 1, 0.5), (3, 4, 0.2)])]
    threshold = training.detector_threshold(docs, outputs)
    brute = max(
        (training.detector_score(docs, outputs, t)["exact_f1"], t) for t in [0.2, 0.5, 0.6, 1.6]
    )
    assert threshold == brute[1]


def test_chain_proposal_recall_uses_pairs_not_cartesian_boundaries():
    docs = [doc([[span(0, 4)]])]
    predictions = [
        dict(doc_id="d", pairs=[(0, 1, 1.0), (3, 4, 1.0)], probabilities=[[0.5] * 6, [0.5] * 6])
    ]
    metrics, records, _ = chain.chain_score(docs, predictions, 0, 0)
    assert metrics["proposal_recall"] == 0
    assert records[0]["individuals"][0]["proposal_hits"] == 0


def test_analytic_unlimited_token_coverage_equals_materialized_enumeration():
    d = doc([[span(s, e) for s in range(11) for e in range(s + 1, 12)]])
    analytic = audit.measure(d, "all_word_spans_analytic")
    explicit = audit.measure(d, "ngrams32")
    assert analytic["candidate_count"] == explicit["candidate_count"]
    for metric in ["exact_recall", "iou50_coverage", "mean_best_iou"]:
        assert np.isclose(analytic[metric], explicit[metric])
    assert len(analytic["misses"]) == len(explicit["misses"])
    for a, b in zip(analytic["misses"], explicit["misses"], strict=True):
        assert np.isclose(a["best_iou"], b["best_iou"])

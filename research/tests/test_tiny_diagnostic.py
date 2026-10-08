"""Check diagnostic threshold decisions against an independent exhaustive scan."""

import importlib

import numpy as np
import pytest

mod = importlib.import_module("research.stages.ST2-l3-rhetoric.tiny_diagnostic")


def row(doc_id, scores, gold, coverage=1):
    pairs = np.array([[0, 1], [1, 3], [2, 4]])
    scores = np.array(scores, dtype=float).reshape(3, 6)
    return dict(
        doc_id=doc_id,
        pairs=pairs,
        scores=scores,
        gold=gold,
        positive=mod.targets(pairs, gold),
        pair_recall=coverage,
    )


def test_common_threshold_uses_empty_article_and_uncensored_scores():
    scores = np.full(18, 0.1)
    scores[0] = 0.25  # Below the old minimum threshold .3.
    positive = row("positive", scores, {(0, 1, 0)})
    empty = row("empty", np.full(18, 0.2), set(), None)
    result = mod.curves([positive, empty], 100)
    assert result["passed"]
    (interval,) = result["successful_intervals"]
    assert interval["lower_exclusive"] == 0.2
    assert interval["upper_inclusive"] == 0.25
    assert result["articles"][1]["correct_empty"]
    assert mod.curves([dict(positive, pair_recall=0.5), empty], 100)["passed"] is False


@pytest.mark.parametrize("cap", [2, 100, None])
def test_threshold_sweep_matches_independent_predictions_with_ties(cap):
    rng = np.random.default_rng(24)
    for _ in range(10):
        rows = [
            row("a", rng.choice([0.1, 0.2, 0.4, 0.8], 18), {(0, 1, 0), (1, 3, 2)}),
            row("b", rng.choice([0.1, 0.2, 0.4, 0.8], 18), set(), None),
        ]
        result = mod.curves(rows, cap)
        candidates = np.unique(np.r_[*[r["scores"].ravel() for r in rows], 1.0])
        best = None
        success = False
        for threshold in candidates:
            values, counts = [], []
            for r in rows:
                flat = r["scores"].ravel()
                order = np.argsort(-flat, kind="stable")
                if cap is not None:
                    order = order[:cap]
                preds = {
                    (int(r["pairs"][i // 6, 0]), int(r["pairs"][i // 6, 1]), int(i % 6))
                    for i in order
                    if flat[i] >= threshold
                }
                score = (
                    2 * len(preds & r["gold"]) / (len(preds) + len(r["gold"]))
                    if r["gold"]
                    else float(not preds)
                )
                values.append(score)
                counts.append(len(preds))
            success |= min(values) >= 0.95
            key = (min(values), sum(values) / len(values), -sum(counts), threshold)
            if best is None or key > best:
                best = key
        assert result["passed"] == success
        assert result["best_threshold"] == best[-1]

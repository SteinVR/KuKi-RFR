import numpy as np

from research.kuki.data import FAMILIES
from research.kuki.span_classifier import answer_pairs, examples, score


def span(label, start=0, end=3):
    return dict(start=start, end=end, label=label)


def test_multilabel_same_boundaries_grouped_only_within_answer():
    answer = {"l3_persuasion": [span(FAMILIES[0]), span(FAMILIES[1]), span(FAMILIES[0])]}
    pairs, y = answer_pairs(answer)
    assert pairs == [(0, 3)] and y.tolist() == [[1, 1, 0, 0, 0, 0]]


def test_conditional_score_equal_article_weight_and_no_empty_training_objects():
    docs = [
        dict(
            doc_id="a",
            text="abcdefgh",
            annotations=[
                dict(annotator_id="x", l3_persuasion=[span(FAMILIES[0]), span(FAMILIES[0], 3, 6)]),
                dict(annotator_id="y", l3_persuasion=[]),
            ],
        ),
        dict(
            doc_id="b",
            text="abcdefgh",
            annotations=[dict(annotator_id="x", l3_persuasion=[span(FAMILIES[1])])],
        ),
    ]
    rows = examples(docs)
    assert len(rows) == 3 and sum(r["weight"] for r in rows if r["doc_id"] == "a") == 1
    p = np.zeros((3, 6))
    p[:, 0] = 1
    m = score(rows, p, 0.5)
    assert m["article_mean_f1"] == 0.5 and m["n_answers"] == 2

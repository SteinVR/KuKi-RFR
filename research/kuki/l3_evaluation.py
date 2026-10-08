"""One-to-one span matching; individual answers averaged within articles."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import FAMILIES


def keys(spans, typed=True):
    return sorted(
        {(s["start"], s["end"], s["label"]) if typed else (s["start"], s["end"]) for s in spans}
    )


def matched(predicted, gold, typed=True, overlap=False):
    p, g = keys(predicted, typed), keys(gold, typed)
    if not overlap:
        return len(set(p) & set(g)), len(p), len(g)
    if not p or not g:
        return 0, len(p), len(g)
    intersection = np.array([[max(0, min(a[1], b[1]) - max(a[0], b[0])) for b in g] for a in p])
    union = np.array([[max(a[1], b[1]) - min(a[0], b[0]) for b in g] for a in p])
    iou = intersection / union
    allowed = iou >= 0.5
    if typed:
        allowed &= np.array([[a[2] == b[2] for b in g] for a in p])
    # Maximize number of admissible matches first; use IoU only to break ties.
    reward = allowed * (1 + iou / (min(len(p), len(g)) + 1))
    left, right = linear_sum_assignment(reward, maximize=True)
    return int(allowed[left, right].sum()), len(p), len(g)


def f1(tp, n_pred, n_gold):
    return 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0.0


def select_spans(output, threshold):
    return [s for s in output["spans"] if s["score"] >= threshold]


def evaluate(docs, outputs, threshold):
    records = []
    family_scores = {f: [] for f in FAMILIES}
    empty_correct, empty_count = 0, 0
    weighted_counts = np.zeros(3)
    for doc, out in zip(docs, outputs, strict=True):
        if out.get("doc_id", doc["doc_id"]) != doc["doc_id"]:
            raise ValueError("Prediction order mismatch")
        predicted = select_spans(out, threshold)
        individual = []
        for annotation in doc["annotations"]:
            gold = annotation["l3_persuasion"]
            exact = matched(predicted, gold)
            weighted_counts += np.asarray(exact) / len(doc["annotations"])
            item = dict(
                annotator_id=annotation["annotator_id"],
                n_gold=len(keys(gold)),
                n_predictions=len(keys(predicted)),
                exact_tp=exact[0],
                exact_f1=f1(*exact),
                exact_precision=exact[0] / exact[1] if exact[1] else 0.0,
                exact_recall=exact[0] / exact[2] if exact[2] else 0.0,
                overlap_f1=f1(*matched(predicted, gold, overlap=True)),
                localization_exact_f1=f1(*matched(predicted, gold, typed=False)),
                localization_overlap_f1=f1(*matched(predicted, gold, typed=False, overlap=True)),
                proposal_hits=sum(
                    s["start"] in out["starts"] and s["end"] in out["ends"] for s in gold
                ),
            )
            item["per_family"] = {
                family: f1(
                    *matched(
                        [s for s in predicted if s["label"] == family],
                        [s for s in gold if s["label"] == family],
                    )
                )
                for family in FAMILIES
            }
            individual.append(item)
            if not gold:
                empty_count += 1
                empty_correct += not predicted
        record = dict(
            doc_id=doc["doc_id"],
            group_id=doc["group_id"],
            individuals=individual,
            n_predictions=len(keys(predicted)),
            n_candidates=out["candidate_count"],
            eligible_before_cap=out["eligible_before_cap"],
        )
        for metric in [
            "exact_f1",
            "exact_precision",
            "exact_recall",
            "overlap_f1",
            "localization_exact_f1",
            "localization_overlap_f1",
        ]:
            record[metric] = float(np.mean([r[metric] for r in individual]))
        nonempty = [r for r in individual if r["n_gold"]]
        record["nonempty_exact_f1"] = (
            float(np.mean([r["exact_f1"] for r in nonempty])) if nonempty else None
        )
        record["nonempty_overlap_f1"] = (
            float(np.mean([r["overlap_f1"] for r in nonempty])) if nonempty else None
        )
        record["proposal_recall"] = (
            float(np.mean([r["proposal_hits"] / r["n_gold"] for r in nonempty]))
            if nonempty
            else None
        )
        for family in FAMILIES:
            family_scores[family].append(np.mean([r["per_family"][family] for r in individual]))
        records.append(record)
    result = {
        metric: float(np.mean([r[metric] for r in records]))
        for metric in [
            "exact_f1",
            "exact_precision",
            "exact_recall",
            "overlap_f1",
            "localization_exact_f1",
            "localization_overlap_f1",
        ]
    }
    for metric in ["nonempty_exact_f1", "nonempty_overlap_f1", "proposal_recall"]:
        eligible = [r[metric] for r in records if r[metric] is not None]
        result[metric] = float(np.mean(eligible)) if eligible else 0.0
    result.update(
        n_articles=len(docs),
        n_answers=sum(len(d["annotations"]) for d in docs),
        threshold=threshold,
        empty_empty_f1=0,
        exact_weighted_micro_f1=f1(*weighted_counts),
        per_family_exact_f1={f: float(np.mean(v)) for f, v in family_scores.items()},
        mean_predictions=float(np.mean([r["n_predictions"] for r in records])),
        nonempty_prediction_articles=sum(r["n_predictions"] > 0 for r in records),
        empty_answers=empty_count,
        correctly_empty_answers=empty_correct,
        false_nonempty_given_empty=(1 - empty_correct / empty_count if empty_count else None),
        empty_answer_rate=empty_count / sum(len(d["annotations"]) for d in docs),
    )
    return result, records


def empty_outputs(docs):
    return [
        dict(
            doc_id=d["doc_id"],
            spans=[],
            starts=[],
            ends=[],
            candidate_count=0,
            eligible_before_cap=0,
        )
        for d in docs
    ]

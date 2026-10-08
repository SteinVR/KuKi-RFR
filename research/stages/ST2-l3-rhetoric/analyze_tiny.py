"""Reproduce component comparisons from saved, uncensored diagnostic arrays."""

import importlib
import json

import numpy as np

from research.kuki.data import write_json

tiny = importlib.import_module("research.stages.ST2-l3-rhetoric.tiny_diagnostic")


def main():
    docs, _, _, _ = tiny.selected_docs()
    root = tiny.STAGE / "experiments/20260928-l3-tiny-verification-01"
    report = {}
    for variant in ["historical_hard", "tiny_three"]:
        arrays = np.load(root / variant / "checkpoints/full-scores-000.npz")
        report[variant] = []
        for i, doc in enumerate(docs):
            gold = {tiny.span_key(s) for s in doc["annotations"][0]["l3_persuasion"]}
            rows = {}
            for kind in ["ordinary", "fixed"]:
                pairs = arrays[f"d{i}_{kind}_pairs"]
                scores = arrays[f"d{i}_{kind}_scores"]
                positive = arrays[f"d{i}_{kind}_positive"]
                logits = arrays[f"d{i}_{kind}_logits"]
                row = dict(
                    doc_id=doc["doc_id"],
                    gold=gold,
                    pairs=pairs,
                    scores=scores,
                    positive=positive,
                    pair_recall=1,
                )
                curve = tiny.curves([row], 100 if kind == "ordinary" else None)
                rows[kind] = dict(
                    best_individual_threshold=curve["best_threshold"],
                    best_individual_metrics=curve["articles"][0],
                    positive_logits=tiny.distribution(logits[positive]),
                    negative_logits=tiny.distribution(logits[~positive]),
                    perfect_score_separation_margin=float(
                        scores[positive].min() - scores[~positive].max()
                    )
                    if positive.any()
                    else None,
                )
            report[variant].append(dict(doc_id=doc["doc_id"], **rows))
    write_json(tiny.SERIES / "component-comparison.json", report)
    print(
        json.dumps(
            {
                name: [
                    dict(
                        doc_id=r["doc_id"],
                        ordinary_best_f1=r["ordinary"]["best_individual_metrics"]["exact_typed_f1"],
                        fixed_best_f1=r["fixed"]["best_individual_metrics"]["exact_typed_f1"],
                    )
                    for r in values
                ]
                for name, values in report.items()
            }
        )
    )


if __name__ == "__main__":
    main()

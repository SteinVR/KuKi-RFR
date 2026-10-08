"""Post-hoc decoder diagnostic using saved top-100 scores; never refits a model."""

import json
from pathlib import Path

import numpy as np

from research.kuki.data import write_json, write_jsonl
from research.kuki.experiment import load_development
from research.kuki.l3_evaluation import evaluate

STAGE = Path("research/stages/ST2-l3-rhetoric")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def suppress_near_duplicates(spans, iou_threshold=0.95):
    selected = []
    for span in sorted(spans, key=lambda s: (-s["score"], s["start"], s["end"], s["label"])):
        redundant = False
        for prior in selected:
            if span["label"] != prior["label"]:
                continue
            intersection = max(
                0, min(span["end"], prior["end"]) - max(span["start"], prior["start"])
            )
            union = max(span["end"], prior["end"]) - min(span["start"], prior["start"])
            if intersection / union >= iou_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(span)
    return selected


def main():
    docs, _ = load_development(read(Path("research/configs/data.json")))
    oracle = {}
    for split, part in docs.items():
        losses, totals = 0, 0
        for d in part:
            for a in d["annotations"]:
                original = [
                    dict(start=s["start"], end=s["end"], label=s["label"], score=1.0)
                    for s in a["l3_persuasion"]
                ]
                totals += len(original)
                losses += len(original) - len(suppress_near_duplicates(original))
        oracle[split] = dict(observed_spans=totals, spans_suppressed=losses)
    results = {}
    for name in ["20260928-l3-pilot-s42-01", "20260928-l3-negative025-s42-01"]:
        run = STAGE / "experiments" / name
        if not (run / "metrics.json").exists():
            continue
        m = read(run / "metrics.json")
        original = rows(run / "dev-raw-predictions.jsonl")
        adjusted = [dict(o, spans=suppress_near_duplicates(o["spans"])) for o in original]
        before_after = {}
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            baseline = evaluate(docs["dev"], original, threshold)[0]
            alternative = evaluate(docs["dev"], adjusted, threshold)[0]
            before_after[str(threshold)] = dict(
                original=baseline, near_duplicate_suppression=alternative
            )
        fixed = before_after[str(m["threshold"])]
        results[name] = dict(
            original_epoch=m["selected_epoch"],
            original_threshold=m["threshold"],
            fixed_threshold=fixed,
            threshold_grid_exploratory=before_after,
            mean_retained_after_suppression=float(np.mean([len(o["spans"]) for o in adjusted])),
        )
        write_jsonl(run / "dev-near-duplicate-diagnostic.jsonl", adjusted)
    write_json(
        STAGE / "decoder-diagnostic.json",
        dict(
            status="exploratory_not_preregistered_not_replacing_primary",
            same_family_iou=0.95,
            input="already_retained_top_100_predictions_no_refilling_after_suppression",
            oracle=oracle,
            runs=results,
            test_scored=False,
        ),
    )
    print(json.dumps({n: r["fixed_threshold"] for n, r in results.items()}))


if __name__ == "__main__":
    main()

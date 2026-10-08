"""Post-hoc strict responsibility baseline, no new fitting or threshold selection."""

import json
from pathlib import Path

import numpy as np

from research.kuki.data import FAMILIES, write_json, write_jsonl
from research.kuki.experiment import load_development
from research.kuki.l3_evaluation import evaluate

ROOT = Path(__file__).parent


def main():
    chain = json.loads((ROOT / "E3-detector/chain.json").read_text())
    detector = json.loads((Path(chain["selected_runs"]["detector"]) / "metrics.json").read_text())
    classifier = json.loads(
        (Path(chain["selected_runs"]["classifier"]) / "metrics.json").read_text()
    )
    dt = detector["dev"]["threshold"]
    ct = classifier["dev"]["threshold"]
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    result = dict(
        post_hoc=True,
        detector_threshold=dt,
        classifier_threshold=ct,
        data_hashes=hashes,
        test_used=False,
        new_training=False,
        new_calibration=False,
    )
    for split in docs:
        scores = [
            json.loads(x)
            for x in (ROOT / "E3-detector" / f"chain-{split}-scores.jsonl").read_text().splitlines()
        ]
        outputs = []
        for p in scores:
            spans = []
            for (s, e, v), prob in zip(p["pairs"][:100], p["probabilities"][:100], strict=True):
                if v < dt:
                    continue
                take = np.asarray(prob) >= ct
                take[np.argmax(prob)] = True
                spans += [
                    dict(start=s, end=e, label=FAMILIES[i], score=v, class_probability=prob[i])
                    for i in np.flatnonzero(take)
                ]
            spans.sort(
                key=lambda s: (
                    -s["score"],
                    -s["class_probability"],
                    s["start"],
                    s["end"],
                    s["label"],
                )
            )
            outputs.append(
                dict(
                    doc_id=p["doc_id"],
                    spans=spans[:100],
                    starts=[],
                    ends=[],
                    candidate_count=sum(v >= dt for _, _, v in p["pairs"][:100]),
                    eligible_before_cap=len(spans),
                )
            )
        metrics, records = evaluate(docs[split], outputs, dt)
        # Here proposal recall refers to the detector-accepted pairs, not the full top500.
        for d, p, r in zip(docs[split], scores, records, strict=True):
            pairs = {(s, e) for s, e, v in p["pairs"][:100] if v >= dt}
            for a, z in zip(d["annotations"], r["individuals"], strict=True):
                z["proposal_hits"] = sum(
                    (s["start"], s["end"]) in pairs for s in a["l3_persuasion"]
                )
            nonempty = [a for a in r["individuals"] if a["n_gold"]]
            r["proposal_recall"] = (
                float(np.mean([a["proposal_hits"] / a["n_gold"] for a in nonempty]))
                if nonempty
                else None
            )
        metrics["proposal_recall"] = float(
            np.mean([r["proposal_recall"] for r in records if r["proposal_recall"] is not None])
        )
        result[split] = metrics
        write_json(ROOT / "E3-detector" / f"strict-{split}-articles.json", records)
        write_jsonl(ROOT / "E3-detector" / f"strict-{split}-predictions.jsonl", outputs)
    write_json(ROOT / "E3-detector/strict-chain.json", result)
    print(json.dumps({s: result[s]["exact_f1"] for s in docs}))


if __name__ == "__main__":
    main()

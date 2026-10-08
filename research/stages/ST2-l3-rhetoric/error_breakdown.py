"""Locate where observed gold spans disappear, without modifying predictions."""

import json
from pathlib import Path

import numpy as np

from research.kuki.data import write_json
from research.kuki.experiment import load_development

STAGE = Path("research/stages/ST2-l3-rhetoric")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    docs, _ = load_development(read(Path("research/configs/data.json")))
    output = {}
    categories = [
        "candidate_miss",
        "correct",
        "different_family_at_exact_bounds",
        "ranking_or_threshold_miss",
    ]
    for name in [
        "20260928-l3-pilot-s42-01",
        "20260928-l3-negative025-s42-01",
        "20260928-l3-hard-s42-01",
    ]:
        run = STAGE / "experiments" / name
        if not (run / "metrics.json").exists():
            continue
        raw = rows(run / "dev-raw-predictions.jsonl")
        pred = rows(run / "dev-predictions.jsonl")
        per_article = []
        for d, r, p in zip(docs["dev"], raw, pred, strict=True):
            assert d["doc_id"] == r["doc_id"] == p["doc_id"]
            predicted = {(s["start"], s["end"], s["label"]) for s in p["spans"]}
            geometry = {(s, e) for s, e, _ in predicted}
            individuals = []
            for answer in d["annotations"]:
                if not answer["l3_persuasion"]:
                    continue
                counts = dict.fromkeys(categories, 0)
                for span in answer["l3_persuasion"]:
                    s, e, f = span["start"], span["end"], span["label"]
                    if s not in r["starts"] or e not in r["ends"]:
                        counts["candidate_miss"] += 1
                    elif (s, e, f) in predicted:
                        counts["correct"] += 1
                    elif (s, e) in geometry:
                        counts["different_family_at_exact_bounds"] += 1
                    else:
                        counts["ranking_or_threshold_miss"] += 1
                n = len(answer["l3_persuasion"])
                individuals.append({k: v / n for k, v in counts.items()})
            if individuals:
                per_article.append(
                    dict(
                        doc_id=d["doc_id"],
                        **{k: float(np.mean([a[k] for a in individuals])) for k in categories},
                    )
                )
        mean = {k: float(np.mean([a[k] for a in per_article])) for k in categories}
        assert abs(sum(mean.values()) - 1) < 1e-12
        output[name] = dict(
            nonempty_article_count=len(per_article), fractions=mean, articles=per_article
        )
    write_json(
        STAGE / "error-breakdown.json",
        dict(
            runs=output,
            normalization="nonempty answers within article, then eligible articles",
            test_scored=False,
        ),
    )
    print(json.dumps({n: r["fractions"] for n, r in output.items()}))


if __name__ == "__main__":
    main()

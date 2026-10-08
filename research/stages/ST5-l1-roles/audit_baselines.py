"""Train/dev-only role audit and fixed CPU baselines."""

import importlib
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from research.kuki.data import ROLES, write_json
from research.kuki.experiment import initialize_run
from research.kuki.l1 import examples

M = importlib.import_module("research.stages.ST5-l1-roles.method")


def audit(docs):
    rows = examples(docs)
    bounds = defaultdict(set)
    for r in rows:
        bounds[r["doc_id"], r["start"], r["end"]].add(r["target"])
    totals = Counter()
    for r in rows:
        totals[r["doc_id"]] += r["weight"]
    assert all(abs(v - 1) < 1e-6 for v in totals.values())
    return dict(
        n_articles=len(docs), articles_with_roles=len(totals),
        empty_articles=[d["doc_id"] for d in docs if d["doc_id"] not in totals],
        empty_answers=sum(not a["l1_roles"] for d in docs for a in d["annotations"]),
        mentions=len(rows), roles=dict(Counter(ROLES[r["target"]] for r in rows)),
        unique_mentions=len(bounds), disputed_exact_mentions=sum(len(v) > 1 for v in bounds.values()),
        per_source={s: dict(articles=len({r["doc_id"] for r in rows if r["source"] == s}),
                           roles=dict(Counter(ROLES[r["target"]] for r in rows if r["source"] == s)))
                    for s in sorted({d["source"] for d in docs})},
    )


def main():
    start = time.perf_counter()
    cfg = M.read(M.ROOT / "configs/baselines.json")
    docs, _, prov = M.inputs(cfg)
    out = M.ROOT / "experiments" / cfg["run_id"]
    initialize_run(out, cfg, prov)
    rows = {s: examples(d) for s, d in docs.items()}
    y = np.array([r["target"] for r in rows["train"]])
    w = np.array([r["weight"] for r in rows["train"]])
    prior = np.bincount(y, weights=w, minlength=3) / w.sum()
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=50000, sublinear_tf=True)),
    ])
    x = features.fit_transform([r["text"] for r in rows["train"]])
    model = LogisticRegression(C=10, max_iter=1000, random_state=42)
    model.fit(x, y, sample_weight=w)
    assert model.classes_.tolist() == [0, 1, 2]
    for split in docs:
        M.persist(out / "prior", split, docs[split], np.tile(prior, (len(rows[split]), 1)))
        prob = model.predict_proba(features.transform([r["text"] for r in rows[split]]))
        M.persist(out / "tfidf", split, docs[split], prob)
    import joblib

    joblib.dump(dict(features=features, model=model, prior=prior), out / "checkpoints/baselines.joblib")
    stats = {s: audit(d) for s, d in docs.items()}
    vocab = {r["mention"].strip().casefold() for r in rows["train"]}
    stats["dev_seen_mention_rows"] = sum(r["mention"].strip().casefold() in vocab for r in rows["dev"])
    stats["test_used"] = False
    write_json(out / "audit.json", stats)
    write_json(out / "runtime.json", dict(seconds=time.perf_counter()-start, device="cpu"))
    print(stats)


if __name__ == "__main__":
    main()

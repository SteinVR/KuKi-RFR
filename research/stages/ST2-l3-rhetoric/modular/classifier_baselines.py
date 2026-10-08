import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from research.kuki.data import write_json
from research.kuki.experiment import load_development
from research.kuki.span_classifier import examples, score, select_threshold

ROOT = Path(__file__).parent


def main():
    begin = time.perf_counter()
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    rows = {s: examples(d) for s, d in docs.items()}
    y = np.asarray([r["target"] for r in rows["train"]])
    weights = np.array([r["weight"] for r in rows["train"]])
    weights = weights / weights.mean()
    freq = np.average(y, axis=0, weights=weights)
    dev = np.tile(freq, (len(rows["dev"]), 1))
    best = select_threshold(rows["dev"], dev)
    result = {
        "frequency": dict(
            dev=best, train=score(rows["train"], np.tile(freq, (len(y), 1)), best["threshold"])
        )
    }
    vectorizers = [
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000),
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=40000),
    ]
    texts = {s: [r["text"] for r in rs] for s, rs in rows.items()}
    xtrain = hstack([v.fit_transform(texts["train"]) for v in vectorizers]).tocsr()
    xdev = hstack([v.transform(texts["dev"]) for v in vectorizers]).tocsr()
    for c in [1.0, 10.0]:
        train_pred = []
        dev_pred = []
        for f in range(y.shape[1]):
            if len(np.unique(y[:, f])) < 2:
                train_pred.append(np.full(len(y), y[0, f]))
                dev_pred.append(np.full(len(rows["dev"]), y[0, f]))
                continue
            model = LogisticRegression(C=c, max_iter=500, random_state=42).fit(
                xtrain, y[:, f], sample_weight=weights
            )
            train_pred.append(model.predict_proba(xtrain)[:, 1])
            dev_pred.append(model.predict_proba(xdev)[:, 1])
        tp = np.array(train_pred).T
        dp = np.array(dev_pred).T
        best = select_threshold(rows["dev"], dp)
        result[f"tfidf_C{c:g}"] = dict(dev=best, train=score(rows["train"], tp, best["threshold"]))
    out = ROOT / "E1-classification"
    write_json(
        out / "baselines.json",
        dict(
            results=result,
            data_hashes=hashes,
            elapsed_seconds=time.perf_counter() - begin,
            test_used=False,
        ),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Fresh-process checkpoint replay and an independent confusion-matrix scorer."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib
import json

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.l1 import RoleClassifier, examples, predict
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST5-l1-roles.method")


def independent(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    cm = np.zeros((3, 3))
    raw = np.zeros((3, 3))
    ce = 0.
    for row in rows:
        y = row["target"]
        pred = int(np.argmax(row["probabilities"]))
        cm[y, pred] += row["weight"]
        raw[y, pred] += 1
        ce -= row["weight"] * np.log(max(row["probabilities"][y], 1e-12))
    def macro(m):
        return float(np.mean(np.divide(2*np.diag(m), m.sum(0)+m.sum(1),
                                       out=np.zeros(3), where=m.sum(0)+m.sum(1)>0)))
    result = dict(macro_f1=macro(cm), mention_macro_f1=macro(raw),
                  article_accuracy=float(np.trace(cm)/cm.sum()),
                  mention_accuracy=float(np.trace(raw)/raw.sum()), cross_entropy=float(ce/cm.sum()))
    expected = M.read(path.with_name(path.name.replace("predictions.jsonl", "metrics.json")))
    for key, value in result.items():
        assert abs(value - expected[key]) < 1e-10, (path, key, value, expected[key])
    return result


def main():
    M.reserve(180)
    out = M.ROOT / "experiments/20260929-l1-verification-s42-01"
    out.mkdir(exist_ok=False)
    results = []
    init = []
    for condition in ["base", "low-lr"]:
        cfg = M.read(M.ROOT / "configs" / f"{condition}.json")
        docs, encoder, prov = M.inputs(cfg)
        run = M.ROOT / "experiments" / cfg["run_id"]
        assert M.read(run / "metadata.json")["provenance"] == prov
        configure(42)
        encoded = encode_documents(M.active(docs["train"])+M.active(docs["dev"]), encoder, cfg)
        model = RoleClassifier(encoder, cfg).cuda()
        init.append(M.initialization(model))
        assert M.read(run / "initialization.json")["trainable_sha256"] == init[-1]
        for state, splits in [("best", ["train", "dev"]), ("last", ["dev"])]:
            payload = torch.load(run / "checkpoints" / f"{state}.pt", weights_only=False,
                                 map_location="cpu")
            assert payload["provenance"] == prov
            M.set_weights(model, payload["trainable"])
            for split in splits:
                path = run / state / f"{split}-predictions.jsonl"
                saved = [json.loads(s) for s in path.read_text().splitlines()]
                rows = examples(docs[split])
                assert [(r["doc_id"],r["annotation_index"],r["start"],r["end"],r["target"],r["weight"])
                        for r in rows] == [(r["doc_id"],r["annotation_index"],r["start"],r["end"],r["target"],r["weight"])
                                          for r in saved]
                p = predict(model, docs[split], encoded, rows)
                assert np.array_equal(p, np.array([r["probabilities"] for r in saved]))
                results.append(dict(condition=condition,state=state,split=split,
                                    exact=True, metrics=independent(path)))
        del model
        torch.cuda.empty_cache()
    assert len(set(init)) == 1
    baseline = M.ROOT / "experiments/20260929-l1-baselines-s42-01"
    for kind in ["prior", "tfidf"]:
        for split in ["train", "dev"]:
            path = baseline / kind / f"{split}-predictions.jsonl"
            results.append(dict(condition=kind,split=split,metrics=independent(path)))
    result = dict(status="passed", results=results, initialization_matched=True, test_used=False)
    write_json(out / "verification.json", result)
    return dict(status="passed", prediction_sets=6, metric_sets=len(results), initialization_matched=True)


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        M.account("verification", STARTED, result)

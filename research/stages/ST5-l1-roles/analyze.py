"""Declared development diagnostics; no new fits or test access."""

import importlib
import json
from collections import Counter

import numpy as np

from research.kuki.data import ROLES, write_json, write_jsonl
from research.kuki.l1 import examples, score

M = importlib.import_module("research.stages.ST5-l1-roles.method")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def bootstrap(rows, first, second):
    groups = sorted({r["group_id"] for r in rows})
    matrices = np.zeros((2,len(groups),3,3))
    for j, p in enumerate([first, second]):
        for r, v in zip(rows, p, strict=True):
            matrices[j,groups.index(r["group_id"]),r["target"],np.argmax(v)] += r["weight"]
    rng = np.random.default_rng(42)
    delta = []
    for _ in range(2000):
        cm = matrices[:,rng.integers(0,len(groups),len(groups))].sum(1)
        den = cm.sum(1)+cm.sum(2)
        f1 = np.divide(2*np.diagonal(cm,axis1=1,axis2=2),den,
                       out=np.zeros_like(den),where=den>0).mean(1)
        delta.append(f1[0]-f1[1])
    return dict(n_groups=len(groups), replicates=2000, seed=42,
                delta_macro_f1=score(rows,first)["macro_f1"]-score(rows,second)["macro_f1"],
                descriptive_percentile_95=np.quantile(delta,[.025,.975]).tolist())


def main():
    candidates = []
    for name in ["base","low-lr"]:
        cfg = M.read(M.ROOT / "configs" / f"{name}.json")
        run = M.ROOT / "experiments" / cfg["run_id"]
        metric = M.read(run / "best/dev-metrics.json")
        candidates.append((name,run,metric))
    chosen = max(candidates,key=lambda x:(x[2]["macro_f1"],-x[2]["cross_entropy"]))
    condition, run, metric = chosen
    cfg = M.read(M.ROOT / "configs" / f"{condition}.json")
    docs, _, _ = M.inputs(cfg)
    rows = examples(docs["dev"])
    saved = read_rows(run / "best/dev-predictions.jsonl")
    assert [(r["doc_id"],r["annotation_index"],r["start"],r["end"]) for r in rows] == [
        (r["doc_id"],r["annotation_index"],r["start"],r["end"]) for r in saved]
    p = np.array([r["probabilities"] for r in saved])
    baseline = read_rows(M.ROOT / "experiments/20260929-l1-baselines-s42-01/tfidf/dev-predictions.jsonl")
    assert [(r["doc_id"],r["annotation_index"],r["start"],r["end"]) for r in baseline] == [
        (r["doc_id"],r["annotation_index"],r["start"],r["end"]) for r in saved]
    q = np.array([r["probabilities"] for r in baseline])
    selection = dict(condition=condition, run_id=cfg["run_id"], checkpoint=str(run / "checkpoints/best.pt"),
                     best_epoch=M.read(run / "progress.json")["best_epoch"], dev=metric,
                     versus_tfidf=bootstrap(rows,p,q), test_used=False)
    write_json(M.ROOT / "selection.json",selection)
    vocabulary = {r["mention"].strip().casefold() for r in examples(docs["train"])}
    masks = {"seen":np.array([r["mention"].strip().casefold() in vocabulary for r in rows])}
    masks["unseen"] = ~masks["seen"]
    masks.update({f"source:{s}":np.array([r["source"]==s for r in rows])
                  for s in sorted({r["source"] for r in rows})})
    diagnostics = {}
    for key, mask in masks.items():
        part = [r for r,m in zip(rows,mask,strict=True) if m]
        # Preserve original within-article weights; slices are descriptive subsets.
        diagnostics[key] = dict(model=score(part,p[mask]),tfidf=score(part,q[mask]),
                                normalization="original article weights, conditional slice")
    write_json(M.ROOT / "diagnostics.json",diagnostics)
    text = {d["doc_id"]:d["text"] for d in docs["dev"]}
    errors = []
    for r,v in zip(rows,p,strict=True):
        pred = int(v.argmax())
        if pred != r["target"]:
            errors.append(dict(**{k:x for k,x in r.items() if k!="text"},
                               gold=ROLES[r["target"]],prediction=ROLES[pred],confidence=float(v[pred]),
                               context=text[r["doc_id"]][max(0,r["start"]-250):r["end"]+250]))
    errors.sort(key=lambda r:(-r["confidence"],r["doc_id"],r["start"]))
    write_jsonl(M.ROOT / "dev-errors.jsonl",errors)
    write_json(M.ROOT / "error-summary.json",dict(
        n_errors=len(errors), directions=dict(Counter(r["gold"]+" -> "+r["prediction"] for r in errors)),
        all_dev_errors=True, no_new_human_labels=True))
    print(json.dumps(selection,ensure_ascii=False))


if __name__ == "__main__":
    main()

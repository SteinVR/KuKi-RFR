"""Fresh replay of all search states, historical base compatibility and CPU metrics."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib
import json

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.l1 import examples, predict
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST5-l1-roles.search.method")
RoleClassifier = importlib.import_module("research.stages.ST5-l1-roles.search.model").TunedRoleClassifier
independent = importlib.import_module("research.stages.ST5-l1-roles.verify").independent
CONDITIONS = ["head-slow", "smooth", "head-slow-smooth", "dropout"]


def replay(model, docs, encoded, path):
    saved = [json.loads(s) for s in path.read_text().splitlines()]
    rows = examples(docs)
    keys = ["doc_id", "annotation_index", "start", "end", "target", "weight"]
    assert [[r[k] for k in keys] for r in rows] == [[r[k] for k in keys] for r in saved]
    probabilities = predict(model, docs, encoded, rows)
    assert np.array_equal(probabilities, np.array([r["probabilities"] for r in saved]))
    return independent(path)


def main():
    M.reserve(180)
    out = M.ROOT / "experiments/20260929-l1-search-verification-s42-01"
    out.mkdir(exist_ok=False)
    original = M.ROOT.parent / "experiments/20260929-l1-base-s42-01"
    original_hash = M.read(original / "initialization.json")["trainable_sha256"]
    results = []
    for condition in CONDITIONS:
        cfg = M.read(M.ROOT / "configs" / f"{condition}.json")
        docs, encoder, prov = M.inputs(cfg)
        run = M.ROOT / "experiments" / cfg["run_id"]
        assert M.read(run / "metadata.json")["provenance"] == prov
        configure(42)
        encoded = encode_documents(M.active(docs["train"])+M.active(docs["dev"]),encoder,cfg)
        model = RoleClassifier(encoder,cfg).cuda()
        assert M.initialization(model) == original_hash
        assert M.read(run / "initialization.json")["trainable_sha256"] == original_hash
        for state, splits in [("best",["train","dev"]),("last",["dev"]),("best-ce",["dev"])]:
            payload = torch.load(run / "checkpoints" / f"{state}.pt",weights_only=False,
                                 map_location="cpu")
            assert payload["provenance"] == prov
            M.set_weights(model,payload["trainable"])
            for split in splits:
                metric = replay(model,docs[split],encoded,run / state / f"{split}-predictions.jsonl")
                results.append(dict(condition=condition,state=state,split=split,exact=True,metrics=metric))
        if condition == "head-slow":
            original_state = torch.load(original / "checkpoints/best.pt",weights_only=False,
                                        map_location="cpu")
            M.set_weights(model,original_state["trainable"])
            metric = replay(model,docs["dev"],encoded,original / "best/dev-predictions.jsonl")
            results.append(dict(condition="original-base",state="best",split="dev",
                                exact=True,metrics=metric))
        del model
        torch.cuda.empty_cache()
    result = dict(status="passed",results=results,initialization_matches_original=True,test_used=False)
    write_json(out / "verification.json",result)
    return dict(status="passed",prediction_sets=len(results),metric_sets=len(results),
                initialization_matches_original=True)


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        M.account("verification",STARTED,result)

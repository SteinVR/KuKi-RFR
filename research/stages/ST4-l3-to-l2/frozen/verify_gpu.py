"""Fresh extraction of every train/dev feature and replay of four saved residual states."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.l2 import frozen_digest
from research.kuki.train_l3 import configure, encode_documents

C = importlib.import_module("research.stages.ST4-l3-to-l2.frozen.common")
E = importlib.import_module("research.stages.ST4-l3-to-l2.frozen.extract")


def main():
    cfg = C.config()
    docs, enc, _ = C.inputs()
    configure(42)
    encoded = encode_documents(docs["train"] + docs["dev"], enc, cfg)
    cc = C.M.candidates(docs)
    stored = C.cache()
    model = E.load_parent(enc, cfg)
    reference = C.read(C.ROOT / "extraction/cache.json")
    assert frozen_digest(model) == reference["frozen_state_sha256"]
    replay = {}
    with torch.no_grad():
        for split, part in docs.items():
            replay[split] = {}
            for d in part:
                assert time.perf_counter() - STARTED < C.remaining() - 10
                did = d["doc_id"]
                item = E.extract_item(model, encoded[did], cc[split][did]["pairs"], len(d["text"]))
                assert all(torch.equal(v, stored[split][did][k]) for k, v in item.items()), did
                replay[split][did] = item
    assert frozen_digest(model) == reference["frozen_state_sha256"]
    del model
    checks = {}
    for condition in ["control", "families"]:
        out = C.ROOT / "experiments" / f"20260929-st4-frozen-{condition}-s42-01"
        model, _, _ = C.model_optimizer(condition, cfg)
        for state in ["best", "last"]:
            saved = torch.load(out / "checkpoints" / f"{state}.pt", weights_only=False)
            model.load_state_dict(saved["trainable"])
            for split, part in docs.items():
                pred = C.predict(model, part, replay[split])
                expected = np.asarray([r["probabilities"] for r in
                                       C.M.jsonl(out / state / f"{split}-l2.jsonl")])
                assert np.array_equal(pred, expected), (condition, state, split)
                checks[f"{condition}/{state}/{split}"] = True
    return dict(status="passed", fresh_feature_articles=176, all_features_exact=True,
                frozen_state_exact=True, prediction_checks=checks, test_used=False)


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        seconds = time.perf_counter() - STARTED
        C.record("fresh_replay", seconds, result)
        write_json(C.ROOT / "gpu-verification.json", dict(seconds=seconds, **result))

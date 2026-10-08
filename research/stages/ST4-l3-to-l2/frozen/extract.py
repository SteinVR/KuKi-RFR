"""One immutable feature extraction; annotations only enter separate L3 evaluation."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import importlib

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.joint import JointModel
from research.kuki.l2 import frozen_digest
from research.kuki.span_classifier import examples
from research.kuki.train_l3 import configure, encode_documents

C = importlib.import_module("research.stages.ST4-l3-to-l2.frozen.common")


def load_parent(enc, cfg):
    model = JointModel(enc, cfg).cuda()
    saved = torch.load(C.M.PARENT / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    C.M.OLD.set_weights(model, saved["trainable"])
    model.requires_grad_(False)
    model.eval()
    return model


def extract_item(model, item, pairs, length):
    pooled, features, prefix = model.encode_shared(item)
    indices = torch.tensor([(s, e) for s, e, _ in pairs], device=pooled.device).reshape(-1, 2)
    assert len(indices) and ((indices[:, 0] >= 0) & (indices[:, 1] <= length)
                             & (indices[:, 1] > indices[:, 0])).all()
    start, end = indices.T
    mean = (prefix[end] - prefix[start]) / (end - start)[:, None]
    probability = model.span_logits(features, prefix, [(s, e) for s, e, _ in pairs]).sigmoid()
    message = torch.cat([mean, probability, indices.float() / length,
                         pooled.new_tensor([v for _, _, v in pairs]).sigmoid()[:, None]], 1)
    return dict(pooled=pooled, message=message, base_logits=model.head["l2"](pooled))


def main():
    assert not C.CACHE.exists(), "Refusing to overwrite feature cache"
    cfg = C.config()
    docs, enc, prov = C.inputs()
    out = C.ROOT / "extraction"
    C.initialize(out, dict(cfg, run_id="20260929-st4-frozen-cache-s42-01"), prov)
    configure(42)
    encoded = encode_documents(docs["train"] + docs["dev"], enc, cfg)
    cc = C.M.candidates(docs)
    model = load_parent(enc, cfg)
    frozen = frozen_digest(model)
    cached, checks = {}, {}
    with torch.no_grad():
        for split, part in docs.items():
            cached[split] = {}
            for d in part:
                assert time.perf_counter() - STARTED < min(180, C.remaining()) - 20
                did = d["doc_id"]
                item = extract_item(model, encoded[did], cc[split][did]["pairs"], len(d["text"]))
                cached[split][did] = {k: v.detach().cpu() for k, v in item.items()}
            pred = C.M.OLD.predict(model, part, encoded, examples(part), cc[split])
            # OOF train candidates differ from the historical train chain. L2 and
            # conditional heads must still be exact; dev automatic chain is exact too.
            for kind in ["l2", "conditional"]:
                wanted = np.asarray([r["probabilities"] for r in
                                     C.M.jsonl(C.M.PARENT / "best" / f"{split}-{kind}.jsonl")])
                assert np.array_equal(pred[kind], wanted), (split, kind)
            if split == "dev":
                assert pred["chain"] == C.M.jsonl(C.M.PARENT / "best/dev-chain.jsonl")
            C.M.OLD.persist_predictions(out, split, part, examples(part), pred)
            checks[split] = dict(l2_exact=True, conditional_exact=True,
                                 automatic_exact=split == "dev", n_articles=len(part))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert frozen_digest(model) == frozen
    C.CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, C.CACHE)
    write_json(out / "cache.json", dict(path=str(C.CACHE), sha256=digest(C.CACHE),
               frozen_state_sha256=frozen, input_provenance=prov, checks=checks,
               tensors=["pooled", "message", "base_logits"], gold_in_messages=False))
    return dict(status="passed", checks=checks, frozen_unchanged=True,
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        C.record("extraction", time.perf_counter() - STARTED, result)

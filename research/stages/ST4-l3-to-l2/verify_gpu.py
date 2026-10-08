"""Fresh process replay of ST4 selected weights and OOF provenance sources."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.evaluation import score
from research.kuki.global_pointer import GlobalPointer
from research.kuki.message import MODES
from research.kuki.span_classifier import examples
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
MOD = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.train")


def main():
    cfg = M.read(M.ROOT / "configs/continue.json")
    docs, enc, _ = M.inputs(cfg)
    configure(42)
    encoded = encode_documents(docs["dev"], enc, cfg)
    cc = M.candidates(docs)
    rows = examples(docs["dev"])
    checks = {}
    diagnostics = {}
    for mode in MODES:
        cfg = M.read(M.ROOT / "configs" / f"{mode}.json")
        out = M.ROOT / "experiments" / cfg["run_id"]
        configure(42)
        model = M.model_from_parent(enc, cfg)
        for state in ["best", "last"]:
            payload = torch.load(
                out / "checkpoints" / f"{state}.pt", map_location="cpu", weights_only=False
            )
            M.OLD.set_weights(model, payload["trainable"])
            pred = M.predict(model, docs["dev"], encoded, rows, cc["dev"])
            for name, key in [("l2", "l2"), ("base-l2", "base"), ("conditional", "conditional")]:
                expected = np.asarray(
                    [r["probabilities"] for r in M.jsonl(out / state / f"dev-{name}.jsonl")]
                )
                assert np.array_equal(expected, pred[key]), (
                    mode,
                    name,
                    float(np.max(abs(expected - pred[key]))),
                )
            assert M.jsonl(out / state / "dev-chain.jsonl") == pred["chain"]
            checks[f"{mode}/{state}"] = dict(
                l2_exact=True, base_exact=True, conditional_exact=True, full_chain_exact=True
            )
            diagnostics[f"{mode}/{state}"] = dict(
                normal=score(docs["dev"], pred["l2"]), off=score(docs["dev"], pred["base"])
            )
            if mode in ["uniform", "confidence", "contextual"]:
                shuffle = M.predict(
                    model, docs["dev"], encoded, rows, cc["dev"], full=False, intervention="shuffle"
                )
                diagnostics[f"{mode}/{state}"]["shuffle"] = score(docs["dev"], shuffle["l2"])
        del model, payload, pred
        torch.cuda.empty_cache()
    for fold in range(3):
        out = M.ROOT / "oof" / f"20260929-oof{fold}-s42-01"
        cfg = M.read(out / "config.json")["config"]
        held = M.jsonl(out / "predictions.jsonl")
        byid = {d["doc_id"]: d for d in docs["train"]}
        chosen = sorted(held, key=lambda r: len(byid[r["doc_id"]]["text"]))
        sample = [chosen[0], chosen[-1]]
        part = [byid[r["doc_id"]] for r in sample]
        configure(42)
        tokens = encode_documents(part, enc, cfg)
        model = GlobalPointer(enc, cfg).cuda()
        payload = torch.load(out / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        M.OLD.set_weights(model, payload["trainable"])
        replay = MOD.detector_predict(model, part, tokens)
        # JSON round trip normalizes tuple/list coordinate representation.
        import json

        assert json.loads(json.dumps(replay)) == sample
        checks[f"oof{fold}"] = dict(two_excluded_articles_exact=True)
        del model, payload
        torch.cuda.empty_cache()
    return dict(status="passed", checks=checks, interventions=diagnostics, test_used=False)


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        seconds = time.perf_counter() - STARTED
        M.ledger_add("fresh_gpu_verification", seconds, result)
        write_json(M.ROOT / "gpu-verification.json", dict(seconds=seconds, **result))
        print(dict(status=result["status"], seconds=seconds), flush=True)

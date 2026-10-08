"""Fresh-process checkpoint replay for all selected ST3 models."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import importlib
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.joint import JointModel
from research.kuki.span_classifier import examples
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path("research/stages/ST3-l2-l3-joint")


def main():
    docs, enc, _ = M.inputs(M.read(ROOT / "configs/joint100.json"))
    candidates = {r["doc_id"]: r for r in M.jsonl(M.DETECTOR / "dev-predictions.jsonl")}
    rows = examples(docs["dev"])
    checks = {}
    cfg = M.read(ROOT / "configs/joint100.json")
    configure(42)
    encoded = encode_documents(docs["dev"], enc, cfg)
    for name in ["l2-only", "l3-only", "joint025", "joint100"]:
        cfg = M.read(ROOT / "configs" / f"{name}.json")
        out = ROOT / "experiments" / cfg["run_id"]
        configure(42)
        model = JointModel(enc, cfg).cuda()
        saved = torch.load(out / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        M.set_weights(model, saved["trainable"])
        result = M.predict(
            model, docs["dev"], encoded, rows, candidates if cfg["l3_weight"] else None
        )
        deltas = {}
        if cfg["l2_weight"]:
            expected = np.asarray([r["probabilities"] for r in M.jsonl(out / "best/dev-l2.jsonl")])
            assert np.array_equal(expected, result["l2"])
            deltas["l2"] = 0.0
        if cfg["l3_weight"]:
            expected = np.asarray(
                [r["probabilities"] for r in M.jsonl(out / "best/dev-conditional.jsonl")]
            )
            assert np.array_equal(expected, result["conditional"])
            cached = M.jsonl(out / "best/dev-chain.jsonl")
            assert cached == result["chain"]
            deltas["conditional"] = 0.0
            deltas["full_chain"] = 0.0
        if cfg["l2_weight"] and cfg["l3_weight"]:
            # L2 output must not depend on the L3 parameters or presence of gold spans.
            model.eval()
            d = docs["dev"][0]
            with torch.no_grad():
                for key in ["token", "char", "boundary_features", "span"]:
                    for p in model.head[key].parameters():
                        p.add_(3.0)
                pooled, _, _ = model.encode_shared(encoded[d["doc_id"]], need_l3=False)
                counter = model.head["l2"](pooled).sigmoid().cpu().numpy()
            assert np.array_equal(counter, result["l2"][0])
            deltas["l2_invariant_to_l3_head_perturbation"] = True
        checks[name] = dict(max_abs_differences=deltas, best_epoch=saved["progress"]["best_epoch"])
        del model, saved, result
        torch.cuda.empty_cache()
    return dict(status="passed", checks=checks, test_used=False)


if __name__ == "__main__":
    try:
        result = main()
    except Exception as exc:
        result = dict(status="failed", error=repr(exc))
        raise
    finally:
        elapsed = time.perf_counter() - STARTED
        ledger = M.read(ROOT / "budget.json")
        ledger["seconds"] += elapsed
        ledger["runs"].append(dict(kind="fresh_reload_verification", seconds=elapsed, **result))
        write_json(ROOT / "budget.json", ledger)
        write_json(ROOT / "gpu-verification.json", dict(seconds=elapsed, **result))
        print(dict(seconds=elapsed, **result))

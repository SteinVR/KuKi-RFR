"""Technical-only ST4 checks; no measured weights feed scientific runs."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.global_pointer import GlobalPointer
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.message import MODES
from research.kuki.train_l2 import make_optimizer
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
OOF = importlib.import_module("research.stages.ST4-l3-to-l2.oof_train")


def main():
    cfg = M.read(M.ROOT / "configs/contextual.json")
    docs, enc, prov = M.inputs(cfg, False)
    ordered = sorted(docs["train"], key=lambda d: len(d["text"]))
    batch = [ordered[i] for i in np.linspace(0, len(ordered) - 1, 8, dtype=int)]
    configure(42)
    encoded = encode_documents(batch, enc, cfg)
    cc = {r["doc_id"]: r for r in M.jsonl(M.OLD.DETECTOR / "train-predictions.jsonl")}
    for doc in batch:
        encoded[doc["doc_id"]]["candidates"] = cc[doc["doc_id"]]["pairs"]
    expected = {r["doc_id"]: r["probabilities"] for r in M.jsonl(M.PARENT / "best/train-l2.jsonl")}
    checks = {}
    out = M.ROOT / "technical"
    out.mkdir(exist_ok=True)
    for mode in MODES:
        cfg = M.read(M.ROOT / "configs" / f"{mode}.json")
        configure(42)
        model = M.model_from_parent(enc, cfg)
        frozen = frozen_digest(model)
        model.eval()
        with torch.no_grad():
            d = batch[-1]
            item = encoded[d["doc_id"]]
            pooled, f, p = model.encode_shared(item)
            base, final = model.l2_outputs(pooled, f, p, item["candidates"], item["length"])
            assert torch.equal(base, final)
            assert np.array_equal(final.sigmoid().cpu().numpy(), expected[d["doc_id"]])
        initial = trainable_state(model)
        opt, sched = make_optimizer(model, cfg)
        torch.cuda.synchronize()
        t = time.perf_counter()
        M.OLD.update(model, opt, sched, batch, encoded)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        if mode != "continue":
            assert not torch.equal(
                initial["head.refine_out.weight"], trainable_state(model)["head.refine_out.weight"]
            )
        if mode == "contextual":
            checkpoint = out / "resume.pt"
            save_checkpoint(checkpoint, model, opt, sched, dict(updates=1), prov)
            M.OLD.update(model, opt, sched, batch, encoded)
            wanted = trainable_state(model)
            optim_expected = opt.state_dict()
            sched_expected = sched.state_dict()
            load_checkpoint(checkpoint, model, opt, sched, prov)
            M.OLD.update(model, opt, sched, batch, encoded)
            assert all(torch.equal(p, wanted[n]) for n, p in trainable_state(model).items())
            assert sched.state_dict() == sched_expected
            for key, state in opt.state_dict()["state"].items():
                for name, value in state.items():
                    reference = optim_expected["state"][key][name]
                    assert (
                        torch.equal(value, reference)
                        if torch.is_tensor(value)
                        else value == reference
                    )
            assert all(
                not torch.equal(initial[n], p)
                for n, p in trainable_state(model).items()
                if "refine_" in n
            )
        assert frozen_digest(model) == frozen
        checks[mode] = dict(
            update_seconds=seconds,
            rough_152_update_seconds=seconds * 152,
            initial_equal=True,
            frozen_unchanged=True,
        )
        del model, opt, sched
        torch.cuda.empty_cache()
    cfg = M.read(M.OLD.DETECTOR / "config.json")["config"]
    configure(42)
    model = GlobalPointer(enc, cfg).cuda()
    opt, sched = make_optimizer(model, cfg)
    t = time.perf_counter()
    OOF.update(model, opt, sched, batch, encoded)
    torch.cuda.synchronize()
    checks["oof"] = dict(update_seconds=time.perf_counter() - t)
    return dict(
        status="passed",
        checks=checks,
        exact_resume=True,
        train_only=True,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        seconds = time.perf_counter() - STARTED
        M.ledger_add("technical", seconds, result)
        write_json(M.ROOT / "profile.json", dict(seconds=seconds, **result))
        print(dict(seconds=seconds, **result))

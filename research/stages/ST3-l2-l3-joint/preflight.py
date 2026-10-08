"""Addressed gradient, forward equivalence, resume and cost checks on training data only."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.kuki.data import write_json
from research.kuki.joint import JointModel, conditional_loss
from research.kuki.l2 import (
    L2Model,
    article_loss,
    frozen_digest,
    load_checkpoint,
    save_checkpoint,
    trainable_state,
)
from research.kuki.span_classifier import SpanClassifier, answer_pairs
from research.kuki.train_l2 import make_optimizer
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path(__file__).parent


def main():
    out = ROOT / "experiments/20260929-preflight-s42-01"
    out.mkdir(exist_ok=False)
    cfg = M.read(ROOT / "configs/joint100.json")
    docs, enc, prov = M.inputs(cfg)
    train = docs["train"]
    lengths = sorted(train, key=lambda d: (len(d["text"]), d["doc_id"]))
    selected = [lengths[i] for i in np.linspace(0, len(lengths) - 1, 8, dtype=int)]
    # Ensure a nonempty first article for direct gradient checks.
    d = next(d for d in train if any(a["l3_persuasion"] for a in d["annotations"]))
    if d not in selected:
        selected.append(d)
    configure(42)
    encoded = encode_documents(selected, enc, cfg)
    model = JointModel(enc, cfg).cuda()
    frozen = frozen_digest(model)
    initial = M.initial_hashes(model)
    model.eval()
    item = encoded[d["doc_id"]]
    with torch.no_grad():
        pooled, features, prefix = model.encode_shared(item)
        ref2 = L2Model.__new__(L2Model)
        nn.Module.__init__(ref2)
        ref2.encoder = model.encoder
        ref2.head = model.head["l2"]
        ref2.eval()
        delta_l2 = float((ref2(item["encoded"]) - model.head["l2"](pooled)).abs().max())
        ref3 = SpanClassifier.__new__(SpanClassifier)
        nn.Module.__init__(ref3)
        ref3.encoder = model.encoder
        ref3.head = model.head
        ref3.cfg = cfg
        ref3.eval()
        pairs = sorted({p for a in d["annotations"] for p in answer_pairs(a)[0]})
        old = ref3.probabilities(item, pairs)
        new = model.span_logits(features, prefix, pairs).sigmoid().cpu().numpy()
        delta_l3 = float(np.abs(old - new).max())
    assert delta_l2 == 0 and delta_l3 == 0
    pooled, features, prefix = model.encode_shared(item)
    loss2 = article_loss(model.head["l2"](pooled), [a["l2"] for a in d["annotations"]])
    loss3 = conditional_loss(model, features, prefix, d["annotations"])
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gradients = {}
    for name, loss in [("l2", loss2), ("l3", loss3)]:
        grads = torch.autograd.grad(
            loss, [p for _, p in named], allow_unused=True, retain_graph=True
        )
        gradients[name] = {
            part: sum(
                float(g.float().square().sum())
                for (n, _), g in zip(named, grads, strict=True)
                if g is not None and (n.startswith(part))
            )
            ** 0.5
            for part in ["encoder.", "head.l2.", "head.span."]
        }
    assert gradients["l2"]["encoder."] > 0 and gradients["l3"]["encoder."] > 0
    assert gradients["l2"]["head.span."] == 0 and gradients["l3"]["head.l2."] == 0
    del loss2, loss3, grads, pooled, features, prefix, ref2, ref3
    optimizer, scheduler = make_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    batch = selected[:8]
    tick = time.perf_counter()
    M.update(model, optimizer, scheduler, batch, encoded)
    update_seconds = time.perf_counter() - tick
    progress = dict(epoch=0, next_position=8, article_order=[x["doc_id"] for x in batch], updates=1)
    save_checkpoint(out / "resume.pt", model, optimizer, scheduler, progress, prov)
    M.update(model, optimizer, scheduler, batch, encoded)
    continuous = trainable_state(model)
    opt_end = optimizer.state_dict()
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    configure(42)
    resumed = JointModel(enc, cfg).cuda()
    assert M.initial_hashes(resumed) == initial
    opt, sch = make_optimizer(resumed, cfg)
    assert load_checkpoint(out / "resume.pt", resumed, opt, sch, prov) == progress
    M.update(resumed, opt, sch, batch, encoded)
    assert all(torch.equal(p, continuous[n]) for n, p in trainable_state(resumed).items())
    assert sch.last_epoch == 2
    for idx, state in opt_end["state"].items():
        for k, v in state.items():
            assert (
                torch.equal(v, opt.state_dict()["state"][idx][k])
                if torch.is_tensor(v)
                else v == opt.state_dict()["state"][idx][k]
            )
    assert frozen_digest(resumed) == frozen
    result = dict(
        status="passed",
        forward_l2_max_abs_diff=delta_l2,
        forward_l3_max_abs_diff=delta_l3,
        gradient_norms=gradients,
        resume_next_update_parameters_exact=True,
        resume_optimizer_exact=True,
        frozen_base_unchanged=True,
        profile_batch_doc_ids=[d["doc_id"] for d in batch],
        profile_batch_char_lengths=[len(d["text"]) for d in batch],
        profile_update_seconds=update_seconds,
        rough_20_epoch_train_seconds=update_seconds * 380,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        technical_optimizer_steps=3,
        test_used=False,
        initialization=initial,
    )
    write_json(out / "verification.json", result)
    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as exc:
        result = dict(status="failed", error=repr(exc))
        raise
    finally:
        seconds = time.perf_counter() - STARTED
        ledger = M.read(ROOT / "budget.json")
        ledger["seconds"] += seconds
        ledger["runs"].append(dict(kind="preflight", seconds=seconds, **result))
        write_json(ROOT / "budget.json", ledger)
        print(dict(seconds=seconds, **result), flush=True)

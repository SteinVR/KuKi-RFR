"""Exactly one bounded continuation; historical checkpoints/caches are read-only."""

# ruff: noqa: E402
# Start the budget clock before heavyweight imports.
import time

STARTED = time.perf_counter()

import importlib
import json
import signal
import traceback
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json, write_jsonl
from research.kuki.experiment import initialize_run, load_development, provenance
from research.kuki.l2 import frozen_digest, save_checkpoint
from research.kuki.span_classifier import SpanClassifier
from research.kuki.train_l2 import epoch_order, learning_rate_multiplier, make_optimizer
from research.kuki.train_l3 import configure, encode_documents

ROOT = Path(__file__).parent
method = importlib.import_module("research.stages.ST2-l3-rhetoric.negative_fit.method")
OPTIONS = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "runs" / OPTIONS["run_id"]


def deadline(signum, frame):
    raise TimeoutError("180 second GPU-process ceiling reached; no restart permitted")


def weights(model, state):
    assert set(state) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[n])


def predict(model, docs, encoded, cache):
    model.eval()
    result = []
    with torch.no_grad():
        for d in docs:
            pairs = cache[d["doc_id"]]["pairs"]
            prob = model.probabilities(encoded[d["doc_id"]], [(s, e) for s, e, _ in pairs])
            result.append(dict(doc_id=d["doc_id"], pairs=pairs, probabilities=prob.tolist()))
    return result


def main():
    remaining = 180 - (time.perf_counter() - STARTED)
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, remaining))
    original = Path(OPTIONS["parent"])
    packed = json.loads((original / "config.json").read_text())
    cfg = dict(
        packed["config"],
        run_id=OPTIONS["run_id"],
        training_recipe=OPTIONS,
        loss="half_positive_BCE_plus_half_negative_BCE; empty_answer_negative_only",
        training_candidates="fixed_detector_top500_negatives_plus_per_answer_gold",
        inference_candidates="fixed_detector_top500",
        conditional_argmax_fallback=False,
        selection="last_update404_no_dev_selection",
    )
    docs, hashes = load_development(packed["data"])
    train_docs = [d for d in docs["train"] if any(a["l3_persuasion"] for a in d["annotations"])]
    assert len(train_docs) == 129
    caches = {}
    for split in ["train", "dev"]:
        p = Path(OPTIONS["candidate_cache"]) / f"chain-{split}-scores.jsonl"
        caches[split] = {x["doc_id"]: x for x in map(json.loads, p.read_text().splitlines())}
    prov = provenance(cfg, packed["data"], hashes, packed["encoder"])
    prov["input_checkpoints"] = {
        str(original / "checkpoints" / n): digest(original / "checkpoints" / n)
        for n in ["best.pt", "last.pt"]
    }
    prov["candidate_caches"] = {
        str(Path(OPTIONS["candidate_cache"]) / f"chain-{s}-scores.jsonl"): digest(
            Path(OPTIONS["candidate_cache"]) / f"chain-{s}-scores.jsonl"
        )
        for s in ["train", "dev"]
    }
    for p in [Path(__file__), ROOT / "method.py", ROOT.parent / "negative-candidates-proposal.md"]:
        prov["code_sha256"][str(p)] = digest(p)
    initialize_run(OUT, cfg, prov)
    configure(42)
    encoded = encode_documents(train_docs + docs["dev"], packed["encoder"], cfg)
    model = SpanClassifier(packed["encoder"], cfg).cuda()
    base_hash = frozen_digest(model)
    start = torch.load(original / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    control = torch.load(original / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert start["updates"] == 340 and control["updates"] == 404
    # Baseline inference only, no baseline optimizer steps.
    weights(model, control["trainable"])
    baseline = predict(model, docs["dev"], encoded, caches["dev"])
    write_jsonl(OUT / "control-dev-scores.jsonl", baseline)
    weights(model, start["trainable"])
    optimizer, scheduler = make_optimizer(model, cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    expected_groups = [
        [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")],
        [f"head.{n}" for n, p in model.head.named_parameters()],
    ]
    assert groups == expected_groups
    for current, saved in zip(
        optimizer.param_groups, start["optimizer"]["param_groups"], strict=True
    ):
        assert len(current["params"]) == len(saved["params"])
        for param, number in zip(current["params"], saved["params"], strict=True):
            state = start["optimizer"]["state"].get(number, {})
            if param.requires_grad:
                assert (
                    state
                    and state["exp_avg"].shape == param.shape
                    and state["exp_avg_sq"].shape == param.shape
                )
            else:
                assert not state
    optimizer.load_state_dict(start["optimizer"])
    scheduler.load_state_dict(start["scheduler"])
    assert scheduler.last_epoch == 340
    expected = [
        cfg["adapter_lr"] * learning_rate_multiplier(340, 41, 408),
        cfg["head_lr"] * learning_rate_multiplier(340, 41, 408),
    ]
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert np.allclose(lrs, expected, rtol=0, atol=1e-15)
    assert lrs == scheduler.get_last_lr()
    write_json(
        OUT / "optimizer-check.json",
        dict(
            parameter_groups=groups,
            shapes_checked=True,
            start_lrs=lrs,
            scheduler_last_epoch=scheduler.last_epoch,
            expected_lrs=expected,
            source_has_rng="rng" in start,
        ),
    )
    print(
        json.dumps(
            dict(
                optimizer_restored=True,
                lrs=lrs,
                scheduler_step=scheduler.last_epoch,
                elapsed=time.perf_counter() - STARTED,
            )
        ),
        flush=True,
    )
    schedule = []
    for epoch in range(20, 24):
        order = epoch_order(129, 42, epoch)
        for pos in range(0, 129, 8):
            schedule.append(dict(epoch_index=epoch, position=pos, indices=order[pos : pos + 8]))
    schedule = schedule[:64]
    assert sum(len(x["indices"]) for x in schedule) == 491
    write_json(
        OUT / "schedule.json", dict(steps=schedule, doc_ids=[d["doc_id"] for d in train_docs])
    )
    configure(42)  # New run RNG is explicit; historic dropout RNG is unavailable.
    progress = dict(
        completed_updates=0,
        total_updates=340,
        article_presentations=0,
        next_schedule_index=0,
        schedule=schedule,
        history=[],
        status="running",
    )
    optimizer.zero_grad(set_to_none=True)
    save_checkpoint(OUT / "checkpoints/start.pt", model, optimizer, scheduler, progress, prov)
    longest = 0.0
    prepared = {}
    for step, item in enumerate(schedule):
        elapsed = time.perf_counter() - STARTED
        if elapsed >= 130 or elapsed + max(4.0, longest + 1.0) >= 133:
            progress["status"] = "incomplete_time_limit"
            break
        epoch = item["epoch_index"]
        if epoch not in prepared:
            prepared[epoch] = {
                d["doc_id"]: method.samples(d, caches["train"][d["doc_id"]], epoch)
                for d in train_docs
            }
        batch = [train_docs[i] for i in item["indices"]]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        tick = time.perf_counter()
        for d in batch:
            loss = method.training_loss(model, encoded[d["doc_id"]], prepared[epoch][d["doc_id"]])
            assert torch.isfinite(loss)
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
        params = [p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        duration = time.perf_counter() - tick
        longest = max(longest, duration)
        progress.update(
            completed_updates=step + 1,
            total_updates=341 + step,
            article_presentations=progress["article_presentations"] + len(batch),
            next_schedule_index=step + 1,
        )
        progress["history"].append(
            dict(
                update=341 + step,
                epoch_index=epoch,
                position=item["position"],
                loss=float(np.mean(losses)),
                lrs=[g["lr"] for g in optimizer.param_groups],
                elapsed_seconds=time.perf_counter() - STARTED,
            )
        )
        assert time.perf_counter() - STARTED < 135, "Training deadline crossed"
        if (step + 1) % 8 == 0:
            save_checkpoint(
                OUT / "checkpoints/latest.pt", model, optimizer, scheduler, progress, prov
            )
            print(json.dumps(progress["history"][-1]), flush=True)
    else:
        progress["status"] = "completed"
    train_finished = time.perf_counter() - STARTED
    assert frozen_digest(model) == base_hash
    save_checkpoint(OUT / "checkpoints/last.pt", model, optimizer, scheduler, progress, prov)
    write_json(OUT / "progress.json", progress)
    if progress["status"] != "completed":
        return dict(
            status=progress["status"],
            updates=progress["completed_updates"],
            training_finished_seconds=train_finished,
            comparison_complete=False,
        )
    assert progress["total_updates"] == 404 and progress["article_presentations"] == 491
    assert [g["lr"] for g in optimizer.param_groups] == [
        g["lr"] for g in control["optimizer"]["param_groups"]
    ]
    prediction = predict(model, docs["dev"], encoded, caches["dev"])
    write_jsonl(OUT / "negative-dev-scores.jsonl", prediction)
    final_lrs = [g["lr"] for g in optimizer.param_groups]
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    configure(42)
    restored = SpanClassifier(packed["encoder"], cfg).cuda()
    saved = torch.load(OUT / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    weights(restored, saved["trainable"])
    repeated = predict(restored, docs["dev"], encoded, caches["dev"])
    assert prediction == repeated
    assert all(k in saved["rng"] for k in ["python", "numpy", "torch", "cuda"])
    torch.cuda.synchronize()
    return dict(
        status="completed",
        comparison_complete=True,
        updates=64,
        total_updates=404,
        article_presentations=491,
        training_finished_seconds=train_finished,
        starting_lrs=lrs,
        final_lrs=final_lrs,
        frozen_base_unchanged=True,
        reloaded_predictions_exact=True,
        rng_and_progress_saved=True,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        test_used=False,
    )


if __name__ == "__main__":
    try:
        result = main()
    except Exception as exc:
        result = dict(
            status="failed",
            comparison_complete=False,
            error=repr(exc),
            traceback=traceback.format_exc(),
            test_used=False,
        )
        print(result["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result["elapsed_seconds"] = time.perf_counter() - STARTED
    result["budget_seconds"] = 180
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "execution.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)

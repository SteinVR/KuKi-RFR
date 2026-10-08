"""Only residual weights train; the original model is absent from this process."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import argparse
import copy
import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.evaluation import save_predictions, score
from research.kuki.l2 import load_checkpoint, save_checkpoint, trainable_state
from research.kuki.train_l2 import epoch_order
from research.kuki.train_l3 import configure

C = importlib.import_module("research.stages.ST4-l3-to-l2.frozen.common")


def profile(docs, features, prov, cfg):
    checks = {}
    out = C.ROOT / "technical/checkpoints"
    out.mkdir(exist_ok=True, parents=True)
    order = epoch_order(len(docs["train"]), 42, 0)
    batch = [docs["train"][i] for i in order[:8]]
    for condition in ["control", "families"]:
        configure(42)
        model, opt, sched = C.model_optimizer(condition, cfg)
        before = trainable_state(model)
        assert np.array_equal(C.predict(model, batch, features["train"]),
                              C.baseline(batch, features["train"]))
        C.update(model, opt, sched, batch, features["train"])
        save_checkpoint(out / f"{condition}.pt", model, opt, sched, dict(updates=1), prov)
        C.update(model, opt, sched, batch, features["train"])
        wanted = trainable_state(model)
        state = copy.deepcopy(opt.state_dict())
        schedule = copy.deepcopy(sched.state_dict())
        load_checkpoint(out / f"{condition}.pt", model, opt, sched, prov)
        C.update(model, opt, sched, batch, features["train"])
        assert all(torch.equal(v, wanted[k]) for k, v in trainable_state(model).items())
        assert sched.state_dict() == schedule
        for k, value in opt.state_dict()["state"].items():
            for name, tensor in value.items():
                assert torch.equal(tensor, state["state"][k][name])
        changed = {k: not torch.equal(v, before[k]) for k, v in wanted.items()}
        # Key bias cancels in softmax and may have only rounding-level gradients.
        assert changed["output.weight"] and changed["query.weight"] and changed["key.weight"]
        assert all(not v.requires_grad and v.grad is None for part in features.values()
                   for item in part.values() for v in item.values())
        checks[condition] = dict(zero_exact=True, resume_exact=True, changed=changed)
    write_json(C.ROOT / "profile.json", dict(status="passed", checks=checks))
    return dict(status="passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=["profile", "control", "families"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = dict(C.config(), condition=args.condition)
    docs, _, prov = C.inputs()
    prov["condition"] = args.condition
    features = C.cache()
    configure(42)
    if args.condition == "profile":
        return args.condition, profile(docs, features, prov, cfg)
    cfg["run_id"] = f"20260929-st4-frozen-{args.condition}-s42-01"
    out = C.ROOT / "experiments" / cfg["run_id"]
    if not args.resume:
        C.initialize(out, cfg, prov)
    model, opt, sched = C.model_optimizer(args.condition, cfg)
    progress = dict(epoch=0, next_position=0, updates=0, articles_seen=0, history=[],
                    best_epoch=0, best_brier=None, active_seconds=0, losses=[], status="running")
    if args.resume:
        progress = load_checkpoint(out / "checkpoints/last.pt", model, opt, sched, prov)
        assert progress["status"] == "paused"
    previous = progress["active_seconds"]
    available = min(cfg["run_budget_seconds"] - previous, C.remaining())

    def save(name):
        progress["active_seconds"] = previous + time.perf_counter() - STARTED
        save_checkpoint(out / "checkpoints" / name, model, opt, sched, progress, prov)

    if not args.resume:
        p = C.predict(model, docs["dev"], features["dev"])
        assert np.array_equal(p, C.baseline(docs["dev"], features["dev"]))
        progress["best_brier"] = score(docs["dev"], p)["brier"]
        write_json(out / "initialization.json", dict(state_sha256=C.M.OLD.state_digest(
            trainable_state(model)), dev=score(docs["dev"], p)))
        save("start.pt")
        save("best.pt")
    while progress["epoch"] < cfg["max_epochs"]:
        order = epoch_order(len(docs["train"]), 42, progress["epoch"])
        progress["article_order"] = order
        while progress["next_position"] < len(order):
            if time.perf_counter() - STARTED > available - 20:
                progress["status"] = "paused"
                save("last.pt")
                write_json(out / "progress.json", progress)
                return args.condition, dict(status="paused", updates=progress["updates"])
            pos = progress["next_position"]
            batch = [docs["train"][i] for i in order[pos:pos + 8]]
            loss = C.update(model, opt, sched, batch, features["train"])
            progress["losses"].append([loss, len(batch)])
            progress["updates"] += 1
            progress["articles_seen"] += len(batch)
            progress["next_position"] += len(batch)
            save("last.pt")
        metrics = score(docs["dev"], C.predict(model, docs["dev"], features["dev"]))
        progress["epoch"] += 1
        progress["history"].append(dict(epoch=progress["epoch"], updates=progress["updates"],
            dev=metrics, train_loss=sum(v * n for v, n in progress["losses"]) / len(docs["train"])))
        progress["next_position"] = 0
        progress["losses"] = []
        if metrics["brier"] < progress["best_brier"]:
            progress["best_brier"] = metrics["brier"]
            progress["best_epoch"] = progress["epoch"]
            save("best.pt")
        save("last.pt")
        write_json(out / "progress.json", progress)
        print(dict(condition=args.condition, epoch=progress["epoch"],
                   brier=metrics["brier"], best=progress["best_epoch"]), flush=True)
    assert progress["updates"] == 570 and progress["articles_seen"] == 4350
    progress["status"] = "completed"
    save("last.pt")
    for state in ["best", "last"]:
        payload = torch.load(out / "checkpoints" / f"{state}.pt", weights_only=False)
        model.load_state_dict(payload["trainable"])
        metrics = {}
        for split, part in docs.items():
            pred = C.predict(model, part, features[split])
            metrics[split] = score(part, pred)
            save_predictions(out / state / f"{split}-l2.jsonl", part, pred, split)
        write_json(out / state / "metrics.json", metrics)
    write_json(out / "progress.json", progress)
    return args.condition, dict(status="completed", best_epoch=progress["best_epoch"],
        best_brier=progress["best_brier"], updates=progress["updates"],
        articles_seen=progress["articles_seen"],
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


if __name__ == "__main__":
    kind, result = "fit", dict(status="failed")
    try:
        kind, result = main()
    finally:
        C.record(kind, time.perf_counter() - STARTED, result)

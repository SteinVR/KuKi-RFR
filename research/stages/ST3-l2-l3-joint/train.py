"""Four predeclared ST3 conditions with per-update restartable state and a shared ledger."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import argparse
import importlib
import json
from pathlib import Path

import torch

from research.kuki.data import write_json, write_jsonl
from research.kuki.experiment import initialize_run
from research.kuki.joint import JointModel
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.span_classifier import examples
from research.kuki.train_l2 import epoch_order, make_optimizer
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", required=True, choices=["l2-only", "l3-only", "joint025", "joint100"]
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = M.read(ROOT / "configs" / f"{args.condition}.json")
    ledger = M.read(ROOT / "budget.json")
    assert ledger["seconds"] + cfg["run_budget_seconds"] <= ledger["limit_seconds"]
    docs, enc, prov = M.inputs(cfg)
    out = ROOT / "experiments" / cfg["run_id"]
    if not args.resume:
        initialize_run(out, cfg, prov)
    else:
        assert M.read(out / "metadata.json")["provenance"] == prov
    configure(cfg["seed"])
    encoded = encode_documents(docs["train"] + docs["dev"], enc, cfg)
    model = JointModel(enc, cfg).cuda()
    initial = M.initial_hashes(model)
    frozen = frozen_digest(model)
    optimizer, scheduler = make_optimizer(model, cfg)
    progress = dict(
        epoch=0,
        next_position=0,
        article_order=[],
        updates=0,
        articles_seen=0,
        l3_articles_seen=0,
        history=[],
        best_key=None,
        best_epoch=None,
        epoch_losses=[],
        active_seconds=0.0,
        status="running",
    )
    if args.resume:
        progress = load_checkpoint(out / "checkpoints/last.pt", model, optimizer, scheduler, prov)
        assert progress["status"] == "paused"
    previous = progress["active_seconds"]
    torch.cuda.reset_peak_memory_stats()
    rows = {s: examples(part) for s, part in docs.items()}

    def save(name):
        progress["active_seconds"] = previous + time.perf_counter() - STARTED
        save_checkpoint(out / "checkpoints" / name, model, optimizer, scheduler, progress, prov)

    if not args.resume:
        save("start.pt")
        write_json(out / "initialization.json", initial)
    while progress["epoch"] < cfg["max_epochs"]:
        if not progress["article_order"]:
            progress["article_order"] = epoch_order(
                len(docs["train"]), cfg["seed"], progress["epoch"]
            )
        order = progress["article_order"]
        while progress["next_position"] < len(order):
            if previous + time.perf_counter() - STARTED > cfg["run_budget_seconds"] - 80:
                progress["status"] = "paused"
                save("last.pt")
                write_json(out / "progress.json", progress)
                return dict(status="paused", updates=progress["updates"], run_id=cfg["run_id"])
            pos = progress["next_position"]
            batch = [docs["train"][i] for i in order[pos : pos + 8]]
            detail = M.update(
                model, optimizer, scheduler, batch, encoded, cfg["gradient_clip_norm"]
            )
            progress["epoch_losses"].append(dict(**detail, n=len(batch)))
            progress["updates"] += 1
            progress["articles_seen"] += len(batch)
            progress["l3_articles_seen"] += (
                sum(any(a["l3_persuasion"] for a in d["annotations"]) for d in batch)
                if cfg["l3_weight"]
                else 0
            )
            progress["next_position"] += len(batch)
            save("last.pt")
        pred = M.predict(model, docs["dev"], encoded, rows["dev"])
        metrics = M.epoch_metrics(docs["dev"], rows["dev"], pred)
        key = M.selection_key(metrics)
        improved = progress["best_key"] is None or tuple(key) > tuple(progress["best_key"])
        progress["epoch"] += 1
        history = dict(
            epoch=progress["epoch"],
            updates=progress["updates"],
            articles_seen=progress["articles_seen"],
            train_losses={
                k: sum(x[k] * x["n"] for x in progress["epoch_losses"]) / len(docs["train"])
                for k in ["l2", "l3"]
            },
            dev=metrics,
            lrs=[g["lr"] for g in optimizer.param_groups],
            elapsed_seconds=previous + time.perf_counter() - STARTED,
        )
        progress["history"].append(history)
        progress["article_order"] = []
        progress["next_position"] = 0
        progress["epoch_losses"] = []
        if improved:
            progress["best_key"] = key
            progress["best_epoch"] = progress["epoch"]
            save("best.pt")
        save("last.pt")
        write_json(out / "progress.json", progress)
        write_jsonl(out / "logs/epochs.jsonl", progress["history"])
        print(
            json.dumps(
                dict(
                    condition=args.condition,
                    epoch=progress["epoch"],
                    best=progress["best_epoch"],
                    brier=metrics["l2"]["brier"] if metrics["l2"] else None,
                    conditional_f1=metrics["conditional"]["article_mean_f1"]
                    if metrics["conditional"]
                    else None,
                    elapsed=history["elapsed_seconds"],
                )
            ),
            flush=True,
        )
    assert progress["updates"] == 380 and progress["articles_seen"] == 2900
    assert frozen_digest(model) == frozen
    progress["status"] = "completed"
    save("last.pt")
    candidates = {
        s: {r["doc_id"]: r for r in M.jsonl(M.DETECTOR / f"{s}-predictions.jsonl")} for s in docs
    }
    for state, splits in [("last", ["dev"]), ("best", ["dev", "train"])]:
        payload = torch.load(
            out / "checkpoints" / f"{state}.pt", map_location="cpu", weights_only=False
        )
        M.set_weights(model, payload["trainable"])
        for split in splits:
            pred = M.predict(
                model,
                docs[split],
                encoded,
                rows[split],
                candidates[split] if cfg["l3_weight"] else None,
            )
            M.persist_predictions(out / state, split, docs[split], rows[split], pred)
    write_json(out / "progress.json", progress)
    params = trainable_state(model)
    return dict(
        status="completed",
        run_id=cfg["run_id"],
        updates=progress["updates"],
        article_presentations=2900,
        l3_article_presentations=progress["l3_articles_seen"],
        best_epoch=progress["best_epoch"],
        base_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        trainable_parameters=sum(p.numel() for p in params.values()),
    )


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
        ledger["runs"].append(dict(kind="fit", seconds=seconds, **result))
        write_json(ROOT / "budget.json", ledger)
        print(json.dumps(dict(seconds=seconds, **result)), flush=True)

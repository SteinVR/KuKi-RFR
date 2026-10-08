"""Matched ST4 continuations; immutable run provenance and exact update checkpoints."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import argparse
import importlib

import torch

from research.kuki.data import write_json, write_jsonl
from research.kuki.evaluation import save_predictions
from research.kuki.experiment import initialize_run
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.message import MODES
from research.kuki.span_classifier import examples
from research.kuki.train_l2 import epoch_order, make_optimizer
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=MODES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = M.read(M.ROOT / "configs" / f"{args.condition}.json")
    docs, enc, prov = M.inputs(cfg)
    out = M.ROOT / "experiments" / cfg["run_id"]
    if not args.resume:
        initialize_run(out, cfg, prov)
    configure(42)
    encoded = encode_documents(docs["train"] + docs["dev"], enc, cfg)
    cc = M.candidates(docs)
    for split in docs:
        for doc in docs[split]:
            encoded[doc["doc_id"]]["candidates"] = cc[split][doc["doc_id"]]["pairs"]
    model = M.model_from_parent(enc, cfg)
    frozen = frozen_digest(model)
    opt, sched = make_optimizer(model, cfg)
    progress = dict(
        epoch=0,
        next_position=0,
        updates=0,
        articles_seen=0,
        history=[],
        best_brier=None,
        best_epoch=0,
        active_seconds=0,
        status="running",
        epoch_losses=[],
    )
    if args.resume:
        progress = load_checkpoint(out / "checkpoints/last.pt", model, opt, sched, prov)
        assert progress["status"] == "paused"
    previous = progress["active_seconds"]
    available = min(cfg["run_budget_seconds"] - previous, M.remaining())
    rows = {s: examples(part) for s, part in docs.items()}

    def save(name):
        progress["active_seconds"] = previous + time.perf_counter() - STARTED
        save_checkpoint(out / "checkpoints" / name, model, opt, sched, progress, prov)

    if not args.resume:
        pred = M.predict(model, docs["dev"], encoded, rows["dev"], cc["dev"], full=False)
        metrics = M.OLD.epoch_metrics(docs["dev"], rows["dev"], pred)
        progress["best_brier"] = metrics["l2"]["brier"]
        write_json(
            out / "initialization.json",
            dict(
                parent_hash=prov["parent_checkpoint_sha256"],
                initial_metrics=metrics,
                hashes=M.OLD.initial_hashes(model),
            ),
        )
        save("start.pt")
        save("best.pt")
    torch.cuda.reset_peak_memory_stats()
    while progress["epoch"] < cfg["max_epochs"]:
        order = epoch_order(len(docs["train"]), 42, progress["epoch"] + 8)
        progress["article_order"] = order
        while progress["next_position"] < len(order):
            if time.perf_counter() - STARTED > available - 40:
                progress["status"] = "paused"
                save("last.pt")
                write_json(out / "progress.json", progress)
                return dict(status="paused", condition=args.condition, updates=progress["updates"])
            p = progress["next_position"]
            batch = [docs["train"][i] for i in order[p : p + 8]]
            detail = M.OLD.update(model, opt, sched, batch, encoded)
            progress["epoch_losses"].append(dict(**detail, n=len(batch)))
            progress["updates"] += 1
            progress["articles_seen"] += len(batch)
            progress["next_position"] += len(batch)
            save("last.pt")
        pred = M.predict(model, docs["dev"], encoded, rows["dev"], cc["dev"], full=False)
        metrics = M.OLD.epoch_metrics(docs["dev"], rows["dev"], pred)
        progress["epoch"] += 1
        progress["history"].append(
            dict(
                epoch=progress["epoch"],
                updates=progress["updates"],
                dev=metrics,
                elapsed_seconds=previous + time.perf_counter() - STARTED,
                train_losses={
                    k: sum(x[k] * x["n"] for x in progress["epoch_losses"]) / len(docs["train"])
                    for k in ["l2", "l3"]
                },
                lrs=[g["lr"] for g in opt.param_groups],
            )
        )
        progress["next_position"] = 0
        progress["epoch_losses"] = []
        if metrics["l2"]["brier"] < progress["best_brier"]:
            progress["best_brier"] = metrics["l2"]["brier"]
            progress["best_epoch"] = progress["epoch"]
            save("best.pt")
        save("last.pt")
        write_json(out / "progress.json", progress)
        write_jsonl(out / "logs/epochs.jsonl", progress["history"])
        print(
            dict(
                condition=args.condition,
                epoch=progress["epoch"],
                best=progress["best_epoch"],
                brier=metrics["l2"]["brier"],
                seconds=time.perf_counter() - STARTED,
            ),
            flush=True,
        )
    assert progress["updates"] == 152 and progress["articles_seen"] == 1160
    assert frozen_digest(model) == frozen
    progress["status"] = "completed"
    save("last.pt")
    for state, splits in [("last", ["dev"]), ("best", ["train", "dev"])]:
        payload = torch.load(
            out / "checkpoints" / f"{state}.pt", map_location="cpu", weights_only=False
        )
        M.OLD.set_weights(model, payload["trainable"])
        for split in splits:
            pred = M.predict(model, docs[split], encoded, rows[split], cc[split])
            M.OLD.persist_predictions(out / state, split, docs[split], rows[split], pred)
            save_predictions(
                out / state / f"{split}-base-l2.jsonl", docs[split], pred["base"], split
            )
    write_json(out / "progress.json", progress)
    result = dict(
        status="completed",
        condition=args.condition,
        updates=152,
        articles_seen=1160,
        best_epoch=progress["best_epoch"],
        base_unchanged=True,
        trainable_parameters=sum(p.numel() for p in trainable_state(model).values()),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    write_json(out / "resources.json", result)
    return result


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        seconds = time.perf_counter() - STARTED
        M.ledger_add("fit", seconds, result)
        print(dict(seconds=seconds, **result), flush=True)

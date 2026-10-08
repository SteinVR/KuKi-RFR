"""Group-excluded proposal generation with fixed epochs and restartable updates."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()
import argparse
import importlib
import math

import torch

from research.kuki.data import write_json, write_jsonl
from research.kuki.experiment import initialize_run
from research.kuki.global_pointer import GlobalPointer
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint
from research.kuki.train_l2 import epoch_order, make_optimizer
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
MOD = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.train")


def update(model, opt, sched, batch, encoded):
    model.train()
    opt.zero_grad(set_to_none=True)
    for doc in batch:
        loss, _ = model.training_loss(encoded[doc["doc_id"]], doc["annotations"])
        assert torch.isfinite(loss)
        (loss / len(batch)).backward()
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    norm = torch.nn.utils.clip_grad_norm_(params, 1)
    assert torch.isfinite(norm)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(3))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = M.read(M.OLD.DETECTOR / "config.json")["config"]
    docs, enc, prov = M.inputs(cfg, need_oof=False)
    membership = M.folds(docs["train"])
    train = [d for d in docs["train"] if membership[d["doc_id"]] != args.fold]
    held = [d for d in docs["train"] if membership[d["doc_id"]] == args.fold]
    assert not {d["group_id"] for d in train} & {d["group_id"] for d in held}
    total = math.ceil(len(train) / 8) * 13
    cfg.update(
        max_epochs=13,
        max_updates=total,
        warmup_updates=math.ceil(total / 10),
        run_budget_seconds=500,
        run_id=f"20260929-oof{args.fold}-s42-01",
    )
    prov["config"] = cfg
    prov["oof"] = dict(
        fold=args.fold,
        train_ids=[d["doc_id"] for d in train],
        prediction_ids=[d["doc_id"] for d in held],
        membership=membership,
    )
    prov["source_initialization"] = "fresh pinned base, no ST3/E3 learned parameters"
    out = M.ROOT / "oof" / cfg["run_id"]
    if not args.resume:
        initialize_run(out, cfg, prov)
    configure(42)
    encoded = encode_documents(train + held, enc, cfg)
    model = GlobalPointer(enc, cfg).cuda()
    frozen = frozen_digest(model)
    opt, sched = make_optimizer(model, cfg)
    progress = dict(
        epoch=0, next_position=0, updates=0, articles_seen=0, status="running", active_seconds=0
    )
    if args.resume:
        progress = load_checkpoint(out / "checkpoints/last.pt", model, opt, sched, prov)
        assert progress["status"] == "paused"
    previous = progress["active_seconds"]
    available = min(cfg["run_budget_seconds"] - previous, M.remaining())

    def save():
        progress["active_seconds"] = previous + time.perf_counter() - STARTED
        save_checkpoint(out / "checkpoints/last.pt", model, opt, sched, progress, prov)
        write_json(out / "progress.json", progress)

    torch.cuda.reset_peak_memory_stats()
    while progress["epoch"] < 13:
        order = epoch_order(len(train), 42, progress["epoch"])
        progress["article_order"] = order
        while progress["next_position"] < len(train):
            if time.perf_counter() - STARTED > available - 35:
                progress["status"] = "paused"
                save()
                return dict(status="paused", fold=args.fold, updates=progress["updates"])
            p = progress["next_position"]
            batch = [train[i] for i in order[p : p + 8]]
            update(model, opt, sched, batch, encoded)
            progress["updates"] += 1
            progress["articles_seen"] += len(batch)
            progress["next_position"] += len(batch)
            save()
        progress["epoch"] += 1
        progress["next_position"] = 0
        save()
        print(
            dict(fold=args.fold, epoch=progress["epoch"], seconds=time.perf_counter() - STARTED),
            flush=True,
        )
    assert frozen_digest(model) == frozen
    assert progress["updates"] == total
    write_jsonl(out / "predictions.jsonl", MOD.detector_predict(model, held, encoded))
    progress["status"] = "completed"
    save()
    return dict(
        status="completed",
        fold=args.fold,
        updates=total,
        articles_seen=progress["articles_seen"],
        base_unchanged=True,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        train_ids=prov["oof"]["train_ids"],
        prediction_ids=prov["oof"]["prediction_ids"],
    )


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        seconds = time.perf_counter() - STARTED
        M.ledger_add("oof", seconds, result)
        print(dict(seconds=seconds, **result), flush=True)

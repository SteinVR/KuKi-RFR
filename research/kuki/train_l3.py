"""ST2 trainer with explicit proposal coverage, negative balance, and full restart state."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import write_json, write_jsonl
from .experiment import initialize_run, load_development, provenance
from .l2 import frozen_digest, load_checkpoint, save_checkpoint, seed_all, trainable_state
from .l3 import L3Model, prepare_l3
from .l3_evaluation import evaluate, select_spans
from .train_l2 import epoch_order, make_optimizer
from .windows import tokenize_document


def configure(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(seed)


def encode_documents(docs, encoder, cfg):
    tok = AutoTokenizer.from_pretrained(encoder["model_id"], revision=encoder["revision"])
    return {
        d["doc_id"]: prepare_l3(
            tokenize_document(tok, d["text"], cfg["window_length"], cfg["overlap"]), d["text"]
        )
        for d in docs
    }


def update(model, optimizer, scheduler, docs, encoded, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    diagnostics = []
    for d in docs:
        loss, detail = model.training_loss(encoded[d["doc_id"]], d["annotations"])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss")
        (loss / len(docs)).backward()
        diagnostics.append(detail)
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise ValueError("Missing or non-finite gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip_norm"])
    if not torch.isfinite(norm):
        raise ValueError("Non-finite gradient norm")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return diagnostics


def predict(model, docs, encoded):
    model.eval()
    with torch.inference_mode():
        return [
            dict(doc_id=d["doc_id"], group_id=d["group_id"], **model.infer(encoded[d["doc_id"]]))
            for d in docs
        ]


def choose_threshold(docs, outputs, grid):
    choices = [evaluate(docs, outputs, threshold)[0] for threshold in grid]
    best = max(choices, key=lambda m: (m["exact_f1"], m["overlap_f1"], m["threshold"]))
    return best, choices


def selection_key(metrics):
    return metrics["exact_f1"], metrics["overlap_f1"], metrics["threshold"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    begin = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads(Path(cfg["data_config"]).read_text())
    encoder = json.loads(Path(cfg["encoder_config"]).read_text())
    encoder["gradient_checkpointing"] = cfg["gradient_checkpointing"]
    docs, hashes = load_development(data_cfg)
    prov = provenance(cfg, data_cfg, hashes, encoder)
    if (
        cfg["max_updates"]
        != int(np.ceil(len(docs["train"]) / cfg["gradient_accumulation_articles"]))
        * cfg["max_epochs"]
    ):
        raise ValueError("Update and epoch budgets disagree")
    if not args.resume:
        initialize_run(args.output, cfg, prov)
    elif json.loads((args.output / "metadata.json").read_text())["provenance"] != prov:
        raise ValueError("Resume provenance changed")
    configure(cfg["seed"])
    encoded = encode_documents(docs["train"] + docs["dev"], encoder, cfg)
    model = L3Model(encoder, cfg).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    progress = dict(
        epoch=0,
        next_position=0,
        order=[],
        updates=0,
        articles_seen=0,
        best=None,
        best_epoch=None,
        patience_anchor=None,
        bad_epochs=0,
        history=[],
        epoch_diagnostics=[],
        active_seconds=0.0,
        status="running",
    )
    checkpoints = args.output / "checkpoints"
    if args.resume:
        progress = load_checkpoint(checkpoints / "last.pt", model, optimizer, scheduler, prov)
        if progress["status"] == "completed":
            raise ValueError("Already completed")
    previous = progress["active_seconds"]
    base_before = frozen_digest(model)
    initial = trainable_state(model)
    torch.cuda.reset_peak_memory_stats()
    stop = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save(name):
        progress["active_seconds"] = previous + time.perf_counter() - begin
        save_checkpoint(checkpoints / name, model, optimizer, scheduler, progress, prov)

    while progress["epoch"] < cfg["max_epochs"]:
        if not progress["order"]:
            progress["order"] = epoch_order(len(docs["train"]), cfg["seed"], progress["epoch"])
        while progress["next_position"] < len(progress["order"]):
            if (
                stop
                or time.perf_counter() - begin > cfg["session_save_stop_minutes"] * 60
                or (
                    args.stop_after_updates is not None
                    and progress["updates"] >= args.stop_after_updates
                )
            ):
                progress["status"] = "paused"
                save("last.pt")
                write_json(args.output / "progress.json", progress)
                print(json.dumps(dict(status="paused", updates=progress["updates"])), flush=True)
                return
            p = progress["next_position"]
            indices = progress["order"][p : p + cfg["gradient_accumulation_articles"]]
            torch.cuda.synchronize()
            started = time.perf_counter()
            details = update(
                model, optimizer, scheduler, [docs["train"][i] for i in indices], encoded, cfg
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            for i, detail in zip(indices, details, strict=True):
                detail.update(
                    doc_id=docs["train"][i]["doc_id"],
                    update_seconds_per_article=seconds / len(indices),
                )
            progress["epoch_diagnostics"].extend(details)
            progress["next_position"] += len(indices)
            progress["updates"] += 1
            progress["articles_seen"] += len(indices)
            progress["status"] = "running"
            save("last.pt")
        outputs = predict(model, docs["dev"], encoded)
        selected, grid = choose_threshold(docs["dev"], outputs, cfg["threshold_grid"])
        improved = progress["best"] is None or selection_key(selected) > selection_key(
            progress["best"]
        )
        if improved:
            progress["best"] = selected
            progress["best_epoch"] = progress["epoch"] + 1
        anchor = progress["patience_anchor"]
        if anchor is None or selected["exact_f1"] > anchor + cfg["early_stopping_min_delta"]:
            progress["patience_anchor"] = selected["exact_f1"]
            progress["bad_epochs"] = 0
        else:
            progress["bad_epochs"] += 1
        diagnostics = progress["epoch_diagnostics"]
        epoch_summary = dict(
            epoch=progress["epoch"] + 1,
            updates=progress["updates"],
            articles_seen=progress["articles_seen"],
            dev=selected,
            thresholds=grid,
            train={
                key: float(np.mean([d[key] for d in diagnostics]))
                for key in ["loss", "boundary_loss", "pair_loss", "n_candidates"]
            },
            train_seconds=sum(d["update_seconds_per_article"] for d in diagnostics),
        )
        write_jsonl(
            args.output / "logs" / f"epoch-{progress['epoch'] + 1:02d}-training.jsonl", diagnostics
        )
        progress["history"].append(epoch_summary)
        progress["epoch"] += 1
        progress["order"] = []
        progress["next_position"] = 0
        progress["epoch_diagnostics"] = []
        if improved:
            save("best.pt")
        save("last.pt")
        write_json(args.output / "progress.json", progress)
        write_jsonl(args.output / "logs" / "epochs.jsonl", progress["history"])
        print(json.dumps(epoch_summary), flush=True)
        if progress["bad_epochs"] >= cfg["early_stopping_patience_epochs"]:
            break
    progress["status"] = "completed"
    save("last.pt")
    assert frozen_digest(model) == base_before
    changed = sum(not torch.equal(v, initial[k]) for k, v in trainable_state(model).items())
    best = load_checkpoint(checkpoints / "best.pt", model, optimizer, scheduler, prov)
    metrics = dict(
        status="completed",
        selected_epoch=best["best_epoch"],
        threshold=best["best"]["threshold"],
        total_updates=progress["updates"],
        articles_seen=progress["articles_seen"],
        history=progress["history"],
        base_unchanged=True,
        trainable_tensors_changed_since_session_start=changed,
        trainable_tensors=len(initial),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        stop_reason="patience"
        if progress["bad_epochs"] >= cfg["early_stopping_patience_epochs"]
        else "epoch_limit",
    )
    for split, part in docs.items():
        outputs = predict(model, part, encoded)
        metrics[split], records = evaluate(part, outputs, metrics["threshold"])
        write_jsonl(args.output / f"{split}-raw-predictions.jsonl", outputs)
        write_jsonl(
            args.output / f"{split}-predictions.jsonl",
            [
                dict(
                    doc_id=d["doc_id"],
                    split=split,
                    group_id=d["group_id"],
                    spans=select_spans(o, metrics["threshold"]),
                )
                for d, o in zip(part, outputs, strict=True)
            ],
        )
        write_jsonl(args.output / f"{split}-article-metrics.jsonl", records)
    metrics["elapsed_seconds"] = previous + time.perf_counter() - begin
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "progress.json", progress)
    print(
        json.dumps(
            dict(
                status="completed",
                epoch=metrics["selected_epoch"],
                dev=metrics["dev"],
                seconds=metrics["elapsed_seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

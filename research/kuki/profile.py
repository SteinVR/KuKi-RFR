"""Bounded technical checks on training articles only; no quality evaluation."""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import digest, write_json
from .l2 import (
    L2Model,
    article_loss,
    frozen_digest,
    load_checkpoint,
    save_checkpoint,
    seed_all,
    trainable_state,
)
from .windows import tokenize_document


def optimizer_for(model, cfg):
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("head.")
                ],
                "lr": cfg["adapter_lr"],
            },
            {"params": model.head.parameters(), "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / 100))
    return optimizer, scheduler


def update(model, optimizer, scheduler, batches):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for encoded, answers in batches:
        loss = article_loss(model(encoded), answers)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss")
        (loss / len(batches)).backward()
        losses.append(float(loss.detach()))
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise ValueError("Missing or non-finite trainable gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(norm):
        raise ValueError("Non-finite gradient norm")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    model_cfg = json.loads(Path(cfg["encoder_config"]).read_text())
    split_path = Path(cfg["split"])
    split = json.loads(split_path.read_text())
    data_path = Path(split["prepared_path"])
    assert digest(data_path) == split["prepared_sha256"]
    if cfg.get("deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(cfg["seed"])
    train_ids = {r["doc_id"] for r in split["rows"] if r["split"] == "train"}
    docs = [json.loads(s) for s in data_path.read_text().splitlines() if s.strip()]
    docs = [d for d in docs if d["doc_id"] in train_ids]
    tok = AutoTokenizer.from_pretrained(model_cfg["model_id"], revision=model_cfg["revision"])
    lengths = {
        d["doc_id"]: len(tok(d["text"], add_special_tokens=False)["input_ids"]) for d in docs
    }
    docs.sort(key=lambda d: (lengths[d["doc_id"]], d["doc_id"]))
    probe = [docs[len(docs) // 2], docs[-1]]
    provenance = dict(
        config=cfg,
        encoder=model_cfg,
        split_sha256=digest(split_path),
        prepared_sha256=digest(data_path),
        uv_lock_sha256=digest(Path(cfg.get("environment_lock", "uv.lock"))),
    )
    write_json(args.output / "config.json", dict(**cfg, encoder=model_cfg))
    write_json(
        args.output / "metadata.json",
        dict(
            provenance=provenance,
            python=platform.python_version(),
            packages={
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "peft", "accelerate", "numpy"]
            },
            gpu=torch.cuda.get_device_name(),
            device_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            cuda=torch.version.cuda,
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            code_sha256={str(p): digest(p) for p in sorted(Path("research/kuki").glob("*.py"))},
        ),
    )
    model = L2Model(model_cfg).cuda()
    initial = trainable_state(model)
    base_before = frozen_digest(model)
    optimizer, scheduler = optimizer_for(model, cfg)
    target_names = [n for n, p in model.named_parameters() if "lora_" in n]
    assert len(target_names) == 22 * 2 * 2 and all(".attn." in n for n in target_names)
    metrics = dict(
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        all_parameters=sum(p.numel() for p in model.parameters()),
        lora_parameter_names=target_names,
        profiles=[],
    )
    for length in model_cfg["window_lengths"]:
        for doc in probe:
            enc = tokenize_document(tok, doc["text"], length, model_cfg["overlap"])
            batch = (enc, [a["l2"] for a in doc["annotations"]])
            torch.cuda.reset_peak_memory_stats()
            timings, losses = [], []
            for _ in range(cfg["updates_per_case"]):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                losses.append(update(model, optimizer, scheduler, [batch]))
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - begin)
            model.eval()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.no_grad():
                pred = model(enc)
                assert torch.isfinite(pred).all()
            torch.cuda.synchronize()
            metrics["profiles"].append(
                dict(
                    window=length,
                    doc_id=doc["doc_id"],
                    tokens=enc["tokens"],
                    windows=len(enc["windows"]),
                    step_seconds=timings,
                    warm_step_seconds=float(np.mean(timings[1:])),
                    eval_seconds=time.perf_counter() - begin,
                    losses=losses,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            )
            write_json(args.output / "metrics.json", metrics)
    metrics["base_unchanged"] = base_before == frozen_digest(model)
    changed = {n: float((p - initial[n]).abs().max()) for n, p in trainable_state(model).items()}
    metrics["all_trainable_tensors_changed"] = all(v > 0 for v in changed.values())
    assert metrics["base_unchanged"] and metrics["all_trainable_tensors_changed"]
    # Resume test uses a fresh optimizer and two articles per update to test accumulation.
    encs = [
        (
            tokenize_document(tok, d["text"], 1024, model_cfg["overlap"]),
            [a["l2"] for a in d["annotations"]],
        )
        for d in probe
    ]
    optimizer, scheduler = optimizer_for(model, cfg)
    order = [1, 0, 0, 1, 1, 0]
    progress = dict(updates=1, article_order=order, next_position=2, epoch=0)
    update(model, optimizer, scheduler, [encs[i] for i in order[:2]])
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "resume-probe.pt"
    save_checkpoint(checkpoint, model, optimizer, scheduler, progress, provenance)
    continuous = [
        update(model, optimizer, scheduler, [encs[i] for i in order[k : k + 2]]) for k in [2, 4]
    ]
    expected = trainable_state(model)
    expected_lr = scheduler.get_last_lr()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model = L2Model(model_cfg).cuda()
    optimizer, scheduler = optimizer_for(model, cfg)
    restored = load_checkpoint(checkpoint, model, optimizer, scheduler, provenance)
    resumed = []
    for k in range(restored["next_position"], len(restored["article_order"]), 2):
        resumed.append(
            update(
                model, optimizer, scheduler, [encs[i] for i in restored["article_order"][k : k + 2]]
            )
        )
    difference = max(
        float((p - expected[n]).abs().max()) for n, p in trainable_state(model).items()
    )
    metrics["resume"] = dict(
        continuous_losses=continuous,
        resumed_losses=resumed,
        max_parameter_difference=difference,
        learning_rates_equal=expected_lr == scheduler.get_last_lr(),
        progress_restored=restored == progress,
        checkpoint_bytes=checkpoint.stat().st_size,
        passed=difference <= 1e-6 and np.allclose(continuous, resumed, atol=1e-6, rtol=0),
    )
    metrics["peak_process_ram_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    metrics["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output / "metrics.json", metrics)
    assert (
        metrics["resume"]["passed"]
        and metrics["resume"]["learning_rates_equal"]
        and metrics["resume"]["progress_restored"]
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "lora_parameter_names"}))


if __name__ == "__main__":
    main()

"""Compare environments on the same train-only L2 work, without quality selection."""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import digest, write_json
from .l2 import L2Model, article_loss, seed_all
from .windows import tokenize_document


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def step(model, optimizer, encoded, answers):
    optimizer.zero_grad(set_to_none=True)
    loss = article_loss(model(encoded), answers)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpointing", choices=["on", "off"], default="on")
    parser.add_argument("--compile", choices=["none", "encoder"], default="none")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(42)
    cfg = json.loads(Path("research/configs/encoder.json").read_text())
    cfg["attention"] = args.attention
    cfg["gradient_checkpointing"] = args.checkpointing == "on"
    cfg["deterministic_flash_attn"] = True
    manifest_path = Path(
        "research/stages/ST0-preparation/experiments/20260928-data-s42-02/split.json"
    )
    manifest = json.loads(manifest_path.read_text())
    train = {r["doc_id"] for r in manifest["rows"] if r["split"] == "train"}
    path = Path(manifest["prepared_path"])
    assert digest(path) == manifest["prepared_sha256"]
    docs = [json.loads(line) for line in path.read_text().splitlines()]
    docs = [d for d in docs if d["doc_id"] in train]
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    begin = time.perf_counter()
    lengths = {
        d["doc_id"]: len(tok(d["text"], add_special_tokens=False)["input_ids"]) for d in docs
    }
    tokenization_seconds = time.perf_counter() - begin
    docs.sort(key=lambda d: (lengths[d["doc_id"]], d["doc_id"]))
    selected = [docs[round((len(docs) - 1) * i / 7)] for i in range(8)]
    result = dict(
        python=platform.python_version(),
        executable=os.sys.executable,
        packages={
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "numpy",
                "tokenizers",
                "triton",
            ]
        },
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        arch_list=torch.cuda.get_arch_list(),
        deterministic=True,
        threads=4,
        checkpointing=args.checkpointing,
        compile=args.compile,
        attention=args.attention,
        model_config=cfg,
        split_sha256=digest(manifest_path),
        code_sha256={str(p): digest(p) for p in Path("research/kuki").glob("*.py")},
        articles=[{"doc_id": d["doc_id"], "tokens": lengths[d["doc_id"]]} for d in selected],
        tokenization_145_articles_seconds=tokenization_seconds,
        cases=[],
    )
    for length in args.lengths:
        seed_all(42)
        model = L2Model(cfg).cuda().train()
        if args.checkpointing == "off":
            model.encoder.gradient_checkpointing_disable()
        if args.compile == "encoder":
            model.encoder = torch.compile(model.encoder, dynamic=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        data = [
            (tokenize_document(tok, d["text"], length, 128), [a["l2"] for a in d["annotations"]])
            for d in selected
        ]
        assert all(
            e["tokens"] == lengths[d["doc_id"]] for (e, _), d in zip(data, selected, strict=True)
        )
        torch.cuda.reset_peak_memory_stats()
        before = gpu_status()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for encoded, answers in data:
            step(model, optimizer, encoded, answers)
        torch.cuda.synchronize()
        warmup = time.perf_counter() - begin
        times = []
        for _ in range(args.rounds):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            losses = [step(model, optimizer, e, a) for e, a in data]
            torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
            assert all(torch.isfinite(loss).item() for loss in losses)
        case = dict(
            window=length,
            warmup_seconds=warmup,
            round_seconds=times,
            median_seconds_per_article=statistics.median(times) / len(data),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_before=before,
            gpu_after=gpu_status(),
        )
        # Diagnose effective attention kernels separately from timed rounds.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            step(model, optimizer, *data[-1])
        case["attention_ops"] = [
            e.key
            for e in prof.key_averages()
            if any(s in e.key.lower() for s in ["attention", "flash", "fmha"])
        ]
        if args.compile != "none":
            case["compile_counters"] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}
        result["cases"].append(case)
        write_json(args.output, result)
        print(json.dumps(case), flush=True)
        del model, optimizer, data
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

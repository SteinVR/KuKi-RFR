"""ST5 shared data, provenance, predictions and resource accounting."""

import hashlib
import json
from pathlib import Path

import torch

from research.kuki.data import digest, write_json, write_jsonl
from research.kuki.experiment import load_development, provenance
from research.kuki.l1 import examples, score
from research.kuki.l2 import trainable_state

ROOT = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text())


def inputs(cfg):
    dc = read(cfg["data_config"])
    docs, hashes = load_development(dc)
    enc = read(cfg["encoder_config"])
    enc["gradient_checkpointing"] = cfg["gradient_checkpointing"]
    prov = provenance(cfg, dc, hashes, enc)
    prov["code_sha256"].update({str(p): digest(p) for p in sorted(ROOT.glob("*.py"))})
    prov["protocol_sha256"] = digest(ROOT / "protocol.md")
    prov["parent_checkpoint"] = None
    prov["adaptation_history"] = "fresh pinned base, fresh LoRA and L1 head; train only"
    return docs, enc, prov


def active(docs):
    return [d for d in docs if any(a["l1_roles"] for a in d["annotations"])]


def initialization(model):
    h = hashlib.sha256()
    for name, tensor in trainable_state(model).items():
        h.update(name.encode())
        h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def set_weights(model, state):
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def persist(out, split, docs, probabilities):
    rows = examples(docs)
    write_jsonl(out / f"{split}-predictions.jsonl", [
        dict(**{k: v for k, v in r.items() if k != "text"}, probabilities=p.tolist())
        for r, p in zip(rows, probabilities, strict=True)
    ])
    metrics = score(rows, probabilities)
    write_json(out / f"{split}-metrics.json", metrics)
    return metrics


def account(kind, started, result):
    import time

    seconds = time.perf_counter() - started
    ledger = read(ROOT / "budget.json")
    ledger["seconds"] += seconds
    ledger["runs"].append(dict(kind=kind, seconds=seconds, **result))
    write_json(ROOT / "budget.json", ledger)
    print(json.dumps(dict(seconds=seconds, **result)), flush=True)


def reserve(seconds):
    ledger = read(ROOT / "budget.json")
    if ledger["seconds"] + seconds > ledger["limit_seconds"]:
        raise RuntimeError("ST5 series budget exhausted")

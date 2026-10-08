"""Standalone frozen-ST3 series: provenance, matched updates and resource ledger."""

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.experiment import initialize_run
from research.kuki.frozen_message import FrozenResidual
from research.kuki.l2 import article_loss
from research.kuki.train_l2 import learning_rate_multiplier

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
ROOT = Path(__file__).parent
CACHE = Path("research/data/st4-frozen/features.pt")
read = M.read


def config():
    return read(ROOT / "config.json")


def inputs():
    cfg = config()
    docs, enc, prov = M.inputs(cfg)
    for p in ROOT.glob("*.py"):
        prov["code_sha256"][str(p)] = digest(p)
    prov["parent_protocol_sha256"] = prov["protocol_sha256"]
    prov["protocol_sha256"] = digest(ROOT / "protocol.md")
    if CACHE.exists():
        prov["cache_sha256"] = digest(CACHE)
    return docs, enc, prov


def initialize(out, cfg, prov):
    # Shared helper's flat snapshots are supplemental; preserve full paths here.
    initialize_run(out, cfg, prov)
    for source, expected in prov["code_sha256"].items():
        path = Path(source)
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        target = out / "code-tree" / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path, target)
        assert digest(target) == expected
    shutil.copy2(ROOT / "protocol.md", out / "protocol.md")


def cache(device="cuda"):
    saved = torch.load(CACHE, map_location="cpu", weights_only=False)
    return {s: {did: {k: v.to(device) for k, v in item.items()}
                for did, item in part.items()} for s, part in saved.items()}


def model_optimizer(condition, cfg):
    model = FrozenResidual(families=condition == "families").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["head_lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: learning_rate_multiplier(step, cfg["warmup_updates"], cfg["max_updates"])
    )
    return model, opt, sched


def update(model, opt, sched, docs, features):
    model.train()
    opt.zero_grad(set_to_none=True)
    losses = []
    for d in docs:
        loss = article_loss(model(features[d["doc_id"]]), [a["l2"] for a in d["annotations"]])
        assert torch.isfinite(loss)
        (loss / len(docs)).backward()
        losses.append(float(loss.detach()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def predict(model, docs, features):
    model.eval()
    with torch.no_grad():
        return np.asarray([model(features[d["doc_id"]]).sigmoid().cpu().numpy() for d in docs])


def baseline(docs, features):
    return np.asarray([features[d["doc_id"]]["base_logits"].sigmoid().cpu().numpy() for d in docs])


def remaining():
    ledger = read(ROOT / "budget.json")
    return ledger["limit_seconds"] - ledger["seconds"]


def record(kind, seconds, result):
    ledger = read(ROOT / "budget.json")
    ledger["seconds"] += seconds
    ledger["runs"].append(dict(kind=kind, seconds=seconds, **result))
    write_json(ROOT / "budget.json", ledger)
    print(json.dumps(dict(kind=kind, seconds=seconds, **result)), flush=True)

"""ST4 input lineage, OOF groups, prediction and a bounded GPU ledger."""

import importlib
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.message import MessageModel

OLD = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path(__file__).parent
PARENT = Path("research/stages/ST3-l2-l3-joint/search/experiments/20260929-joint200-s42-01")
read, jsonl = OLD.read, OLD.jsonl


def folds(docs):
    groups = sorted({d["group_id"] for d in docs})
    order = np.random.default_rng(42).permutation(len(groups))
    mapping = {groups[i]: j % 3 for j, i in enumerate(order)}
    return {d["doc_id"]: mapping[d["group_id"]] for d in docs}


def inputs(cfg, need_oof=True):
    docs, enc, prov = OLD.inputs(cfg)
    for p in ROOT.glob("*.py"):
        prov["code_sha256"][str(p)] = digest(p)
    # Historical module hashes belong to lineage, not flattened source snapshots.
    prov["dependency_code_sha256"] = {str(p): digest(p) for p in OLD.ROOT.glob("*.py")}
    prov["code_sha256"] = {
        p: h for p, h in prov["code_sha256"].items() if not p.startswith(str(OLD.ROOT))
    }
    prov["protocol_sha256"] = digest(ROOT / "protocol.md")
    prov["parent_checkpoint_sha256"] = digest(PARENT / "checkpoints/best.pt")
    if need_oof:
        prov["oof_predictions_sha256"] = digest(ROOT / "oof/train-predictions.jsonl")
        prov["oof_manifest_sha256"] = digest(ROOT / "oof/manifest.json")
    return docs, enc, prov


def candidates(docs):
    result = {}
    for split in docs:
        path = (
            ROOT / "oof/train-predictions.jsonl"
            if split == "train"
            else OLD.DETECTOR / "dev-predictions.jsonl"
        )
        rows = jsonl(path)
        assert len(rows) == len({r["doc_id"] for r in rows})
        assert {r["doc_id"] for r in rows} == {d["doc_id"] for d in docs[split]}
        result[split] = {r["doc_id"]: r for r in rows}
    return result


def model_from_parent(enc, cfg):
    model = MessageModel(enc, cfg).cuda()
    saved = torch.load(PARENT / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    current = dict(model.named_parameters())
    assert all(n in current and current[n].shape == p.shape for n, p in saved["trainable"].items())
    assert {n for n, p in current.items() if p.requires_grad and "refine_" not in n} == set(
        saved["trainable"]
    )
    with torch.no_grad():
        for name, p in saved["trainable"].items():
            current[name].copy_(p)
    return model


def predict(model, docs, encoded, rows, cc, full=True, intervention=None):
    model.eval()
    pairs = {}
    for row in rows:
        pairs.setdefault(row["doc_id"], set()).add((row["start"], row["end"]))
    lookup, chain, l2, base = {}, [], [], []
    with torch.no_grad():
        for doc in docs:
            did = doc["doc_id"]
            pooled, feat, prefix = model.encode_shared(encoded[did])
            pp = cc[did]["pairs"]
            b, f = model.l2_outputs(pooled, feat, prefix, pp, len(doc["text"]), intervention)
            base.append(b.sigmoid().cpu().numpy())
            l2.append(f.sigmoid().cpu().numpy())
            gold = sorted(pairs.get(did, set()))
            if gold:
                probabilities = model.span_logits(feat, prefix, gold).sigmoid().cpu().numpy()
                lookup[did] = dict(zip(gold, probabilities, strict=True))
            if full:
                probabilities = (
                    model.span_logits(feat, prefix, [(s, e) for s, e, _ in pp])
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
                chain.append(dict(doc_id=did, pairs=pp, probabilities=probabilities.tolist()))
    conditional = np.asarray([lookup[r["doc_id"]][(r["start"], r["end"])] for r in rows])
    return dict(l2=np.asarray(l2), base=np.asarray(base), conditional=conditional, chain=chain)


def ledger_add(kind, seconds, result):
    ledger = read(ROOT / "budget.json")
    ledger["seconds"] += seconds
    ledger["runs"].append(dict(kind=kind, seconds=seconds, **result))
    write_json(ROOT / "budget.json", ledger)


def remaining():
    ledger = read(ROOT / "budget.json")
    return ledger["limit_seconds"] - ledger["seconds"]

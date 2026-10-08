"""Shared ST3 mechanics; no detector training and no held-out test access."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json, write_jsonl
from research.kuki.evaluation import save_predictions
from research.kuki.evaluation import score as l2_score
from research.kuki.experiment import load_development, provenance
from research.kuki.l2 import trainable_state
from research.kuki.span_classifier import examples, score, select_threshold

ROOT = Path(__file__).parent
CHAIN = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.chain")
DIAGNOSTICS = importlib.import_module("research.stages.ST2-l3-rhetoric.negative_fit.method")
DETECTOR = Path("research/stages/ST2-l3-rhetoric/experiments/20260928-detector-base-s42-01")


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def inputs(cfg):
    data = read(cfg["data_config"])
    enc = read(cfg["encoder_config"])
    docs, hashes = load_development(data)
    prov = provenance(cfg, data, hashes, enc)
    for p in sorted(ROOT.glob("*.py")):
        prov["code_sha256"][str(p)] = digest(p)
    prov["protocol_sha256"] = digest(ROOT / "protocol.md")
    prov["detector_inputs"] = {
        str(DETECTOR / f"{s}-predictions.jsonl"): digest(DETECTOR / f"{s}-predictions.jsonl")
        for s in docs
    }
    prov["detector_checkpoint_sha256"] = digest(DETECTOR / "checkpoints/best.pt")
    return docs, enc, prov


def state_digest(state):
    h = hashlib.sha256()
    for name, p in sorted(state.items()):
        h.update(name.encode())
        h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def set_weights(model, state):
    assert set(state) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n])


def update(model, optimizer, scheduler, docs, encoded, clip=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for d in docs:
        loss, detail = model.training_loss(encoded[d["doc_id"]], d["annotations"])
        assert torch.isfinite(loss)
        (loss / len(docs)).backward()
        losses.append(detail)
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    norm = torch.nn.utils.clip_grad_norm_(params, clip)
    assert torch.isfinite(norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(
        l2=float(np.mean([x["l2"] for x in losses])),
        l3=float(np.mean([x["l3"] for x in losses])),
        gradient_norm=float(norm),
    )


def predict(model, docs, encoded, rows, candidates=None):
    model.eval()
    row_pairs = {}
    for r in rows:
        row_pairs.setdefault(r["doc_id"], set()).add((r["start"], r["end"]))
    l2 = []
    lookup = {}
    chain = []
    with torch.no_grad():
        for d in docs:
            did = d["doc_id"]
            pooled, features, prefix = model.encode_shared(encoded[did], model.l3_weight > 0)
            if model.l2_weight:
                l2.append(model.head["l2"](pooled).sigmoid().cpu().numpy())
            if model.l3_weight:
                pairs = sorted(row_pairs.get(did, set()))
                if pairs:
                    prob = model.span_logits(features, prefix, pairs).sigmoid().cpu().numpy()
                    lookup[did] = dict(zip(pairs, prob, strict=True))
                if candidates is not None:
                    pp = candidates[did]["pairs"]
                    probs = (
                        model.span_logits(features, prefix, [(s, e) for s, e, _ in pp])
                        .sigmoid()
                        .cpu()
                        .numpy()
                    )
                    chain.append(dict(doc_id=did, pairs=pp, probabilities=probs.tolist()))
    cond = (
        np.asarray([lookup[r["doc_id"]][(r["start"], r["end"])] for r in rows])
        if model.l3_weight
        else None
    )
    return dict(l2=np.asarray(l2) if model.l2_weight else None, conditional=cond, chain=chain)


def epoch_metrics(docs, rows, pred):
    return dict(
        l2=l2_score(docs, pred["l2"]) if pred["l2"] is not None else None,
        conditional=select_threshold(rows, pred["conditional"])
        if pred["conditional"] is not None
        else None,
    )


def selection_key(metrics):
    if metrics["l2"] is not None:
        return (-metrics["l2"]["brier"],)
    m = metrics["conditional"]
    return (m["article_mean_f1"], m["macro_family_f1"], -m["brier"])


def persist_predictions(out, split, docs, rows, pred):
    out.mkdir(exist_ok=True, parents=True)
    if pred["l2"] is not None:
        save_predictions(out / f"{split}-l2.jsonl", docs, pred["l2"], split)
    if pred["conditional"] is not None:
        write_jsonl(
            out / f"{split}-conditional.jsonl",
            [
                dict(
                    doc_id=r["doc_id"],
                    annotation_index=r["annotation_index"],
                    start=r["start"],
                    end=r["end"],
                    probabilities=p.tolist(),
                )
                for r, p in zip(rows, pred["conditional"], strict=True)
            ],
        )
        write_jsonl(out / f"{split}-chain.jsonl", pred["chain"])


def saved_metrics(out, docs, cfg):
    result = {}
    for split, part in docs.items():
        m = {}
        if (out / f"{split}-l2.jsonl").exists():
            p = jsonl(out / f"{split}-l2.jsonl")
            assert [r["doc_id"] for r in p] == [d["doc_id"] for d in part]
            m["l2"] = l2_score(part, np.asarray([r["probabilities"] for r in p], dtype=np.float32))
        if (out / f"{split}-conditional.jsonl").exists():
            rows = examples(part)
            stored = jsonl(out / f"{split}-conditional.jsonl")
            assert [(r["doc_id"], r["annotation_index"], r["start"], r["end"]) for r in rows] == [
                (r["doc_id"], r["annotation_index"], r["start"], r["end"]) for r in stored
            ]
            probs = np.asarray([r["probabilities"] for r in stored], dtype=np.float32)
            devrows = examples(docs["dev"])
            devprobs = np.asarray(
                [r["probabilities"] for r in jsonl(out / "dev-conditional.jsonl")], dtype=np.float32
            )
            threshold = select_threshold(devrows, devprobs)["threshold"]
            m["conditional"] = score(rows, probs, threshold)
            cc = jsonl(out / f"{split}-chain.jsonl")
            m["chain_fixed"], articles, outputs = CHAIN.chain_score(
                part, cc, cfg["alpha"], cfg["threshold"]
            )
            write_json(out / f"{split}-chain-articles.json", articles)
            write_jsonl(out / f"{split}-chain-decoded.jsonl", outputs)
            m["candidate_diagnostics"] = DIAGNOSTICS.candidate_diagnostics(
                part, cc, cfg["alpha"], cfg["threshold"]
            )
            devcc = jsonl(out / "dev-chain.jsonl")
            tuned, _ = CHAIN.exact_threshold(docs["dev"], CHAIN.ranked(devcc, cfg["alpha"]))
            m["chain_recalibrated"] = CHAIN.chain_score(part, cc, cfg["alpha"], tuned)[0]
            m["diagnostic_dev_threshold"] = tuned
        result[split] = m
    write_json(out / "metrics.json", result)
    return result


def initial_hashes(model):
    return dict(
        encoder=state_digest(
            {n: p for n, p in trainable_state(model).items() if n.startswith("encoder.")}
        ),
        all_heads=state_digest(
            {n: p.detach().cpu() for n, p in model.named_parameters() if n.startswith("head.")}
        ),
    )

"""Frozen detector/classifier composition, explicit calibration and reload verification."""

import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import FAMILIES, write_json, write_jsonl
from research.kuki.experiment import load_development
from research.kuki.global_pointer import GlobalPointer
from research.kuki.l3_evaluation import evaluate
from research.kuki.span_classifier import SpanClassifier, examples
from research.kuki.train_l3 import configure, encode_documents

training = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.train")
ROOT = Path(__file__).parent
RUNS = ROOT.parent / "experiments"


def load_model(run, cls):
    packed = json.loads((run / "config.json").read_text())
    cfg = packed["config"]
    configure(cfg["seed"])
    model = cls(packed["encoder"], cfg).cuda()
    weights = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)[
        "trainable"
    ]
    assert set(weights) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(weights[n])
    return model.eval(), packed


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ranked(predictions, alpha):
    outputs = []
    for p in predictions:
        prob = np.clip(np.asarray(p["probabilities"]), 1e-7, 1 - 1e-7)
        scores = (
            np.log(prob / (1 - prob)) + alpha * np.asarray([v for _, _, v in p["pairs"]])[:, None]
        )
        spans = [
            dict(start=s, end=e, label=family, score=float(scores[i, j]))
            for i, (s, e, _) in enumerate(p["pairs"])
            for j, family in enumerate(FAMILIES)
        ]
        spans.sort(key=lambda s: (-s["score"], s["start"], s["end"], s["label"]))
        outputs.append(
            dict(
                doc_id=p["doc_id"],
                spans=spans[:100],
                starts=[],
                ends=[],
                candidate_count=len(p["pairs"]),
                eligible_before_cap=0,
                all_scores=scores.ravel(),
            )
        )
    return outputs


def exact_threshold(docs, outputs):
    """Exact article/answer F1 sweep over states after the fixed top100 output cap."""
    values = [s["score"] for o in outputs for s in o["spans"]]
    thresholds = np.unique(values + [max(values) + 1])
    curve = np.zeros(len(thresholds))
    for d, o in zip(docs, outputs, strict=True):
        spans = sorted(o["spans"], key=lambda s: s["score"])
        v = np.asarray([s["score"] for s in spans])
        pos = np.searchsorted(v, thresholds, side="left")
        count = len(spans) - pos
        for a in d["annotations"]:
            gold = {(s["start"], s["end"], s["label"]) for s in a["l3_persuasion"]}
            hits = [(s["start"], s["end"], s["label"]) in gold for s in spans]
            cumul = np.r_[0, np.cumsum(hits)]
            tp = cumul[-1] - cumul[pos]
            curve += 2 * tp / np.maximum(1, count + len(gold)) / len(d["annotations"]) / len(docs)
    best = max(range(len(thresholds)), key=lambda i: (curve[i], thresholds[i]))
    return float(thresholds[best]), float(curve[best])


def chain_score(docs, predictions, alpha, threshold):
    outputs = ranked(predictions, alpha)
    for o in outputs:
        o["eligible_before_cap"] = int((o.pop("all_scores") >= threshold).sum())
    metrics, records = evaluate(docs, outputs, threshold)
    # Original evaluate assumes a Cartesian boundary set. Replace with actual paired proposals.
    for d, p, r in zip(docs, predictions, records, strict=True):
        pairs = {(s, e) for s, e, _ in p["pairs"]}
        for a, result in zip(d["annotations"], r["individuals"], strict=True):
            result["proposal_hits"] = sum(
                (s["start"], s["end"]) in pairs for s in a["l3_persuasion"]
            )
        nonempty = [a for a in r["individuals"] if a["n_gold"]]
        r["proposal_recall"] = (
            float(np.mean([a["proposal_hits"] / a["n_gold"] for a in nonempty]))
            if nonempty
            else None
        )
    metrics["proposal_recall"] = float(
        np.mean([r["proposal_recall"] for r in records if r["proposal_recall"] is not None])
    )
    metrics["alpha"] = alpha
    return metrics, records, outputs


def main():
    begin = time.perf_counter()
    ledger = json.loads((ROOT / "budget.json").read_text())
    assert ledger["seconds"] < 2370
    dirs = {
        kind: [RUNS / f"20260928-{kind}-{variant}-s42-01" for variant in ["base", "low-lr"]]
        for kind in ["classifier", "detector"]
    }

    def criteria(path):
        m = json.loads((path / "metrics.json").read_text())["dev"]
        return (
            (m["article_mean_f1"], m["macro_family_f1"], -m["brier"])
            if "article_mean_f1" in m
            else (m["recall"]["200"], m["exact_f1"], m["overlap_f1"])
        )

    selected = {k: max(v, key=criteria) for k, v in dirs.items()}
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    model, packed = load_model(selected["classifier"], SpanClassifier)
    encoded = encode_documents(docs["train"] + docs["dev"], packed["encoder"], packed["config"])
    rows = examples(docs["dev"])
    with torch.no_grad():
        actual = training.classifier_predict(model, docs["dev"], encoded, rows)
    saved = np.asarray(
        [r["probabilities"] for r in read_jsonl(selected["classifier"] / "dev-predictions.jsonl")]
    )
    verify = dict(classifier_max_abs_diff=float(np.max(np.abs(actual - saved))))
    assert np.array_equal(actual, saved)
    predictions = {}
    with torch.no_grad():
        for split in docs:
            pp = read_jsonl(selected["detector"] / f"{split}-predictions.jsonl")
            for p in pp:
                pairs = [(s, e) for s, e, _ in p["pairs"]]
                p["probabilities"] = model.probabilities(encoded[p["doc_id"]], pairs).tolist()
            predictions[split] = pp
            write_jsonl(ROOT / "E3-detector" / f"chain-{split}-scores.jsonl", pp)
    del model
    torch.cuda.empty_cache()
    detector, _ = load_model(selected["detector"], GlobalPointer)
    actual_pairs = training.detector_predict(detector, docs["dev"], encoded)
    saved_pairs = read_jsonl(selected["detector"] / "dev-predictions.jsonl")
    verify["detector_pair_indices_equal"] = all(
        [(s, e) for s, e, _ in a["pairs"]] == [(s, e) for s, e, _ in b["pairs"]]
        for a, b in zip(actual_pairs, saved_pairs, strict=True)
    )
    verify["detector_max_abs_diff"] = max(
        abs(x[2] - y[2])
        for a, b in zip(actual_pairs, saved_pairs, strict=True)
        for x, y in zip(a["pairs"], b["pairs"], strict=True)
    )
    assert verify["detector_pair_indices_equal"] and verify["detector_max_abs_diff"] == 0
    del detector
    settings = []
    for alpha in [0.0, 0.5, 1.0]:
        threshold, f1 = exact_threshold(docs["dev"], ranked(predictions["dev"], alpha))
        metrics, _, _ = chain_score(docs["dev"], predictions["dev"], alpha, threshold)
        assert abs(metrics["exact_f1"] - f1) < 1e-12
        settings.append(dict(alpha=alpha, threshold=threshold, dev=metrics))
    choice = max(settings, key=lambda s: (s["dev"]["exact_f1"], s["threshold"], -s["alpha"]))
    result = dict(
        selected_runs={k: str(v) for k, v in selected.items()},
        selection=choice,
        calibration=settings,
        data_hashes=hashes,
        test_used=False,
        verification=verify,
    )
    for split in docs:
        metrics, records, outputs = chain_score(
            docs[split], predictions[split], choice["alpha"], choice["threshold"]
        )
        result[split] = metrics
        write_json(ROOT / "E3-detector" / f"chain-{split}-articles.json", records)
        write_jsonl(ROOT / "E3-detector" / f"chain-{split}-predictions.jsonl", outputs)
    seconds = time.perf_counter() - begin
    result["elapsed_seconds"] = seconds
    ledger["seconds"] += seconds
    ledger["runs"].append(
        dict(run_id="frozen-chain-verification", kind="evaluation", seconds=seconds, updates=0)
    )
    write_json(ROOT / "budget.json", ledger)
    write_json(ROOT / "E3-detector" / "chain.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

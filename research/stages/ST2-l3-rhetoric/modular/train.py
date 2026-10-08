"""Bounded independent E1/E3 training with immutable run directories and shared budget."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import FAMILIES, write_json, write_jsonl
from research.kuki.experiment import initialize_run, load_development, provenance
from research.kuki.global_pointer import GlobalPointer
from research.kuki.l2 import frozen_digest, trainable_state
from research.kuki.l3_evaluation import f1, matched
from research.kuki.span_classifier import (
    SpanClassifier,
    answer_pairs,
    examples,
    score,
    select_threshold,
)
from research.kuki.train_l2 import epoch_order, make_optimizer
from research.kuki.train_l3 import configure, encode_documents

ROOT = Path(__file__).parent


def classifier_predict(model, docs, encoded, rows):
    model.eval()
    lookup = {}
    with torch.no_grad():
        for d in docs:
            pairs = sorted({p for a in d["annotations"] for p in answer_pairs(a)[0]})
            if not pairs:
                continue
            probs = model.probabilities(encoded[d["doc_id"]], pairs)
            lookup[d["doc_id"]] = dict(zip(pairs, probs, strict=True))
    return np.asarray([lookup[r["doc_id"]][(r["start"], r["end"])] for r in rows])


def detector_predict(model, docs, encoded):
    model.eval()
    with torch.no_grad():
        return [
            dict(
                doc_id=d["doc_id"],
                pairs=model.infer_pairs(encoded[d["doc_id"]], 500),
                candidate_universe=len(d["text"]) * (len(d["text"]) + 1) // 2,
            )
            for d in docs
        ]


def detector_threshold(docs, outputs):
    thresholds = np.unique(
        np.r_[
            [v for o in outputs for _, _, v in o["pairs"][:100]],
            max(v for o in outputs for _, _, v in o["pairs"]) + 1,
        ]
    )
    curve = np.zeros(len(thresholds))
    for d, o in zip(docs, outputs, strict=True):
        pp = o["pairs"][:100]
        values = np.array([v for _, _, v in pp])
        order = np.argsort(values, kind="stable")
        values = values[order]
        before = np.searchsorted(values, thresholds, side="left")
        count = len(pp) - before
        for a in d["annotations"]:
            gold = {(s["start"], s["end"]) for s in a["l3_persuasion"]}
            positive = np.array([(pp[i][0], pp[i][1]) in gold for i in order])
            cumulative = np.r_[0, np.cumsum(positive)]
            tp = cumulative[-1] - cumulative[before]
            curve += 2 * tp / np.maximum(1, count + len(gold)) / len(d["annotations"]) / len(docs)
    best = max(range(len(thresholds)), key=lambda i: (curve[i], thresholds[i]))
    return float(thresholds[best])


def detector_score(docs, outputs, threshold):
    records = []
    ks = [50, 100, 200, 500]
    for d, o in zip(docs, outputs, strict=True):
        pp = o["pairs"]
        pred = [dict(start=s, end=e, label=FAMILIES[0]) for s, e, v in pp[:100] if v >= threshold]
        individual = []
        for a in d["annotations"]:
            gold = [
                dict(start=s["start"], end=s["end"], label=FAMILIES[0]) for s in a["l3_persuasion"]
            ]
            goldpairs = {(s["start"], s["end"]) for s in gold}
            exact = matched(pred, gold, typed=False)
            individual.append(
                dict(
                    annotator_id=a["annotator_id"],
                    n_gold=len(goldpairs),
                    exact_f1=f1(*exact),
                    overlap_f1=f1(*matched(pred, gold, typed=False, overlap=True)),
                    recall={
                        str(k): len(goldpairs & {(s, e) for s, e, _ in pp[:k]}) / len(goldpairs)
                        if goldpairs
                        else None
                        for k in ks
                    },
                    correct_empty=not pred if not gold else None,
                )
            )
        records.append(
            dict(
                doc_id=d["doc_id"],
                n_predictions=len(pred),
                exact_f1=float(np.mean([a["exact_f1"] for a in individual])),
                overlap_f1=float(np.mean([a["overlap_f1"] for a in individual])),
                recall={
                    str(k): float(np.mean([a["recall"][str(k)] for a in individual if a["n_gold"]]))
                    if any(a["n_gold"] for a in individual)
                    else None
                    for k in ks
                },
                individuals=individual,
            )
        )
    empty = [a for r in records for a in r["individuals"] if not a["n_gold"]]
    return dict(
        exact_f1=float(np.mean([r["exact_f1"] for r in records])),
        overlap_f1=float(np.mean([r["overlap_f1"] for r in records])),
        recall={
            str(k): float(
                np.mean([r["recall"][str(k)] for r in records if r["recall"][str(k)] is not None])
            )
            for k in ks
        },
        mean_predictions=float(np.mean([r["n_predictions"] for r in records])),
        correctly_empty_answers=sum(a["correct_empty"] for a in empty),
        empty_answers=len(empty),
        threshold=threshold,
        articles=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    begin = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    kind = cfg["kind"]
    ledger_path = ROOT / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["seconds"] < 2400
    data_cfg = json.loads(Path(cfg["data_config"]).read_text())
    enc = json.loads(Path(cfg["encoder_config"]).read_text())
    docs, hashes = load_development(data_cfg)
    configure(cfg["seed"])
    encoded = encode_documents(docs["train"] + docs["dev"], enc, cfg)
    out = ROOT.parent / "experiments" / cfg["run_id"]
    prov = provenance(cfg, data_cfg, hashes, enc)
    for path in [Path(__file__), ROOT / "protocol.md"]:
        import hashlib

        prov["code_sha256"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    initialize_run(out, cfg, prov)
    model = (SpanClassifier if kind == "classifier" else GlobalPointer)(enc, cfg).cuda()
    base = frozen_digest(model)
    train_docs = (
        [d for d in docs["train"] if any(a["l3_persuasion"] for a in d["annotations"])]
        if kind == "classifier"
        else docs["train"]
    )
    assert cfg["max_updates"] == int(np.ceil(len(train_docs) / 8)) * cfg["max_epochs"]
    rows = {s: examples(p) for s, p in docs.items()}
    optimizer, scheduler = make_optimizer(model, cfg)
    history = []
    best_key = None
    best_epoch = None
    bad = 0
    updates = 0
    seen = 0
    status = "epoch_limit"
    best_metrics = None
    for epoch in range(cfg["max_epochs"]):
        order = epoch_order(len(train_docs), cfg["seed"], epoch)
        losses = []
        t = time.perf_counter()
        stopped = False
        for p in range(0, len(order), 8):
            if (
                time.perf_counter() - begin > cfg["run_budget_seconds"]
                or ledger["seconds"] + time.perf_counter() - begin > 2300
            ):
                stopped = True
                status = "time_limit"
                break
            batch = [train_docs[i] for i in order[p : p + 8]]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for d in batch:
                loss, detail = model.training_loss(encoded[d["doc_id"]], d["annotations"])
                assert torch.isfinite(loss)
                (loss / len(batch)).backward()
                losses.append(detail["loss"])
            params = [p for p in model.parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip_norm"])
            assert torch.isfinite(norm)
            optimizer.step()
            scheduler.step()
            updates += 1
            seen += len(batch)
        train_seconds = time.perf_counter() - t
        if kind == "classifier":
            predicted = classifier_predict(model, docs["dev"], encoded, rows["dev"])
            m = select_threshold(rows["dev"], predicted)
            key = (m["article_mean_f1"], m["macro_family_f1"], -m["brier"])
        else:
            predicted = detector_predict(model, docs["dev"], encoded)
            m = detector_score(docs["dev"], predicted, detector_threshold(docs["dev"], predicted))
            key = (m["recall"]["200"], m["exact_f1"], m["overlap_f1"])
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_epoch = epoch + 1
            best_metrics = m
            bad = 0
            torch.save(
                dict(
                    trainable=trainable_state(model),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    updates=updates,
                    epoch=epoch + 1,
                ),
                out / "checkpoints/best.pt",
            )
        else:
            bad += 1
        entry = dict(
            epoch=epoch + 1,
            updates=updates,
            article_presentations=seen,
            loss=float(np.mean(losses)) if losses else None,
            train_seconds=train_seconds,
            dev=m,
        )
        history.append(entry)
        write_json(out / "history.json", history)
        print(
            json.dumps(
                dict(
                    epoch=epoch + 1,
                    updates=updates,
                    loss=entry["loss"],
                    train_seconds=train_seconds,
                    dev={k: v for k, v in m.items() if k not in ["articles", "article_f1"]},
                    best_epoch=best_epoch,
                )
            ),
            flush=True,
        )
        if stopped:
            break
        if bad >= cfg["early_stopping_patience_epochs"]:
            status = "patience"
            break
    optimizer.zero_grad(set_to_none=True)
    torch.save(
        dict(
            trainable=trainable_state(model),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            updates=updates,
            history=history,
        ),
        out / "checkpoints/last.pt",
    )
    assert frozen_digest(model) == base
    selected = torch.load(out / "checkpoints/best.pt", map_location="cpu", weights_only=False)[
        "trainable"
    ]
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(selected[name])
    final = {}
    for split, part in docs.items():
        if kind == "classifier":
            prob = classifier_predict(model, part, encoded, rows[split])
            m = score(rows[split], prob, best_metrics["threshold"])
            final[split] = m
            write_jsonl(
                out / f"{split}-predictions.jsonl",
                [
                    dict(
                        **{k: v for k, v in r.items() if k not in ["text", "target"]},
                        target=r["target"].tolist(),
                        probabilities=p.tolist(),
                    )
                    for r, p in zip(rows[split], prob, strict=True)
                ],
            )
        else:
            prediction = detector_predict(model, part, encoded)
            m = detector_score(part, prediction, best_metrics["threshold"])
            final[split] = m
            write_jsonl(out / f"{split}-predictions.jsonl", prediction)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    ledger["seconds"] += seconds
    ledger["runs"].append(
        dict(run_id=cfg["run_id"], kind=kind, seconds=seconds, updates=updates, status=status)
    )
    write_json(ledger_path, ledger)
    write_json(
        out / "metrics.json",
        dict(
            status=status,
            kind=kind,
            selected_epoch=best_epoch,
            updates=updates,
            article_presentations=seen,
            elapsed_seconds=seconds,
            base_unchanged=True,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            excluded_all_empty_train_articles=len(docs["train"]) - len(train_docs),
            test_used=False,
            **final,
        ),
    )
    print(
        json.dumps(
            dict(
                completed=cfg["run_id"],
                seconds=seconds,
                total_seconds=ledger["seconds"],
                selected_epoch=best_epoch,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

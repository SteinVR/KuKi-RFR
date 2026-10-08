"""Bounded train-only ST2 diagnostics. Does not change the historical trainer/model."""

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import research.kuki.l3 as l3_module
import research.kuki.l3_losses as losses_module
from research.kuki.data import FAMILIES, write_json
from research.kuki.experiment import initialize_run, load_development, provenance
from research.kuki.l2 import frozen_digest, trainable_state
from research.kuki.l3 import L3Model, candidate_pairs, prepare_l3, propose, span_key
from research.kuki.train_l3 import configure, encode_documents

STAGE = Path(__file__).parent
SERIES = STAGE / "diagnostics/20260928-tiny-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select():
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    choices = []
    for d in docs["train"]:
        for i, a in sorted(
            enumerate(d["annotations"]), key=lambda x: (str(x[1]["annotator_id"]), x[0])
        ):
            spans = a["l3_persuasion"]
            text = d["text"]
            overlap = any(
                min(s["end"], t["end"]) > max(s["start"], t["start"])
                for j, s in enumerate(spans)
                for t in spans[:j]
            )
            special = any(
                text[s["start"]].isspace()
                or text[s["end"] - 1].isspace()
                or any(
                    0 < p < len(text) and text[p - 1].isalnum() and text[p].isalnum()
                    for p in [s["start"], s["end"]]
                )
                for s in spans
            )
            choices.append((d, i, a, overlap, special))
    first = next(
        c
        for c in choices
        if 500 <= len(c[0]["text"]) <= 4000
        and 3 <= len({span_key(s) for s in c[2]["l3_persuasion"]}) <= 30
        and len({s["label"] for s in c[2]["l3_persuasion"]}) >= 2
    )
    second = next(
        c
        for c in choices
        if 500 <= len(c[0]["text"]) <= 4000
        and c[0]["doc_id"] != first[0]["doc_id"]
        and 3 <= len({span_key(s) for s in c[2]["l3_persuasion"]}) <= 30
        and c[3]
        and c[4]
    )
    third = next(
        c
        for c in sorted(
            choices,
            key=lambda c: (len(c[0]["text"]), c[0]["doc_id"], str(c[2]["annotator_id"]), c[1]),
        )
        if c[0]["doc_id"] not in {first[0]["doc_id"], second[0]["doc_id"]}
        and not c[2]["l3_persuasion"]
    )
    records = []
    for d, i, a, overlap, special in [first, second, third]:
        gold = {span_key(s) for s in a["l3_persuasion"]}
        n = len(d["text"])
        assert len(gold) == len(a["l3_persuasion"]), "Duplicate typed gold span"
        for s in a["l3_persuasion"]:
            assert 0 <= s["start"] < s["end"] <= n and d["text"][s["start"] : s["end"]] == s["text"]
        assert (
            len(gold) <= 100
            and len({s for s, _, _ in gold}) <= 128
            and len({e for _, e, _ in gold}) <= 128
        )
        bounds = set(np.linspace(0, n, 33, dtype=int).tolist())
        bounds |= {
            p + delta
            for s, e, _ in gold
            for p in [s, e]
            for delta in [-1, 0, 1]
            if 0 <= p + delta <= n
        }
        pairs = candidate_pairs(sorted(bounds - {n}), sorted(bounds - {0}))
        assert {(s, e) for s, e, _ in gold} <= set(pairs)
        rec = dict(
            doc_id=d["doc_id"],
            annotation_index=i,
            annotator_id=a["annotator_id"],
            text_sha256=d["text_sha256"],
            annotation_sha256=hashlib.sha256(
                json.dumps(a, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            source_metadata=a["source_metadata"],
            characters=n,
            typed_spans=len(gold),
            starts=len({s for s, _, _ in gold}),
            ends=len({e for _, e, _ in gold}),
            families=sorted({s["label"] for s in a["l3_persuasion"]}),
            overlapping=overlap,
            special_boundary=special,
            fixed_pairs=pairs,
        )
        rec["fixed_pairs_sha256"] = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
        records.append(rec)
    write_json(
        SERIES / "manifest.json",
        dict(design_sha256=sha(SERIES / "design.json"), data_hashes=hashes, records=records),
    )
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ["fixed_pairs", "source_metadata"]}
                for r in records
            ]
        ),
        flush=True,
    )


def selected_docs():
    manifest = json.loads((SERIES / "manifest.json").read_text())
    data_cfg = json.loads(Path("research/configs/data.json").read_text())
    docs, hashes = load_development(data_cfg)
    assert hashes == manifest["data_hashes"]
    by_id = {d["doc_id"]: d for d in docs["train"]}
    chosen = []
    for r in manifest["records"]:
        d = copy.deepcopy(by_id[r["doc_id"]])
        a = d["annotations"][r["annotation_index"]]
        assert (
            hashlib.sha256(json.dumps(a, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            == r["annotation_sha256"]
        )
        assert d["text_sha256"] == r["text_sha256"]
        d["annotations"] = [a]
        chosen.append(d)
    return chosen, manifest, data_cfg, hashes


def targets(pairs, gold):
    return np.array(
        [[(s, e, f) in gold for f in range(len(FAMILIES))] for s, e in pairs], dtype=bool
    )


def distribution(values):
    return (
        dict(n=int(values.size), quantiles=np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist())
        if values.size
        else dict(n=0, quantiles=[])
    )


def curves(rows, cap):
    """Exact common threshold sweep. Scores below cap cannot change the output."""
    selected = []
    for r in rows:
        flat = r["scores"].ravel()
        order = np.argsort(-flat, kind="stable")
        if cap is not None:
            order = order[:cap]
        selected.append((flat[order], r["positive"].ravel()[order]))
    thresholds = np.unique(np.concatenate([v[0] for v in selected] + [np.array([1.0])]))
    f1s, numbers, truepos = [], [], []
    for r, (scores, positive) in zip(rows, selected, strict=True):
        ix = np.argsort(scores, kind="stable")
        scores, positive = scores[ix], positive[ix]
        before = np.searchsorted(scores, thresholds, side="left")
        pred = len(scores) - before
        cumulative = np.r_[0, np.cumsum(positive)]
        tp = cumulative[-1] - cumulative[before]
        g = len(r["gold"])
        f1s.append(2 * tp / np.maximum(1, pred + g) if g else (pred == 0).astype(float))
        numbers.append(pred)
        truepos.append(tp)
    f1s, numbers, truepos = np.array(f1s), np.array(numbers), np.array(truepos)
    passed = (f1s >= 0.95).all(axis=0)
    if cap is not None and not all(r["pair_recall"] == 1 or not r["gold"] for r in rows):
        passed[:] = False
    intervals = []
    padded = np.r_[False, passed, False]
    for a, b in zip(
        np.flatnonzero(np.diff(padded.astype(int)) == 1),
        np.flatnonzero(np.diff(padded.astype(int)) == -1),
        strict=True,
    ):
        lower = float(thresholds[a - 1]) if a else 0.0
        upper = float(thresholds[b - 1])

        def logit(x):
            return float(np.log(x / (1 - x))) if 0 < x < 1 else None

        intervals.append(
            dict(
                lower_exclusive=lower,
                upper_inclusive=upper,
                width=upper - lower,
                lower_logit=logit(lower),
                upper_logit=logit(upper),
            )
        )
    # Best diagnostic threshold maximizes worst-document score, then mean, then precision proxy.
    best = max(
        range(len(thresholds)),
        key=lambda i: (
            float(f1s[:, i].min()),
            float(f1s[:, i].mean()),
            -int(numbers[:, i].sum()),
            float(thresholds[i]),
        ),
    )
    t = float(thresholds[best])
    metrics = []
    for j, r in enumerate(rows):
        scores, pos = selected[j]
        indices = np.argsort(-r["scores"].ravel(), kind="stable")
        if cap is not None:
            indices = indices[:cap]
        indices = indices[r["scores"].ravel()[indices] >= t]
        pairs = {tuple(r["pairs"][i // len(FAMILIES)]) for i in indices}
        goldpairs = {(s, e) for s, e, _ in r["gold"]}
        untyped = 2 * len(pairs & goldpairs) / max(1, len(pairs) + len(goldpairs))
        metrics.append(
            dict(
                doc_id=r["doc_id"],
                exact_typed_f1=float(f1s[j, best]) if r["gold"] else 0.0,
                correct_empty=not len(indices) if not r["gold"] else None,
                exact_untyped_f1=untyped,
                predictions=int(numbers[j, best]),
                true_positive=int(truepos[j, best]),
                gold=len(r["gold"]),
            )
        )
    return dict(
        passed=bool(passed.any()),
        successful_intervals=intervals,
        best_threshold=t,
        articles=metrics,
        distinct_output_states=len(thresholds),
        threshold_scope="all raw candidates; top-cap applied only for ordinary output",
    )


def evaluate(model, docs, encoded, records, output, step):
    model.eval()
    ordinary, oracle, details = [], [], []
    arrays = {}
    with torch.no_grad():
        for index, d in enumerate(docs):
            item = encoded[d["doc_id"]]
            features, boundary, prefix = model.encode(item)
            starts, ends = propose(boundary, 128, item["length"])
            pairs = candidate_pairs(starts, ends)
            gold = {span_key(s) for s in d["annotations"][0]["l3_persuasion"]}
            goldpairs = {(s, e) for s, e, _ in gold}
            gs, ge = {s for s, _, _ in gold}, {e for _, e, _ in gold}
            coverage = dict(
                start_recall=len(gs & set(starts)) / len(gs) if gs else None,
                end_recall=len(ge & set(ends)) / len(ge) if ge else None,
                pair_recall=len(goldpairs & set(pairs)) / len(goldpairs) if goldpairs else None,
            )
            b = boundary.cpu().numpy()
            ranks = []
            for positions, col, slice_ in [(gs, 0, b[:-1, 0]), (ge, 1, b[1:, 1])]:
                ranked = np.argsort(-slice_, kind="stable") + (1 if col else 0)
                inv = {int(p): k + 1 for k, p in enumerate(ranked)}
                ranks.append([inv[p] for p in sorted(positions)])
            detail = dict(
                doc_id=d["doc_id"],
                **coverage,
                gold_start_ranks=ranks[0],
                gold_end_ranks=ranks[1],
                candidate_pairs=len(pairs),
            )
            for kind, pp, container in [
                ("ordinary", pairs, ordinary),
                ("fixed", records[index]["fixed_pairs"], oracle),
            ]:
                logits_t = model.span_logits(features, prefix, pp)
                logits = logits_t.cpu().numpy()
                scores = logits_t.sigmoid().cpu().numpy()
                mask = targets(pp, gold)
                hardpairs = np.array(
                    [
                        any(abs(s - gs_) <= 1 and abs(e - ge_) <= 1 for gs_, ge_, _ in gold)
                        for s, e in pp
                    ]
                )
                hard = (~mask) & hardpairs[:, None]
                regular = (~mask) & (~hard)
                detail[kind] = {
                    name: dict(logits=distribution(logits[m]), scores=distribution(scores[m]))
                    for name, m in [
                        ("positive", mask),
                        ("hard_negative", hard),
                        ("ordinary_negative", regular),
                    ]
                }
                container.append(
                    dict(
                        doc_id=d["doc_id"],
                        pairs=np.asarray(pp),
                        scores=scores,
                        positive=mask,
                        gold=gold,
                        pair_recall=coverage["pair_recall"] if kind == "ordinary" else 1,
                    )
                )
                for name, values in [
                    ("pairs", pp),
                    ("logits", logits),
                    ("scores", scores),
                    ("positive", mask),
                    ("hard_negative", hard),
                ]:
                    arrays[f"d{index}_{kind}_{name}"] = values
            arrays[f"d{index}_boundary_logits"] = b
            details.append(detail)
    result = dict(
        update=step, ordinary=curves(ordinary, 100), oracle=curves(oracle, None), details=details
    )
    np.savez_compressed(output / "checkpoints" / f"full-scores-{step:03}.npz", **arrays)
    write_json(output / f"eval-{step:03}.json", result)
    return result


def audit(model, docs, encoded, records):
    results = []
    original_balanced, original_hard = l3_module.balanced_bce, losses_module.hard_negative_bce
    for d, _r in zip(docs, records, strict=True):
        item = encoded[d["doc_id"]]
        gold = {span_key(s) for s in d["annotations"][0]["l3_persuasion"]}
        features, boundary, prefix = model.encode(item)
        s, e = propose(boundary, 128, item["length"])
        pairs = candidate_pairs(
            sorted(set(s) | {a for a, _, _ in gold}), sorted(set(e) | {b for _, b, _ in gold})
        )
        target = targets(pairs, gold)
        expected_b = np.zeros((item["length"] + 1, 2), dtype=bool)
        for a, b, _ in gold:
            expected_b[a, 0] = True
            expected_b[b, 1] = True
        expected_b = np.r_[expected_b[:-1, 0], expected_b[1:, 1]]
        captured = {}

        def cb(logits, t, weight=1, expected_b=expected_b, captured=captured):
            assert np.array_equal(t.detach().cpu().numpy().astype(bool), expected_b)
            value = original_balanced(logits, t, weight)
            captured["boundary"] = value
            return value

        def ch(logits, t, *args, target=target, captured=captured):
            assert np.array_equal(t.detach().cpu().numpy().astype(bool), target)
            value = original_hard(logits, t, *args)
            captured["span"] = value[0]
            return value

        with (
            patch.object(l3_module, "balanced_bce", cb),
            patch.object(losses_module, "hard_negative_bce", ch),
        ):
            loss, detail = model.training_loss(item, d["annotations"])
        assert torch.isfinite(loss)
        grads = {}
        for component, value in captured.items():
            model.zero_grad(set_to_none=True)
            value.backward(retain_graph=True)
            groups = {}
            for name, p in model.named_parameters():
                group = (
                    "boundary_head"
                    if name.startswith("head.boundary.")
                    else "span_head"
                    if name.startswith("head.span.")
                    else "shared_head"
                    if name.startswith("head.")
                    else "lora"
                )
                if not p.requires_grad:
                    continue
                g = groups.setdefault(
                    group, dict(tensors=0, with_gradient=0, nonzero=0, norm_squared=0.0)
                )
                g["tensors"] += 1
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
                    g["with_gradient"] += 1
                    g["nonzero"] += int(bool(p.grad.count_nonzero()))
                    g["norm_squared"] += float(p.grad.square().sum())
            assert (
                groups[component + "_head" if component == "boundary" else "span_head"]["nonzero"]
                > 0
            )
            assert groups["shared_head"]["nonzero"] > 0 and groups["lora"]["nonzero"] > 0
            grads[component] = groups
        distances = []
        collisions = []
        f = features.detach().cpu().numpy()
        for p in sorted({a for a, b, _ in gold} | {b for a, b, _ in gold}):
            for q in [p - 1, p + 1]:
                if 0 <= q < len(f):
                    dist = float(np.linalg.norm(f[p] - f[q]))
                    distances.append(dist)
                    if dist == 0:
                        collisions.append([p, q])
        if not gold:
            distances = np.linalg.norm(f[:-1] - f[1:], axis=1).tolist()
        # Rebuilding index features from tokenizer offsets must agree exactly.
        assert prepare_l3(item["encoded"], d["text"]) == item
        results.append(
            dict(
                doc_id=d["doc_id"],
                targets_and_masks_exact=True,
                no_duplicate_typed_targets=True,
                cap_feasible=True,
                all_gold_in_training_candidates=True,
                adjacent_feature_distances=distribution(np.asarray(distances)),
                exact_gold_neighbor_feature_collisions=collisions,
                gradients=grads,
                loss=detail,
                characters=item["length"],
                tokens=item["encoded"]["tokens"],
            )
        )
        model.zero_grad(set_to_none=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "audit", "one", "three"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.mode == "select":
        select()
        return
    begin = time.perf_counter()
    ledger_path = SERIES / "budget.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else dict(gpu_process_seconds=0.0, fits=0, processes=[])
    )
    assert ledger["gpu_process_seconds"] < 1200
    if args.mode != "audit":
        assert ledger["fits"] < 3
    docs, manifest, data_cfg, hashes = selected_docs()
    records = manifest["records"]
    if args.mode == "one":
        docs, records = docs[:1], records[:1]
    cfg = json.loads((STAGE / "configs/hard-negative-v1.json").read_text())
    cfg.update(
        head_dropout=0.0,
        max_updates=200,
        warmup_updates=0,
        diagnostic_mode=args.mode,
        gradient_accumulation_articles=len(docs),
        scheduler="constant",
        early_stop="first checkpoint meeting all criteria",
    )
    enc_cfg = json.loads(Path(cfg["encoder_config"]).read_text())
    enc_cfg["lora"]["dropout"] = 0.0
    configure(42)
    encoded = encode_documents(docs, enc_cfg, cfg)
    run_id = args.run_id or f"20260928-l3-tiny-{args.mode}-s42-01"
    output = STAGE / "experiments" / run_id
    prov = provenance(cfg, data_cfg, hashes, enc_cfg)
    prov["code_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    initialize_run(output, cfg, prov)
    shutil.copy2(SERIES / "manifest.json", output / "manifest.json")
    shutil.copy2(SERIES / "design.json", output / "design.json")
    model = L3Model(enc_cfg, cfg).cuda()
    disabled = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            disabled.append(dict(name=name, previous_p=module.p))
            module.p = 0.0
    for name, module in model.named_modules():
        for field in ["attention_dropout", "hidden_dropout", "embedding_dropout", "mlp_dropout"]:
            if isinstance(getattr(module, field, None), (int, float)):
                disabled.append(dict(name=name, field=field, previous_p=getattr(module, field)))
                setattr(module, field, 0.0)
    write_json(output / "dropout.json", disabled)
    model.train()
    base_hash = frozen_digest(model)
    status = "running"
    history = []
    updates = 0
    if args.mode == "audit":
        result = audit(model, docs, encoded, records)
        write_json(output / "audit.json", result)
        status = "audit_passed"
        print(json.dumps(dict(status=status, results=result)), flush=True)
    else:
        ledger["fits"] += 1
        groups = [
            dict(
                params=[
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("encoder.")
                ],
                lr=cfg["adapter_lr"],
            ),
            dict(params=list(model.head.parameters()), lr=cfg["head_lr"]),
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        train = []
        for step in range(201):
            if step % 20 == 0:
                result = evaluate(model, docs, encoded, records, output, step)
                history.append(result)
                torch.save(trainable_state(model), output / "checkpoints" / f"update-{step:03}.pt")
                print(
                    json.dumps(
                        dict(
                            update=step,
                            ordinary=result["ordinary"],
                            oracle=result["oracle"],
                            coverage=[
                                {
                                    k: d[k]
                                    for k in ["doc_id", "start_recall", "end_recall", "pair_recall"]
                                }
                                for d in result["details"]
                            ],
                        )
                    ),
                    flush=True,
                )
                if result["ordinary"]["passed"]:
                    status = "passed"
                    break
            if step == 200:
                status = "update_limit"
                break
            if ledger["gpu_process_seconds"] + time.perf_counter() - begin > 1170:
                status = "time_limit"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            details = []
            for d in docs:
                loss, detail = model.training_loss(encoded[d["doc_id"]], d["annotations"])
                assert torch.isfinite(loss)
                (loss / len(docs)).backward()
                details.append(dict(doc_id=d["doc_id"], **detail))
            params = [p for p in model.parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip_norm"])
            assert torch.isfinite(norm)
            optimizer.step()
            updates = step + 1
            train.append(dict(update=updates, gradient_norm=float(norm), articles=details))
        if history[-1]["update"] != updates:
            history.append(evaluate(model, docs, encoded, records, output, updates))
        optimizer.zero_grad(set_to_none=True)
        torch.save(
            dict(
                trainable=trainable_state(model), optimizer=optimizer.state_dict(), updates=updates
            ),
            output / "checkpoints" / "final.pt",
        )
        write_json(output / "training.json", train)
    assert frozen_digest(model) == base_hash
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    ledger["gpu_process_seconds"] += elapsed
    ledger["processes"].append(
        dict(run_id=run_id, mode=args.mode, elapsed_seconds=elapsed, updates=updates, status=status)
    )
    write_json(ledger_path, ledger)
    metrics = dict(
        status=status,
        updates=updates,
        elapsed_seconds=elapsed,
        total_gpu_process_seconds=ledger["gpu_process_seconds"],
        fits=ledger["fits"],
        base_unchanged=True,
        test_used=False,
        history=history,
    )
    write_json(output / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()

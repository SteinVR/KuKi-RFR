"""Fixed-cache per-answer negative sampling and diagnostics, no threshold search."""

import hashlib

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from research.kuki.data import FAMILIES
from research.kuki.span_classifier import answer_pairs


def samples(doc, cached, epoch, seed=42):
    pp = [(s, e) for s, e, _ in cached["pairs"]]
    confidence = np.asarray(cached["probabilities"]).max(1)
    results = []
    for ai, a in enumerate(doc["annotations"]):
        positive, target = answer_pairs(a)
        gold = set(positive)
        eligible = [i for i, p in enumerate(pp) if p not in gold]
        assert len(eligible) >= 64
        ordered = sorted(eligible, key=lambda i: (-confidence[i], pp[i]))
        hard = ordered[:32]
        entropy = hashlib.sha256(f"{seed}|{doc['doc_id']}|{ai}|{epoch}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(entropy[:8], "little"))
        random = rng.choice(ordered[32:], 32, replace=False).tolist()
        negatives = [pp[i] for i in hard + random]
        results.append(dict(positive=positive, target=target, negative=negatives))
    return results


def answer_loss(positive_logits, target, negative_logits):
    neg = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    if not len(positive_logits):
        return neg
    pos = F.binary_cross_entropy_with_logits(positive_logits, target)
    return 0.5 * pos + 0.5 * neg


def training_loss(model, item, answers):
    features, _, prefix = model.encode(item)
    losses = []
    for a in answers:
        p = model.span_logits(features, prefix, a["positive"])
        n = model.span_logits(features, prefix, a["negative"])
        target = torch.as_tensor(a["target"], device=p.device)
        losses.append(answer_loss(p, target, n))
    return torch.stack(losses).mean()


def candidate_diagnostics(docs, outputs, alpha, threshold):
    records = []
    excluded = []
    for d, o in zip(docs, outputs, strict=True):
        assert d["doc_id"] == o["doc_id"]
        pairs = [(s, e) for s, e, _ in o["pairs"]]
        probs = np.clip(np.asarray(o["probabilities"]), 1e-7, 1 - 1e-7)
        scores = (
            np.log(probs / (1 - probs)) + alpha * np.asarray([v for _, _, v in o["pairs"]])[:, None]
        )
        accepted = scores.max(1) >= threshold
        individual = []
        for ai, a in enumerate(d["annotations"]):
            gold = {(s["start"], s["end"], s["label"]) for s in a["l3_persuasion"]}
            y = np.array([[(s, e, f) in gold for f in FAMILIES] for s, e in pairs])
            pos = y.any(1)
            neg = ~pos
            eligible = bool(y.any())
            ap = float(average_precision_score(y.ravel(), scores.ravel())) if eligible else None
            if not eligible:
                excluded.append(
                    dict(
                        doc_id=d["doc_id"],
                        annotation_index=ai,
                        annotator_id=a["annotator_id"],
                        observed_gold_count=len(gold),
                    )
                )
            individual.append(
                dict(
                    annotation_index=ai,
                    annotator_id=a["annotator_id"],
                    candidate_positive_typed=int(y.sum()),
                    candidate_positive_pairs=int(pos.sum()),
                    ap=ap,
                    negative_pair_acceptance=float(accepted[neg].mean()) if neg.any() else None,
                    positive_pair_retention=float(accepted[pos].mean()) if pos.any() else None,
                    positive_typed_retention=float((scores[y] >= threshold).mean())
                    if eligible
                    else None,
                    accepted_pairs_before_cap=int(accepted.sum()),
                )
            )
        record = dict(doc_id=d["doc_id"], individuals=individual)
        for metric in [
            "ap",
            "negative_pair_acceptance",
            "positive_pair_retention",
            "positive_typed_retention",
            "accepted_pairs_before_cap",
        ]:
            vals = [a[metric] for a in individual if a[metric] is not None]
            record[metric] = float(np.mean(vals)) if vals else None
        records.append(record)
    result = {}
    for metric in [
        "ap",
        "negative_pair_acceptance",
        "positive_pair_retention",
        "positive_typed_retention",
        "accepted_pairs_before_cap",
    ]:
        vals = [r[metric] for r in records if r[metric] is not None]
        result[metric] = float(np.mean(vals)) if vals else None
    result.update(
        ap_eligible_answers=sum(a["ap"] is not None for r in records for a in r["individuals"]),
        ap_eligible_articles=sum(r["ap"] is not None for r in records),
        ap_excluded_answers=excluded,
        articles=records,
    )
    return result

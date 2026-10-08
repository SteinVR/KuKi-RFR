"""Conditional multi-label family classifier: spans supplied explicitly, no localization."""

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from .data import FAMILIES
from .l3 import L3Model, span_key


def answer_pairs(answer):
    mapping = {}
    for span in answer["l3_persuasion"]:
        s, e, label = span_key(span)
        mapping.setdefault((s, e), np.zeros(len(FAMILIES), dtype=np.float32))[label] = 1
    pairs = sorted(mapping)
    return pairs, np.asarray([mapping[p] for p in pairs], dtype=np.float32).reshape(
        -1, len(FAMILIES)
    )


def examples(docs):
    rows = []
    for d in docs:
        nonempty = [a for a in d["annotations"] if a["l3_persuasion"]]
        for ai, a in enumerate(d["annotations"]):
            pairs, target = answer_pairs(a)
            for pair, y in zip(pairs, target, strict=True):
                s, e = pair
                rows.append(
                    dict(
                        doc_id=d["doc_id"],
                        annotator_id=a["annotator_id"],
                        annotation_index=ai,
                        start=s,
                        end=e,
                        text=d["text"][s:e]
                        + " [CONTEXT] "
                        + d["text"][max(0, s - 200) : min(len(d["text"]), e + 200)],
                        target=y,
                        weight=1 / (len(nonempty) * len(pairs)),
                    )
                )
    return rows


def score(rows, prob, threshold):
    y = np.asarray([r["target"] for r in rows])
    weights = np.array([r["weight"] for r in rows])
    pred = prob >= threshold
    pred[np.arange(len(pred)), np.argmax(prob, axis=1)] = True
    by_answer = {}
    by_doc = {}
    for r, g, p in zip(rows, y, pred, strict=True):
        key = (r["doc_id"], r["annotation_index"])
        counts = by_answer.setdefault(key, np.zeros(3))
        counts += np.array([(g * p).sum(), p.sum(), g.sum()])
    for (doc, _), (tp, npred, ngold) in by_answer.items():
        by_doc.setdefault(doc, []).append(2 * tp / (npred + ngold))
    article = {d: float(np.mean(v)) for d, v in by_doc.items()}
    return dict(
        article_mean_f1=float(np.mean(list(article.values()))),
        macro_family_f1=float(
            f1_score(y, pred, average="macro", sample_weight=weights, zero_division=0)
        ),
        per_family_f1=dict(
            zip(
                FAMILIES,
                f1_score(y, pred, average=None, sample_weight=weights, zero_division=0).tolist(),
                strict=True,
            )
        ),
        micro_f1=float(f1_score(y, pred, average="micro", sample_weight=weights, zero_division=0)),
        brier=float(np.average(np.mean((prob - y) ** 2, axis=1), weights=weights)),
        threshold=float(threshold),
        n_articles=len(article),
        n_answers=len(by_answer),
        n_spans=len(rows),
        family_support=dict(zip(FAMILIES, y.sum(0).astype(int).tolist(), strict=True)),
        article_f1=article,
    )


def select_threshold(rows, prob):
    results = [score(rows, prob, t) for t in np.linspace(0.1, 0.9, 17)]
    return max(results, key=lambda r: (r["article_mean_f1"], r["macro_family_f1"], r["threshold"]))


class SpanClassifier(L3Model):
    def __init__(self, encoder, cfg):
        super().__init__(encoder, cfg)
        self.head["boundary"].requires_grad_(False)

    def training_loss(self, item, annotations):
        features, _, prefix = self.encode(item)
        losses = []
        for answer in annotations:
            pairs, target = answer_pairs(answer)
            if not pairs:
                continue
            logits = self.span_logits(features, prefix, pairs)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits, torch.tensor(target, device=logits.device)
                )
            )
        if not losses:
            raise ValueError("Conditional classifier cannot train on an empty answer alone")
        loss = torch.stack(losses).mean()
        return loss, dict(loss=float(loss.detach()))

    def probabilities(self, item, pairs):
        features, _, prefix = self.encode(item)
        return self.span_logits(features, prefix, pairs).sigmoid().cpu().numpy()

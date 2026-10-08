"""Conditional three-role classification with individual answers and exact input bounds."""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F

from .data import ROLES
from .l3 import L3Model


def role_pairs(answer):
    pairs = {}
    for span in answer["l1_roles"]:
        key = (span["start"], span["end"])
        label = ROLES.index(span["label"])
        if key in pairs and pairs[key] != label:
            raise ValueError("Conflicting within-answer roles must be masked by ST0")
        pairs[key] = label
    return sorted(pairs), [pairs[p] for p in sorted(pairs)]


def examples(docs):
    rows = []
    for doc in docs:
        active = sum(bool(a["l1_roles"]) for a in doc["annotations"])
        for ai, answer in enumerate(doc["annotations"]):
            pairs, labels = role_pairs(answer)
            for (start, end), target in zip(pairs, labels, strict=True):
                assert 0 <= start < end <= len(doc["text"])
                rows.append(dict(
                    doc_id=doc["doc_id"], group_id=doc["group_id"], source=doc["source"],
                    annotator_id=answer["annotator_id"], annotation_index=ai,
                    start=start, end=end, mention=doc["text"][start:end], target=target,
                    weight=1 / (active * len(pairs)),
                    text=doc["text"][start:end] + " [CONTEXT] "
                    + doc["text"][max(0, start - 200):min(len(doc["text"]), end + 200)],
                ))
    return rows


def score(rows, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if not rows or probabilities.shape != (len(rows), 3):
        raise ValueError("Need nonempty rows and three probabilities per mention")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
        raise ValueError("Invalid categorical probabilities")
    y = np.array([r["target"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    pred = probabilities.argmax(1)  # Fixed tie order: ROLES.
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(3), sample_weight=w, zero_division=0
    )
    return dict(
        macro_f1=float(f1.mean()),
        mention_macro_f1=float(f1_score(y, pred, labels=range(3), average="macro",
                                      zero_division=0)),
        article_accuracy=float(np.average(pred == y, weights=w)),
        mention_accuracy=float(np.mean(pred == y)),
        cross_entropy=float(np.average(-np.log(np.maximum(
            probabilities[np.arange(len(y)), y], 1e-12)), weights=w)),
        brier=float(np.average(((probabilities - np.eye(3)[y]) ** 2).mean(1), weights=w)),
        per_role={label: dict(precision=float(precision[i]), recall=float(recall[i]),
                             f1=float(f1[i]), weighted_support=float(support[i]),
                             mentions=int((y == i).sum())) for i, label in enumerate(ROLES)},
        confusion=confusion_matrix(y, pred, labels=range(3)).tolist(),
        weighted_confusion=confusion_matrix(y, pred, labels=range(3), sample_weight=w).tolist(),
        n_articles=len({r["doc_id"] for r in rows}),
        n_answers=len({(r["doc_id"], r["annotation_index"]) for r in rows}),
        n_mentions=len(rows), role_order=ROLES,
    )


def conditional_loss(logits, pairs, annotations):
    index = {p: i for i, p in enumerate(pairs)}
    losses = []
    for answer in annotations:
        ap, labels = role_pairs(answer)
        if ap:
            losses.append(F.cross_entropy(
                logits[[index[p] for p in ap]],
                torch.tensor(labels, device=logits.device),
            ))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


class RoleClassifier(L3Model):
    """Fresh base/LoRA; reuse checked character geometry, never L3 weights or labels."""

    def __init__(self, encoder, cfg):
        super().__init__(encoder, cfg)
        self.head["boundary"].requires_grad_(False)  # Unused inherited boundary proposal layer.
        dim = cfg["span_dimension"]
        self.head["span"] = nn.Sequential(
            nn.Linear(dim * 4 + 1, dim), nn.GELU(), nn.Dropout(cfg["head_dropout"]),
            nn.Linear(dim, len(ROLES)),
        )

    def role_logits(self, item, pairs):
        features, _, prefix = self.encode(item)
        bounds = torch.tensor(pairs, device=features.device, dtype=torch.long).reshape(-1, 2)
        start, end = bounds.T
        length = (end - start).float()
        mean = (prefix[end] - prefix[start]) / length[:, None]
        context = (prefix[-1] / item["length"]).expand(len(pairs), -1)
        return self.head["span"](torch.cat([
            features[start], features[end], mean, context, torch.log1p(length[:, None]) / 10,
        ], dim=1))

    def training_loss(self, item, annotations):
        pairs = sorted({p for a in annotations for p in role_pairs(a)[0]})
        if not pairs:
            raise ValueError("No observed role: mask this article before standalone training")
        logits = self.role_logits(item, pairs)
        loss = conditional_loss(logits, pairs, annotations)
        return loss, dict(loss=float(loss.detach()))


def predict(model, docs, encoded, rows):
    output = {}
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            # Positions are task inputs; no roles or annotator IDs enter the model.
            pairs = sorted({(r["start"], r["end"]) for r in rows if r["doc_id"] == doc["doc_id"]})
            if pairs:
                values = model.role_logits(encoded[doc["doc_id"]], pairs).softmax(-1).cpu().numpy()
                output.update({(doc["doc_id"], *p): v for p, v in zip(pairs, values, strict=True)})
    return np.asarray([output[r["doc_id"], r["start"], r["end"]] for r in rows])

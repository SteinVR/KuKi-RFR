"""Article-balanced L2 evaluation against individual answers and strict majority."""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .data import FRAMES, write_jsonl


def votes(docs):
    return np.asarray([np.mean([a['l2'] for a in d['annotations']], axis=0) for d in docs])


def article_brier(docs, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(docs), len(FRAMES)) or not np.isfinite(p).all():
        raise ValueError('Invalid probability shape or non-finite values')
    if (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities outside [0, 1]')
    return np.asarray([
        np.mean((np.asarray([a['l2'] for a in d['annotations']]) - row) ** 2, axis=0)
        for d, row in zip(docs, p, strict=True)
    ])


def score(docs, probabilities, threshold=0.5):
    p = np.asarray(probabilities)
    fractions = votes(docs)
    target = fractions > 0.5
    predicted = p >= threshold
    brier = article_brier(docs, p)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, average=None, zero_division=0
    )
    return dict(
        n_articles=len(docs), n_answers=sum(len(d['annotations']) for d in docs),
        brier=float(brier.mean()),
        vote_fraction_mse=float(np.mean((p - fractions) ** 2)),
        observed_disagreement_floor=float(np.mean(fractions * (1 - fractions))),
        macro_f1=float(f1_score(target, predicted, average='macro', zero_division=0)),
        micro_f1=float(f1_score(target, predicted, average='micro', zero_division=0)),
        threshold=threshold, f1_reference='strict_majority_votes_gt_half_empty_allowed',
        per_frame=[dict(frame=name, brier=float(brier[:, i].mean()),
                        precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
                        majority_support=int(support[i]), predicted_positive=int(predicted[:, i].sum()),
                        positive_vote_mass=float(fractions[:, i].sum()),
                        ties=int((fractions[:, i] == 0.5).sum()))
                   for i, name in enumerate(FRAMES)],
    )


def save_predictions(path, docs, probabilities, split, threshold=0.5):
    brier = article_brier(docs, probabilities)
    write_jsonl(path, [dict(
        doc_id=d['doc_id'], group_id=d['group_id'], split=split,
        probabilities=list(map(float, row)), vote_fractions=fractions.tolist(),
        n_annotators=len(d['annotations']),
        majority_labels=[f for f, v in zip(FRAMES, fractions, strict=True) if v > 0.5],
        predicted_labels=[f for f, v in zip(FRAMES, row, strict=True) if v >= threshold],
        brier=float(error.mean()),
    ) for d, row, fractions, error in zip(docs, probabilities, votes(docs), brier, strict=True)])

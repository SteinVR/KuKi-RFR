"""Character-exact boundary proposals and unrestricted-length span classification."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import FAMILIES
from .l2 import L2Model


def span_key(span):
    return (span["start"], span["end"], FAMILIES.index(span["label"]))


def prepare_l3(encoded, text):
    starts, ends = np.asarray(encoded["offsets"]).T
    positions = np.arange(len(text) + 1)
    # Adjacent contextual token representations, including token-interior and
    # whitespace positions. No trimming, snapping, sentence or window cap.
    left = np.clip(np.searchsorted(starts, positions, side="left") - 1, 0, len(starts) - 1)
    right = np.clip(np.searchsorted(ends, positions, side="right"), 0, len(ends) - 1)
    geometry = np.stack(
        [
            positions - starts[left],
            ends[left] - positions,
            positions - starts[right],
            ends[right] - positions,
        ],
        axis=1,
    )
    geometry = np.clip(geometry / 32, -1, 1).astype("float32")
    chars = np.array([0] + [ord(c) % 2048 + 1 for c in text] + [0])
    return dict(
        encoded=encoded,
        length=len(text),
        left=left.tolist(),
        right=right.tolist(),
        geometry=geometry.tolist(),
        before=chars[:-1].tolist(),
        after=chars[1:].tolist(),
    )


def balanced_bce(logits, target, negative_weight=1.0):
    """Intentional class-balanced risk, not an estimate of full-universe BCE."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = target.bool()
    negative = ~positive
    parts, weights = [], []
    if positive.any():
        parts.append(loss[positive].mean())
        weights.append(1.0)
    if negative.any():
        parts.append(loss[negative].mean() * negative_weight)
        weights.append(negative_weight)
    if not parts:
        return logits.sum() * 0
    return sum(parts) / sum(weights)


def propose(boundary_logits, k, length):
    # Stable CPU sort provides an explicit tie rule (lower offset first).
    scores = boundary_logits.detach().float().cpu().numpy()
    starts = np.argsort(-scores[:-1, 0], kind="stable")[:k].tolist()
    ends = (np.argsort(-scores[1:, 1], kind="stable")[:k] + 1).tolist()
    assert all(0 <= s < length for s in starts) and all(0 < e <= length for e in ends)
    return sorted(starts), sorted(ends)


def candidate_pairs(starts, ends):
    return [(s, e) for s in starts for e in ends if s < e]


class L3Model(nn.Module):
    def __init__(self, encoder_config, cfg):
        super().__init__()
        base = L2Model(encoder_config)  # Reuses only pinned base + fresh LoRA; no ST1 weights.
        self.encoder = base.encoder
        hidden = base.head.in_features
        dim = cfg["span_dimension"]
        self.head = nn.ModuleDict(
            {
                "token": nn.Linear(hidden, dim),
                "char": nn.Embedding(2049, 8),
                "boundary_features": nn.Sequential(nn.Linear(dim * 2 + 20, dim), nn.GELU()),
                "boundary": nn.Linear(dim, 2),
                "span": nn.Sequential(
                    nn.Linear(dim * 3 + 1, dim),
                    nn.GELU(),
                    nn.Dropout(cfg["head_dropout"]),
                    nn.Linear(dim, len(FAMILIES)),
                ),
            }
        )
        self.cfg = cfg

    def encode(self, item):
        device = next(self.head.parameters()).device
        enc = item["encoded"]
        tokens = None
        for window in enc["windows"]:
            ids = torch.tensor([window["input_ids"]], device=device)
            h = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[
                0
            ]
            weights = torch.tensor(window["weights"], device=device)
            valid = weights > 0
            projected = self.head["token"](h[valid].float()) * weights[valid, None]
            padded = F.pad(projected, (0, 0, window["start"], enc["tokens"] - window["end"]))
            tokens = padded if tokens is None else tokens + padded
        left = torch.tensor(item["left"], device=device)
        right = torch.tensor(item["right"], device=device)
        features = self.head["boundary_features"](
            torch.cat(
                [
                    tokens[left],
                    tokens[right],
                    self.head["char"](torch.tensor(item["before"], device=device)),
                    self.head["char"](torch.tensor(item["after"], device=device)),
                    torch.tensor(item["geometry"], device=device),
                ],
                dim=1,
            )
        )
        boundary = self.head["boundary"](features)
        # Prefix mean of character-context representations retains interior
        # evidence and permits spans crossing arbitrary encoder windows.
        interior = tokens[right[:-1]]
        prefix = F.pad(interior.cumsum(0), (0, 0, 1, 0))
        return features, boundary, prefix

    def span_logits(self, features, prefix, pairs):
        indices = torch.tensor(pairs, device=features.device, dtype=torch.long).reshape(-1, 2)
        start, end = indices.T
        length = (end - start).float()
        mean = (prefix[end] - prefix[start]) / length[:, None]
        return self.head["span"](
            torch.cat(
                [
                    features[start],
                    features[end],
                    mean,
                    torch.log1p(length[:, None]) / 10,
                ],
                dim=1,
            )
        )

    def training_loss(self, item, annotations):
        features, boundary, prefix = self.encode(item)
        starts, ends = propose(boundary, self.cfg["proposal_top_k"], item["length"])
        raw_pairs = set(candidate_pairs(starts, ends))
        # Training augmentation only; deployment proposals never see annotations.
        gold = {span_key(s) for a in annotations for s in a["l3_persuasion"]}
        starts = sorted(set(starts) | {s for s, _, _ in gold})
        ends = sorted(set(ends) | {e for _, e, _ in gold})
        pairs = candidate_pairs(starts, ends)
        pair_index = {p: i for i, p in enumerate(pairs)}
        logits = self.span_logits(features, prefix, pairs)
        boundary_losses, pair_losses = [], []
        positives = []
        pos_bce, neg_bce = [], []
        hard_diagnostics = []
        for annotation in annotations:
            target_boundary = torch.zeros_like(boundary)
            target = torch.zeros_like(logits)
            for s in annotation["l3_persuasion"]:
                start, end, label = span_key(s)
                target_boundary[start, 0] = 1
                target_boundary[end, 1] = 1
                target[pair_index[(start, end)], label] = 1
            # Omit invalid start at len(text) and invalid end at zero.
            valid_boundary = torch.cat([boundary[:-1, 0], boundary[1:, 1]])
            valid_target = torch.cat([target_boundary[:-1, 0], target_boundary[1:, 1]])
            boundary_losses.append(
                balanced_bce(valid_boundary, valid_target, self.cfg["negative_weight"])
            )
            if self.cfg.get("pair_negative_policy") == "hard_top_max64_8P":
                from .l3_losses import hard_negative_bce

                objective, hard_detail = hard_negative_bce(
                    logits,
                    target,
                    self.cfg["hard_negative_ratio"],
                    self.cfg["hard_negative_minimum"],
                    self.cfg["negative_weight"],
                )
                pair_losses.append(objective)
                hard_diagnostics.append(hard_detail)
            else:
                pair_losses.append(balanced_bce(logits, target, self.cfg["negative_weight"]))
            loss_elements = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            ).detach()
            mask = target.bool()
            positives.append(int(mask.sum()))
            pos_bce.append(float(loss_elements[mask].mean()) if mask.any() else None)
            neg_bce.append(float(loss_elements[~mask].mean()) if (~mask).any() else None)
        boundary_loss = torch.stack(boundary_losses).mean()
        pair_loss = torch.stack(pair_losses).mean()
        loss = self.cfg["boundary_loss_weight"] * boundary_loss + pair_loss
        diagnostic = dict(
            loss=float(loss.detach()),
            boundary_loss=float(boundary_loss.detach()),
            pair_loss=float(pair_loss.detach()),
            n_candidates=len(pairs),
            positive_decisions=positives,
            negative_decisions=[len(pairs) * len(FAMILIES) - n for n in positives],
            positive_bce=pos_bce,
            negative_bce=neg_bce,
            raw_proposal_gold_hits=sum((s, e) in raw_pairs for s, e, _ in gold),
            gold_union_size=len(gold),
        )
        if hard_diagnostics:
            diagnostic["hard_negatives"] = hard_diagnostics
        return loss, diagnostic

    def infer(self, item):
        # Intentionally no annotation argument.
        features, boundary, prefix = self.encode(item)
        starts, ends = propose(boundary, self.cfg["proposal_top_k"], item["length"])
        pairs = candidate_pairs(starts, ends)
        scores = self.span_logits(features, prefix, pairs).sigmoid().float().cpu().numpy()
        # Keep the finite prediction budget explicit. No nonmaximum suppression:
        # overlapping, nested, or identically bounded distinct families survive.
        eligible = np.flatnonzero(scores.ravel() >= min(self.cfg["threshold_grid"]))
        ordered = eligible[np.argsort(-scores.ravel()[eligible], kind="stable")]
        retained = ordered[: self.cfg["max_predictions"]]
        spans = [
            dict(
                start=pairs[i // len(FAMILIES)][0],
                end=pairs[i // len(FAMILIES)][1],
                label=FAMILIES[i % len(FAMILIES)],
                score=float(scores.ravel()[i]),
            )
            for i in retained
        ]
        return dict(
            spans=spans,
            starts=starts,
            ends=ends,
            candidate_count=len(pairs),
            eligible_before_cap=len(eligible),
        )

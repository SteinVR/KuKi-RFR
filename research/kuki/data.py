"""Lossless text and explicit annotation policy for KuKi RU."""

import json
import re
from collections import Counter, defaultdict
from hashlib import sha256

FRAMES = [
    "Economic",
    "Capacity & resources",
    "Morality",
    "Fairness & equality",
    "Legality, constitutionality, jurisprudence",
    "Policy prescription & evaluation",
    "Crime & punishment",
    "Security & defense",
    "Health & safety",
    "Quality of life",
    "Cultural identity",
    "Public opinion",
    "Political",
    "External regulation & reputation",
]
ROLES = ["PROTAGONIST", "ANTAGONIST", "INNOCENT"]
FAMILIES = [
    "Attack on Reputation",
    "Justification",
    "Simplification",
    "Distraction",
    "Call",
    "Manipulative Wording",
]
LABELS = {"l1_roles": ROLES, "l3_persuasion": FAMILIES, "l4_coded": ["Coded Phrase"]}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def normalize_for_dedup(text):
    # This is a comparison key ONLY, never model input or offset space.
    return re.sub(r"\s+", " ", text).strip().casefold()


def prepare_document(raw):
    text = raw["content"]
    assert raw["lang"] == "ru" and text
    assert raw["n_annotators"] == len(raw["annotations"])
    assert set(map(str, raw["annotators"])) == set(raw["annotations"])
    result = {k: raw[k] for k in ["doc_id", "lang", "source", "block", "published_at", "title"]}
    result.update(text=text, text_sha256=sha256(text.encode()).hexdigest(), annotations=[])
    events = []
    for aid, annotation in sorted(raw["annotations"].items()):

        def event(kind, aid=aid, **details):
            events.append(dict(doc_id=raw["doc_id"], annotator_id=aid, kind=kind, **details))

        frames = annotation["frames"]
        if set(frames) - set(FRAMES):
            raise ValueError(f"Unknown frame in {raw['doc_id']}: {frames}")
        if not frames:
            event("empty_l2_observed_negative", policy="retain_all_zero_answer")
        if len(frames) != len(set(frames)):
            event("duplicate_frame", policy="set_membership")
        layers = {layer: [] for layer in LABELS}
        seen = {}
        for index, span in enumerate(annotation["spans"]):
            if not (0 <= span["start"] < span["end"] <= len(text)):
                raise ValueError(f"Invalid span bounds: {raw['doc_id']}, {aid}, {index}")
            if text[span["start"] : span["end"]] != span["text"]:
                raise ValueError(f"Span text mismatch: {raw['doc_id']}, {aid}, {index}")
            if span["label"] not in LABELS.get(span["layer"], []):
                event("invalid_layer_label", span_index=index, span=span, policy="quarantine")
                continue
            key = (span["layer"], span["label"], span["start"], span["end"])
            if key in seen:
                seen[key]["source_span_indices"].append(index)
                event(
                    "duplicate_span",
                    span_index=index,
                    retained_index=seen[key]["source_span_indices"][0],
                )
                continue
            entry = dict(span, source_span_indices=[index])
            seen[key] = entry
            layers[span["layer"]].append(entry)
        roles = defaultdict(list)
        for span in layers["l1_roles"]:
            roles[span["start"], span["end"]].append(span)
        excluded = []
        for bounds, spans in roles.items():
            if len({s["label"] for s in spans}) > 1:
                excluded.extend(spans)
                event(
                    "conflicting_l1_roles",
                    start=bounds[0],
                    end=bounds[1],
                    spans=spans,
                    policy="mask_this_mention_for_this_annotator",
                )
        layers["l1_roles"] = [s for s in layers["l1_roles"] if s not in excluded]
        result["annotations"].append(
            dict(
                annotator_id=aid,
                source_metadata={
                    k: v for k, v in annotation.items() if k not in ["frames", "spans"]
                },
                l2=[int(f in frames) for f in FRAMES],
                **layers,
            )
        )
    return result, events


def label_features(doc):
    features = {f"source:{doc['source']}"}
    for a in doc["annotations"]:
        features.add(f"annotator:{a['annotator_id']}")
        features.update("l2:" + f for f, v in zip(FRAMES, a["l2"], strict=True) if v)
        for layer in ["l1_roles", "l3_persuasion"]:
            features.update(layer + ":" + s["label"] for s in a[layer])
    return features


def group_duplicates(docs, threshold=0.8, must_link=()):
    """Conservative connected components of exact keys or word 5-gram similarity.

    Does not claim to recover event groups or paraphrased copies.
    """
    parent = list(range(len(docs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = [normalize_for_dedup(d["text"]) for d in docs]
    shingles = []
    for text in keys:
        words = re.findall(r"\w+", text)
        shingles.append(set(zip(*(words[i:] for i in range(5)), strict=False)))
    pairs, nearest = [], []
    for i in range(len(docs)):
        for j in range(i):
            a, b = shingles[i], shingles[j]
            intersection = len(a & b)
            jac = intersection / max(1, len(a | b))
            containment = intersection / max(1, min(len(a), len(b)))
            # Containment catches excerpts only when at least 100 shingles match.
            grouped = (
                keys[i] == keys[j]
                or jac >= threshold
                or (containment >= 0.9 and intersection >= 100)
            )
            item = dict(
                left=docs[j]["doc_id"],
                right=docs[i]["doc_id"],
                jaccard=jac,
                containment=containment,
                grouped=grouped,
            )
            nearest.append(item)
            if grouped:
                parent[root(i)] = root(j)
                pairs.append(item)
    indices = {d["doc_id"]: i for i, d in enumerate(docs)}
    for left, right in must_link:
        parent[root(indices[left])] = root(indices[right])
        pairs.append(
            dict(left=left, right=right, grouped=True, reason="audited_shared_report_excerpt")
        )
    groups = defaultdict(list)
    for i, doc in enumerate(docs):
        groups[root(i)].append(doc["doc_id"])
    groups = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    return groups, pairs, sorted(nearest, key=lambda x: x["jaccard"], reverse=True)[:20]


def split_groups(docs, groups, seed=42):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    by_id = {d["doc_id"]: d for d in docs}
    features = [set.union(*(label_features(by_id[i]) for i in g)) for g in groups]
    names = sorted(set.union(*features))
    y = np.array([[int(n in fs) for n in names] for fs in features])
    # One prescribed draw; never select among seeds or fit models to choose a split.
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, hold = next(first.split(np.zeros(len(groups)), y))
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    dev, test = next(second.split(np.zeros(len(hold)), y[hold]))
    assignment = {}
    for name, indices in [("train", train), ("dev", hold[dev]), ("test", hold[test])]:
        for i in indices:
            for doc_id in groups[i]:
                assignment[doc_id] = (name, groups[i][0])
    return assignment


def distributions(docs, assignment):
    result = {}
    for split in ["train", "dev", "test"]:
        subset = [d for d in docs if assignment[d["doc_id"]][0] == split]
        features = Counter(f for d in subset for f in label_features(d))
        result[split] = dict(
            documents=len(subset),
            annotations=sum(len(d["annotations"]) for d in subset),
            document_presence=dict(sorted(features.items())),
            annotation_counts=dict(
                Counter(a["annotator_id"] for d in subset for a in d["annotations"])
            ),
        )
    return result

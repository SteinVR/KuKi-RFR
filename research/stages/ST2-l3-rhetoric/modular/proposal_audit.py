"""E2: fixed candidate generators, exact/IoU oracle coverage and measured cost."""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from razdel import sentenize, tokenize

from research.kuki.data import write_json
from research.kuki.experiment import load_development

ROOT = Path(__file__).parent
METHODS = [
    "sentences",
    "clauses_sentences",
    "ngrams8",
    "ngrams32",
    "multiscale",
    "all_word_spans_analytic",
]


def candidates(text, method):
    tokens = list(tokenize(text))
    sents = list(sentenize(text))
    if method == "all_word_spans_analytic":
        return tokens, None
    result = set()
    if method in ["sentences", "clauses_sentences", "multiscale"]:
        result |= {(s.start, s.stop) for s in sents}
    if method in ["clauses_sentences", "multiscale"]:
        for sent in sents:
            parts = []
            start = sent.start
            for match in re.finditer(r"[,;:!?—–]|\n", text[sent.start : sent.stop]):
                end = sent.start + match.end()
                parts.append((start, end))
                start = end
            parts.append((start, sent.stop))
            for i, (a, b) in enumerate(parts):
                while a < b and text[a].isspace():
                    a += 1
                while b > a and text[b - 1].isspace():
                    b -= 1
                if a < b:
                    result.add((a, b))
                if method == "multiscale":
                    for j in [i + 1, i + 2]:
                        if j < len(parts):
                            result.add((a, parts[j][1]))
    if method == "multiscale":
        result |= {(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text) if m.group().strip()}
        for a, b in zip(sents, sents[1:], strict=False):
            result.add((a.start, b.stop))
    limit = {"ngrams8": 8, "ngrams32": 32, "multiscale": 8}.get(method, 0)
    for i, t in enumerate(tokens):
        for j in range(i, min(len(tokens), i + limit)):
            result.add((t.start, tokens[j].stop))
    return tokens, np.asarray(sorted(result), dtype=np.int64).reshape(-1, 2)


def measure(doc, method):
    begin = time.perf_counter()
    tokens, pairs = candidates(doc["text"], method)
    seconds = time.perf_counter() - begin
    starts = {t.start for t in tokens}
    ends = {t.stop for t in tokens}
    count = len(tokens) * (len(tokens) + 1) // 2 if pairs is None else len(pairs)
    pset = set() if pairs is None else set(map(tuple, pairs))
    answers = []
    misses = []
    for ai, ann in enumerate(doc["annotations"]):
        spans = ann["l3_persuasion"]
        values = []
        for span in spans:
            s, e = span["start"], span["end"]
            hit = s in starts and e in ends if pairs is None else (s, e) in pset
            if pairs is not None:
                overlap = pairs[(pairs[:, 0] < e) & (pairs[:, 1] > s)]
                iou = (
                    (np.minimum(overlap[:, 1], e) - np.maximum(overlap[:, 0], s))
                    / (np.maximum(overlap[:, 1], e) - np.minimum(overlap[:, 0], s))
                    if len(overlap)
                    else np.array([0.0])
                )
                best = float(iou.max())
            else:
                # Only nearest starts/ends can maximize interval IoU on the unconstrained word lattice.
                ss = sorted(starts)
                ee = sorted(ends)
                si = np.searchsorted(ss, s)
                ei = np.searchsorted(ee, e)
                possible = [
                    (ss[i], ee[j])
                    for i in [si - 1, si]
                    for j in [ei - 1, ei]
                    if 0 <= i < len(ss) and 0 <= j < len(ee) and ss[i] < ee[j]
                ]
                best = max(
                    [max(0, min(b, e) - max(a, s)) / (max(b, e) - min(a, s)) for a, b in possible],
                    default=0.0,
                )
            values.append((hit, best >= 0.5, best))
            if not hit:
                misses.append(
                    dict(
                        annotation_index=ai,
                        label=span["label"],
                        start=s,
                        end=e,
                        length=e - s,
                        best_iou=best,
                        word_aligned=s in starts and e in ends,
                    )
                )
        if values:
            answers.append(np.mean(values, axis=0).tolist())
    return dict(
        doc_id=doc["doc_id"],
        characters=len(doc["text"]),
        words=len(tokens),
        candidate_count=count,
        generation_seconds=seconds,
        exact_recall=float(np.mean([a[0] for a in answers])) if answers else None,
        iou50_coverage=float(np.mean([a[1] for a in answers])) if answers else None,
        mean_best_iou=float(np.mean([a[2] for a in answers])) if answers else None,
        misses=misses,
    )


def main():
    begin = time.perf_counter()
    out = ROOT / "E2-proposals"
    out.mkdir(exist_ok=True)
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    result = {}
    for method in METHODS:
        result[method] = {}
        for split, part in docs.items():
            records = [measure(d, method) for d in part]
            numbers = np.array([r["candidate_count"] for r in records])
            summary = dict(
                n_articles=len(part),
                exact_recall=float(
                    np.mean([r["exact_recall"] for r in records if r["exact_recall"] is not None])
                ),
                iou50_coverage=float(
                    np.mean(
                        [r["iou50_coverage"] for r in records if r["iou50_coverage"] is not None]
                    )
                ),
                mean_best_iou=float(
                    np.mean([r["mean_best_iou"] for r in records if r["mean_best_iou"] is not None])
                ),
                candidate_count_quantiles=np.quantile(numbers, [0, 0.5, 0.9, 1]).tolist(),
                candidate_count_total=int(numbers.sum()),
                generation_seconds=sum(r["generation_seconds"] for r in records),
                empty_gold_articles=sum(r["exact_recall"] is None for r in records),
            )
            result[method][split] = summary
            write_json(out / f"{method}-{split}.json", records)
            print(json.dumps(dict(method=method, split=split, **summary)), flush=True)
    write_json(
        out / "metrics.json",
        dict(
            results=result,
            elapsed_seconds=time.perf_counter() - begin,
            data_hashes=hashes,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            test_used=False,
            overlap_note="oracle coverage, not one-to-one prediction F1; multiple gold spans may share one candidate",
        ),
    )


if __name__ == "__main__":
    main()

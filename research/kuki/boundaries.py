"""Audit representation requirements; does not pretend to implement the L3 decoder."""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from .data import digest, write_json
from .windows import tokenize_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.model_config.read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    docs = [json.loads(s) for s in args.data.read_text().splitlines()]
    counts = Counter()
    lengths, by_length = [], {str(n): Counter() for n in cfg["window_lengths"]}
    for d in docs:
        encodings = {
            n: tokenize_document(tok, d["text"], n, cfg["overlap"]) for n in cfg["window_lengths"]
        }
        first = next(iter(encodings.values()))
        starts = {a for a, b in first["offsets"]}
        ends = {b for a, b in first["offsets"]}
        lengths.append(
            dict(
                doc_id=d["doc_id"],
                tokens=first["tokens"],
                windows={str(n): len(e["windows"]) for n, e in encodings.items()},
            )
        )
        for a in d["annotations"]:
            spans = a["l3_persuasion"]
            for i, s in enumerate(spans):
                counts["l3_spans"] += 1
                text = d["text"]
                if any(
                    0 < pos < len(text) and text[pos - 1].isalnum() and text[pos].isalnum()
                    for pos in [s["start"], s["end"]]
                ):
                    counts["l3_boundary_inside_word"] += 1
                assert d["text"][s["start"] : s["end"]] == s["text"]
                if s["start"] not in starts or s["end"] not in ends:
                    counts["l3_not_exact_token_boundary"] += 1
                for n, enc in encodings.items():
                    off = enc["offsets"]
                    if not any(
                        off[w["start"]][0] <= s["start"] and off[w["end"] - 1][1] >= s["end"]
                        for w in enc["windows"]
                    ):
                        by_length[str(n)]["l3_not_contained_in_any_window"] += 1
                for t in spans[:i]:
                    if min(s["end"], t["end"]) > max(s["start"], t["start"]):
                        counts["l3_overlapping_pairs"] += 1
                        if s["label"] == t["label"]:
                            counts["l3_same_family_overlapping_pairs"] += 1
                        elif (s["start"], s["end"]) == (t["start"], t["end"]):
                            counts["l3_same_bounds_different_family"] += 1
    write_json(
        args.output,
        dict(
            model=cfg,
            prepared_sha256=digest(args.data),
            counts=dict(counts),
            by_window_length={n: dict(c) for n, c in by_length.items()},
            documents=lengths,
            status="raw_character_spans_roundtrip_verified; token_only_or_window_local_L3_not_approved",
        ),
    )
    print(
        json.dumps(
            {
                "counts": dict(counts),
                "by_window_length": by_length,
                "max_tokens": max(x["tokens"] for x in lengths),
            }
        )
    )


if __name__ == "__main__":
    main()

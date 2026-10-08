"""Post-hoc CPU check of token-aligned proposal fractions on the first detector."""

import json
from pathlib import Path

import numpy as np
from razdel import tokenize

from research.kuki.data import digest, write_json
from research.kuki.experiment import load_development

ROOT = Path(__file__).parent


def main():
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    result = dict(data_hashes=hashes, model="20260928-detector-base-s42-01", test_used=False)
    for split in docs:
        path = ROOT.parent / f"experiments/20260928-detector-base-s42-01/{split}-predictions.jsonl"
        predictions = [json.loads(x) for x in path.read_text().splitlines()]
        records = []
        for d, p in zip(docs[split], predictions, strict=True):
            tokens = list(tokenize(d["text"]))
            starts = {t.start for t in tokens}
            ends = {t.stop for t in tokens}
            records.append(
                dict(
                    doc_id=d["doc_id"],
                    token_aligned_fraction=sum(s in starts and e in ends for s, e, _ in p["pairs"])
                    / len(p["pairs"]),
                )
            )
        result[split] = dict(
            predictions_sha256=digest(path),
            mean_token_aligned_fraction=float(
                np.mean([r["token_aligned_fraction"] for r in records])
            ),
            articles=records,
        )
    write_json(ROOT / "E3-detector/token-alignment-diagnostic.json", result)
    print({s: result[s]["mean_token_aligned_fraction"] for s in docs})


if __name__ == "__main__":
    main()

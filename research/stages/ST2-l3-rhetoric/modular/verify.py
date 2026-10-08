"""Independent CPU replay of all saved model metrics and artifact provenance."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from research.kuki.data import write_json
from research.kuki.experiment import load_development
from research.kuki.span_classifier import score

ROOT = Path(__file__).parent
train = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.train")
chain = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.chain")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    records = []
    for kind in ["classifier", "detector"]:
        for variant in ["base", "low-lr"]:
            run = ROOT.parent / "experiments" / f"20260928-{kind}-{variant}-s42-01"
            m = json.loads((run / "metrics.json").read_text())
            meta = json.loads((run / "metadata.json").read_text())
            history = json.loads((run / "history.json").read_text())
            assert m["base_unchanged"] and m["test_used"] is False
            assert history[-1]["updates"] == m["updates"]
            assert history[-1]["article_presentations"] == m["article_presentations"]
            assert meta["provenance"]["data_hashes"] == hashes
            assert digest(run / "uv.lock") == meta["provenance"]["uv_lock_sha256"]
            for p, h in meta["provenance"]["code_sha256"].items():
                assert digest(run / "code" / Path(p).name) == h
            diffs = {}
            for split in docs:
                pred = chain.read_jsonl(run / f"{split}-predictions.jsonl")
                if kind == "classifier":
                    for r in pred:
                        r["target"] = np.asarray(r["target"], dtype=np.float32)
                    mm = score(
                        pred,
                        np.asarray([r["probabilities"] for r in pred], dtype=np.float32),
                        m[split]["threshold"],
                    )
                    fields = ["article_mean_f1", "macro_family_f1", "brier"]
                else:
                    assert [p["doc_id"] for p in pred] == [d["doc_id"] for d in docs[split]]
                    for d, p in zip(docs[split], pred, strict=True):
                        assert len(p["pairs"]) == 500
                        assert len({(s, e) for s, e, _ in p["pairs"]}) == 500
                        assert all(0 <= s < e <= len(d["text"]) for s, e, _ in p["pairs"])
                    mm = train.detector_score(docs[split], pred, m[split]["threshold"])
                    fields = ["exact_f1", "overlap_f1", "mean_predictions"]
                    assert mm["recall"] == m[split]["recall"]
                diffs[split] = {f: abs(mm[f] - m[split][f]) for f in fields}
                assert max(diffs[split].values()) < 1e-12, (run.name, split, diffs[split])
            records.append(
                dict(run=run.name, metric_absolute_differences=diffs, code_snapshots_verified=True)
            )
    saved = json.loads((ROOT / "E3-detector/chain.json").read_text())
    for split in docs:
        pred = chain.read_jsonl(ROOT / "E3-detector" / f"chain-{split}-scores.jsonl")
        mm, _, _ = chain.chain_score(
            docs[split], pred, saved["selection"]["alpha"], saved["selection"]["threshold"]
        )
        assert mm == saved[split]
    extended = json.loads((ROOT / "E3-detector/calibration-extension.json").read_text())
    strict = json.loads((ROOT / "E3-detector/strict-chain.json").read_text())
    for split in docs:
        pred = chain.read_jsonl(ROOT / "E3-detector" / f"chain-{split}-scores.jsonl")
        mm, _, _ = chain.chain_score(
            docs[split], pred, extended["selection"]["alpha"], extended["selection"]["threshold"]
        )
        assert mm == extended[split]
        outputs = chain.read_jsonl(ROOT / "E3-detector" / f"strict-{split}-predictions.jsonl")
        sm, _ = chain.evaluate(docs[split], outputs, strict["detector_threshold"])
        for field in ["exact_f1", "overlap_f1", "mean_predictions", "correctly_empty_answers"]:
            assert sm[field] == strict[split][field]
    e2 = json.loads((ROOT / "E2-proposals/metrics.json").read_text())
    assert e2["data_hashes"] == hashes and e2["test_used"] is False
    assert digest(ROOT / "proposal_audit.py") == e2["code_sha256"]
    historical = json.loads((ROOT / "prior-artifact-hashes.json").read_text())
    for name, expected in historical.items():
        if name.endswith("ST2-l3-rhetoric/README.md"):
            continue  # Navigation update only; all prior research artifacts remain immutable.
        assert digest(Path(name)) == expected
    ledger = json.loads((ROOT / "budget.json").read_text())
    assert abs(sum(r["seconds"] for r in ledger["runs"]) - ledger["seconds"]) < 1e-8
    assert ledger["seconds"] <= ledger["limit_seconds"]
    write_json(
        ROOT / "verification.json",
        dict(
            passed=True,
            data_hashes=hashes,
            model_runs=records,
            chain_metrics_exactly_recomputed=True,
            post_hoc_decoders_recomputed=True,
            reloaded_model_predictions=saved["verification"],
            e2_code_hash_verified=True,
            historical_artifacts_unchanged=len(historical) - 1,
            test_used=False,
            gpu_seconds=ledger["seconds"],
            gpu_limit_seconds=ledger["limit_seconds"],
        ),
    )
    print("All saved model metrics, code snapshots, chain, hashes and budget verified.")


if __name__ == "__main__":
    main()

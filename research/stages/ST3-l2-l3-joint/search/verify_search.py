"""Independent set-based exact metrics, L2 squared error and tie-aware AP replay."""

import importlib
from pathlib import Path

import numpy as np

from research.kuki.data import FAMILIES, digest, write_json

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path("research/stages/ST3-l2-l3-joint/search")


def manual_ap(target, score):
    order = np.argsort(-score, kind="stable")
    y = np.asarray(target)[order]
    s = np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum((np.diff(np.r_[0, tp]) / y.sum()) * (tp / (ends + 1))))


def main():
    prior = M.read(ROOT / "prior-artifact-hashes.json")
    changed = [p for p, h in prior.items() if digest(Path(p)) != h]
    assert not changed, changed
    historical = M.read(ROOT.parent / "prior-artifact-hashes.json")
    assert all(digest(Path(p)) == h for p, h in historical.items())
    cfg = M.read(ROOT / "configs/joint200.json")
    docs, _, _ = M.inputs(cfg)
    checks = {}
    for name in ["joint200", "joint400"]:
        cfg = M.read(ROOT / "configs" / f"{name}.json")
        out = ROOT / "experiments" / cfg["run_id"]
        metadata = M.read(out / "metadata.json")["provenance"]
        # Verify the exact frozen source snapshot used by this run, independent of
        # analysis scripts added after training.
        for path, h in metadata["code_sha256"].items():
            assert digest(out / "code" / Path(path).name) == h
        assert metadata["data_hashes"] == M.inputs(cfg)[2]["data_hashes"]
        for path, h in metadata["detector_inputs"].items():
            assert digest(Path(path)) == h
        assert M.read(out / "progress.json")["updates"] == 380
        for state, parts in [("best", docs), ("last", {"dev": docs["dev"]})]:
            metrics = M.read(out / state / "metrics.json")
            for split, part in parts.items():
                key = f"{name}/{state}/{split}"
                record = {}
                if cfg["l2_weight"]:
                    saved = M.jsonl(out / state / f"{split}-l2.jsonl")
                    perdoc = []
                    for d, r in zip(part, saved, strict=True):
                        assert d["doc_id"] == r["doc_id"]
                        p = np.array(r["probabilities"])
                        perdoc.append(
                            np.mean(
                                [np.mean((p - np.array(a["l2"])) ** 2) for a in d["annotations"]]
                            )
                        )
                    delta = abs(float(np.mean(perdoc)) - metrics[split]["l2"]["brier"])
                    assert delta < 1e-12
                    record["brier_delta"] = delta
                if cfg["l3_weight"]:
                    cached = M.jsonl(out / state / f"{split}-chain.jsonl")
                    f1s = []
                    aps = []
                    for d, r in zip(part, cached, strict=True):
                        assert d["doc_id"] == r["doc_id"]
                        probs = np.clip(np.array(r["probabilities"]), 1e-7, 1 - 1e-7)
                        scores = (
                            np.log(probs / (1 - probs))
                            + cfg["alpha"] * np.array([v for _, _, v in r["pairs"]])[:, None]
                        )
                        scored = [
                            (float(scores[i, j]), s, e, f)
                            for i, (s, e, _) in enumerate(r["pairs"])
                            for j, f in enumerate(FAMILIES)
                        ]
                        scored.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
                        prediction = {
                            (s, e, f) for v, s, e, f in scored[:100] if v >= cfg["threshold"]
                        }
                        af = []
                        ap = []
                        for a in d["annotations"]:
                            gold = {(s["start"], s["end"], s["label"]) for s in a["l3_persuasion"]}
                            af.append(
                                2 * len(prediction & gold) / max(1, len(prediction) + len(gold))
                            )
                            labels = np.array(
                                [[(s, e, f) in gold for f in FAMILIES] for s, e, _ in r["pairs"]]
                            )
                            if labels.any():
                                ap.append(manual_ap(labels.ravel(), scores.ravel()))
                        f1s.append(np.mean(af))
                        if ap:
                            aps.append(np.mean(ap))
                    df = abs(float(np.mean(f1s)) - metrics[split]["chain_fixed"]["exact_f1"])
                    da = abs(float(np.mean(aps)) - metrics[split]["candidate_diagnostics"]["ap"])
                    assert df < 1e-12 and da < 1e-12
                    record.update(exact_f1_delta=df, ap_delta=da)
                checks[key] = record
    budget = M.read(ROOT / "budget.json")
    assert budget["seconds"] <= budget["limit_seconds"]
    gpu = M.read(ROOT / "gpu-verification.json")
    assert gpu["status"] == "passed"
    result = dict(
        status="passed",
        checks=checks,
        prior_files_unchanged=len(prior),
        gpu_reload=gpu,
        total_gpu_process_seconds=budget["original_seconds"] + budget["seconds"],
        test_used=False,
    )
    write_json(ROOT / "verification.json", result)
    print(
        dict(
            status="passed",
            cases=len(checks),
            historical_files=len(prior),
            seconds=budget["seconds"],
        )
    )


if __name__ == "__main__":
    main()

"""Independent CPU AP implementation and run/checkpoint/provenance checks."""

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.experiment import load_development
from research.kuki.train_l2 import epoch_order, learning_rate_multiplier

ROOT = Path(__file__).parent
opts = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "runs" / opts["run_id"]
chain = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.chain")


def manual_ap(y, score):
    order = np.argsort(-score, kind="stable")
    y = y[order]
    s = score[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    increments = np.diff(np.r_[0, tp])
    return float(np.sum(precision * increments) / y.sum())


def main():
    a = json.loads((OUT / "analysis.json").read_text())
    p = json.loads((OUT / "progress.json").read_text())
    assert (
        a["execution"]["status"] == "completed"
        and p["total_updates"] == 404
        and p["completed_updates"] == 64
        and p["article_presentations"] == 491
    )
    assert [x["update"] for x in p["history"]] == list(range(341, 405))
    for h in p["history"]:
        multiplier = learning_rate_multiplier(h["update"], 41, 408)
        assert np.allclose(h["lrs"], [5e-5 * multiplier, 1e-3 * multiplier], rtol=0, atol=1e-15)
    schedule = json.loads((OUT / "schedule.json").read_text())
    for step in schedule["steps"]:
        assert (
            step["indices"]
            == epoch_order(129, 42, step["epoch_index"])[step["position"] : step["position"] + 8]
        )
    state = torch.load(OUT / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert state["progress"] == p and state["scheduler"]["last_epoch"] == 404
    assert set(state["rng"]) == {"python", "numpy", "torch", "cuda"}
    prov = json.loads((OUT / "metadata.json").read_text())["provenance"]
    for name, h in prov["code_sha256"].items():
        assert digest(OUT / "code" / Path(name).name) == h
    for name, h in (prov["input_checkpoints"] | prov["candidate_caches"]).items():
        assert digest(Path(name)) == h
    docs, _ = load_development(json.loads(Path("research/configs/data.json").read_text()))
    differences = {}
    fullsets = {}
    for condition in ["control", "negative"]:
        full = chain.read_jsonl(OUT / f"{condition}-full-composite-scores.jsonl")
        perdoc = []
        eligible = []
        maxdiff = 0.0
        for d, o, saved in zip(
            docs["dev"], full, a[condition]["candidates"]["articles"], strict=True
        ):
            assert o["doc_id"] == d["doc_id"] == saved["doc_id"]
            vals = []
            for ai, ann in enumerate(d["annotations"]):
                gold = {(z["start"], z["end"], z["label"]) for z in ann["l3_persuasion"]}
                y = np.array(
                    [[(s, e, f) in gold for f in o["families"]] for s, e in o["pairs"]]
                ).ravel()
                if not y.any():
                    continue
                value = manual_ap(y, np.asarray(o["composite_scores"]).ravel())
                maxdiff = max(maxdiff, abs(value - saved["individuals"][ai]["ap"]))
                vals.append(value)
                eligible.append((d["doc_id"], ai))
            if vals:
                perdoc.append(float(np.mean(vals)))
        assert maxdiff < 1e-12
        assert abs(np.mean(perdoc) - a[condition]["candidates"]["ap"]) < 1e-12
        differences[condition] = maxdiff
        fullsets[condition] = eligible
        scores = chain.read_jsonl(OUT / f"{condition}-dev-scores.jsonl")
        metrics, _, _ = chain.chain_score(docs["dev"], scores, opts["alpha"], opts["threshold"])
        assert metrics == a[condition]["metrics"]
    assert fullsets["control"] == fullsets["negative"]
    previous = json.loads((ROOT.parent / "modular/manifest.json").read_text())
    assert all(digest(Path(name)) == h for name, h in previous["files"].items())
    write_json(
        OUT / "verification.json",
        dict(
            passed=True,
            manual_ap_max_abs_difference=differences,
            ap_eligible_answers=len(fullsets["control"]),
            historical_modular_files_unchanged=len(previous["files"]),
            historical_checkpoints_unchanged=True,
            lr_and_schedule_verified=True,
            rng_and_progress_verified=True,
            fixed_threshold_metrics_recomputed=True,
            test_used=False,
        ),
    )
    print("CPU verification passed: manual AP, metrics, LR, schedule, RNG, and historical hashes.")


if __name__ == "__main__":
    main()

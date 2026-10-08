"""Verify committed predictions independently and reload both selected GPU models."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from research.kuki.data import digest, write_json
from research.kuki.experiment import load_development
from research.kuki.l2 import load_checkpoint
from research.kuki.l3 import L3Model
from research.kuki.train_l2 import make_optimizer
from research.kuki.train_l3 import configure, encode_documents, predict

STAGE = Path("research/stages/ST2-l3-rhetoric")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def independent_metrics(docs, outputs):
    exact_articles, overlap_articles = [], []
    for d, out in zip(docs, outputs, strict=True):
        assert d["doc_id"] == out["doc_id"]
        p = {(s["start"], s["end"], s["label"]) for s in out["spans"]}
        for s, e, _ in p:
            assert 0 <= s < e <= len(d["text"])
        exact, overlap = [], []
        for a in d["annotations"]:
            g = {(s["start"], s["end"], s["label"]) for s in a["l3_persuasion"]}
            denominator = len(p) + len(g)
            exact.append(2 * len(p & g) / denominator if denominator else 0.0)
            # Independent cardinality-only assignment verifies matched-count
            # correctness without reusing the production matching function.
            admissible = np.array(
                [
                    [
                        s[2] == t[2]
                        and max(0, min(s[1], t[1]) - max(s[0], t[0]))
                        / (max(s[1], t[1]) - min(s[0], t[0]))
                        >= 0.5
                        for t in g
                    ]
                    for s in p
                ]
            )
            if p and g:
                left, right = linear_sum_assignment(admissible.astype(int), maximize=True)
                matches = int(admissible[left, right].sum())
            else:
                matches = 0
            overlap.append(2 * matches / denominator if denominator else 0.0)
        exact_articles.append(np.mean(exact))
        overlap_articles.append(np.mean(overlap))
    return float(np.mean(exact_articles)), float(np.mean(overlap_articles))


def main():
    begin = time.perf_counter()
    docs, _ = load_development(read(Path("research/configs/data.json")))
    results = {}
    encoded = None
    for name in ["20260928-l3-pilot-s42-01", "20260928-l3-negative025-s42-01"]:
        run = STAGE / "experiments" / name
        m = read(run / "metrics.json")
        prov = read(run / "metadata.json")["provenance"]
        for path, sha in prov["code_sha256"].items():
            assert digest(Path(path)) == sha == digest(run / "code" / Path(path).name)
        assert digest(run / "uv.lock") == prov["uv_lock_sha256"] == digest(Path("uv.lock"))
        for split, part in docs.items():
            saved = rows(run / f"{split}-predictions.jsonl")
            assert all(p["split"] == split for p in saved)
            exact, overlap = independent_metrics(part, saved)
            assert abs(exact - m[split]["exact_f1"]) < 1e-12
            assert abs(overlap - m[split]["overlap_f1"]) < 1e-12
        configure(prov["config"]["seed"])
        if encoded is None:
            encoded = encode_documents(docs["dev"], prov["encoder"], prov["config"])
        model = L3Model(prov["encoder"], prov["config"]).cuda()
        optimizer, scheduler = make_optimizer(model, prov["config"])
        state = load_checkpoint(run / "checkpoints/best.pt", model, optimizer, scheduler, prov)
        regenerated = predict(model, docs["dev"], encoded)
        assert regenerated == rows(run / "dev-raw-predictions.jsonl")
        assert state["best_epoch"] == m["selected_epoch"]
        assert state["best"]["threshold"] == m["threshold"]
        last = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        assert last["progress"]["updates"] == m["total_updates"]
        assert last["progress"]["articles_seen"] == m["articles_seen"]
        assert m["articles_seen"] == last["progress"]["epoch"] * len(docs["train"])
        results[name] = dict(
            independent_exact_and_overlap_metrics=True,
            fresh_gpu_raw_predictions_identical=True,
            code_and_lock_match=True,
            counters_and_selected_checkpoint_correct=True,
            trainable_tensors_changed=m["trainable_tensors_changed_since_session_start"],
            trainable_tensors=m["trainable_tensors"],
        )
        assert (
            m["base_unchanged"]
            and m["trainable_tensors_changed_since_session_start"] == m["trainable_tensors"]
        )
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(
        status="passed",
        runs=results,
        test_scored=False,
        gpu_resume_max_difference=read(
            STAGE / "experiments/20260928-l3-preflight-s42-01/metrics.json"
        )["resume_max_parameter_difference"],
        verify_script_sha256=digest(Path(__file__)),
        elapsed_seconds=time.perf_counter() - begin,
    )
    write_json(STAGE / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Read-only independent decoding plus GPU reload on the fixed diagnostic articles."""

import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.kuki.data import write_json
from research.kuki.experiment import initialize_run, provenance
from research.kuki.l2 import trainable_state
from research.kuki.l3 import L3Model
from research.kuki.train_l3 import configure, encode_documents

tiny = importlib.import_module("research.stages.ST2-l3-rhetoric.tiny_diagnostic")
STAGE = Path(__file__).parent


def independent_decode(arrays, docs, threshold):
    result = []
    for i, d in enumerate(docs):
        pairs = arrays[f"d{i}_ordinary_pairs"]
        scores = arrays[f"d{i}_ordinary_scores"].ravel()
        indices = sorted(range(len(scores)), key=lambda k: (-float(scores[k]), k))[:100]
        pred = {
            (int(pairs[k // 6, 0]), int(pairs[k // 6, 1]), k % 6)
            for k in indices
            if scores[k] >= threshold
        }
        gold = {
            (s["start"], s["end"], tiny.FAMILIES.index(s["label"]))
            for s in d["annotations"][0]["l3_persuasion"]
        }
        tp = len(pred & gold)
        result.append(
            dict(
                doc_id=d["doc_id"],
                f1=2 * tp / max(1, len(pred) + len(gold)),
                predictions=len(pred),
                tp=tp,
                gold=len(gold),
                correct_empty=not pred if not gold else None,
            )
        )
    return result


def verify_saved(run, docs):
    metrics = json.loads((run / "metrics.json").read_text())
    ev = metrics["history"][-1]
    arrays = np.load(run / "checkpoints" / f"full-scores-{ev['update']:03}.npz")
    decoded = independent_decode(arrays, docs, ev["ordinary"]["best_threshold"])
    for a, b in zip(decoded, ev["ordinary"]["articles"], strict=True):
        assert (
            a["f1"] == b["exact_typed_f1"]
            and a["predictions"] == b["predictions"]
            and a["tp"] == b["true_positive"]
        )
    checked = []
    for interval in ev["ordinary"]["successful_intervals"]:
        midpoint = (interval["lower_exclusive"] + interval["upper_inclusive"]) / 2
        d = independent_decode(arrays, docs, midpoint)
        assert all(r["f1"] >= 0.95 if r["gold"] else r["correct_empty"] for r in d)
        checked.append(dict(threshold=midpoint, articles=d))
    grid = {str(t): independent_decode(arrays, docs, t) for t in [0.3, 0.5, 0.7, 0.9]}
    return dict(
        run_id=run.name,
        independent_metrics_equal=True,
        interval_midpoints_checked=checked,
        original_threshold_grid=grid,
    )


def main():
    begin = time.perf_counter()
    docs, manifest, data_cfg, hashes = tiny.selected_docs()
    ledger_path = tiny.SERIES / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["gpu_process_seconds"] < 1100
    cpu = [
        verify_saved(STAGE / "experiments/20260928-l3-tiny-one-s42-01", docs[:1]),
        verify_saved(STAGE / "experiments/20260928-l3-tiny-three-s42-01", docs),
    ]
    cfg = json.loads((STAGE / "configs/hard-negative-v1.json").read_text())
    enc = json.loads(Path(cfg["encoder_config"]).read_text())
    configure(42)
    encoded = encode_documents(docs, enc, cfg)
    output = STAGE / "experiments/20260928-l3-tiny-verification-01"
    prov = provenance(cfg, data_cfg, hashes, enc)
    for path in [Path(__file__), Path(tiny.__file__)]:
        prov["code_sha256"][str(path.resolve())] = tiny.sha(path)
    initialize_run(output, cfg, prov)
    model = L3Model(enc, cfg).cuda()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    sources = [
        ("historical_hard", STAGE / "experiments/20260928-l3-hard-s42-01/checkpoints/best.pt"),
        ("tiny_three", STAGE / "experiments/20260928-l3-tiny-three-s42-01/checkpoints/final.pt"),
    ]
    comparison = {}
    for label, path in sources:
        state = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
        assert set(state) == set(trainable_state(model))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(state[name])
        sub = output / label
        sub.mkdir()
        (sub / "checkpoints").mkdir()
        ev = tiny.evaluate(model, docs, encoded, manifest["records"], sub, 0)
        arrays = np.load(sub / "checkpoints/full-scores-000.npz")
        independent = independent_decode(arrays, docs, ev["ordinary"]["best_threshold"])
        for a, b in zip(independent, ev["ordinary"]["articles"], strict=True):
            assert a["f1"] == b["exact_typed_f1"] and a["predictions"] == b["predictions"]
        if label == "tiny_three":
            old = np.load(
                STAGE / "experiments/20260928-l3-tiny-three-s42-01/checkpoints/full-scores-120.npz"
            )
            assert set(old.files) == set(arrays.files)
            assert all(np.array_equal(old[k], arrays[k]) for k in old.files)
        # Production infer and uncensored diagnostic decoding agree at .3.
        with torch.no_grad():
            for i, d in enumerate(docs):
                out = model.infer(encoded[d["doc_id"]])
                raw = arrays[f"d{i}_ordinary_scores"]
                pp = arrays[f"d{i}_ordinary_pairs"]
                order = np.argsort(-raw.ravel(), kind="stable")
                expected = order[raw.ravel()[order] >= 0.3][:100]
                actual = [
                    (s["start"], s["end"], tiny.FAMILIES.index(s["label"]), s["score"])
                    for s in out["spans"]
                ]
                desired = [
                    (int(pp[k // 6, 0]), int(pp[k // 6, 1]), int(k % 6), float(raw.ravel()[k]))
                    for k in expected
                ]
                assert actual == desired
        comparison[label] = dict(
            checkpoint_sha256=tiny.sha(path),
            ordinary=ev["ordinary"],
            oracle=ev["oracle"],
            coverage=[
                {k: r[k] for k in ["doc_id", "start_recall", "end_recall", "pair_recall"]}
                for r in ev["details"]
            ],
            original_threshold_grid={
                str(t): independent_decode(arrays, docs, t) for t in [0.3, 0.5, 0.7, 0.9]
            },
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    ledger["gpu_process_seconds"] += elapsed
    ledger["processes"].append(
        dict(
            run_id=output.name,
            mode="verification_no_training",
            elapsed_seconds=elapsed,
            updates=0,
            status="passed",
        )
    )
    write_json(ledger_path, ledger)
    result = dict(
        status="passed",
        cpu_checks=cpu,
        comparison=comparison,
        full_scores_bitwise_equal_after_reload=True,
        production_infer_equal_to_full_score_decode=True,
        elapsed_seconds=elapsed,
        total_gpu_process_seconds=ledger["gpu_process_seconds"],
        updates=0,
        test_used=False,
    )
    write_json(output / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

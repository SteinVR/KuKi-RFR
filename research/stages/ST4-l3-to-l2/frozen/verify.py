"""Independent L2 calculations and artifact/provenance audit; no GPU computation."""

import importlib
from pathlib import Path

import numpy as np

from research.kuki.data import digest, write_json

C = importlib.import_module("research.stages.ST4-l3-to-l2.frozen.common")


def main():
    docs, _, live = C.inputs()
    prior = C.read(C.ROOT / "prior-artifact-hashes.json")
    assert all(digest(Path(p)) == h for p, h in prior.items())
    cache = C.read(C.ROOT / "extraction/cache.json")
    assert digest(C.CACHE) == cache["sha256"]
    checks = {}
    initial = []
    for condition in ["control", "families"]:
        out = C.ROOT / "experiments" / f"20260929-st4-frozen-{condition}-s42-01"
        initial.append(C.read(out / "initialization.json")["state_sha256"])
        prov = C.read(out / "metadata.json")["provenance"]
        assert prov["condition"] == condition
        assert prov["cache_sha256"] == cache["sha256"]
        for key in ["data_hashes", "uv_lock_sha256", "parent_checkpoint_sha256",
                    "oof_predictions_sha256", "oof_manifest_sha256", "detector_inputs",
                    "detector_checkpoint_sha256", "protocol_sha256"]:
            assert prov[key] == live[key], key
        for source, expected in prov["code_sha256"].items():
            source = Path(source)
            relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
            assert digest(out / "code-tree" / relative) == expected
        assert digest(out / "protocol.md") == prov["protocol_sha256"]
        progress = C.read(out / "progress.json")
        assert progress["updates"] == 570 and progress["articles_seen"] == 4350
        assert len(progress["history"]) == 30
        values = [C.read(out / "initialization.json")["dev"]["brier"]] + [
            r["dev"]["brier"] for r in progress["history"]]
        assert progress["best_epoch"] == int(np.argmin(values))
        for state in ["best", "last"]:
            metrics = C.read(out / state / "metrics.json")
            for split, part in docs.items():
                saved = C.M.jsonl(out / state / f"{split}-l2.jsonl")
                errors, binary_targets, binary_preds = [], [], []
                for d, row in zip(part, saved, strict=True):
                    assert row["doc_id"] == d["doc_id"]
                    p = np.array(row["probabilities"])
                    ys = np.array([a["l2"] for a in d["annotations"]])
                    errors.append(np.mean([np.mean((p-y)**2) for y in ys]))
                    binary_targets.append(ys.mean(0) > .5)
                    binary_preds.append(p >= .5)
                y, p = np.array(binary_targets), np.array(binary_preds)
                tp, fp, fn = (y & p).sum(0), (~y & p).sum(0), (y & ~p).sum(0)
                f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
                delta = abs(float(np.mean(errors)) - metrics[split]["brier"])
                assert delta < 1e-12
                assert abs(f1.mean() - metrics[split]["macro_f1"]) < 1e-12
                checks[f"{condition}/{state}/{split}"] = dict(brier_delta=delta, macro_f1_exact=True)
    assert initial[0] == initial[1]
    gpu = C.read(C.ROOT / "gpu-verification.json")
    assert gpu["status"] == "passed"
    ledger = C.read(C.ROOT / "budget.json")
    assert ledger["seconds"] < ledger["limit_seconds"]
    write_json(C.ROOT / "verification.json", dict(status="passed", checks=checks,
        identical_initial_residuals=True, historical_files_unchanged=len(prior),
        fresh_gpu_replay=gpu, gpu_process_seconds=ledger["seconds"], test_used=False))
    print(dict(status="passed", cases=len(checks), historical_files=len(prior)))


if __name__ == "__main__":
    main()

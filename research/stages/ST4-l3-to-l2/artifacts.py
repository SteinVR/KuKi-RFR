"""Lossless path-preserving snapshots and group-excluded proposal assembly."""

import importlib
import shutil
from pathlib import Path

from research.kuki.data import digest, write_json, write_jsonl

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")


def snapshot(out):
    prov = M.read(out / "metadata.json")["provenance"]
    for source, h in prov["code_sha256"].items():
        flat = out / "code" / Path(source).name
        candidate = flat if flat.exists() and digest(flat) == h else Path(source)
        assert digest(candidate) == h, f"Cannot reconstruct exact snapshot: {source}"
        relative = Path(source).relative_to(Path.cwd()) if Path(source).is_absolute() else Path(source)
        target = out / "code-tree" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
    write_json(
        out / "snapshot-layout.json",
        dict(
            authoritative="code-tree/<original relative path>",
            hashes_verified=True,
            reason="Preserves same-basename files from shared and stage modules",
        ),
    )


def assemble():
    cfg = M.read(M.ROOT / "configs/continue.json")
    docs, _, _ = M.inputs(cfg, False)
    membership = M.folds(docs["train"])
    manifest = dict(membership=membership, folds=[], test_used=False)
    predictions = []
    for i in range(3):
        out = M.ROOT / "oof" / f"20260929-oof{i}-s42-01"
        progress = M.read(out / "progress.json")
        assert progress["status"] == "completed"
        prov = M.read(out / "metadata.json")["provenance"]
        fold = prov["oof"]
        assert set(fold["train_ids"]).isdisjoint(fold["prediction_ids"])
        assert set(fold["prediction_ids"]) == {k for k, v in membership.items() if v == i}
        lookup = {d["doc_id"]: d["group_id"] for d in docs["train"]}
        assert not {lookup[k] for k in fold["train_ids"]} & {
            lookup[k] for k in fold["prediction_ids"]
        }
        rr = M.jsonl(out / "predictions.jsonl")
        assert {r["doc_id"] for r in rr} == set(fold["prediction_ids"])
        predictions.extend(rr)
        snapshot(out)
        manifest["folds"].append(
            dict(
                **fold,
                checkpoint_sha256=digest(out / "checkpoints/last.pt"),
                predictions_sha256=digest(out / "predictions.jsonl"),
                updates=progress["updates"],
                articles_seen=progress["articles_seen"],
            )
        )
    assert len(predictions) == len({r["doc_id"] for r in predictions}) == len(docs["train"])
    predictions.sort(key=lambda r: r["doc_id"])
    write_jsonl(M.ROOT / "oof/train-predictions.jsonl", predictions)
    manifest["train_predictions_sha256"] = digest(M.ROOT / "oof/train-predictions.jsonl")
    write_json(M.ROOT / "oof/manifest.json", manifest)
    print(dict(oof_complete=True, articles=len(predictions)))


if __name__ == "__main__":
    assemble()

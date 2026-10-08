"""A bounded, explicitly post-primary dev threshold refinement, without retraining."""

import json
from pathlib import Path

from research.kuki.data import digest, write_json, write_jsonl
from research.kuki.experiment import load_development
from research.kuki.l3_evaluation import evaluate, select_spans

STAGE = Path("research/stages/ST2-l3-rhetoric")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    cfg_path = STAGE / "configs/decoder-v2.json"
    cfg = read(cfg_path)
    docs, _ = load_development(read(Path("research/configs/data.json")))
    results = {}
    for name in ["20260928-l3-pilot-s42-01", "20260928-l3-negative025-s42-01"]:
        run = STAGE / "experiments" / name
        if not (run / "metrics.json").exists():
            continue
        original = read(run / "metrics.json")
        raw = rows(run / "dev-raw-predictions.jsonl")
        grid = [evaluate(docs["dev"], raw, threshold)[0] for threshold in cfg["threshold_grid"]]
        selected = max(grid, key=lambda m: (m["exact_f1"], m["overlap_f1"], m["threshold"]))
        result = dict(
            parent_run=name,
            parent_checkpoint_sha256=digest(run / "checkpoints/best.pt"),
            selected_epoch=original["selected_epoch"],
            threshold=selected["threshold"],
            dev_grid=grid,
            dev=selected,
        )
        for split, part in docs.items():
            raw = rows(run / f"{split}-raw-predictions.jsonl")
            result[split], detail = evaluate(part, raw, selected["threshold"])
            write_jsonl(
                run / f"{split}-decoder-v2-predictions.jsonl",
                [
                    dict(
                        doc_id=d["doc_id"],
                        group_id=d["group_id"],
                        split=split,
                        spans=select_spans(o, selected["threshold"]),
                    )
                    for d, o in zip(part, raw, strict=True)
                ],
            )
            write_jsonl(run / f"{split}-decoder-v2-article-metrics.jsonl", detail)
        write_json(run / "decoder-v2-metrics.json", result)
        results[name] = result
    write_json(
        STAGE / "decoder-v2.json", dict(config=cfg, config_sha256=digest(cfg_path), runs=results)
    )
    print(json.dumps({n: dict(threshold=r["threshold"], dev=r["dev"]) for n, r in results.items()}))


if __name__ == "__main__":
    main()

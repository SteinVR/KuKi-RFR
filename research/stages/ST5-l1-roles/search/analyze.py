"""Predeclared development selection with explicit F1/probability tradeoffs."""

import importlib
import json

import numpy as np

from research.kuki.data import write_json
from research.kuki.l1 import examples

M = importlib.import_module("research.stages.ST5-l1-roles.search.method")
bootstrap = importlib.import_module("research.stages.ST5-l1-roles.analyze").bootstrap


def probabilities(run):
    return np.array([json.loads(s)["probabilities"] for s in
                     (run / "best/dev-predictions.jsonl").read_text().splitlines()])


def main():
    original = M.ROOT.parent / "experiments/20260929-l1-base-s42-01"
    base = M.read(original / "best/dev-metrics.json")
    candidates = []
    for name in ["head-slow","smooth","head-slow-smooth","dropout"]:
        cfg = M.read(M.ROOT / "configs" / f"{name}.json")
        run = M.ROOT / "experiments" / cfg["run_id"]
        m = M.read(run / "best/dev-metrics.json")
        p = M.read(run / "progress.json")
        candidates.append(dict(condition=name,run_id=cfg["run_id"],run=str(run),
                               best_epoch=p["best_epoch"],best_ce_epoch=p["best_ce_epoch"],dev=m,
                               dev_best_ce=M.read(run / "best-ce/dev-metrics.json"),
                               last=M.read(run / "last/dev-metrics.json"),
                               train=M.read(run / "best/train-metrics.json")))
    def key(x):
        return x["dev"]["macro_f1"], -x["dev"]["cross_entropy"]
    eligible = [c for c in candidates if c["dev"]["macro_f1"] >= base["macro_f1"]+.01
                and c["dev"]["cross_entropy"] <= base["cross_entropy"]
                and c["dev"]["brier"] <= base["brier"]]
    selected = max(eligible,key=key) if eligible else None
    soft = [c for c in candidates if c["dev"]["macro_f1"] >= base["macro_f1"]-.01
            and c["dev"]["cross_entropy"] <= base["cross_entropy"]-.1
            and c["dev"]["brier"] <= base["brier"]]
    probability_candidate = min(soft,key=lambda c:(c["dev"]["cross_entropy"],-c["dev"]["macro_f1"])) if soft else None
    docs,_,_ = M.inputs(M.read(M.ROOT / "configs/head-slow.json"))
    rows = examples(docs["dev"])
    baseline = probabilities(original)
    from pathlib import Path

    for c in candidates:
        c["versus_original_base"] = bootstrap(rows,probabilities(Path(c["run"])),baseline)
    result = dict(original_base=base,candidates=candidates,
                  absolute_best_new_f1=max(candidates,key=key)["condition"],
                  selected_condition=selected["condition"] if selected else "original-base",
                  selected_checkpoint=str(Path(selected["run"]) / "checkpoints/best.pt") if selected
                  else str(original / "checkpoints/best.pt"),
                  probability_candidate=probability_candidate["condition"] if probability_candidate else None,
                  original_series_unchanged=True,test_used=False)
    write_json(M.ROOT / "selection.json",result)
    print(json.dumps(dict(selected=result["selected_condition"],
                          probability_candidate=result["probability_candidate"],
                          results=[(c["condition"],c["dev"]["macro_f1"],c["dev"]["cross_entropy"],
                                    c["dev"]["brier"]) for c in candidates])))


if __name__ == "__main__":
    main()

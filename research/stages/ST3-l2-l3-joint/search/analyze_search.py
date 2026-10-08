"""Analyse the extension without rewriting historical run artifacts."""

import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.kuki.data import write_json

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
A = importlib.import_module("research.stages.ST3-l2-l3-joint.analyze")
ROOT = Path(__file__).parent
NEW = ["joint200", "joint400"]
JOINT = ["joint025", "joint100", *NEW]


def dominates(left, right):
    a = (left["l2"]["brier"], -left["conditional"]["article_mean_f1"])
    b = (right["l2"]["brier"], -right["conditional"]["article_mean_f1"])
    return all(x <= y for x, y in zip(a, b, strict=True)) and a != b


def main():
    cfg = M.read(ROOT / "configs/joint200.json")
    docs, _, _ = M.inputs(cfg)
    original = M.read(ROOT.parent / "analysis.json")
    metrics = dict(original["best"])
    last = dict(original["last_dev"])
    runs = {
        n: ROOT.parent / "experiments" / M.read(ROOT.parent / "configs" / f"{n}.json")["run_id"]
        for n in ["l2-only", "l3-only", "joint025", "joint100"]
    }
    runs.update(
        {n: ROOT / "experiments" / M.read(ROOT / "configs" / f"{n}.json")["run_id"] for n in NEW}
    )
    progress = {n: M.read(p / "progress.json") for n, p in runs.items()}
    comparisons = dict(original["comparisons"])
    l2control = M.jsonl(runs["l2-only"] / "best/dev-l2.jsonl")
    l2previous = M.jsonl(runs["joint100"] / "best/dev-l2.jsonl")
    oldexact = M.read(runs["l3-only"] / "best/dev-chain-articles.json")
    for name in NEW:
        path = runs[name]
        c = M.read(path / "config.json")["config"]
        p = progress[name]
        assert p["status"] == "completed"
        assert (p["updates"], p["articles_seen"], p["l3_articles_seen"]) == (380, 2900, 2580)
        metrics[name] = M.saved_metrics(path / "best", docs, c)
        last[name] = M.saved_metrics(path / "last", {"dev": docs["dev"]}, c)["dev"]
        current = M.jsonl(path / "best/dev-l2.jsonl")
        assert [r["doc_id"] for r in current] == [r["doc_id"] for r in l2control]
        comp = dict(
            brier_vs_matched=A.bootstrap(
                docs["dev"], [r["brier"] for r in current], [r["brier"] for r in l2control]
            ),
            brier_vs_previous_joint100=A.bootstrap(
                docs["dev"], [r["brier"] for r in current], [r["brier"] for r in l2previous]
            ),
        )
        jc = metrics[name]["dev"]["conditional"]["article_f1"]
        lc = metrics["l3-only"]["dev"]["conditional"]["article_f1"]
        ids = [d for d in docs["dev"] if d["doc_id"] in jc]
        comp["conditional_vs_matched"] = A.bootstrap(
            ids, [jc[d["doc_id"]] for d in ids], [lc[d["doc_id"]] for d in ids]
        )
        newexact = M.read(path / "best/dev-chain-articles.json")
        assert [r["doc_id"] for r in newexact] == [r["doc_id"] for r in oldexact]
        comp["exact_vs_matched"] = A.bootstrap(
            docs["dev"], [r["exact_f1"] for r in newexact], [r["exact_f1"] for r in oldexact]
        )
        comp["criterion"] = dict(
            brier_gain_at_least_003=comp["brier_vs_matched"]["mean"] <= -0.003,
            conditional_drop_at_most_02=comp["conditional_vs_matched"]["mean"] >= -0.02,
            exact_drop_at_most_005=comp["exact_vs_matched"]["mean"] >= -0.005,
        )
        comp["criterion_passed"] = all(comp["criterion"].values())
        comparisons[name] = comp
    selected = min(JOINT, key=lambda n: metrics[n]["dev"]["l2"]["brier"])
    pareto = [
        n
        for n in JOINT
        if not any(dominates(metrics[k]["dev"], metrics[n]["dev"]) for k in JOINT if k != n)
    ]
    feasible = [n for n in JOINT if comparisons[n]["criterion_passed"]]
    initial = [M.read(p / "initialization.json") for p in runs.values()]
    assert all(x == initial[0] for x in initial)
    packages = [M.read(p / "metadata.json")["packages"] for p in runs.values()]
    assert all(x == packages[0] for x in packages)
    study = M.read(ROOT / "study.json")
    assert len(study["trials"]) == 2
    for trial in study["trials"]:
        assert trial["state"] == "COMPLETE"
        name = {2.0: "joint200", 4.0: "joint400"}[trial["params"]["l3_weight"]]
        assert trial["attrs"]["selected_epoch"] == progress[name]["best_epoch"]
        assert trial["values"] == [
            metrics[name]["dev"]["l2"]["brier"],
            metrics[name]["dev"]["conditional"]["article_mean_f1"],
        ]
    curves = {}
    for n, p in progress.items():
        hh = p["history"]
        curves[n] = dict(selected_epoch=p["best_epoch"], last_epoch=p["epoch"])
        if n != "l2-only":
            best = max(hh, key=lambda h: h["dev"]["conditional"]["article_mean_f1"])
            curves[n].update(
                max_conditional_f1=best["dev"]["conditional"]["article_mean_f1"],
                max_conditional_epoch=best["epoch"],
            )
        if n in JOINT:
            curves[n]["epochs_passing_L2_and_conditional_guards_exploratory"] = [
                h["epoch"]
                for h in hh
                if h["dev"]["l2"]["brier"] <= metrics["l2-only"]["dev"]["l2"]["brier"] - 0.003
                and h["dev"]["conditional"]["article_mean_f1"]
                >= metrics["l3-only"]["dev"]["conditional"]["article_mean_f1"] - 0.02
            ]
    subgroups = {}
    for source in sorted({d["source"] for d in docs["dev"]}):
        ii = [i for i, d in enumerate(docs["dev"]) if d["source"] == source]
        subgroups[source] = {"n": len(ii)}
        for n in ["l2-only", *JOINT]:
            rr = M.jsonl(runs[n] / "best/dev-l2.jsonl")
            subgroups[source][n] = float(np.mean([rr[i]["brier"] for i in ii]))
    result = dict(
        best=metrics,
        last_dev=last,
        comparisons=comparisons,
        selected_joint=selected,
        pareto_joint=pareto,
        feasible_joint=feasible,
        curves=curves,
        source_subgroups_exploratory=subgroups,
        initialization_identical=True,
        training_package_versions_identical=True,
        test_used=False,
        selection="minimum_dev_L2_Brier_across_four_weights",
        limitation="One seed; repeatedly selected dev; bootstrap excludes search and training variance",
    )
    write_json(ROOT / "analysis.json", result)
    write_json(
        ROOT / "selection.json",
        dict(
            selected_joint=selected,
            feasible_joint=feasible,
            full_criterion_passed=comparisons[selected]["criterion_passed"],
            pareto_joint=pareto,
            test_used=False,
            ST4_started=False,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, p in progress.items():
        hh = p["history"]
        epochs = [h["epoch"] for h in hh]
        if n != "l3-only":
            axes[0].plot(epochs, [h["dev"]["l2"]["brier"] for h in hh], label=n)
        if n != "l2-only":
            axes[1].plot(epochs, [h["dev"]["conditional"]["article_mean_f1"] for h in hh], label=n)
    for ax, title in zip(
        axes,
        ["L2: Brier на dev (меньше — лучше)", "L3: F1 при заданных границах (больше — лучше)"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Эпоха")
        ax.set_xticks([1, 4, 8, 12, 16, 20])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "learning-curves.png", dpi=170)
    plt.close(fig)
    print(
        {
            n: dict(
                epoch=progress[n]["best_epoch"],
                brier=metrics[n]["dev"]["l2"]["brier"],
                conditional=metrics[n]["dev"]["conditional"]["article_mean_f1"],
                exact=metrics[n]["dev"]["chain_fixed"]["exact_f1"],
                passes=comparisons[n]["criterion_passed"],
            )
            for n in JOINT
        }
    )


if __name__ == "__main__":
    main()

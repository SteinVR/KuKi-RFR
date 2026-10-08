"""Predeclared comparisons, descriptive grouped intervals, all-article diagnostics."""

import importlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.kuki.data import write_json
from research.kuki.evaluation import article_brier, score

C = importlib.import_module("research.stages.ST4-l3-to-l2.frozen.common")


def probabilities(path):
    return np.asarray([r["probabilities"] for r in C.M.jsonl(path)])


def paired(docs, a, b):
    delta = article_brier(docs, a).mean(1) - article_brier(docs, b).mean(1)
    groups = sorted({d["group_id"] for d in docs})
    members = [np.array([i for i, d in enumerate(docs) if d["group_id"] == g]) for g in groups]
    rng = np.random.default_rng(42)
    draws = [float(delta[np.concatenate([members[j] for j in
             rng.integers(0, len(groups), size=len(groups))])].mean()) for _ in range(10000)]
    return dict(delta=float(delta.mean()), descriptive_ci95=np.quantile(draws, [.025, .975]).tolist(),
                n_groups=len(groups), n_articles=len(docs), repeats=10000)


def main():
    docs, _, _ = C.inputs()
    dev = docs["dev"]
    base = probabilities(C.ROOT / "extraction/dev-l2.jsonl")
    base_metrics = score(dev, base)
    models, predictions, histories = {}, {}, {}
    for condition in ["control", "families"]:
        out = C.ROOT / "experiments" / f"20260929-st4-frozen-{condition}-s42-01"
        progress = C.read(out / "progress.json")
        assert progress["status"] == "completed"
        histories[condition] = progress["history"]
        models[condition] = dict(best_epoch=progress["best_epoch"], states={})
        for state in ["best", "last"]:
            p = probabilities(out / state / "dev-l2.jsonl")
            predictions[f"{condition}/{state}"] = p
            models[condition]["states"][state] = dict(
                metrics=score(dev, p), vs_base=paired(dev, p, base),
                mean_abs_probability_change=float(abs(p - base).mean()))
    full = predictions["families/best"]
    ctrl = predictions["control/best"]
    comparisons = dict(families_vs_control=paired(dev, full, ctrl),
                       families_vs_base=paired(dev, full, base),
                       control_vs_base=paired(dev, ctrl, base),
                       last_families_vs_control=paired(dev, predictions["families/last"],
                                                      predictions["control/last"]))
    criterion = comparisons["families_vs_control"]["delta"] < 0 and comparisons[
        "families_vs_base"]["delta"] < 0
    selected = min([("baseline", base_metrics["brier"])] + [
        (c, models[c]["states"]["best"]["metrics"]["brier"]) for c in models], key=lambda x: x[1])[0]
    articles = []
    for i, d in enumerate(dev):
        errors = {"baseline": float(article_brier([d], base[i:i+1]).mean())}
        errors.update({c: float(article_brier([d], p[i:i+1]).mean())
                       for c, p in [("families", full), ("control", ctrl)]})
        articles.append(dict(doc_id=d["doc_id"], group_id=d["group_id"],
                             brier=errors, families_minus_control=errors["families"]-errors["control"],
                             families_minus_baseline=errors["families"]-errors["baseline"],
                             probabilities=dict(baseline=base[i].tolist(), families=full[i].tolist(),
                                                control=ctrl[i].tolist())))
    l3 = C.read(C.M.PARENT / "best/metrics.json")["dev"]
    result = dict(baseline=base_metrics, models=models, comparisons=comparisons, articles=articles,
                  l3_fixed={k: l3[k] for k in ["conditional", "chain_fixed", "candidate_diagnostics"]})
    write_json(C.ROOT / "analysis.json", result)
    write_json(C.ROOT / "selection.json", dict(criterion_met=criterion,
        best_dev_system=selected, selected_epochs={c: models[c]["best_epoch"] for c in models},
        rule="families best Brier strictly better than baseline AND control best",
        confirmatory_evidence=False, test_used=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for c, h in histories.items():
        ax[0].plot([0] + [r["epoch"] for r in h], [base_metrics["brier"]] + [r["dev"]["brier"] for r in h], label=c)
        ax[1].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label=c)
    ax[0].axhline(base_metrics["brier"], color="black", linestyle="--", label="frozen ST3")
    for a, title in zip(ax, ["Dev Brier (lower is better)", "Train BCE"], strict=True):
        a.set(xlabel="Residual training epoch", title=title)
        a.grid(alpha=.2)
        a.legend()
    fig.savefig(C.ROOT / "learning-curves.png", dpi=160)
    print(dict(selected=selected, criterion=criterion, comparisons=comparisons))


if __name__ == "__main__":
    main()

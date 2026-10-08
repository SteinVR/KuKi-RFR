"""Recompute ST3 metrics and descriptive group-bootstrap comparisons from saved predictions."""

import importlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.kuki.data import write_json

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path("research/stages/ST3-l2-l3-joint")
NAMES = ["l2-only", "l3-only", "joint025", "joint100"]


def bootstrap(docs, left, right):
    delta = np.asarray(left) - np.asarray(right)
    assert len(delta) == len(docs)
    groups = sorted({d["group_id"] for d in docs})
    members = [np.array([i for i, d in enumerate(docs) if d["group_id"] == g]) for g in groups]
    sums = np.array([delta[ii].sum() for ii in members])
    counts = np.array([len(ii) for ii in members])
    idx = np.random.default_rng(42).integers(0, len(groups), (10000, len(groups)))
    dist = sums[idx].sum(1) / counts[idx].sum(1)
    return dict(
        mean=float(delta.mean()),
        interval95=np.quantile(dist, [0.025, 0.975]).tolist(),
        articles=len(docs),
        groups=len(groups),
        draws=10000,
        seed=42,
        limitation="Descriptive selected dev predictions; excludes model-selection and training variability",
    )


def table(rows):
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def num(x):
    return "—" if x is None else f"{x:.4f}"


def main():
    cfg = M.read(ROOT / "configs/joint100.json")
    docs, enc, prov = M.inputs(cfg)
    runs = {
        n: ROOT / "experiments" / M.read(ROOT / "configs" / f"{n}.json")["run_id"] for n in NAMES
    }
    metrics = {}
    last = {}
    progress = {}
    for name, path in runs.items():
        c = M.read(path / "config.json")["config"]
        progress[name] = M.read(path / "progress.json")
        assert progress[name]["status"] == "completed"
        metrics[name] = M.saved_metrics(path / "best", docs, c)
        last[name] = M.saved_metrics(path / "last", {"dev": docs["dev"]}, c)["dev"]
    selected = min(["joint025", "joint100"], key=lambda n: metrics[n]["dev"]["l2"]["brier"])
    comparisons = {}
    l2control = M.jsonl(runs["l2-only"] / "best/dev-l2.jsonl")
    oldpath = Path("research/stages/ST1-l2-frames/experiments/20260928-l2-pilot-s42-01")
    old = M.jsonl(oldpath / "dev-predictions.jsonl")
    for name in ["joint025", "joint100"]:
        curr = M.jsonl(runs[name] / "best/dev-l2.jsonl")
        assert (
            [r["doc_id"] for r in curr]
            == [r["doc_id"] for r in l2control]
            == [r["doc_id"] for r in old]
        )
        comp = {
            "brier_vs_matched": bootstrap(
                docs["dev"], [r["brier"] for r in curr], [r["brier"] for r in l2control]
            ),
            "brier_vs_historical_ST1": bootstrap(
                docs["dev"], [r["brier"] for r in curr], [r["brier"] for r in old]
            ),
        }
        jc = metrics[name]["dev"]["conditional"]["article_f1"]
        lc = metrics["l3-only"]["dev"]["conditional"]["article_f1"]
        ids = [d for d in docs["dev"] if d["doc_id"] in jc]
        comp["conditional_vs_matched"] = bootstrap(
            ids, [jc[d["doc_id"]] for d in ids], [lc[d["doc_id"]] for d in ids]
        )
        jj = M.read(runs[name] / "best/dev-chain-articles.json")
        ll = M.read(runs["l3-only"] / "best/dev-chain-articles.json")
        comp["exact_vs_matched"] = bootstrap(
            docs["dev"], [r["exact_f1"] for r in jj], [r["exact_f1"] for r in ll]
        )
        lastcurr = M.jsonl(runs[name] / "last/dev-l2.jsonl")
        lastctrl = M.jsonl(runs["l2-only"] / "last/dev-l2.jsonl")
        comp["last_brier_vs_matched"] = bootstrap(
            docs["dev"], [r["brier"] for r in lastcurr], [r["brier"] for r in lastctrl]
        )
        comp["criterion"] = dict(
            brier_gain_at_least_003=comp["brier_vs_matched"]["mean"] <= -0.003,
            conditional_drop_at_most_02=comp["conditional_vs_matched"]["mean"] >= -0.02,
            exact_drop_at_most_005=comp["exact_vs_matched"]["mean"] >= -0.005,
        )
        comp["criterion_passed"] = all(comp["criterion"].values())
        comparisons[name] = comp
    oldm = M.read(oldpath / "metrics.json")["dev"]
    historical = dict(
        l2_ST1=oldm,
        l3_E1=M.read(
            Path(
                "research/stages/ST2-l3-rhetoric/experiments/20260928-classifier-low-lr-s42-01/metrics.json"
            )
        )["dev"],
        chain_exact_f1=0.06172670411667743,
        chain_overlap_f1=0.10015586763044322,
    )
    currows = {
        n: M.jsonl(runs[n] / "best/dev-l2.jsonl") for n in ["l2-only", "joint025", "joint100"]
    }
    subgroups = {}
    for source in sorted({d["source"] for d in docs["dev"]}):
        ii = [i for i, d in enumerate(docs["dev"]) if d["source"] == source]
        subgroups[source] = dict(
            n=len(ii),
            **{n: float(np.mean([rr[i]["brier"] for i in ii])) for n, rr in currows.items()},
        )
    error = []
    for i, d in enumerate(docs["dev"]):
        joint = currows[selected][i]
        control = currows["l2-only"][i]
        target = np.asarray(joint["vote_fractions"]) > 0.5
        p = np.asarray(joint["probabilities"]) >= 0.5
        from research.kuki.data import FRAMES

        error.append(
            dict(
                doc_id=d["doc_id"],
                title=d["title"],
                source=d["source"],
                brier_joint=joint["brier"],
                brier_control=control["brier"],
                delta=joint["brier"] - control["brier"],
                false_positive=[
                    f for f, t, z in zip(FRAMES, target, p, strict=True) if z and not t
                ],
                false_negative=[
                    f for f, t, z in zip(FRAMES, target, p, strict=True) if t and not z
                ],
            )
        )
    initial = [M.read(p / "initialization.json") for p in runs.values()]
    assert all(x == initial[0] for x in initial)
    summary = dict(
        selected_joint=selected,
        selection="minimum_dev_L2_Brier_among_two_predeclared_weights",
        comparisons=comparisons,
        best=metrics,
        last_dev=last,
        historical=historical,
        source_subgroups_exploratory=subgroups,
        l2_article_errors=sorted(error, key=lambda r: r["delta"]),
        initialization_identical=True,
        test_used=False,
        criterion_is_development_heuristic=True,
    )
    write_json(ROOT / "analysis.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n, p in progress.items():
        hh = p["history"]
        epochs = [h["epoch"] for h in hh]
        if n != "l3-only":
            axes[0].plot(epochs, [h["dev"]["l2"]["brier"] for h in hh], label=n)
        if n != "l2-only":
            axes[1].plot(epochs, [h["dev"]["conditional"]["article_mean_f1"] for h in hh], label=n)
    for ax, title in zip(
        axes,
        ["L2: Brier на dev (меньше — лучше)", "L3: F1 на dev при заданных границах"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "learning-curves.png", dpi=170)
    plt.close(fig)
    for name, path in runs.items():
        p = progress[name]
        m = metrics[name]
        c = M.read(path / "config.json")["config"]
        text = [
            "# ST3 — " + name,
            "",
            f"Завершено20эпох,380обновлений,2900предъявлений статьи; L3-непустых предъявлений {p['l3_articles_seen']}. Выбрана эпоха{p['best_epoch']}. Seed42, свежие LoRA и головы, начальные параметры совпадают во всех четырёх условиях.",
            "",
            "## Постановка",
            "",
            f"Вес L2={c['l2_weight']}, вес L3={c['l3_weight']}. [Общий протокол](../../protocol.md). L2 выбирает лучший checkpoint по dev Brier; отдельный L3 — по условному F1. Отчёт использует одно выбранное состояние для обоих выходов.",
            "",
            "## Метрики",
            "",
            "| Выход / метрика | Train best | Dev best | Dev last20 |",
            "|---|---:|---:|---:|",
        ]
        for task, key, label in [
            ("l2", "brier", "L2 Brier ↓"),
            ("l2", "macro_f1", "L2 macro-F1"),
            ("l2", "micro_f1", "L2 micro-F1"),
            ("conditional", "article_mean_f1", "L3 F1 с заданными границами"),
            ("conditional", "macro_family_f1", "L3 macro-family F1"),
            ("chain_fixed", "exact_f1", "L3 сквозной exact-F1"),
            ("chain_fixed", "overlap_f1", "L3 сквозной overlap-F1"),
            ("chain_fixed", "exact_precision", "L3 precision"),
            ("chain_fixed", "exact_recall", "L3 recall"),
            ("candidate_diagnostics", "ap", "AP внутри top500×6"),
            ("chain_fixed", "mean_predictions", "Среднее число выдач"),
        ]:
            if task in m["dev"]:
                text.append(
                    f"| {label} | {num(m['train'][task][key])} | {num(m['dev'][task][key])} | {num(last[name][task][key])} |"
                )
        if "conditional" in m["dev"]:
            text += [
                "",
                f"Условный dev-порог={m['dev']['conditional']['threshold']}; автоматический порог фиксирован {cfg['threshold']}. После отдельной диагностической калибровки: порог{m['dev']['diagnostic_dev_threshold']:.6f}, dev exact-F1={m['dev']['chain_recalibrated']['exact_f1']:.4f}. Калибровка не меняет выбранные веса модели.",
                f"Правильных пустых ответов: {m['dev']['chain_fixed']['correctly_empty_answers']}/{m['dev']['chain_fixed']['empty_answers']}. AP исключает ответы без положительной типизированной пары внутри кандидатов: {len(m['dev']['candidate_diagnostics']['ap_excluded_answers'])}; их идентификаторы сохранены в metrics.json. Из сквозной оценки они не исключаются.",
                "",
                "| Семейство | F1 при заданных границах | Сквозной F1 |",
                "|---|---:|---:|",
            ]
            for f, v in m["dev"]["conditional"]["per_family_f1"].items():
                # Detailed end-to-end family structure retained verbatim in metrics JSON.
                text.append(
                    f"| {f} | {v:.4f} | {m['dev']['chain_fixed']['per_family_exact_f1'][f]:.4f} |"
                )
        if "l2" in m["dev"]:
            text += [
                "",
                "| Фрейм | Brier dev | F1 dev | Majority support |",
                "|---|---:|---:|---:|",
            ]
            for f in m["dev"]["l2"]["per_frame"]:
                text.append(
                    f"| {f['frame']} | {f['brier']:.4f} | {f['f1']:.4f} | {f['majority_support']} |"
                )
        text += [
            "",
            "## Воспроизводимость и предел вывода",
            "",
            "Полные start/best/last checkpoints включают optimizer, scheduler, RNG и позицию. Замороженная база проверена. Код и окружение сохранены в code/, metadata.json и uv.lock. [Кривые всей серии](../../learning-curves.png). Прогнозы best/train и best/dev, а также last/dev сохранены независимо. Финальная проверка общей серии — verification.json на два уровня выше.",
            "Один seed и многократно использованный dev; это разработческое наблюдение. Отсутствие L3 на статье маскирует условную ошибку, а не означает доказанное отсутствие риторики. Фиксированный генератор имеет ограниченный охват; положительное обучение классификатора не решает отказ от ложных кандидатов. Test остаётся закрыт.",
            "",
        ]
        (path / "report.md").write_text(
            re.sub(r"(?<=\d)(?=[А-Яа-я])|(?<=[А-Яа-я])(?=\d)", " ", "\n".join(text))
        )
    result = []
    for n in NAMES:
        d = metrics[n]["dev"]
        result.append(
            dict(
                condition=n,
                epoch=progress[n]["best_epoch"],
                brier=d.get("l2", {}).get("brier"),
                conditional=d.get("conditional", {}).get("article_mean_f1"),
                exact=d.get("chain_fixed", {}).get("exact_f1"),
            )
        )
    print(
        json.dumps(
            dict(results=result, comparisons=comparisons, selected=selected), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()

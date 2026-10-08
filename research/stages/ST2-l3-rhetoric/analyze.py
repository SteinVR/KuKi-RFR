"""Rebuild stage comparison and error diagnostics from final train/dev predictions."""

import json
from pathlib import Path

import numpy as np

from research.kuki.data import FAMILIES, write_json
from research.kuki.experiment import load_development
from research.kuki.l3_evaluation import empty_outputs, evaluate, f1, matched, select_spans

STAGE = Path("research/stages/ST2-l3-rhetoric")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def bootstrap(left, right, records, seed=42):
    difference = np.asarray(left) - np.asarray(right)
    groups = sorted({r["group_id"] for r in records})
    indices = [[i for i, r in enumerate(records) if r["group_id"] == g] for g in groups]
    sums = np.asarray([difference[i].sum() for i in indices])
    sizes = np.asarray([len(i) for i in indices])
    draws = np.random.default_rng(seed).integers(0, len(groups), (10000, len(groups)))
    distribution = sums[draws].sum(1) / sizes[draws].sum(1)
    return dict(
        difference=float(difference.mean()),
        interval_95=np.quantile(distribution, [0.025, 0.975]).tolist(),
        groups=len(groups),
        draws=10000,
        seed=seed,
        limitation="descriptive selected-dev fixed-prediction interval, no training-seed or selection correction",
    )


def family_metrics(docs, predictions):
    output = {}
    for family in FAMILIES:
        exact_nonempty, overlap_nonempty, counts = [], [], np.zeros(3)
        n_spans = n_positive_answers = n_pred = 0
        for doc, predicted in zip(docs, predictions, strict=True):
            p = [s for s in predicted if s["label"] == family]
            n_pred += len(p)
            ex, ov = [], []
            for answer in doc["annotations"]:
                g = [s for s in answer["l3_persuasion"] if s["label"] == family]
                counts += np.asarray(matched(p, g)) / len(doc["annotations"])
                n_spans += len(g)
                n_positive_answers += bool(g)
                if g:
                    ex.append(f1(*matched(p, g)))
                    ov.append(f1(*matched(p, g, overlap=True)))
            if ex:
                exact_nonempty.append(np.mean(ex))
                overlap_nonempty.append(np.mean(ov))
        output[family] = dict(
            observed_spans=n_spans,
            positive_answers=n_positive_answers,
            predicted_spans=n_pred,
            weighted_micro_exact_f1=f1(*counts),
            nonempty_article_mean_exact_f1=float(np.mean(exact_nonempty))
            if exact_nonempty
            else 0.0,
            nonempty_article_mean_overlap_f1=float(np.mean(overlap_nonempty))
            if overlap_nonempty
            else 0.0,
        )
    return output


def main():
    docs, _ = load_development(read(Path("research/configs/data.json")))
    runs = [
        STAGE / "experiments" / n
        for n in [
            "20260928-l3-pilot-s42-01",
            "20260928-l3-negative025-s42-01",
            "20260928-l3-hard-s42-01",
        ]
    ]
    agreement = []
    for d in docs["dev"]:
        answers = d["annotations"]
        if len(answers) < 2:
            continue
        pairs = []
        for i, left in enumerate(answers):
            for right in answers[:i]:
                pairs.append(
                    dict(
                        annotators=[left["annotator_id"], right["annotator_id"]],
                        exact=f1(*matched(left["l3_persuasion"], right["l3_persuasion"])),
                        overlap=f1(
                            *matched(left["l3_persuasion"], right["l3_persuasion"], overlap=True)
                        ),
                    )
                )
        agreement.append(
            dict(
                doc_id=d["doc_id"],
                pairs=pairs,
                exact=float(np.mean([r["exact"] for r in pairs])),
                overlap=float(np.mean([r["overlap"] for r in pairs])),
            )
        )
    write_json(
        STAGE / "dev-human-agreement.json",
        dict(
            n_articles=len(agreement),
            n_pairs=sum(len(r["pairs"]) for r in agreement),
            mean_exact=float(np.mean([r["exact"] for r in agreement])),
            mean_overlap=float(np.mean([r["overlap"] for r in agreement])),
            records=agreement,
            limitation="Multiply annotated dev subset only; neither a ceiling nor directly comparable to all-dev model quality",
        ),
    )
    summary = {}
    records = {}
    for run in runs:
        if not (run / "metrics.json").exists():
            continue
        m = read(run / "metrics.json")
        raw = rows(run / "dev-raw-predictions.jsonl")
        score, detail = evaluate(docs["dev"], raw, m["threshold"])
        assert score == m["dev"]
        pred = [select_spans(o, m["threshold"]) for o in raw]
        summary[run.name] = dict(
            epoch=m["selected_epoch"],
            threshold=m["threshold"],
            elapsed_seconds=m["elapsed_seconds"],
            dev=score,
            train=m["train"],
            families=family_metrics(docs["dev"], pred),
            cap_saturated_articles=sum(len(p) == 100 for p in pred),
            mean_candidates=float(np.mean([o["candidate_count"] for o in raw])),
            trainable_parameters=m["trainable_parameters"],
        )
        records[run.name] = detail
    if not summary:
        raise ValueError("No completed model runs")
    primary = "20260928-l3-pilot-s42-01"
    empty, empty_detail = evaluate(docs["dev"], empty_outputs(docs["dev"]), 0.5)
    comparisons = {}
    for name, detail in records.items():
        comparisons[name + "_minus_empty"] = bootstrap(
            [r["exact_f1"] for r in detail], [r["exact_f1"] for r in empty_detail], detail
        )
    alternate = "20260928-l3-negative025-s42-01"
    if alternate in records:
        comparisons["negative025_minus_primary"] = bootstrap(
            [r["exact_f1"] for r in records[alternate]],
            [r["exact_f1"] for r in records[primary]],
            records[primary],
        )
    hard = "20260928-l3-hard-s42-01"
    if hard in records:
        comparisons["hard_minus_primary"] = bootstrap(
            [r["exact_f1"] for r in records[hard]],
            [r["exact_f1"] for r in records[primary]],
            records[primary],
        )
    examples = []
    run = STAGE / "experiments" / primary
    if (run / "metrics.json").exists():
        raw = rows(run / "dev-raw-predictions.jsonl")
        threshold = summary[primary]["threshold"]
        for d, out, detail in zip(docs["dev"], raw, records[primary], strict=True):
            predicted = select_spans(out, threshold)
            examples.append(
                dict(
                    doc_id=d["doc_id"],
                    title=d["title"],
                    source=d["source"],
                    **{
                        k: detail[k]
                        for k in ["exact_f1", "overlap_f1", "proposal_recall", "n_predictions"]
                    },
                    predicted_examples=[
                        dict(**s, text=d["text"][s["start"] : s["end"]]) for s in predicted[:3]
                    ],
                )
            )
    write_json(
        STAGE / "analysis.json",
        dict(
            runs=summary,
            empty_baseline=empty,
            comparisons=comparisons,
            examples=examples,
            test_scored=False,
        ),
    )
    lines = [
        "# ST2 — сравнение на dev",
        "",
        "31 статья, 40 индивидуальных ответов. F1: среднее по ответам внутри статьи, затем по статьям. "
        "Пусто/пусто = 0; IoU требует взаимно-однозначного сопоставления и того же семейства.",
        "",
        "| Вариант | Эпоха / порог | Exact F1 | IoU≥0,5 F1 | Exact precision | Exact recall | Охват кандидатов на непустых | Прогнозов/статью |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        "| Пустой | — | 0 | 0 | 0 | 0 | 0 | 0 |",
    ]
    for name, r in summary.items():
        m = r["dev"]
        lines.append(
            f"| {name} | {r['epoch']} / {r['threshold']} | {m['exact_f1']:.4f} | {m['overlap_f1']:.4f} | "
            f"{m['exact_precision']:.4f} | {m['exact_recall']:.4f} | {m['proposal_recall']:.4f} | {m['mean_predictions']:.1f} |"
        )
    lines += [
        "",
        "## Семейства",
        "",
        "Условный F1 ниже рассчитан только по ответам, где данное семейство непусто, "
        "с усреднением по соответствующим статьям. Это дополнение к основной оценке по всем статьям.",
        "",
        "| Вариант / семейство | Dev отметок | Ответов с отметками | Прогнозов | F1 exact, непустые | F1 IoU, непустые |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in summary.items():
        for family, m in r["families"].items():
            lines.append(
                f"| {name} / {family} | {m['observed_spans']} | {m['positive_answers']} | {m['predicted_spans']} | "
                f"{m['nonempty_article_mean_exact_f1']:.4f} | {m['nonempty_article_mean_overlap_f1']:.4f} |"
            )
    lines += ["", "## Пустые ответы и ограничения кандидатов", ""]
    for name, r in summary.items():
        m = r["dev"]
        lines.append(
            f"- {name}: правильных пустых {m['correctly_empty_answers']}/{m['empty_answers']}; "
            f"ложный непустой ответ при пустой разметке {m['false_nonempty_given_empty']:.1%}; "
            f"статей на лимите 100 прогнозов: {r['cap_saturated_articles']}/31. "
            f"В среднем {r['mean_candidates']:.0f} пар кандидатов до классификации."
        )
    lines += ["", "## Описательная неопределённость", ""]
    for name, c in comparisons.items():
        lo, hi = c["interval_95"]
        lines.append(f"- {name}: Δ exact-F1 {c['difference']:+.5f}, 95% [{lo:+.5f}; {hi:+.5f}].")
    lines += [
        "",
        "Интервалы по фиксированным прогнозам на dev после выбора эпохи/порога; "
        "не учитывают подбор, seed или неизвестные событийные зависимости. Test не оценивался.",
        "",
    ]
    (STAGE / "comparison.md").write_text("\n".join(lines))
    print(json.dumps({name: r["dev"] for name, r in summary.items()}))


if __name__ == "__main__":
    main()

"""CPU-only metrics, article comparisons and reports for the predeclared ST4 matrix."""

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.kuki.data import FAMILIES, FRAMES, write_json
from research.kuki.evaluation import score as l2_score
from research.kuki.message import MODES
from research.kuki.span_classifier import examples, score, select_threshold

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
A = importlib.import_module("research.stages.ST3-l2-l3-joint.analyze")


def metrics(path, docs, cfg):
    devrows = examples(docs["dev"])
    devp = np.asarray(
        [r["probabilities"] for r in M.jsonl(path / "dev-conditional.jsonl")], dtype=np.float32
    )
    threshold = select_threshold(devrows, devp)["threshold"]
    result = {}
    for split, part in docs.items():
        rows = examples(part)
        saved = M.jsonl(path / f"{split}-conditional.jsonl")
        assert [(r["doc_id"], r["annotation_index"], r["start"], r["end"]) for r in saved] == [
            (r["doc_id"], r["annotation_index"], r["start"], r["end"]) for r in rows
        ]
        p = np.asarray([r["probabilities"] for r in saved], dtype=np.float32)
        chain = M.jsonl(path / f"{split}-chain.jsonl")
        fixed, articles, decoded = M.OLD.CHAIN.chain_score(
            part, chain, cfg["alpha"], cfg["threshold"]
        )
        write_json(path / f"{split}-chain-articles.json", articles)
        l2 = M.jsonl(path / f"{split}-l2.jsonl")
        base = M.jsonl(path / f"{split}-base-l2.jsonl")
        assert (
            [r["doc_id"] for r in l2] == [r["doc_id"] for r in chain] == [d["doc_id"] for d in part]
        )
        result[split] = dict(
            l2=l2_score(part, np.asarray([r["probabilities"] for r in l2], dtype=np.float32)),
            base_l2=l2_score(
                part, np.asarray([r["probabilities"] for r in base], dtype=np.float32)
            ),
            conditional=score(rows, p, threshold),
            chain_fixed=fixed,
            candidate_diagnostics=M.OLD.DIAGNOSTICS.candidate_diagnostics(
                part, chain, cfg["alpha"], cfg["threshold"]
            ),
        )
    write_json(path / "metrics.json", result)
    return result


def main():
    cfg = M.read(M.ROOT / "configs/continue.json")
    docs, _, _ = M.inputs(cfg)
    best, last, progress, runs = {}, {}, {}, {}
    for mode in MODES:
        cfg = M.read(M.ROOT / "configs" / f"{mode}.json")
        path = M.ROOT / "experiments" / cfg["run_id"]
        runs[mode] = path
        progress[mode] = M.read(path / "progress.json")
        assert progress[mode]["status"] == "completed"
        best[mode] = metrics(path / "best", docs, cfg)
        last[mode] = metrics(path / "last", {"dev": docs["dev"]}, cfg)["dev"]
    winner = min(
        ["uniform", "confidence", "contextual"], key=lambda n: best[n]["dev"]["l2"]["brier"]
    )
    predictions = {m: M.jsonl(runs[m] / "best/dev-l2.jsonl") for m in MODES}
    parent = M.jsonl(M.PARENT / "best/dev-l2.jsonl")
    comparisons = {}
    for mode in MODES:
        m = best[mode]["dev"]
        comp = {}
        for reference in ["continue", "self", "hidden", "ST3"]:
            ref = parent if reference == "ST3" else predictions[reference]
            comp[f"brier_vs_{reference}"] = A.bootstrap(
                docs["dev"], [r["brier"] for r in predictions[mode]], [r["brier"] for r in ref]
            )
        control = best["continue"]["dev"]
        comp["conditional_delta"] = (
            m["conditional"]["article_mean_f1"] - control["conditional"]["article_mean_f1"]
        )
        comp["exact_delta"] = m["chain_fixed"]["exact_f1"] - control["chain_fixed"]["exact_f1"]
        comp["criterion"] = dict(
            brier=comp["brier_vs_continue"]["mean"] <= -0.003,
            conditional=comp["conditional_delta"] >= -0.02,
            exact=comp["exact_delta"] >= -0.005,
        )
        comp["criterion_passed"] = all(comp["criterion"].values())
        left = m["conditional"]["article_f1"]
        right = control["conditional"]["article_f1"]
        eligible = [d for d in docs["dev"] if d["doc_id"] in left]
        assert set(left) == set(right)
        comp["conditional_vs_continue"] = A.bootstrap(
            eligible, [left[d["doc_id"]] for d in eligible], [right[d["doc_id"]] for d in eligible]
        )
        comp["exact_vs_continue"] = A.bootstrap(
            docs["dev"],
            [r["exact_f1"] for r in M.read(runs[mode] / "best/dev-chain-articles.json")],
            [r["exact_f1"] for r in M.read(runs["continue"] / "best/dev-chain-articles.json")],
        )
        comp["last_brier_vs_continue"] = A.bootstrap(
            docs["dev"],
            [r["brier"] for r in M.jsonl(runs[mode] / "last/dev-l2.jsonl")],
            [r["brier"] for r in M.jsonl(runs["continue"] / "last/dev-l2.jsonl")],
        )
        comparisons[mode] = comp
    errors = []
    for i, d in enumerate(docs["dev"]):
        new = predictions[winner][i]
        old = predictions["continue"][i]
        truth = np.asarray(new["vote_fractions"]) > 0.5
        pred = np.asarray(new["probabilities"]) >= 0.5
        errors.append(
            dict(
                doc_id=d["doc_id"],
                title=d["title"],
                source=d["source"],
                brier=new["brier"],
                control_brier=old["brier"],
                delta=new["brier"] - old["brier"],
                false_positive=[
                    f for f, t, p in zip(FRAMES, truth, pred, strict=True) if p and not t
                ],
                false_negative=[
                    f for f, t, p in zip(FRAMES, truth, pred, strict=True) if t and not p
                ],
            )
        )
    subgroups = {}
    for source in sorted({d["source"] for d in docs["dev"]}):
        ii = [i for i, d in enumerate(docs["dev"]) if d["source"] == source]
        subgroups[source] = dict(
            n=len(ii),
            **{m: float(np.mean([rr[i]["brier"] for i in ii])) for m, rr in predictions.items()},
        )
    diagnostics = (
        M.read(M.ROOT / "gpu-verification.json")
        if (M.ROOT / "gpu-verification.json").exists()
        else None
    )
    last_changes = {}
    message_amplitude = {}
    last_control = M.jsonl(runs["continue"] / "last/dev-l2.jsonl")
    for mode in MODES:
        actual = M.jsonl(runs[mode] / "last/dev-l2.jsonl")
        base = M.jsonl(runs[mode] / "last/dev-base-l2.jsonl")
        message_amplitude[mode] = dict(
            mean_absolute_probability_change=float(
                np.mean(
                    [
                        np.abs(np.asarray(a["probabilities"]) - np.asarray(b["probabilities"]))
                        for a, b in zip(actual, base, strict=True)
                    ]
                )
            ),
            brier_on_minus_off=float(
                np.mean([a["brier"] - b["brier"] for a, b in zip(actual, base, strict=True)])
            ),
        )
        last_changes[mode] = [
            dict(
                doc_id=d["doc_id"],
                title=d["title"],
                source=d["source"],
                delta_vs_continue=a["brier"] - b["brier"],
                delta_message_on_vs_off=a["brier"] - off["brier"],
            )
            for d, a, b, off in zip(docs["dev"], actual, last_control, base, strict=True)
        ]
    result = dict(
        selected_message=winner,
        last_article_diagnostics=last_changes,
        last_message_amplitude=message_amplitude,
        selected_initial_modes=[m for m in MODES if progress[m]["best_epoch"] == 0],
        best=best,
        last_dev=last,
        comparisons=comparisons,
        source_subgroups_exploratory=subgroups,
        article_errors=sorted(errors, key=lambda r: r["delta"]),
        gpu_diagnostics=diagnostics,
        test_used=False,
    )
    write_json(M.ROOT / "analysis.json", result)
    write_json(
        M.ROOT / "selection.json",
        dict(
            selected_message=winner,
            exact_ties=[
                n
                for n in ["uniform", "confidence", "contextual"]
                if best[n]["dev"]["l2"]["brier"] == best[winner]["dev"]["l2"]["brier"]
            ],
            criterion_passed=comparisons[winner]["criterion_passed"],
            selection="Minimum dev L2 Brier among uniform/confidence/contextual, single selected checkpoint",
            joint_ST3_retained=True,
            test_used=False,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode in MODES:
        hh = progress[mode]["history"]
        initial = M.read(runs[mode] / "initialization.json")["initial_metrics"]
        axes[0].plot(
            [0] + [r["epoch"] for r in hh],
            [initial["l2"]["brier"]] + [r["dev"]["l2"]["brier"] for r in hh],
            label=mode,
        )
        axes[1].plot(
            [0] + [r["epoch"] for r in hh],
            [initial["conditional"]["article_mean_f1"]]
            + [r["dev"]["conditional"]["article_mean_f1"] for r in hh],
            label=mode,
        )
    for ax, title in zip(axes, ["L2: Brier на dev ↓", "L3: условный F1 на dev ↑"], strict=True):
        ax.set_title(title)
        ax.set_xlabel("Дополнительная эпоха")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(M.ROOT / "learning-curves.png", dpi=160)
    plt.close(fig)
    table = [
        "| Вариант | Эпоха | Brier ↓ | macro-F1 | Условный L3 F1 | Сквозной exact-F1 | Last Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        m = best[mode]["dev"]
        table.append(
            f"| {mode} | {progress[mode]['best_epoch']} | {m['l2']['brier']:.4f} | {m['l2']['macro_f1']:.4f} | {m['conditional']['article_mean_f1']:.4f} | {m['chain_fixed']['exact_f1']:.4f} | {last[mode]['l2']['brier']:.4f} |"
        )
        text = [
            f"# ST4 — {mode}",
            "",
            f"Выполнено 8 дополнительных эпох, 152 updates, 1160 предъявлений статьи; выбрана эпоха {progress[mode]['best_epoch']} (0 — исходное состояние). Начало: joint200 best, ещё 152 исторических updates и 1160 предъявлений. Один seed42.",
            "",
            "[Протокол](../../protocol.md). Общий энкодер и L3 loss продолжают обучаться; основной отбор по L2 Brier. Test закрыт.",
            "",
            "| Метрика | Train best | Dev best | Dev last |",
            "|---|---:|---:|---:|",
        ]
        for task, key, label in [
            ("l2", "brier", "L2 Brier"),
            ("l2", "macro_f1", "L2 macro-F1"),
            ("conditional", "article_mean_f1", "Условный L3 F1"),
            ("chain_fixed", "exact_f1", "Сквозной exact-F1"),
            ("chain_fixed", "overlap_f1", "Сквозной overlap-F1"),
            ("candidate_diagnostics", "ap", "AP внутри кандидатов"),
        ]:
            text.append(
                f"| {label} | {best[mode]['train'][task][key]:.4f} | {m[task][key]:.4f} | {last[mode][task][key]:.4f} |"
            )
        text += [
            "",
            f"Brier при отключённой поправке: {m['base_l2']['brier']:.4f}. Это диагностическое вмешательство, не отдельно выбранная модель.",
            "",
            "Train использует OOF-кандидаты трёх детекторов, dev — исторический full-train детектор. Сравнение train/dev не изолирует переобучение классификатора. Порог условного L3 выбран на dev данного checkpoint; автоматический decoder неизменен.",
            "",
            f"Ресурсы: {M.read(runs[mode] / 'resources.json')}",
            f"Время процесса с GPU: {next(r['seconds'] for r in M.read(M.ROOT / 'budget.json')['runs'] if r.get('condition') == mode):.2f}с.",
            "",
            "Конфигурация, точный снимок исходников code-tree, metadata, checkpoints и прогнозы сохранены рядом. Групповые сравнения, разбор ошибок и ограничения — в [общем отчёте](../../report.md).",
        ]
        text += [
            "",
            "## Знаменатели, пустые ответы и редкие семейства",
            "",
            f"Условный L3: {m['conditional']['n_articles']} статей, {m['conditional']['n_answers']} непустых ответов, {m['conditional']['n_spans']} пар. Порог {m['conditional']['threshold']:.2f}; аргмакс применяется только к условной оценке.",
            f"Автоматический dev L3: {m['chain_fixed']['n_articles']} статья, {m['chain_fixed']['n_answers']} ответов. Правильных отказов {m['chain_fixed']['correctly_empty_answers']}/{m['chain_fixed']['empty_answers']}; empty/empty F1=0. Фиксированный decoder alpha={cfg['alpha']}, threshold={cfg['threshold']}, cap100.",
            f"Dev AP: {m['candidate_diagnostics']['ap_eligible_answers']} ответов из {m['chain_fixed']['n_answers']}, {m['candidate_diagnostics']['ap_eligible_articles']} статей. Исключено {len(m['candidate_diagnostics']['ap_excluded_answers'])} ответов без положительных в top500×6; их список — metrics.json. Это ранжирование внутри списка, не сквозная полнота.",
            "",
            "| Семейство | Условный F1 best | Условный F1 last | Автоматический exact-F1 best | Автоматический exact-F1 last |",
            "|---|---:|---:|---:|---:|",
        ]
        for family in FAMILIES:
            text.append(
                f"| {family} | {m['conditional']['per_family_f1'][family]:.4f} | {last[mode]['conditional']['per_family_f1'][family]:.4f} | {m['chain_fixed']['per_family_exact_f1'][family]:.4f} | {last[mode]['chain_fixed']['per_family_exact_f1'][family]:.4f} |"
            )
        (runs[mode] / "report.md").write_text("\n".join(text) + "\n")
    c = comparisons[winner]
    budget = M.read(M.ROOT / "budget.json")
    if all(progress[m]["best_epoch"] == 0 for m in MODES):
        lead = "**Ни один из шести вариантов не улучшил исходный ST3 по dev Brier: во всех ветках выбран начальный checkpoint. Локальный критерий ST4 не выполнен.** Техническое разрешение равенства между тремя message-вариантами в selection.json не означает превосходства одного из них."
    else:
        lead = f"Выбранный вариант передачи по dev Brier: **{winner}**. Локальный критерий ST4 относительно continue: **{'выполнен' if c['criterion_passed'] else 'не выполнен'}**."
    text = [
        "# ST4: явная передача L3 → L2",
        "",
        lead,
        "",
        "Это ограниченная разработческая серия. Для вывода о пользе семантического канала необходимо учитывать self/hidden, неизменный ST3, оба выхода и многократный выбор по dev. Новый научный метод не заявляется.",
        "",
        "[Исследовательский разбор и следующий возможный шаг](interpretation.md). [Итоговый self-review](self-review.md).",
        "",
        "## Что сравнивается",
        "",
        "[Протокол до обучения](protocol.md): шесть веток из одного joint200, одинаковые 8 дополнительных эпох и сброс optimizer/scheduler. continue — без сообщения; self — собственные признаки текста; hidden — признаки предложенных span; uniform/confidence/contextual — явные семейства, положение и оценка детектора с разной агрегацией. Contextual использует стандартное контекстное внимание.",
        "",
        "OOF: три независимых детектора с чистой базы, по 13 эпох. Каждая train-статья получает предложения источника, не обученного на её группе. Для dev сохранён прежний full-train детектор. Внутренний классификатор семейств совместный; gold-фрагменты не входят в L2 forward. [Manifest](oof/manifest.json).",
        "",
        "## Результаты",
        "",
        *table,
        "",
        f"Для {winner}: ΔBrier относительно continue {c['brier_vs_continue']['mean']:+.5f}, описательный 95% интервал {c['brier_vs_continue']['interval95']}; относительно ST3 {c['brier_vs_ST3']['mean']:+.5f}; self {c['brier_vs_self']['mean']:+.5f}; hidden {c['brier_vs_hidden']['mean']:+.5f}.",
        "",
        "Столбец last показывает одинаковые 152 дополнительные updates. Best имеет одинаковый бюджет выбора, но может иметь различное число фактических updates. Исходное состояние участвует в выборе во всех ветках.",
        "",
        "![Кривые](learning-curves.png)",
        "",
        "## Полезная и вредная передача",
        "",
        "Ниже — крайние изменения Brier относительно continue, выбранные после расчёта как иллюстрации, не независимые проверки. Разбивка всех 31 статей и четырёх источников — в analysis.json.",
    ]
    changed_errors = [r for r in errors if abs(r["delta"]) > 1e-12]
    if not changed_errors:
        text[-1] = (
            "Выбранное состояние даёт те же прогнозы, что continue. Поэтому случаев полезной или вредной передачи в выбранном состоянии нет. Ниже отдельно рассматриваются обученные last-состояния; они не заменяют исходный критерий выбора."
        )
    for row in (
        sorted(changed_errors, key=lambda r: r["delta"])[:3]
        + sorted(changed_errors, key=lambda r: r["delta"])[-3:]
    ):
        text.append(
            f"- {row['doc_id']} ({row['source']}): ΔBrier {row['delta']:+.4f}; {row['title']}."
        )
    initial_modes = [m for m in MODES if progress[m]["best_epoch"] == 0]
    text += [
        "",
        f"Исходное состояние выбрано у: {', '.join(initial_modes) or 'ни одной ветки'}. Совпадение начальных прогнозов не является свидетельством эквивалентности обученных механизмов.",
        "",
    ]
    text += [
        "| Last после 152 updates | Brier | Brier без поправки | Улучшено / ухудшено статей против continue |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        changes = last_changes[mode]
        better = sum(r["delta_vs_continue"] < -1e-12 for r in changes)
        worse = sum(r["delta_vs_continue"] > 1e-12 for r in changes)
        text.append(
            f"| {mode} | {last[mode]['l2']['brier']:.5f} | {last[mode]['base_l2']['brier']:.5f} | {better} / {worse} |"
        )
    if diagnostics and diagnostics.get("status") == "passed":
        text += [
            "",
            "Диагностика последнего обученного состояния: циклическая перестановка явных семейств/позиций/оценок между кандидатами одной статьи. Это последующая проверка чувствительности, не критерий отбора.",
            "",
        ]
        for mode in ["uniform", "confidence", "contextual"]:
            dm = diagnostics["interventions"][f"{mode}/last"]
            text.append(
                f"- {mode}: Brier обычный {dm['normal']['brier']:.6f}, shuffle {dm['shuffle']['brier']:.6f}."
            )
        text += [
            "",
            "Перестановка сохраняет наборы и средние значения семантических компонентов. Она проверяет их привязку к конкретным фрагментам, а не полное удаление семантики; слабая чувствительность не доказывает неиспользование семейств.",
        ]
    text += [
        "",
        "## Ресурс, проверки и ограничения",
        "",
        f"Расход: {budget['seconds'] / 60:.2f} из 60 минут GPU-процессов, включая OOF и технические проверки. CPU-анализ не является GPU-временем. Подробности: [бюджет](budget.json), [профиль](profile.json), [верификация](verification.json).",
        "",
        "Один seed; dev 31 статья многократно использован для выбора ST2/ST3/ST4. Bootstrap не учитывает этот выбор и изменчивость обучения. OOF-модели обучены на меньшем корпусе и имеют другое расписание, чем исторический full-train источник; OOF train/dev-сдвиг сохраняется. Гиперпараметры OOF перенесены из исторической разработки, это не полностью вложенная независимая оценка. Автоматический L3 остаётся отдельным ограничением. Градиент L2 через явные вероятности семейств отключён, через скрытые признаки разрешён; результат относится к этому режиму. Контроли различаются числом параметров, FLOPs не уравнены. Test закрыт, ST5/ST6 не начаты.",
        "",
        "В отличие от L2 в ST3, модель с сообщением требует детектор и при применении. Здесь инференс L2/L3 использует сохранённые предложения; полная задержка детектор→сообщение→L2 отдельно не измерялась. В стоимость серии входят новые OOF-источники; историческое обучение full-train детектора и ST3 повторно не выполнялось.",
        "",
        "Диагностики off/shuffle хранятся в gpu-verification.json и analysis.json; они не меняют выбор. Прогнозы, посемейные/пофреймовые метрики и индивидуальные отчёты сохранены в experiments/.",
    ]
    (M.ROOT / "report.md").write_text("\n".join(text) + "\n")
    print(
        dict(
            selected=winner,
            criterion=c["criterion_passed"],
            brier=best[winner]["dev"]["l2"]["brier"],
        )
    )


if __name__ == "__main__":
    main()

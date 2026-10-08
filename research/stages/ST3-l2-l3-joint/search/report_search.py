"""Russian research reports for each new condition and the bounded search."""

import importlib
from pathlib import Path

from research.kuki.data import write_json

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path(__file__).parent
JOINT = ["joint025", "joint100", "joint200", "joint400"]
NEW = ["joint200", "joint400"]


def n(v):
    return "—" if v is None else f"{v:.4f}"


def main():
    a = M.read(ROOT / "analysis.json")
    budget = M.read(ROOT / "budget.json")
    metrics = a["best"]
    runs = {
        name: ROOT / "experiments" / M.read(ROOT / "configs" / f"{name}.json")["run_id"]
        for name in NEW
    }
    for name, path in runs.items():
        c = M.read(path / "config.json")["config"]
        p = M.read(path / "progress.json")
        comp = a["comparisons"][name]
        md = metrics[name]
        last = a["last_dev"][name]
        chosen = next(h for h in p["history"] if h["epoch"] == p["best_epoch"])
        final = p["history"][-1]
        text = [
            f"# ST3: {name}, вес L3 = {c['l3_weight']}",
            "",
            "## Исследовательский вопрос",
            "",
            f"Улучшает ли увеличение веса условной ошибки L3 до {c['l3_weight']} компромисс двух выходов по сравнению с λ=1? Исходная гипотеза: L3 в совместном обучении недополучает относительный вес. Это не гипотеза о недостаточном числе эпох: исходные кривые уже показывали ухудшение после пика.",
            "",
            "## Метод и воспроизводимость",
            "",
            "[Протокол расширения](../../protocol.md). Отдельное обучение со свежими LoRA и головами, один общий encoder RuModernBERT, seed 42. Train/dev: 145/31 статья; тот же групповой split, тест закрыт. 20 эпох, 380 updates, 2900 предъявлений статей, 2580 предъявлений с непустой L3-разметкой. Один разметчик и одна статья не превращаются в множество независимых наблюдений из-за числа фрагментов.",
            "",
            "Вес L2=1. LoRA rank 8, alpha 16, dropout 0.1; attention Wqkv/Wo; замороженная BF16-база. Adapter LR=0.0002, head LR=0.001, AdamW decay 0.01, warmup 38 updates, линейное снижение LR до нуля, clip norm 1. Accumulation 8 статей с корректной нормировкой последнего batch. Полная статья через окна 1024/128, перекрытие учитывается. Инициализация и версии библиотек обучения совпали со всей исходной серией.",
            "",
            "Условный L3-классификатор обучается на размеченных фрагментах. При отсутствии размеченной L3 на статье условная ошибка маскируется; это не доказательство отсутствия риторики. Фиксированный GlobalPointer предлагает до 500 пар границ. Автоматическая цепочка использует logit вероятности семейства + 2 × logit детектора, порог −5.172313813576109 и максимум 100 выходных фрагментов.",
            "",
            f"Checkpoint выбран по минимуму L2 dev Brier: эпоха {p['best_epoch']}, {chosen['updates']} updates. **Все основные L3-метрики ниже относятся к тому же checkpoint.** Максимум L3 на другой эпохе не подставляется в результат.",
            "",
            "## Результаты",
            "",
            "| Метрика | Train выбранной модели | Dev выбранной модели | Dev после 20 эпох |",
            "|---|---:|---:|---:|",
        ]
        for task, key, label in [
            ("l2", "brier", "L2 Brier ↓"),
            ("l2", "macro_f1", "L2 macro-F1 ↑"),
            ("l2", "micro_f1", "L2 micro-F1 ↑"),
            ("conditional", "article_mean_f1", "L3 условный F1 ↑"),
            ("conditional", "macro_family_f1", "L3 macro-family F1 ↑"),
            ("chain_fixed", "exact_f1", "Сквозной exact-F1 ↑"),
            ("chain_fixed", "overlap_f1", "Сквозной overlap-F1 ↑"),
            ("chain_fixed", "exact_precision", "Сквозной precision ↑"),
            ("chain_fixed", "exact_recall", "Сквозной recall ↑"),
            ("candidate_diagnostics", "ap", "AP внутри top500 × 6 ↑"),
            ("chain_fixed", "mean_predictions", "Среднее число выдач"),
        ]:
            text.append(
                f"| {label} | {n(md['train'][task][key])} | {n(md['dev'][task][key])} | {n(last[task][key])} |"
            )
        text += [
            "",
            "## Сравнение и предзаданные критерии",
            "",
            "| Сравнение | Разность | Описательный 95% интервал |",
            "|---|---:|---|",
        ]
        for key, label in [
            ("brier_vs_matched", "Brier − отдельный L2"),
            ("brier_vs_previous_joint100", "Brier − прежний joint100"),
            ("conditional_vs_matched", "Условный F1 − отдельный L3"),
            ("exact_vs_matched", "Exact-F1 − отдельный L3"),
        ]:
            v = comp[key]
            lo, hi = v["interval95"]
            text.append(f"| {label} | {v['mean']:+.4f} | [{lo:+.4f}; {hi:+.4f}] |")
        text += [
            "",
            "Bootstrap: 10 000 выборок групп статей, seed 42. Интервалы описывают фиксированные выбранные dev-прогнозы; они не учитывают поиск веса, выбор эпохи и вариативность обучения.",
            "",
            f"Прирост Brier ≥0.003: **{comp['criterion']['brier_gain_at_least_003']}**. Потеря условного F1 ≤0.02: **{comp['criterion']['conditional_drop_at_most_02']}**. Потеря exact-F1 ≤0.005: **{comp['criterion']['exact_drop_at_most_005']}**. Общий критерий: **{comp['criterion_passed']}**.",
            "",
            "## Калибровка и ограничение генератора",
            "",
            f"Условный порог на dev: {md['dev']['conditional']['threshold']}. Дополнительный подбор только порога автоматической цепочки на dev даёт exact-F1={md['dev']['chain_recalibrated']['exact_f1']:.4f}, порог {md['dev']['diagnostic_dev_threshold']:.6f}. Это диагностический максимум после просмотра dev, а не замена основного фиксированного decoder и не результат на test.",
            f"Правильных пустых ответов: {md['dev']['chain_fixed']['correctly_empty_answers']}/{md['dev']['chain_fixed']['empty_answers']}. Для AP исключены {len(md['dev']['candidate_diagnostics']['ap_excluded_answers'])} ответа без положительной типизированной пары внутри кандидатов; они сохранены в сквозной оценке. Идентификаторы исключений находятся в metrics.json.",
            "",
            "## Динамика обучения",
            "",
            f"Минимум Brier выбран на эпохе {p['best_epoch']}. Максимум условного L3 F1 по всей кривой: {a['curves'][name]['max_conditional_f1']:.4f} на эпохе {a['curves'][name]['max_conditional_epoch']}; это отдельная диагностика. К эпохе 20 learning rate достигает нуля.",
            f"Ошибка обучения L2: {chosen['train_losses']['l2']:.4f} на выбранной эпохе → {final['train_losses']['l2']:.4f} на последней; L3: {chosen['train_losses']['l3']:.4f} → {final['train_losses']['l3']:.4f}. [Кривые сравнения](../../learning-curves.png).",
            "",
            "## Разбивка по классам",
            "",
            "| Семейство | Условный F1 | Сквозной exact-F1 |",
            "|---|---:|---:|",
        ]
        for f, v in md["dev"]["conditional"]["per_family_f1"].items():
            text.append(
                f"| {f} | {v:.4f} | {md['dev']['chain_fixed']['per_family_exact_f1'][f]:.4f} |"
            )
        text += ["", "| Фрейм L2 | Brier | F1 | Majority support |", "|---|---:|---:|---:|"]
        for f in md["dev"]["l2"]["per_frame"]:
            text.append(
                f"| {f['frame']} | {f['brier']:.4f} | {f['f1']:.4f} | {f['majority_support']} |"
            )
        text += [
            "",
            "## Ошибки по статьям",
            "",
            "Пять наибольших ухудшений Brier относительно отдельного L2; это диагностика dev, а не новая выборка.",
            "",
            "| Статья | Brier модели | Brier L2-only | Разность |",
            "|---|---:|---:|---:|",
        ]
        rr = M.jsonl(path / "best/dev-l2.jsonl")
        ctrl = M.jsonl(ROOT.parent / "experiments/20260929-l2-only-s42-01/best/dev-l2.jsonl")
        err = [
            dict(
                doc_id=r["doc_id"],
                brier=r["brier"],
                control=o["brier"],
                delta=r["brier"] - o["brier"],
            )
            for r, o in zip(rr, ctrl, strict=True)
        ]
        write_json(path / "l2-article-deltas.json", sorted(err, key=lambda x: x["delta"]))
        for e in sorted(err, key=lambda x: x["delta"], reverse=True)[:5]:
            text.append(
                f"| {e['doc_id']} | {e['brier']:.4f} | {e['control']:.4f} | {e['delta']:+.4f} |"
            )
        execution = M.read(path / "execution.json")
        text += [
            "",
            "## Проверка и предел вывода",
            "",
            f"GPU-процесс: {execution['seconds']:.2f} секунд ({execution['seconds'] / 60:.2f} минуты). Start/best/last checkpoints включают optimizer, scheduler, RNG и позицию. Полные прогнозы, снимки кода, lockfile и metadata сохранены рядом. [Независимая проверка](../../verification.json), [свежая загрузка на GPU](../../gpu-verification.json).",
            "Один seed и небольшой многократно использованный dev. Сравнение подтверждает только наблюдаемую разницу в этой разработческой серии. Общая модель не признана пригодной заменой отдельных моделей при провале хотя бы одного предзаданного критерия. Ещё больше эпох с этим же завершённым scheduler не является обоснованным способом улучшения.",
            "",
        ]
        (path / "report.md").write_text(
            "\n".join(text)
            .replace("**True**", "**выполнен**")
            .replace("**False**", "**не выполнен**")
            .replace("| False |", "| нет |")
            .replace("| True |", "| да |")
        )
    chosen = a["selected_joint"]
    feasible = a["feasible_joint"]
    passed = a["comparisons"][chosen]["criterion_passed"]
    before = metrics["joint100"]["dev"]
    after = metrics[chosen]["dev"]
    text = [
        "# ST3: расширение подбора веса L3 через Optuna",
        "",
        f"Лучший dev Brier среди четырёх весов: **{chosen}**, {metrics[chosen]['dev']['l2']['brier']:.4f}. Общий предзаданный критерий двух выходов для него: **{passed}**. Допустимые условия среди всех проверенных: **{', '.join(feasible) if feasible else 'нет'}**.",
        "",
        "## Что проверено",
        "",
        f"По сравнению с предыдущим выбранным joint100: Brier {before['l2']['brier']:.4f} → {after['l2']['brier']:.4f}; условный L3 F1 {before['conditional']['article_mean_f1']:.4f} → {after['conditional']['article_mean_f1']:.4f}; сквозной exact-F1 {before['chain_fixed']['exact_f1']:.4f} → {after['chain_fixed']['exact_f1']:.4f}. Это измеренный резерв первоначального диапазона весов. Результат сильнее прежнего, но не устанавливает максимум архитектуры.",
        "",
        "После исходных λ=0.25 и 1 добавлены ровно два отдельных обучения: λ=2 и 4. Смена единственного веса проверяет конкретную гипотезу о недостаточном относительном весе L3. Все эпохи, порядок данных, LR, инициализация и критерии остались прежними. [Протокол до запуска](protocol.md); исходный ST3 сохранён без переписывания.",
        "",
        "| Модель | Выбранная эпоха | L2 Brier ↓ | Условный L3 F1 ↑ | Сквозной exact-F1 ↑ | AP ↑ | Все критерии |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ["l2-only", "l3-only", *JOINT]:
        m = metrics[name]["dev"]
        cur = a["curves"][name]
        text.append(
            f"| {name} | {cur['selected_epoch']} | {n(m.get('l2', {}).get('brier'))} | {n(m.get('conditional', {}).get('article_mean_f1'))} | {n(m.get('chain_fixed', {}).get('exact_f1'))} | {n(m.get('candidate_diagnostics', {}).get('ap'))} | {a['comparisons'][name]['criterion_passed'] if name in JOINT else 'контроль'} |"
        )
    text += [
        "",
        "Условный L3 F1 оценивает классификацию при заданных человеком границах; сквозной F1 оценивает реальные фрагменты с типом, выданные фиксированной автоматической цепочкой. Эти метрики не взаимозаменяемы. Для каждой совместной модели обе метрики сняты с одного выбранного по L2 checkpoint.",
        "",
        f"Победа по основному Brier не означает улучшения каждого показателя L2: macro-F1 меняется {before['l2']['macro_f1']:.4f} → {after['l2']['macro_f1']:.4f}. Для сравнения Brier нового победителя с joint100 описательный 95% интервал разности включает ноль: [{a['comparisons'][chosen]['brier_vs_previous_joint100']['interval95'][0]:+.4f}; {a['comparisons'][chosen]['brier_vs_previous_joint100']['interval95'][1]:+.4f}]. Устойчивость небольшого дополнительного прироста ещё не подтверждена.",
        "",
        "## Что сделала Optuna",
        "",
        "Optuna 5.0.0 сохранила две новые попытки в SQLite-study с параметрами, временем, выбранной эпохой и двумя целями: Brier вниз и условный F1 вверх. Использован GridSampler с заранее заданными точками 2 и 4, а не TPE. Это воспроизводимый ограниченный поиск; улучшение, если оно есть, нельзя приписывать интеллектуальному алгоритму выбора Optuna.",
        "",
        f"Pareto-набор среди всех четырёх совместных вариантов: **{', '.join(a['pareto_joint'])}**. В него входят варианты, у которых нельзя улучшить один из двух выбранных показателей другим проверенным вариантом, не ухудшив другой. Этот набор не заменяет защитный критерий сквозного exact-F1.",
        "",
        "Для действительно адаптивного поиска нужен больший бюджет: сначала пробные конфигурации, затем TPE предлагает следующие по накопленным наблюдениям. По умолчанию TPESampler использует 10 стартовых trials; это настраивается, но два новых обучения не дают содержательной проверки его преимущества. Hyperband экономит ресурс ранним отсевом, однако в нашей серии L2 и L3 достигают пиков в разные эпохи; ранний отсев по L2 рискует удалять полезные для L3 варианты.",
        "",
        "Источники: [TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [GridSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html), [multi-objective](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html), [Hyperband](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html).",
        "",
        "## Хватило ли эпох",
        "",
        "![Кривые обучения](learning-curves.png)",
        "",
        "| Условие | Выбранная L2 эпоха | Максимум условного L3 F1 / эпоха | L2 Brier на эпохе 20 | L3 F1 на эпохе 20 |",
        "|---|---:|---|---:|---:|",
    ]
    for name in JOINT:
        cur = a["curves"][name]
        last = a["last_dev"][name]
        text.append(
            f"| {name} | {cur['selected_epoch']} | {cur['max_conditional_f1']:.4f} / {cur['max_conditional_epoch']} | {last['l2']['brier']:.4f} | {last['conditional']['article_mean_f1']:.4f} |"
        )
    text += [
        "",
        "Конец 20-й эпохи соответствует нулевому LR. Продолжение потребовало бы другого расписания. По этим данным нельзя объявить достигнутым максимум архитектуры; проверено конкретное расписание и четыре веса, а не всё пространство гиперпараметров.",
        "",
        "Дополнительная проверка всей истории (диагностическая, без изменения правила выбора): эпохи, одновременно прошедшие требования Brier и условного F1: "
        + "; ".join(
            f"{name}: {a['curves'][name]['epochs_passing_L2_and_conditional_guards_exploratory']}"
            for name in JOINT
        )
        + ". Сквозная цепочка для всех эпох не вычислялась; эта проверка охватывает только первые два требования.",
        "",
        "## Сравнение по источникам",
        "",
        "Разбивка является описательной: малые группы не используются для новых решений или заявлений об обобщении.",
        "",
        "| Источник | Статей | L2-only | λ=.25 | λ=1 | λ=2 | λ=4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, s in a["source_subgroups_exploratory"].items():
        text.append(
            "| " + " | ".join([source, str(s["n"])] + [n(s[k]) for k in ["l2-only", *JOINT]]) + " |"
        )
    text += [
        "",
        "## Ресурсы и проверки",
        "",
        f"Исходная серия: {budget['original_seconds'] / 60:.2f} мин. Расширение, включая свежую GPU-проверку: {budget['seconds'] / 60:.2f} мин. Всего: {(budget['original_seconds'] + budget['seconds']) / 60:.2f} из 60 минут. Бюджет не расширен. CPU-пересчёт метрик не учитывается как GPU-время.",
        "Начальные параметры и версии torch/transformers/peft и остальных библиотек обучения совпали. Optuna добавила только пять новых пакетов: optuna, alembic, colorlog, mako, sqlalchemy. Исторические пакеты не обновлялись. Независимый пересчёт Brier, exact-F1 и AP и свежий replay checkpoint перечислены в verification.json.",
        "",
        "## Отдельные эксперименты и артефакты",
        "",
    ]
    for name, path in runs.items():
        text.append(
            f"- [{name}: подробный исследовательский отчёт](experiments/{path.name}/report.md)."
        )
    text += [
        "- [Все метрики и bootstrap](analysis.json), [решение](selection.json), [Optuna study export](study.json), [бюджет](budget.json), [проверки](verification.json).",
        "",
        "## Следующий содержательный поиск",
        "",
        "Рекомендация для отдельной будущей серии: TPE по небольшому пространству веса L3 и learning rate адаптера, с заранее ограниченным числом полных обучений. Именно сочетание веса и скорости обновления может изменить наблюдаемый компромисс; это гипотеза, ещё не проверенный результат. Отбор проводить на внутренних групповых разбиениях train, затем проверить лидеров на нескольких seed. Число запусков, правила допуска L3 и финальный способ выбора модели закрепить до поиска. Бесконечный перебор по нынешнему dev будет увеличивать риск подгонки к 31 статье.",
        "",
        "## Ограничения вывода",
        "",
        "Seed один. Dev содержит 31 статью и уже многократно использовался для выбора. Описательные bootstrap-интервалы не устраняют переоптимизацию dev. Test не использовался и ST4 не запускался. Генератор границ остаётся фиксированным, поэтому улучшение классификации не гарантирует улучшения всей цепочки. Для более широкого поиска следует заранее выделить новый бюджет и внутренние групповые разбиения в train; закрытый test не превращать в очередную поверхность настройки.",
        "",
    ]
    (ROOT / "report.md").write_text(
        "\n".join(text)
        .replace("**True**", "**выполнен**")
        .replace("**False**", "**не выполнен**")
        .replace("| False |", "| нет |")
        .replace("| True |", "| да |")
    )


if __name__ == "__main__":
    main()

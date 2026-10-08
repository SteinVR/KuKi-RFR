"""Write E1/E3 research reports after completed independent runs and chain evaluation."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.kuki.data import FAMILIES, write_json
from research.kuki.span_classifier import score

ROOT = Path(__file__).parent
RUNS = ROOT.parent / "experiments"


def read(path):
    return json.loads(path.read_text())


def runs(kind):
    result = []
    for variant in ["base", "low-lr"]:
        p = RUNS / f"20260928-{kind}-{variant}-s42-01"
        result.append(
            dict(
                path=p,
                metrics=read(p / "metrics.json"),
                history=read(p / "history.json"),
                config=read(p / "config.json")["config"],
            )
        )
    return result


def e1():
    rr = runs("classifier")
    selected = max(
        rr,
        key=lambda r: (
            r["metrics"]["dev"]["article_mean_f1"],
            r["metrics"]["dev"]["macro_family_f1"],
            -r["metrics"]["dev"]["brier"],
        ),
    )
    base = read(ROOT / "E1-classification/baselines.json")
    m = selected["metrics"]
    out = ROOT / "E1-classification"
    baseline = max(
        base["results"], key=lambda name: base["results"][name]["dev"]["article_mean_f1"]
    )
    a = m["dev"]["article_f1"]
    b = base["results"][baseline]["dev"]["article_f1"]
    assert a.keys() == b.keys()
    diffs = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(42)
    boot = diffs[rng.integers(0, len(diffs), (5000, len(diffs)))].mean(1)
    diagnostic = dict(
        selected_run=str(selected["path"]),
        best_baseline=baseline,
        paired_dev_difference=float(diffs.mean()),
        bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
        warning="descriptive post-selection dev interval; fixed fitted models; does not include training/selection uncertainty",
    )
    rows = [
        json.loads(x) for x in (selected["path"] / "dev-predictions.jsonl").read_text().splitlines()
    ]
    prob = np.array([r["probabilities"] for r in rows])
    y = np.array([r["target"] for r in rows])
    prediction = prob >= m["dev"]["threshold"]
    prediction[np.arange(len(rows)), prob.argmax(1)] = True
    weights = np.array([r["weight"] for r in rows])
    diagnostic["raw_family_counts"] = {
        f: dict(
            tp=int((prediction[:, i] & y[:, i].astype(bool)).sum()),
            fp=int((prediction[:, i] & ~y[:, i].astype(bool)).sum()),
            fn=int((~prediction[:, i] & y[:, i].astype(bool)).sum()),
        )
        for i, f in enumerate(FAMILIES)
    }
    diagnostic["exact_labelset_accuracy_article_weighted"] = float(
        np.average(np.all(prediction == y, axis=1), weights=weights)
    )
    # Independently recompute saved conditional metrics from prediction rows.
    for r in rows:
        r["target"] = np.array(r["target"])
    recomputed = score(rows, prob, m["dev"]["threshold"])
    assert abs(recomputed["article_mean_f1"] - m["dev"]["article_mean_f1"]) < 1e-12
    write_json(out / "analysis.json", diagnostic)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    for r in rr:
        h = r["history"]
        x = [z["epoch"] for z in h]
        label = f"LoRA LR {r['config']['adapter_lr']:g}"
        axes[0].plot(x, [z["loss"] for z in h], label=label)
        axes[1].plot(x, [z["dev"]["article_mean_f1"] for z in h], label=label)
        axes[2].plot(x, [z["dev"]["macro_family_f1"] for z in h], label=label)
    for ax, title in zip(
        axes, ["Train loss", "Dev conditional F1", "Dev macro family F1"], strict=True
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning-curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "# E1. Классификация семейств при заданных точных фрагментах\n",
        f"Выбранный классификатор достигает dev F1 **{m['dev']['article_mean_f1']:.4f}**, macro-F1 **{m['dev']['macro_family_f1']:.4f}**. Лучший TF-IDF контроль по основной метрике: **{base['results'][baseline]['dev']['article_mean_f1']:.4f}**. Это условная классификация: модель получает настоящие границы, и здесь не проверяется способность находить фрагменты.\n",
        """## Вопрос, гипотеза и единица наблюдения\n
Можно ли содержательно оценивать семейство риторики отдельно от локализации? Гипотеза: контекстный энкодер распознаёт семейства лучше частот и лексического контроля, но качество на редких семействах может оставаться слабым.\n
Прежний split145/31/31, только русский корпус, один seed42. При нескольких разметчиках каждый ответ сохраняется отдельно. Разные семейства на одной паре границ объединяются в multi-hot в пределах одного ответа; ответы разных людей не объединяются. Статья имеет одинаковый суммарный вес вне зависимости от числа её ответов/фрагментов. Статьи с полностью пустыми ответами не содержат объектов условной классификации; это не отрицательные примеры E1. Негативные фрагменты и отказ проверяются в цепочке E3.\n""",
        f"В условной train-части{m['train']['n_articles']} статей,{m['train']['n_answers']} непустых ответов,{m['train']['n_spans']} пар «фрагмент × ответ»; dev —{m['dev']['n_articles']} статей,{m['dev']['n_answers']} ответов,{m['dev']['n_spans']} пар. Исключены16 полностью пустых train-статей и2 dev-статьи. Test не использован.\n",
        """## Сравниваемые методы\n
Частотный контроль оценивает частоты семейств на train с равным весом статей и ответов. TF-IDF использует текст фрагмента и соседний контекст ±200символов: word1–2 (до30000 признаков), char3–5 (до40000), min_df2; отдельная логистическая регрессия на семейство, C∈{1,10}, max_iter500. Словари и коэффициенты обучены только на train.\n
Нейронная модель: закреплённый RuModernBERT `8aee6f818db9070b635a549580dc64dcfa03e39a`; окна1024 BPE-токена с перекрытием128. База BF16 заморожена, LoRA rank8/alpha16/dropout0.1 в attention. Для пары точных символьных границ используются контекст начала/конца, среднее представление внутренней части и длина; MLP размерности64 с dropout0.1 выдаёт6 независимых sigmoid-оценок. Границы не предсказываются. Контекст за пределами отдельного окна не получает полноценного межоконного attention — перекрывающиеся окна агрегируются.\n
Loss: средний BCE по парам/семействам каждого непустого ответа, затем по ответам и статьям. Это разные бинарные семейства, поэтому несколько меток допустимы. В условной задаче гарантируется хотя бы одна метка через argmax; этот fallback не переносится в автоматическую цепочку.\n
## Настройка и выбор\n
Два независимых fit из одной исходной базы: LoRA LR2e-4 и5e-5, head LR1e-3, AdamW decay0.01, accumulation8 статей, clip1, warmup41updates и линейное снижение на408updates. Максимум24эпохи, patience6 без улучшений, ограничение процесса500с до финальной оценки. Порог выбирается на dev по17 значениям0.1…0.9. Основной критерий — micro-F1 внутри индивидуального ответа, среднее между ответами статьи и статьями; tie-break macro-F1, затем Brier. Отбор по dev выполняется после каждой эпохи.\n
Это ограниченная проверка чувствительности к LR и длительности обучения, не исчерпывающая оптимизация. Seed не варьировался. В первом конфиге сохранены унаследованные неиспользуемые поля старого ST2; эффективное поведение зафиксировано в протоколе и исходном коде, у второго конфигурация очищена.\n
## Результаты\n
| Метод | Train F1 | Dev F1 | Dev macro-F1 | Dev Brier | Порог |\n|---|---:|---:|---:|---:|---:|""",
    ]
    all_results = [(name, r) for name, r in base["results"].items()] + [
        (f"RuModernBERT LR{r['config']['adapter_lr']:g}", r["metrics"]) for r in rr
    ]
    for name, r in all_results:
        d = r["dev"]
        lines.append(
            f"| {name} | {r['train']['article_mean_f1']:.4f} | {d['article_mean_f1']:.4f} | {d['macro_family_f1']:.4f} | {d['brier']:.4f} | {d['threshold']:.2f} |"
        )
    lines += [
        "\nBrier — средняя квадратичная ошибка6 вероятностей, меньше лучше. Macro-F1 придаёт семействам одинаковый вес; внутри каждого семейства используются веса статей/ответов/пар. Он не совпадает с усреднённым по статьям F1 отдельных семейств в сквозном ST2.\n",
        "| LoRA LR | Выбрана эпоха | Выполнено эпох | Обновлений | Предъявлений статей | Остановка | Время процесса, с |\n|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in rr:
        z = r["metrics"]
        lines.append(
            f"| {r['config']['adapter_lr']:g} | {z['selected_epoch']} | {len(r['history'])} | {z['updates']} | {z['article_presentations']} | {z['status']} | {z['elapsed_seconds']:.2f} |"
        )
    lines += [
        "\nПри time_limit последняя эпоха может быть неполной: у low-LR выполнено 404 из 408 запланированных updates. Кривая показывает оценку после этой части эпохи; выбранная эпоха20 полностью завершена. Потенциальное продолжение не входит в текущий результат.\n",
        "\n![Кривые обучения](learning-curves.png)\n",
        "## Семейства и ошибки\n",
        "| Семейство | Train, сырая поддержка | Dev, сырая поддержка | Dev F1 | TP / FP / FN, сырые |\n|---|---:|---:|---:|---|",
    ]
    for f in FAMILIES:
        c = diagnostic["raw_family_counts"][f]
        lines.append(
            f"| {f} | {m['train']['family_support'][f]} | {m['dev']['family_support'][f]} | {m['dev']['per_family_f1'][f]:.4f} | {c['tp']} / {c['fp']} / {c['fn']} |"
        )
    lo, hi = diagnostic["bootstrap95"]
    lines += [
        f"\nТочное совпадение всего набора меток фрагмента с весами статьи/ответа: {diagnostic['exact_labelset_accuracy_article_weighted']:.4f}. Сырые поддержки/ошибки не являются числом независимых документов. У Distraction в dev всего4 метки: нулевой либо нестабильный F1 этой категории нельзя скрывать общей метрикой.\n",
        f"Парная разница выбранной модели и{baseline} по29 dev-статьям: {diffs.mean():+.4f}; описательный bootstrap-интервал95% [{lo:+.4f}; {hi:+.4f}],5000 повторов, seed42. Модели и пороги уже выбраны на этом dev: интервал не учитывает отбор, повторные запуски и обучение, поэтому это не независимое подтверждение эффекта.\n",
        """## Интерпретация и ограничения\n
Разделение ответственности позволяет увидеть качество распознавания семейства отдельно от ошибок границ. Польза контекстного классификатора оценивается относительно лексического контроля, а способность всей системы выделять риторику остаётся вопросом E3.\n
В обучении E1 отсутствуют ошибочные предложения генератора. Его вероятности условны на предоставленный фрагмент; уверенный ответ на ложном кандидате возможен. В E3 используется независимый балл детектора и возможность не выдавать метку. Дообучение классификатора на out-of-fold предложениях и отрицательных примерах — отдельный последующий эксперимент, здесь не выполнено.\n
При малом корпусе остаются редкие классы, неоднозначность разметки и разрыв train/dev; один seed и общий dev для настройки ограничивают силу выводов. Test закрыт, результат не называется финальным качеством на новых данных.\n
## Воспроизведение и самопроверка\n
```bash
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.train --config research/stages/ST2-l3-rhetoric/modular/E1-classification/base.json
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.train --config research/stages/ST2-l3-rhetoric/modular/E1-classification/low-lr.json
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.classifier_baselines
```

Trainer запрещает перезапись существующего Run ID. Для независимого повторения нужен новый Run ID и отдельный бюджетный ledger; исходные архивы не удалять. В каталоге каждого запуска: конфигурация, версии/хэши, снимок кода и lock, история, best/last веса+optimizer/scheduler, train/dev прогнозы и метрики. Эти checkpoints сохраняют выбранные веса; точное возобновление новой modular-серии посреди эпохи не реализовано и не заявляется.\n""",
        f"Выбранный запуск: [{selected['path'].name}](../../experiments/{selected['path'].name}/metrics.json). [Контроли](baselines.json), [анализ](analysis.json), [протокол](../protocol.md), [классификатор](../../../../kuki/span_classifier.py). GPU-время двух fit: {sum(r['metrics']['elapsed_seconds'] for r in rr):.2f}с; CPU-контроли: {base['elapsed_seconds']:.2f}с. Проверка загрузки выбранных весов входит в бюджет E3 и записывается в `E3-detector/chain.json`.\n",
        """Отдельная self-review: метрика пересчитана из сохранённых прогнозов; проверены исключение пустых объектов, multi-label внутри ответа, отсутствие смешивания разметчиков и test, неизменность frozen base. Реализованы тесты весов/группировки и отдельные тесты дальнейшей калибровки. Результаты не выдаются за автоматическую детекцию.\n""",
    ]
    (out / "report.md").write_text(
        re.sub(r"(?<=[А-Яа-яЁё])(?=\d)|(?<=\d)(?=[А-Яа-яЁё])", " ", "\n".join(lines))
    )


if __name__ == "__main__":
    e1()

"""Render the bounded ST5 search report from persisted, verified artifacts."""
import json
import re
from pathlib import Path

from research.kuki.data import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    selection = read(ROOT / 'selection.json')
    confidence = read(ROOT / 'confidence-diagnostics.json')['results']
    budget = read(ROOT / 'budget.json')
    old_budget = read(ROOT.parent / 'budget.json')
    verify = read(ROOT / 'experiments/20260929-l1-search-verification-s42-01/verification.json')
    assert verify['status'] == 'passed'
    candidates = selection['candidates']
    chosen_name = selection['selected_condition']
    chosen = next((x for x in candidates if x['condition'] == chosen_name), None)
    original = selection['original_base']
    selected = chosen['dev'] if chosen else original
    table = [f"| original-base | 13 | {original['macro_f1']:.4f} | {original['mention_macro_f1']:.4f} | {original['article_accuracy']:.4f} | {original['cross_entropy']:.4f} | {original['brier']:.4f} |"]
    ce_rows, runtime_rows, class_rows, error_rows = [], [], [], []
    for c in candidates:
        m, ce, run = c['dev'], c['dev_best_ce'], Path(c['run'])
        table.append(f"| {c['condition']} | {c['best_epoch']} | {m['macro_f1']:.4f} | {m['mention_macro_f1']:.4f} | {m['article_accuracy']:.4f} | {m['cross_entropy']:.4f} | {m['brier']:.4f} |")
        ce_rows.append(f"| {c['condition']} | {c['best_ce_epoch']} | {ce['macro_f1']:.4f} | {ce['cross_entropy']:.4f} | {c['last']['macro_f1']:.4f} |")
        rt, progress = read(run / 'runtime.json'), read(run / 'progress.json')
        runtime_rows.append(f"| {c['condition']} | {progress['epoch']} | {rt['updates']} | {rt['article_presentations']} | {rt['seconds']:.2f} | {rt['stop_reason']} |")
        (run / 'report.md').write_text(f'''# ST5 search: {c['condition']}

Выбран best-F1 epoch {c['best_epoch']}: macro-F1 с весами статей {m['macro_f1']:.6f}, обычный macro-F1 {m['mention_macro_f1']:.6f}, CE {m['cross_entropy']:.6f}, Brier {m['brier']:.6f}. Train macro-F1 {c['train']['macro_f1']:.6f}. Last macro-F1 {c['last']['macro_f1']:.6f}.

Диагностический best-CE epoch {c['best_ce_epoch']}: F1 {ce['macro_f1']:.6f}, CE {ce['cross_entropy']:.6f}. Этот checkpoint не подменяет основное правило выбора по F1.

{progress['epoch']} эпох, {rt['updates']} optimizer updates, {rt['article_presentations']} предъявлений, {rt['seconds']:.2f}с процесса; остановка {rt['stop_reason']}. Исходная база заморожена; начальные параметры совпали с историческим base. Веса, optimizer/scheduler/RNG и порядок данных сохранены локально в checkpoints/; полный снимок исходников — code-tree/. Метрики и прогнозы — best/, best-ce/, last/. Свежий replay и независимый пересчёт пройдены.

[Общий отчёт](../../report.md), [протокол](../../protocol.md). Test закрыт, seed 42; результат не устанавливает преимущество на независимом тесте.
''')
    for role in ['PROTAGONIST', 'ANTAGONIST', 'INNOCENT']:
        a, b = original['per_role'][role]['f1'], selected['per_role'][role]['f1']
        class_rows.append(f'| {role} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    for name, x in confidence.items():
        error_rows.append(f"| {name} | {x['train']['macro_f1']:.4f} | {x['errors']} | {x['confidence_slices']['0.99']['errors']} |")
    change = selected['macro_f1'] - original['macro_f1']
    ci_text = 'Новая замена исходному base по правилу протокола не найдена.'
    tradeoff = ''
    if chosen:
        interval = chosen['versus_original_base']['descriptive_percentile_95']
        ci_text = f"Δmacro-F1 {change:+.4f}; парный описательный 95% bootstrap-интервал [{interval[0]:+.4f}; {interval[1]:+.4f}] по 29 группам статей, 2000 повторов. Интервал {'включает' if interval[0] <= 0 <= interval[1] else 'не включает'} ноль."
        early = confidence[chosen_name]['best_ce']
        em = early['metrics']
        tradeoff = f"""У выбранного {chosen_name} есть существенный компромисс. Основной best-F1 даёт {selected['macro_f1']:.4f}, но ошибок с уверенностью≥99% у него {confidence[chosen_name]['confidence_slices']['0.99']['errors']} против {confidence['original-base']['confidence_slices']['0.99']['errors']} у исходного base. Прохождение ограничений на средние CE/Brier не означает устранения переуверенности.

Заранее сохранённый best-CE той же конфигурации, эпоха {early['epoch']}, имеет F1 **{em['macro_f1']:.4f}**, CE **{em['cross_entropy']:.4f}**, Brier **{em['brier']:.4f}**; число уверенных ошибок — **{early['errors_at99']}**. Для контроля классификации текущий выбор по протоколу остаётся best-F1. Раннее состояние — отдельная сохранённая альтернатива для последующей проверки вероятностей; оно не подменяет основной результат. Необходимость такого разделения — важный результат этой настройки, а не основание объявить найденный максимум F1 лучшим во всех отношениях."""
    report = f'''# ST5: дооптимизация обучения L1

**Завершено 29 сентября 2026.** По заранее заданному правилу выбран **{chosen_name}**: macro-F1 с равным весом статей **{selected['macro_f1']:.4f}** против **{original['macro_f1']:.4f}**, CE **{selected['cross_entropy']:.4f}** против **{original['cross_entropy']:.4f}**, Brier **{selected['brier']:.4f}** против **{original['brier']:.4f}**. Это дооптимизация на dev; независимое обобщение не проверено. Итоговый test закрыт.

## Зачем понадобился поиск

Первая серия сравнила только две скорости LoRA при одинаковом Head LR 1e-3. У исходного base ошибка вероятностей dev начинает расти после 4–5-й эпохи, тогда как F1 продолжает улучшаться. Значит, исходные две точки не давали достаточных оснований считать обучение настроенным. Новая серия проверяет скорость головы и регуляризацию, сохраняя исходную архитектуру и метод оценки.

[Протокол до fit](protocol.md) фиксировал четыре новых варианта: head-slow (Head LR 3e-4), smooth (ε=0.1), их сочетание и dropout (Head dropout 0.3). Все прочие параметры совпадают с base: LoRA LR 2e-4, rank8/alpha16/dropout0.1, decay0.01, warmup34, линейное расписание, batch8, окна1024/overlap128. Инициализация и порядок статей одинаковы, seed42. Это конечная локальная матрица, не полный поиск оптимума. Каждая модель обучена заново от той же закреплённой базы, без продолжения переобученного checkpoint.

Smoothing смешивает целевую метку с равномерным распределением: правильная роль 0.9333, остальные по0.0333. Исходные аннотации не меняются; все оценочные метрики используют исходные ответы. Основание для проверки — [Müller et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html); это гипотеза регуляризации, а не гарантия калибровки или пригодности сообщений L1. Train-loss сглаженных и обычных вариантов имеет разную цель и не сравнивается напрямую как одинаковая CE.

## Основное сравнение: best по F1

135 активных train-статей и29 dev-статей, 1234/200 индивидуальных отметок, то же разбиение ST0. Основной F1 и accuracy используют прежние веса упоминание→непустой ответ→статья. Macro-F1 по отметкам приведён отдельно. CE и Brier меньше — лучше.

| Вариант | Эпоха | Macro-F1, веса статей | Macro-F1 по отметкам | Article accuracy | CE | Brier |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Ранжирование зависит от весов: у smooth обычный macro-F1 по отметкам выше, чем у dropout, и поштучных ошибок меньше. Основная оценка даёт каждой статье одинаковый вес независимо от числа упоминаний и предпочитает dropout. Правило было зафиксировано до поиска; победитель по нему не объявляется лучшим по всем альтернативным метрикам.

Рекомендуемая замена требует прироста F1 хотя бы0.01 и отсутствия ухудшения CE/Brier относительно исходного base. Среди прошедших ограничение выбирается максимальный F1, затем меньшая CE. Абсолютно лучший новый F1 получен у **{selection['absolute_best_new_f1']}**. Выбранный checkpoint: `{selection['selected_checkpoint']}`. Историческая [первая серия](../report.md) и её selection.json не переписаны; текущий выбор этой дооптимизации — [selection.json](selection.json).

{ci_text} Эти интервалы не учитывают поиск четырёх конфигураций, выбор эпох и случайность обучения. Они не заменяют оценку на независимом test или сравнение нескольких seed. Порог0.01 — инженерное правило, не проверка статистической значимости.

Отдельный кандидат по ошибке вероятностей при F1 не ниже base−0.01: **{selection['probability_candidate']}**. Он выбран по CE при улучшении хотя бы0.10 и Brier не хуже base; такой статус сам по себе не означает лучшую классификацию или доказанную калибровку.

## Роль выбора эпохи

Сохранены диагностические best-CE и последние состояния. Основной результат выше остаётся выбранным по F1; эти дополнительные состояния не используются для скрытой смены критерия.

| Вариант | Best-CE эпоха | F1 этого состояния | CE этого состояния | F1 после полного обучения |
|---|---:|---:|---:|---:|
{chr(10).join(ce_rows)}

У исходного base минимальная CE была на эпохе4 (0.8041; F1 0.6218), но соответствующие веса тогда не сохранялись. Этот исторический факт взят из logs/epochs.jsonl; он не выдаётся за вновь проверенный checkpoint. Для новых вариантов best-CE заранее включён в протокол и воспроизведён в свежем процессе.

## Классы, переобучение и уверенные ошибки

| Роль | F1 исходного base | F1 выбранной модели | Разность |
|---|---:|---:|---:|
{chr(10).join(class_rows)}

| Вариант / best-F1 | Train macro-F1 | Ошибок на200 dev-отметках | Ошибок с уверенностью≥99% |
|---|---:|---:|---:|
{chr(10).join(error_rows)}

{tradeoff}

Число уверенных ошибок — дополнительная описательная диагностика, не критерий поиска; пороги не настраивались. Все индивидуальные отметки здесь считаются поштучно. Эти показатели описывают поведение вероятностей на этих данных, но не устанавливают универсальную калибровку. Высокий train F1 и сохраняющийся разрыв с dev требуют осторожности даже при улучшении метрик. Полные кривые, вероятностные срезы и хеши входов — [confidence-diagnostics.json](confidence-diagnostics.json).

## Проверки и ресурс

70 тестов прошли до fit, включая точное совпадение loss при ε=0, формулу/градиенты ε=0.1 и нормировку по ответам. GPU-проверка сглаженной цели подтвердила изменения LoRA A/B и активных голов, неизменность базы и точное восстановление следующего update вместе с optimizer. Начальные параметры всех вариантов совпали с исходным base.

Свежий процесс воспроизвёл 17 наборов прогнозов и независимо пересчитал их основные метрики: best-F1 train/dev, best-CE dev и last dev четырёх моделей плюс исторический base dev в новой обёртке. [Проверка](experiments/20260929-l1-search-verification-s42-01/verification.json). Полный code-tree каждого нового запуска сохраняет реальные пути источников и проверен по хешам.

| Вариант | Эпох | Updates | Предъявлений | Время процесса, с | Остановка |
|---|---:|---:|---:|---:|---|
{chr(10).join(runtime_rows)}

Учтённое время новой серии **{budget['seconds']:.2f}с / {budget['seconds']/60:.2f} из50 GPU-минут**, включая технические процессы, загрузку, оценки, сохранение и replay. Первая серия отдельно заняла {old_budget['seconds']/60:.2f}минут; суммарно две серии ST5 — {(budget['seconds']+old_budget['seconds'])/60:.2f}минут. CPU-анализ и написание кода сюда не входят. [Полный ресурс](budget.json).

Первая техническая попытка завершилась до GPU и до optimizer.step из-за абсолютного пути при копировании code-tree. Ошибка исправлена в архивировании; повторная проверка прошла. Неудачная попытка и её время сохранены, исходные научные артефакты не перезаписывались.

## Границы вывода

Подбор оправдан как проверка качества исходного контроля: скорость головы и регуляризация оказались отдельными существенными настройками, которые нельзя было заменить единственным изменением LoRA LR. Однако четыре новых точки не устанавливают глобальный оптимум; weight decay, rank, warmup, batch и иные значения smoothing/dropout не перебирались. Один seed, 29 многократно использованных dev-статей и заданные человеком позиции ограничивают вывод.

Новая выбранная модель служит более обоснованным самостоятельным контролем для следующего сравнения. Польза её ролей или вероятностей для L2/L3 пока не проверялась; ST6 не запускался. Проверку обмена следует сопоставлять с настроенными отдельными моделями и общей моделью без обмена, чтобы эффект механизма не смешивался со слабой настройкой контроля.
'''
    report = re.sub(r'(\d)([А-Яа-яЁё])', r'\1 \2', report)
    report = re.sub(r'([А-Яа-яЁё])(\d)', r'\1 \2', report)
    report = re.sub(r'\b(rank|alpha|dropout|decay|warmup|batch|overlap|seed)(\d)', r'\1 \2', report)
    (ROOT / 'report.md').write_text(report)
    write_json(ROOT / 'report-inputs.json', dict(selection='selection.json',
        confidence='confidence-diagnostics.json', budget='budget.json',
        generation_script='analysis/report.py', generated_after_replay=True))
    print(chosen_name, selected['macro_f1'], selected['cross_entropy'], budget['seconds'])


if __name__ == '__main__':
    main()

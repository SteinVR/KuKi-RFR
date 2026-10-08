"""Compact, source-backed comparison of the two declared balances and fixed controls."""

import importlib
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from research.kuki.data import digest, write_json

M = importlib.import_module('research.stages.ST6-l1-l2-l3.balance.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def summarize(condition, run, refs):
    progress = M.read(run / 'progress.json')
    history = progress['history']
    chosen = max(history, key=lambda h: tuple(J.selection_key(h['dev'], refs)))
    assert chosen['epoch'] == progress['best_epoch']
    metrics = M.read(run / 'best/dev-metrics.json')
    assert metrics == chosen['dev']
    peaks = {
        'l1': max(history, key=lambda h: (h['dev']['l1']['macro_f1'], -h['dev']['l1']['cross_entropy'])),
        'l2': max(history, key=lambda h: (-h['dev']['l2']['brier'], h['dev']['l2']['macro_f1'])),
    }
    diagnostics = {}
    for task, record in peaks.items():
        path = run / 'checkpoints' / f'best-{task}.pt'
        if condition == 'joint':
            path = run / 'checkpoints/best.pt' if record['epoch'] == chosen['epoch'] else None
        diagnostics[task] = dict(epoch=record['epoch'], metrics=record['dev'],
                                 checkpoint=str(path) if path else None,
                                 checkpoint_sha256=digest(path) if path else None)
    return dict(condition=condition, run=str(run), epoch=chosen['epoch'],
                dev=metrics, train=M.read(run / 'best/train-metrics.json'),
                last=M.read(run / 'last/dev-metrics.json'),
                selection_key=J.selection_key(metrics, refs), criterion_passed=J.accepted(metrics, refs),
                passing_epochs=[h['epoch'] for h in history if J.accepted(h['dev'], refs)],
                diagnostics=diagnostics, checkpoint=str(run / 'checkpoints/best.pt'),
                checkpoint_sha256=digest(run / 'checkpoints/best.pt'),
                completed_epochs=progress['epoch'], updates=progress['updates'], status=progress['status'],
                equal_budget=progress['updates'] == 380 and progress['epoch'] == 20,
                article_presentations=progress['articles_seen'], l1_presentations=progress['l1_articles_seen'])


def main():
    refs = M.read(M.ROOT / 'references.json')
    verification = M.read(M.ROOT / 'verification.json')
    assert verification['status'] == 'passed'
    runs = {'joint': Path(refs['joint_run'])}
    for condition in M.CONDITIONS:
        cfg = M.read(M.ROOT / 'configs' / f'{condition}.json')
        M.inputs(cfg)  # Recheck immutable sources, data integrity and exact allowed config changes.
        runs[condition] = M.ROOT / 'experiments' / cfg['run_id']
    original_metadata = M.read(runs['joint'] / 'metadata.json')
    original_provenance = original_metadata['provenance']
    for condition in M.CONDITIONS:
        metadata = M.read(runs[condition] / 'metadata.json')
        provenance = metadata['provenance']
        assert metadata['packages'] == original_metadata['packages']
        assert provenance['uv_lock_sha256'] == original_provenance['uv_lock_sha256']
        assert provenance['data_hashes'] == original_provenance['data_hashes']
        assert provenance['encoder'] == original_provenance['encoder']
        assert all(provenance['code_sha256'][p] == h for p, h in original_provenance['code_sha256'].items())
    records = [summarize(condition, run, refs) for condition, run in runs.items()]
    eligible = [r for r in records if r['equal_budget'] and r['criterion_passed']]
    winner = max(eligible, key=lambda r: tuple(r['selection_key'])) if eligible else None
    all_passing = {r['condition']: r['passing_epochs'] for r in records}
    any_passing = any(all_passing.values())
    write_json(M.ROOT / 'selection.json', dict(
        selected_condition=winner['condition'] if winner else None,
        selected_checkpoint=winner['checkpoint'] if winner else None,
        any_selected_checkpoint_passes=winner is not None, any_epoch_passes=any_passing,
        all_passing_epochs=all_passing, records=records, test_used=False,
        controls_changed=False, shared_code_packages_data_matched=True,
        planned_fits=2, series_complete=all(r['equal_budget'] for r in records)))
    labels = {'joint': '1:1 (1.0 / 1.0)', 'roles': '4:1 (1.6 / 0.4)', 'frames': '1:4 (0.4 / 1.6)'}
    names = {'joint': 'Исходный joint 1:1', 'roles': 'Приоритет ролей 4:1', 'frames': 'Приоритет фреймов 1:4'}
    colors = {'joint': '#64748b', 'roles': '#cc6430', 'frames': '#007e9e'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for r in records:
        history = M.read(Path(r['run']) / 'progress.json')['history']
        epochs = [h['epoch'] for h in history]
        series = [[h['dev']['l1']['macro_f1'] for h in history],
                  [h['dev']['l2']['brier'] for h in history],
                  [h['train_l1'] for h in history], [h['train_l2'] for h in history]]
        for ax, ys in zip(axes.flat, series, strict=True):
            ax.plot(epochs, ys, color=colors[r['condition']], label=labels[r['condition']], linewidth=1.8)
            ax.scatter([r['epoch']], [ys[r['epoch']-1]], color=colors[r['condition']],
                       s=55, edgecolors='black', zorder=5)
    for ax, value, tolerance in [(axes[0, 0], refs['l1_f1'], .01), (axes[0, 1], refs['l2_brier'], .003)]:
        ax.axhline(value, color='black', linestyle='--', linewidth=1, label='Standalone reference')
        ax.axhspan(value-tolerance, value+tolerance, color='gray', alpha=.12)
    for ax, title, ylabel in zip(axes.flat,
                                 ['L1 dev: higher is better', 'L2 dev: lower is better',
                                  'L1 training objective (before task weight)', 'L2 training objective (before task weight)'],
                                 ['Article-balanced macro-F1', 'Article-balanced Brier', 'CE', 'BCE'], strict=True):
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(M.ROOT / 'learning-curves.png', dpi=160)
    plt.close(fig)
    budget = M.read(M.ROOT / 'budget.json')
    fits = [r for r in budget['runs'] if r['kind'] == 'fit']
    assert len({r['run_id'] for r in fits}) <= 2
    assert budget['seconds'] <= budget['limit_seconds']
    histories = {r['condition']: M.read(Path(r['run']) / 'progress.json')['history'] for r in records}
    source_files = [M.ROOT / 'analyze.py', M.ROOT / 'references.json', M.ROOT / 'verification.json', M.ROOT / 'budget.json']
    for run in runs.values():
        source_files += [run / 'progress.json', run / 'best/dev-metrics.json', run / 'best/train-metrics.json',
                         run / 'last/dev-metrics.json']
    write_json(M.ROOT / 'analysis-provenance.json', dict(files={str(p): digest(p) for p in source_files}, test_used=False))
    verdict = (f"Да: **{names[winner['condition']]}**, эпоха **{winner['epoch']}**, проходит критерий."
               if winner else 'Основные выбранные checkpoint не проходят критерий; принятой замены отдельных моделей нет.')
    report = f'''# ST6: проверка баланса L1–L2

**29 сентября 2026.** {verdict}
Выполнены ровно два новых обучения. Результаты исходного 1:1 и отдельных моделей переиспользованы.
Проверялась одна гипотеза: изменение баланса обучающих ошибок позволит получить
одно состояние с приемлемым качеством обоих выходов. [Протокол до fit](protocol.md).

## Основное сравнение

| Модель | Коэффициенты L1 / L2 | Выбранная эпоха | L1 macro-F1 ↑ | L2 Brier ↓ | Общий критерий |
|---|---|---:|---:|---:|---|
| Отдельный L1, ST5 dropout | — | 16 | {refs['l1_f1']:.6f} | — | контроль |
| Отдельный L2, ST3 l2-only | — | 8 | — | {refs['l2_brier']:.6f} | контроль |
'''
    for r in records:
        weights = {'joint': '1 / 1', 'roles': '1,6 / 0,4', 'frames': '0,4 / 1,6'}[r['condition']]
        report += (f"| {names[r['condition']]} | {weights} | {r['epoch']} | {r['dev']['l1']['macro_f1']:.6f} | "
                   f"{r['dev']['l2']['brier']:.6f} | {'да' if r['criterion_passed'] else 'нет'} |\n")
    report += '''
L1 F1 даёт каждой статье одинаковый суммарный вес; L2 Brier считается по всем
индивидуальным ответам с равными весами статей. Обе метрики joint получены
из **одного checkpoint**. Более высокий F1 и более низкий Brier — лучше.

Критерий прежний: F1 ≥ 0,755617 и Brier ≤ 0,151920, а также F1 ≥ 0,775617
либо Brier ≤ 0,145920. В коде используются полные значения контрольных метрик,
без округления таблицы. Правило выбора: максимум `(min(g1,g2), g1+g2, −L1_CE)`,
где g1=ΔF1/0,01 и g2=−ΔBrier/0,003; при равенстве остаётся ранняя эпоха.

## Кривые и полная история критерия

![Кривые трёх совместных вариантов](learning-curves.png)

Точки с чёрной обводкой — выбранные эпохи. Серые полосы показывают допуски
±0,01 F1 и ±0,003 Brier вокруг отдельных контролей; сама полоса ещё не означает
успеха, поскольку нужно также улучшение хотя бы одной задачи.
Нижние графики показывают исходные ошибки задач до умножения на коэффициенты.

| Вариант | Проверенных эпох | Эпохи, прошедшие полный критерий |
|---|---:|---|
'''
    for r in records:
        report += f"| {names[r['condition']]} | {len(histories[r['condition']])} | {r['passing_epochs'] or 'нет'} |\n"
    report += '''
## Диагностические лучшие состояния

Они показываются отдельно и не подменяют основной выбор. Для обоих новых
запусков сохранены веса best-l1 и best-l2, включая случаи совпадения с best.
У исходного 1:1 позднее восстановление несохранённых промежуточных весов не выполнялось.

| Новый вариант | Диагностический выбор | Эпоха | L1 F1 | L2 Brier | Веса |
|---|---|---:|---:|---:|---|
'''
    for r in records[1:]:
        for task, d in r['diagnostics'].items():
            link = Path(d['checkpoint']).relative_to(M.ROOT)
            report += (f"| {names[r['condition']]} | best-{task} | {d['epoch']} | "
                       f"{d['metrics']['l1']['macro_f1']:.6f} | {d['metrics']['l2']['brier']:.6f} | [checkpoint]({link}) |\n")
    report += '\n## Ответы на исследовательские вопросы\n\n'
    if not any_passing:
        report += '**1. Один checkpoint, проходящий критерий, не получен.** Ни одна эпоха трёх вариантов его не прошла.\n\n'
    elif winner:
        report += f"**1. Такой checkpoint получен:** {names[winner['condition']]}, эпоха {winner['epoch']}. Это dev-результат, не независимое подтверждение.\n\n"
    else:
        report += '**1. В истории есть прошедшие эпохи, но основной заранее заданный выбор их не предпочёл.** Они перечислены выше; правило выбора не менялось после результатов.\n\n'
    base = records[0]
    report += '**2. Изменение баланса дало следующие разности относительно 1:1 в выбранных состояниях:**\n\n'
    for r in records[1:]:
        a = r['dev']['l1']['macro_f1'] - base['dev']['l1']['macro_f1']
        b = r['dev']['l2']['brier'] - base['dev']['l2']['brier']
        report += f"- {names[r['condition']]}: ΔL1 F1 {a:+.6f}; ΔL2 Brier {b:+.6f}.\n"
    report += '\nЭпохи выбирались отдельно одним и тем же правилом; это сравнение результатов обучения, а не гарантированное монотонное управление качеством через вес.\n\n'
    if winner:
        report += '**3. Есть основание сохранить прошедший вариант как кандидата для следующего сравнения.** Независимую замену отдельных моделей ещё не подтверждает один seed на многократно использованном dev.\n'
    else:
        report += '**3. Основания заменять отдельные модели по принятому критерию нет.** Фиксируется отрицательный результат этой ограниченной проверки; отдельные модели сохраняются.\n'
    report += '''

## Сопоставимость и ограничения

Три joint одинаковы по начальным параметрам и RNG, архитектуре, dropout, LR,
порядку статей, seed42, split, расписанию и нормировкам. Новые коэффициенты:
1,6/0,4 и 0,4/1,6, сумма 2. Не менялась нормировка L1 145/135; отсутствующая
разметка не превращалась в отрицательный класс. Для всех joint назначены
20 эпох / 380 обновлений / 2900 предъявлений статей, из них2700 с L1.
Сумма коэффициентов сохраняет номинальный масштаб цели, но не гарантирует
одинаковые нормы градиентов или шагов после клиппирования и AdamW.

У отдельного L1 было **340 обновлений**, у совместных — **380**. Сравнение
трёх joint согласовано; сравнение с самостоятельным L1 не изолирует все
различия обучения. Нулевые инициализации использованы заново, обученные best
ST5/ST6 не продолжались. Другие результаты настройки ST5 не подставлялись.
L3 и обмен прогнозами не добавлялись. Test остался закрыт.
Данные dev малы (31 статья, 29 с L1), использовались многократно, seed один.
Отрицательный результат не опровергает все возможные режимы совместного обучения.

## Проверки и ресурс

До fit прошёл 81 тест, включая оба коэффициента, аналитический градиент цели,
пустой L1 и запрет незаявленных изменений настроек. На GPU для обоих вариантов
проверены все активные головы и LoRAA/B, неизменность базы, точное продолжение
следующего обновления и optimizer. Начальные параметры и RNG побайтно совпали
с первым ST6. [Техническая проверка](preflight.json).

Свежий процесс проверил основные, диагностические и последние состояния;
метрики F1/Brier независимо пересчитаны. Проверены выбор эпох по всей истории,
соответствие весов нужным эпохам, снимки кода и неизменность исторических файлов.
'''
    report += f"Всего **{len(verification['sets'])} наборов прогнозов**. [Результаты проверки](verification.json).\n\n"
    report += '| Запуск | Эпох | Updates | Статей / с L1 | Полный бюджет обновлений | GPU-процесс, с |\n|---|---:|---:|---|---|---:|\n'
    for r in records[1:]:
        seconds = sum(f['seconds'] for f in fits if f['run_id'] == Path(r['run']).name)
        report += (f"| {names[r['condition']]} | {r['completed_epochs']} | {r['updates']} | "
                   f"{r['article_presentations']} / {r['l1_presentations']} | {'да' if r['equal_budget'] else 'НЕТ — неполный запуск'} | {seconds:.2f} |\n")
    report += f'''
Вся новая серия, включая загрузку, технические шаги, оценки, сохранение и replay:
**{budget['seconds']:.2f} с / {budget['seconds']/60:.2f} из 30 минут**.
Технических optimizer updates — 6, научных новых — {sum(r['updates'] for r in records[1:])}.
Исторические обучения не повторялись, их стоимость учтена ранее. CPU-анализ
и подготовка отчёта в GPU-время не включены. [Бюджет](budget.json).

После двух заданных вариантов серия остановлена. Дополнительных весов задач,
seed, архитектур или обучений не добавлено. [Все состояния и решение](selection.json).

Отдельный self-review: проверены точное число fit, неизменные контроли и нулевые
веса, сумма коэффициентов и нормировка, равенство бюджетов трёх joint,
один checkpoint для обоих выходов, вся история критерия, диагностические веса,
свежий replay, фактический ресурс, ограничение L1 340/380 и закрытый test.
'''
    (M.ROOT / 'report.md').write_text(report)
    for r in records[1:]:
        (Path(r['run']) / 'report.md').write_text(
            f"# {Path(r['run']).name}\n\n"
            f"Выбрана эпоха {r['epoch']}: L1 F1 {r['dev']['l1']['macro_f1']:.6f}, "
            f"L2 Brier {r['dev']['l2']['brier']:.6f}. Критерий: {'пройден' if r['criterion_passed'] else 'не пройден'}.\n\n"
            "[Полное сравнение и ограничения](../../report.md) · [Журнал эпох](logs/epochs.jsonl) · "
            "[Метрики](best/dev-metrics.json) · [Ресурс](runtime.json).\n")
    print(dict(selected_condition=winner['condition'] if winner else None, any_epoch_passes=any_passing,
               minutes=budget['seconds']/60, rows=[(r['condition'], r['epoch'], r['dev']['l1']['macro_f1'],
                                                    r['dev']['l2']['brier']) for r in records]))


if __name__ == '__main__':
    main()

"""Report the predeclared one-run comparison; no model fitting or test scoring."""

import importlib
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from research.kuki.data import ROLES, digest, write_json, write_jsonl
from research.kuki.evaluation import article_brier
from research.kuki.evaluation import score as score_l2
from research.kuki.l1 import examples
from research.kuki.l1 import score as score_l1

M = importlib.import_module('research.stages.ST6-l1-l2-l3.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def probs(path):
    return np.asarray([r['probabilities'] for r in M.jsonl(path)])


def bootstrap_l1(rows, p, q):
    groups = sorted({r['group_id'] for r in rows})
    matrices = np.zeros((2, len(groups), 3, 3))
    for i, values in enumerate([p, q]):
        for r, v in zip(rows, values, strict=True):
            matrices[i, groups.index(r['group_id']), r['target'], v.argmax()] += r['weight']
    rng, deltas = np.random.default_rng(42), []
    for _ in range(2000):
        cm = matrices[:, rng.integers(len(groups), size=len(groups))].sum(1)
        den = cm.sum(1) + cm.sum(2)
        f1 = np.divide(2*np.diagonal(cm, axis1=1, axis2=2), den,
                       out=np.zeros_like(den), where=den > 0).mean(1)
        deltas.append(f1[0] - f1[1])
    return dict(delta=score_l1(rows, p)['macro_f1'] - score_l1(rows, q)['macro_f1'],
                descriptive_percentile_95=np.quantile(deltas, [.025, .975]).tolist(),
                groups=len(groups), replicates=2000, seed=42)


def bootstrap_l2(docs, p, q):
    delta = article_brier(docs, p).mean(1) - article_brier(docs, q).mean(1)
    groups = sorted({d['group_id'] for d in docs})
    indices = [[i for i, d in enumerate(docs) if d['group_id'] == g] for g in groups]
    rng, values = np.random.default_rng(42), []
    for _ in range(2000):
        idx = [i for g in rng.integers(len(groups), size=len(groups)) for i in indices[g]]
        values.append(delta[idx].mean())
    return dict(delta=float(delta.mean()), descriptive_percentile_95=np.quantile(values, [.025, .975]).tolist(),
                groups=len(groups), replicates=2000, seed=42)


def confidence(rows, p):
    y = np.asarray([r['target'] for r in rows])
    wrong = p.argmax(1) != y
    return dict(errors=int(wrong.sum()), wrong_confidence_ge_90=int((wrong & (p.max(1) >= .9)).sum()),
                wrong_confidence_ge_99=int((wrong & (p.max(1) >= .99)).sum()))


def main():
    cfg, refs = M.read(M.ROOT / 'configs/joint.json'), M.read(M.ROOT / 'references.json')
    docs, _, _ = M.inputs(cfg)
    run = M.ROOT / 'experiments' / cfg['run_id']
    progress = M.read(run / 'progress.json')
    verify = M.read(M.ROOT / 'verification.json')
    assert progress['status'] == 'completed' and verify['status'] == 'passed'
    history = progress['history']
    selected = max(history, key=lambda h: tuple(J.selection_key(h['dev'], refs)))
    assert selected['epoch'] == progress['best_epoch']
    best, train, last = [M.read(run / state / f'{split}-metrics.json')
                         for state, split in [('best', 'dev'), ('best', 'train'), ('last', 'dev')]]
    assert best == selected['dev']
    l1base = M.read(M.L1 / 'best/dev-metrics.json')
    l2base = M.read(M.L2 / 'best/metrics.json')['dev']['l2']
    l1last = M.read(M.L1 / 'last/dev-metrics.json')
    l2last = M.read(M.L2 / 'last/metrics.json')['dev']['l2']
    peak_l1 = max(history, key=lambda h: h['dev']['l1']['macro_f1'])
    peak_l2 = min(history, key=lambda h: h['dev']['l2']['brier'])
    rows = examples(docs['dev'])
    p1, p2 = probs(run / 'best/dev-l1.jsonl'), probs(run / 'best/dev-l2.jsonl')
    q1, q2 = probs(M.L1 / 'best/dev-predictions.jsonl'), probs(M.L2 / 'best/dev-l2.jsonl')
    bootstrap = dict(l1=bootstrap_l1(rows, p1, q1), l2=bootstrap_l2(docs['dev'], p2, q2))
    selection = dict(condition='joint', run_id=cfg['run_id'], checkpoint=str(run / 'checkpoints/best.pt'),
                     checkpoint_sha256=digest(run / 'checkpoints/best.pt'), epoch=progress['best_epoch'],
                     dev=best, gains=J.joint_gains(best, refs), criterion_passed=J.accepted(best, refs),
                     epochs_passing_criterion=[h['epoch'] for h in history if J.accepted(h['dev'], refs)],
                     bootstrap=bootstrap, l1_comparison='historical, same labeled exposures, unequal update schedule',
                     l2_comparison='matched historical control, exact initial update verified', test_used=False)
    write_json(M.ROOT / 'selection.json', selection)
    sources = {}
    for source in sorted({d['source'] for d in docs['dev']}):
        mask1 = np.asarray([r['source'] == source for r in rows])
        mask2 = np.asarray([d['source'] == source for d in docs['dev']])
        rr = [r for r, keep in zip(rows, mask1, strict=True) if keep]
        dd = [d for d, keep in zip(docs['dev'], mask2, strict=True) if keep]
        sources[source] = dict(n_articles=len(dd), l1_joint=score_l1(rr, p1[mask1]),
                               l1_reference=score_l1(rr, q1[mask1]),
                               l2_joint=score_l2(dd, p2[mask2]), l2_reference=score_l2(dd, q2[mask2]))
    errors, text = [], {d['doc_id']: d['text'] for d in docs['dev']}
    for r, new, old in zip(rows, p1, q1, strict=True):
        if new.argmax() != r['target'] or old.argmax() != r['target']:
            errors.append(dict(**{k: v for k, v in r.items() if k != 'text'},
                               gold=ROLES[r['target']], joint=ROLES[new.argmax()], reference=ROLES[old.argmax()],
                               joint_confidence=float(new.max()), reference_confidence=float(old.max()),
                               context=text[r['doc_id']][max(0, r['start']-200):r['end']+200]))
    write_jsonl(M.ROOT / 'dev-errors.jsonl', errors)
    y = np.asarray([r['target'] for r in rows])
    error_summary = dict(joint=confidence(rows, p1), reference=confidence(rows, q1),
                         corrected=int(((q1.argmax(1) != y) & (p1.argmax(1) == y)).sum()),
                         introduced=int(((q1.argmax(1) == y) & (p1.argmax(1) != y)).sum()),
                         directions=dict(Counter(e['gold']+' -> '+e['joint'] for e in errors
                                                 if e['gold'] != e['joint'])))
    article_rows = []
    e1, e2 = article_brier(docs['dev'], p2).mean(1), article_brier(docs['dev'], q2).mean(1)
    for d, new, old in zip(docs['dev'], e1, e2, strict=True):
        mask = np.asarray([r['doc_id'] == d['doc_id'] for r in rows])
        rr = [r for r, keep in zip(rows, mask, strict=True) if keep]
        article_rows.append(dict(doc_id=d['doc_id'], group_id=d['group_id'], source=d['source'],
                                 l2_joint=float(new), l2_reference=float(old), delta_brier=float(new-old),
                                 l1_joint=score_l1(rr, p1[mask]) if rr else None,
                                 l1_reference=score_l1(rr, q1[mask]) if rr else None))
    write_json(M.ROOT / 'diagnostics.json', dict(sources=sources, errors=error_summary,
                                                articles=article_rows, train=train, last=last))
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    axes[0].plot(epochs, [h['dev']['l1']['macro_f1'] for h in history], marker='.')
    axes[0].axhline(refs['l1_f1'], color='gray', linestyle='--', label='ST5 reference')
    axes[0].axhspan(refs['l1_f1']-.01, refs['l1_f1']+.01, alpha=.1)
    axes[0].set(title='L1 dev macro-F1 (higher is better)', ylabel='Article-balanced F1')
    axes[1].plot(epochs, [h['dev']['l2']['brier'] for h in history], marker='.')
    axes[1].axhline(refs['l2_brier'], color='gray', linestyle='--', label='ST3 L2 reference')
    axes[1].axhspan(refs['l2_brier']-.003, refs['l2_brier']+.003, alpha=.1)
    axes[1].set(title='L2 dev Brier (lower is better)', ylabel='Brier')
    axes[2].plot(epochs, [h['train_l1'] for h in history], label='L1 CE')
    axes[2].plot(epochs, [h['train_l2'] for h in history], label='L2 BCE')
    axes[2].set(title='Training objectives', ylabel='Loss per labeled article')
    for ax in axes:
        ax.axvline(progress['best_epoch'], color='tab:red', alpha=.6, linestyle=':', label='Selected epoch')
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(M.ROOT / 'learning-curves.png', dpi=160)
    plt.close(fig)
    budget = M.read(M.ROOT / 'budget.json')
    runtime = M.read(run / 'runtime.json')
    d1, d2 = bootstrap['l1']['delta'], bootstrap['l2']['delta']
    passed = 'выполнен' if selection['criterion_passed'] else 'не выполнен'
    ci1, ci2 = bootstrap['l1']['descriptive_percentile_95'], bootstrap['l2']['descriptive_percentile_95']
    report = f'''# ST6-A: совместное обучение ролей L1 и фреймов L2

**Завершено 29 сентября 2026.** Один совместный fit; готовые самостоятельные модели
переиспользованы. Предзаданный критерий обоих выходов **{passed}**.
Выбран один checkpoint эпохи **{progress['best_epoch']}**: L1 macro-F1
**{best['l1']['macro_f1']:.4f}**, L2 Brier **{best['l2']['brier']:.4f}**.
Это разработческая оценка одного seed. Test закрыт.

## Вопрос и фактически выполненная серия

Проверено, что даёт общий энкодер для ролей заданных упоминаний и фреймов статьи
без передачи прогнозов между задачами. [Протокол до fit](protocol.md) фиксирует
одну конфигурацию, веса1:1, правила выбора и предел1500с. L3 и явный обмен не включались.
Обучение началось от закреплённой предобученной базы с новыми LoRA/головами:
для воспроизводимости взяты сохранённые **нулевые**, ещё не обученные инициализации
ST5/ST3. Дообученные best-веса ST5 не служили стартом.

Готовые контроли: ST5 dropout0.3 epoch16 и ST3 l2-only epoch8.
Их выбранные веса, метрики, прогнозы и нулевые состояния закреплены хешами в
[references.json](references.json). Повторных отдельных обучений нет.
Настройки L1 соответствуют выбранному ST5: dropout0.3, smoothing0,
LoRA LR2e-4, head LR1e-3, dim64. Окна1024/overlap128, вся статья,
LoRA rank8/alpha16, AdamW decay0.01, clip1, batch8, warmup38/380.

145train/31dev, прежний split и seed42. L1 имеет135/29 активных статей и1234/200
индивидуальных отметок. Пустая ролевая разметка маскируется; отрицательного
четвёртого класса нет. L1 усредняется по упоминаниям, непустым ответам и статьям;
множитель145/135 выравнивает глобальный вес наблюдаемой задачи. L2 использует
BCE по индивидуальным ответам. Наблюдаемые роли используются лишь в обучающей
цели; L2 при применении зависит только от текста, даже позиции L1 ему не нужны.

## Основной результат из одного состояния

| Выход | Готовый отдельный контроль | Joint, эпоха{progress['best_epoch']} | Joint − контроль |
|---|---:|---:|---:|
| L1 article-balanced macro-F1 ↑ | {l1base['macro_f1']:.6f} | {best['l1']['macro_f1']:.6f} | {d1:+.6f} |
| L1 macro-F1 по отметкам ↑ | {l1base['mention_macro_f1']:.4f} | {best['l1']['mention_macro_f1']:.4f} | {best['l1']['mention_macro_f1']-l1base['mention_macro_f1']:+.4f} |
| L1 CE ↓ | {l1base['cross_entropy']:.4f} | {best['l1']['cross_entropy']:.4f} | {best['l1']['cross_entropy']-l1base['cross_entropy']:+.4f} |
| L1 Brier ↓ | {l1base['brier']:.4f} | {best['l1']['brier']:.4f} | {best['l1']['brier']-l1base['brier']:+.4f} |
| L2 Brier ↓ | {l2base['brier']:.6f} | {best['l2']['brier']:.6f} | {d2:+.6f} |
| L2 macro-F1 ↑ | {l2base['macro_f1']:.4f} | {best['l2']['macro_f1']:.4f} | {best['l2']['macro_f1']-l2base['macro_f1']:+.4f} |

L1 оценивается с равным суммарным весом каждой статьи. L2 Brier учитывает все
индивидуальные ответы, F1 — строгое большинство голосов и фиксированный порог0.5.
Для joint не склеивались лучшие L1 и L2 из разных эпох.
Ключ выбора: максимум худшего из двух нормированных улучшений
(ΔF1/0.01, −ΔBrier/0.003), затем их суммы, затем меньшая L1 CE.
Рабочий критерий допускает потерю L1≤0.01 и рост Brier≤0.003 при улучшении
хотя бы одного выхода на соответствующий шаг. Он **{passed}**;
эпохи, прошедшие его диагностически: {selection['epochs_passing_criterion']}.
Это инженерные допуски, а не доказательство статистической значимости.

Парный bootstrap по группам статей,2000повторов, seed42: ΔL1 F1
{d1:+.4f}, описательный95%интервал [{ci1[0]:+.4f};{ci1[1]:+.4f}] ({bootstrap['l1']['groups']}групп);
ΔL2 Brier {d2:+.4f}, интервал [{ci2[0]:+.4f};{ci2[1]:+.4f}] ({bootstrap['l2']['groups']}групп).
Интервалы не учитывают поиск по dev, выбор эпох и случайность обучения.

## Сравнимость и существенное ограничение L1

Для L2 совпадают начальные веса, порядок145статей,380updates, расписание LR,
20эпох и2900предъявлений. Новая обёртка **побайтно воспроизвела один update**
старого L2-only из его start.pt с тем же RNG. В joint добавляется цель L1,
включая её влияние на общий градиент и клиппирование.

L1 сохраняет архитектуру, начальные параметры, настройки и2700предъявлений
размеченных статей за20эпох. Но ST5 обучался340updates только на135активных
статьях, joint —380updates на145. Меняются смешение батчей и расписание
общего энкодера. Поэтому разность L1 описывает **две готовые системы**,
а не изолированный причинный эффект совместного обучения при полностью равных
условиях. Повторный согласованный L1 не выполнен. Отрицательный результат
этой одной конфигурации не закрывает всю гипотезу L1–L2.

## Обучение и ошибки

![Кривые обучения](learning-curves.png)

Joint на train в выбранном состоянии: L1 F1 {train['l1']['macro_f1']:.4f},
L2 Brier {train['l2']['brier']:.4f}. Последнее состояние после{progress['updates']}updates:
dev L1 F1 {last['l1']['macro_f1']:.4f}, L2 Brier {last['l2']['brier']:.4f}.
Кривые всех эпох сохранены в журнале; train loss не заменяет dev-качество.

Диагностически лучший L1 в этой же серии: эпоха{peak_l1['epoch']},
F1 {peak_l1['dev']['l1']['macro_f1']:.4f}, при L2 Brier {peak_l1['dev']['l2']['brier']:.4f}.
Лучший L2: эпоха{peak_l2['epoch']}, Brier {peak_l2['dev']['l2']['brier']:.4f},
при L1 F1 {peak_l2['dev']['l1']['macro_f1']:.4f}. Эти разные состояния не подменяют
единый выбранный checkpoint и не складываются в искусственно лучшую модель.

| После20эпох | Отдельный контроль | Joint |
|---|---:|---:|
| L1 macro-F1 | {l1last['macro_f1']:.4f} | {last['l1']['macro_f1']:.4f} |
| L2 Brier | {l2last['brier']:.4f} | {last['l2']['brier']:.4f} |

У отдельного L1 по-прежнему340updates, у L2 и joint380. Последняя эпоха не является
новым критерием выбора и не устраняет отмеченное различие условий L1.

| Роль | F1 ST5 | F1 joint |
|---|---:|---:|
'''
    for role in ROLES:
        report += f"| {role} | {l1base['per_role'][role]['f1']:.4f} | {best['l1']['per_role'][role]['f1']:.4f} |\n"
    report += f'''
Из200индивидуальных отметок ошибок: ST5 **{error_summary['reference']['errors']}**,
joint **{error_summary['joint']['errors']}**. Исправлено прежних ошибок{error_summary['corrected']},
внесено новых{error_summary['introduced']}. Уверенных ошибочных ответов≥99%:
{error_summary['reference']['wrong_confidence_ge_99']}→{error_summary['joint']['wrong_confidence_ge_99']}.
Эти числа считаются по отметкам; они не подменяют основную метрику с весами статей.
Лучшие CE/Brier не означают автоматически доказанную калибровку.

| Источник | Статей dev | L1 ST5 | L1 joint | L2 Brier контроль | L2 Brier joint |
|---|---:|---:|---:|---:|---:|
'''
    for source, v in sources.items():
        report += (f"| {source} | {v['n_articles']} | {v['l1_reference']['macro_f1']:.4f} | "
                   f"{v['l1_joint']['macro_f1']:.4f} | {v['l2_reference']['brier']:.4f} | {v['l2_joint']['brier']:.4f} |\n")
    report += '''
Подгруппы описательные и малы, без новых правил выбора. Полные ошибки упоминаний
с контекстом — [dev-errors.jsonl](dev-errors.jsonl), метрики по статьям,
источникам, train и last — [diagnostics.json](diagnostics.json).

| Фрейм | Brier контроль | Brier joint | F1 контроль | F1 joint |
|---|---:|---:|---:|---:|
'''
    for a, b in zip(l2base['per_frame'], best['l2']['per_frame'], strict=True):
        assert a['frame'] == b['frame']
        report += f"| {a['frame']} | {a['brier']:.4f} | {b['brier']:.4f} | {a['f1']:.4f} | {b['f1']:.4f} |\n"
    if (not selection['criterion_passed'] and peak_l1['dev']['l1']['macro_f1'] < l1base['macro_f1']
            and peak_l2['dev']['l2']['brier'] < l2base['brier']):
        report += f'''

Содержательный итог: на ранней эпохе{peak_l2['epoch']} ролевая обучающая цель
сопровождается улучшением L2 относительно отдельного контроля, но L1 в этом
состоянии слабее ST5. Даже максимальный L1 этой серии не достигает выбранного
ST5. Поэтому одного checkpoint, пригодного для замены двух самостоятельных
моделей по принятому критерию, здесь нет. Это не объясняется только неудачным
выбором одной эпохи: критерий проверен на всей истории. Причина отставания L1
не изолирована от различий расписания и смешения батчей; конфликт градиентов
не измерялся и не объявляется установленной причиной.
'''
    report += f'''

## Проверки, ресурс и решение

75CPU-тестов прошли до fit. GPU-проверки: градиенты LoRAA/B и всех активных
голов, пустой L1, неизменность базы, точное продолжение параметров и optimizer,
идентичность старого L2 update и L1-вычислений. Свежий процесс точно воспроизвёл
**8наборов прогнозов**: joint best train/dev и last dev для двух выходов плюс
оба исторических best dev. Основные F1/Brier пересчитаны независимо.
Изменение всех параметров L1-головы не изменило L2. [Проверка](verification.json).
Хеши всех источников контролей и снимков кода сверены.

Fit: {runtime['seconds']:.2f}с, {progress['updates']}updates,
{progress['articles_seen']}предъявлений статей/{progress['l1_articles_seen']}с L1,
остановка `{progress['stop_reason']}`. Пик reserved GPU
{runtime['peak_reserved_bytes']/1024**3:.2f}GiB.
Вся новая GPU-серия с загрузкой, оценкой, техническими шагами и replay:
**{budget['seconds']:.2f}с / {budget['seconds']/60:.2f}мин из25мин**.
Отдельно5техническихupdates. Исторические контроли повторно не обучались;
их предыдущая стоимость уже записана в ST3/ST5. [Бюджет](budget.json).

Совместный checkpoint сохранён как результат ограниченной проверки.
Критерий замены двух отдельных выходов **{passed}**. Решение касается этой
конфигурации и dev; автоматического продолжения подбора, подключения L3,
явного обмена или открытия test нет. Отдельные модели сохраняются.

Отдельный self-review: сверены заявленный вопрос и выполненный fit, отсутствие
дообученного родителя, переиспользование обоих контролей, ограничение расписания
L1, единый checkpoint, независимые метрики, фактический ресурс и закрытый test.
'''
    (M.ROOT / 'report.md').write_text(report)
    (run / 'report.md').write_text(
        f"# {cfg['run_id']}\n\n"
        f"Завершён: {progress['updates']}updates, best epoch{progress['best_epoch']}. "
        f"L1 F1 {best['l1']['macro_f1']:.6f}, L2 Brier {best['l2']['brier']:.6f}. "
        f"Предзаданный критерий {passed}.\n\n"
        "[Полный отчёт, сравнения и ограничения](../../report.md) · "
        "[Dev-метрики выбранного состояния](best/dev-metrics.json) · "
        "[Журнал эпох](logs/epochs.jsonl) · [Ресурс](runtime.json).\n")
    analysis_files = [M.ROOT / 'analyze.py', M.ROOT / 'verification.json',
                      M.ROOT / 'references.json', run / 'progress.json',
                      M.L1 / 'last/dev-metrics.json', M.L2 / 'last/metrics.json']
    analysis_files += list((run / 'best').glob('*.json*'))
    write_json(M.ROOT / 'analysis-provenance.json', dict(
        files={str(p): digest(p) for p in analysis_files}, test_used=False))
    print(dict(epoch=selection['epoch'], l1_f1=best['l1']['macro_f1'], l2_brier=best['l2']['brier'],
               criterion_passed=selection['criterion_passed'], minutes=budget['seconds']/60))


if __name__ == '__main__':
    main()

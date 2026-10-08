"""Rebuild the descriptive ST1 comparison from stored train/dev predictions."""

import json
from pathlib import Path

import numpy as np

from research.kuki.data import FRAMES, write_json
from research.kuki.experiment import load_development

STAGE = Path('research/stages/ST1-l2-frames')


def read(path):
    return json.loads(path.read_text())


def predictions(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare(left, right, seed=42, repeats=10000):
    assert [r['doc_id'] for r in left] == [r['doc_id'] for r in right]
    delta = np.array([a['brier'] - b['brier'] for a, b in zip(left, right, strict=True)])
    groups = sorted({r['group_id'] for r in left})
    members = [np.array([i for i, r in enumerate(left) if r['group_id'] == g]) for g in groups]
    sums = np.array([delta[idx].sum() for idx in members])
    counts = np.array([len(idx) for idx in members])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repeats, len(groups)))
    distribution = sums[draws].sum(1) / counts[draws].sum(1)
    return dict(mean_brier_difference=float(delta.mean()),
                descriptive_95_percentile_interval=np.quantile(distribution, [0.025, 0.975]).tolist(),
                n_articles=len(delta), n_groups=len(groups), bootstrap_draws=repeats, seed=seed,
                favorable_articles=int((delta < 0).sum()), unfavorable_articles=int((delta > 0).sum()),
                limitation='Selected dev predictions; no selection or training-seed uncertainty correction')


def main():
    docs, _ = load_development(read(Path('research/configs/data.json')))
    baseline = STAGE / 'experiments/20260928-l2-baselines-s42-01'
    runs = {'frequency': baseline, 'tfidf_logistic': baseline}
    for path in sorted((STAGE / 'experiments').glob('*pilot*')):
        if (path / 'metrics.json').exists():
            runs[path.name] = path
    metrics, pred = {}, {}
    for name, path in runs.items():
        data = read(path / 'metrics.json')
        metrics[name] = data[name] if name in ['frequency', 'tfidf_logistic'] else data
        pred[name] = predictions(path / (f'{name}-dev-predictions.jsonl'
                                        if name in ['frequency', 'tfidf_logistic'] else 'dev-predictions.jsonl'))
    selected = min((n for n in runs if n.startswith('2026')), key=lambda n: metrics[n]['dev']['brier'])
    comparisons = {f'{name}_minus_tfidf': compare(rows, pred['tfidf_logistic'])
                   for name, rows in pred.items() if name != 'tfidf_logistic'}
    errors = []
    for d, row, linear in zip(docs['dev'], pred[selected], pred['tfidf_logistic'], strict=True):
        assert d['doc_id'] == row['doc_id'] == linear['doc_id']
        target = np.array(row['vote_fractions']) > 0.5
        labels = np.array(row['probabilities']) >= 0.5
        errors.append(dict(doc_id=d['doc_id'], source=d['source'], title=d['title'],
                           n_annotators=row['n_annotators'], brier=row['brier'],
                           delta_vs_linear=row['brier'] - linear['brier'],
                           false_positive=[f for f, t, p in zip(FRAMES, target, labels, strict=True) if p and not t],
                           false_negative=[f for f, t, p in zip(FRAMES, target, labels, strict=True) if t and not p]))
    errors.sort(key=lambda r: -r['brier'])
    subgroups = {}
    for name in sorted({str(d['source']) for d in docs['dev']}):
        idx = [i for i, d in enumerate(docs['dev']) if str(d['source']) == name]
        subgroups[name] = dict(n_articles=len(idx), **{
            model: float(np.mean([rows[i]['brier'] for i in idx])) for model, rows in pred.items()})
    analysis = dict(selected_neural_run=selected, comparisons=comparisons,
                    errors=errors, exploratory_source_groups=subgroups,
                    test_scored=False)
    write_json(STAGE / 'analysis.json', analysis)
    lines = ['# ST1 — сравнение на dev', '',
             'Только разработческая выборка: 31 статья. Меньший Brier лучше; больший F1 лучше.', '',
             '| Модель | Brier train | Brier dev | Macro-F1 dev | Micro-F1 dev |',
             '|---|---:|---:|---:|---:|']
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['train']['brier']:.4f} | {m['dev']['brier']:.4f} | "
                     f"{m['dev']['macro_f1']:.4f} | {m['dev']['micro_f1']:.4f} |")
    lines += ['', f'Выбранная нейросетевая конфигурация: `{selected}`.', '',
              '## По фреймам', '',
              '| Фрейм | Train majority + | Dev majority + | Dev ties | TF-IDF Brier | Neural Brier | Neural F1 | Neural прогноз + |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for i, frame in enumerate(FRAMES):
        t = metrics[selected]['train']['per_frame'][i]
        n = metrics[selected]['dev']['per_frame'][i]
        b = metrics['tfidf_logistic']['dev']['per_frame'][i]
        lines.append(f"| {frame} | {t['majority_support']} | {n['majority_support']} | {n['ties']} | "
                     f"{b['brier']:.4f} | {n['brier']:.4f} | {n['f1']:.4f} | {n['predicted_positive']} |")
    lines += ['', '## Наибольшие ошибки выбранной нейросети', '',
              'Лишние и пропущенные фреймы указаны относительно строгого большинства, которое само может быть неоднозначным.', '',
              '| Статья | Ответов | Brier | Разница с TF-IDF | Лишние фреймы | Пропущенные фреймы |',
              '|---|---:|---:|---:|---|---|']
    for r in errors[:10]:
        lines.append(f"| {r['doc_id']} | {r['n_annotators']} | {r['brier']:.4f} | {r['delta_vs_linear']:+.4f} | "
                     f"{'; '.join(r['false_positive']) or '—'} | {'; '.join(r['false_negative']) or '—'} |")
    lines += ['', '## Описательная неопределённость', '']
    for name, c in comparisons.items():
        lo, hi = c['descriptive_95_percentile_interval']
        lines.append(f"- {name}: Δ Brier {c['mean_brier_difference']:+.4f}, "
                     f"парный групповой bootstrap 95% [{lo:+.4f}; {hi:+.4f}].")
    lines += ['', 'Это интервалы фиксированных прогнозов на dev после выбора модели. Они не учитывают подбор, '
              'разные seed или возможную зависимость различных статей об одном событии. '
              'Группировка покрывает только связанные тексты, установленные ST0. '
              'Результат не является подтверждающим тестом или доказательством пользы обмена задач.', '']
    (STAGE / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(dict(selected=selected, comparisons=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()

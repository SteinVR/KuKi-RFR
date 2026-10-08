"""Descriptive confidence and curve diagnostics; never used to select hyperparameters."""
import json
from pathlib import Path

import numpy as np

from research.kuki.data import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    runs = {'original-base': ROOT.parent / 'experiments/20260929-l1-base-s42-01'}
    for name in ['head-slow', 'smooth', 'head-slow-smooth', 'dropout']:
        runs[name] = ROOT / 'experiments' / read(ROOT / 'configs' / f'{name}.json')['run_id']
    outputs = {}
    inputs = {}
    for name, run in runs.items():
        path = run / 'best/dev-predictions.jsonl'
        inputs[str(path)] = digest(path)
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        p = np.array([r['probabilities'] for r in rows])
        y = np.array([r['target'] for r in rows])
        w = np.array([r['weight'] for r in rows])
        confidence = p.max(1)
        correct = p.argmax(1) == y
        outputs[name] = dict(
            errors=int((~correct).sum()),
            confidence_slices={str(t): dict(
                mentions=int((confidence >= t).sum()),
                errors=int(((confidence >= t) & ~correct).sum()),
                original_article_weight=float(w[confidence >= t].sum()),
            ) for t in [.9, .99, .999]},
            train=read(run / 'best/train-metrics.json'),
            dev=read(run / 'best/dev-metrics.json'),
            last=read(run / 'last/dev-metrics.json'),
            curve=read(run / 'progress.json')['history'],
        )
        ce_path = run / 'best-ce/dev-predictions.jsonl'
        if ce_path.exists():
            ce_rows = [json.loads(s) for s in ce_path.read_text().splitlines()]
            ce_wrong = [r for r in ce_rows if np.argmax(r['probabilities']) != r['target']]
            inputs[str(ce_path)] = digest(ce_path)
            outputs[name]['best_ce'] = dict(
                epoch=read(run / 'progress.json')['best_ce_epoch'],
                metrics=read(run / 'best-ce/dev-metrics.json'), errors=len(ce_wrong),
                errors_at99=sum(max(r['probabilities']) >= .99 for r in ce_wrong),
            )
    write_json(ROOT / 'confidence-diagnostics.json', dict(
        results=outputs, input_sha256=inputs, code_sha256=digest(Path(__file__)),
        exploratory_only=True, thresholds_not_used_for_selection=True,
        confidence_slice_counts_are_unweighted=True, test_used=False,
    ))
    print({name: dict(errors=x['errors'], at99=x['confidence_slices']['0.99'])
           for name, x in outputs.items()})


if __name__ == '__main__':
    main()

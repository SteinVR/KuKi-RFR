"""Fresh replay and independent metrics for main, diagnostic and last checkpoints."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.l1 import examples
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module('research.stages.ST6-l1-l2-l3.balance.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')
V = importlib.import_module('research.stages.ST6-l1-l2-l3.verify')


def main():
    M.reserve(180)
    results, scopes = [], []
    for condition in M.CONDITIONS:
        cfg = M.read(M.ROOT / 'configs' / f'{condition}.json')
        docs, enc, prov = M.inputs(cfg)
        run = M.ROOT / 'experiments' / cfg['run_id']
        if not (run / 'progress.json').exists():
            scopes.append(dict(condition=condition, status='not_started'))
            continue
        progress = M.read(run / 'progress.json')
        scopes.append(dict(condition=condition, status=progress['status'], updates=progress['updates'],
                           equal_budget=progress['updates'] == 380 and progress['epoch'] == 20))
        assert M.read(run / 'metadata.json')['provenance'] == prov
        for source, expected in prov['code_sha256'].items():
            assert digest(run / 'code-tree' / Path(source).resolve().relative_to(Path.cwd())) == expected
        configure(42)
        encoded = encode_documents(docs['train'] + docs['dev'], enc, cfg)
        model = J.JointRoles(enc, cfg).cuda()
        initial_hash = M.initialize_weights(model)
        assert M.read(run / 'initialization.json')['trainable_sha256'] == initial_hash
        start = torch.load(run / 'checkpoints/start.pt', map_location='cpu', weights_only=False)
        assert M.state_hash(start['trainable']) == initial_hash and start['progress']['updates'] == 0
        if not progress['history']:
            del model
            torch.cuda.empty_cache()
            continue
        refs = M.read(M.ROOT / 'references.json')
        best = max(progress['history'], key=lambda h: tuple(J.selection_key(h['dev'], refs)))
        one = max(progress['history'], key=lambda h: (h['dev']['l1']['macro_f1'], -h['dev']['l1']['cross_entropy']))
        two = max(progress['history'], key=lambda h: (-h['dev']['l2']['brier'], h['dev']['l2']['macro_f1']))
        expected_epochs = {'best': best['epoch'], 'best-l1': one['epoch'], 'best-l2': two['epoch'], 'last': progress['epoch']}
        assert best['epoch'] == progress['best_epoch']
        assert one['epoch'] == progress['best_l1_epoch'] and two['epoch'] == progress['best_l2_epoch']
        for state, splits in [('best', ['train', 'dev']), ('last', ['dev']), ('best-l1', ['dev']), ('best-l2', ['dev'])]:
            payload = torch.load(run / 'checkpoints' / f'{state}.pt', map_location='cpu', weights_only=False)
            assert payload['provenance'] == prov
            assert payload['progress']['epoch'] == expected_epochs[state]
            M.set_weights(model, payload['trainable'])
            for split in splits:
                saved_metrics = M.read(run / state / f'{split}-metrics.json')
                pred = M.predict(model, docs[split], encoded)
                if state != 'last' and split == 'dev':
                    expected_metrics = next(h['dev'] for h in progress['history'] if h['epoch'] == expected_epochs[state])
                    assert saved_metrics == expected_metrics
                for task in ['l1', 'l2']:
                    stored = M.jsonl(run / state / f'{split}-{task}.jsonl')
                    assert np.array_equal(pred[task], np.asarray([r['probabilities'] for r in stored]))
                    if task == 'l1':
                        rows = examples(docs[split])
                        keys = ['doc_id', 'annotation_index', 'start', 'end', 'target', 'weight']
                        assert [[r[k] for k in keys] for r in rows] == [[r[k] for k in keys] for r in stored]
                        value = V.independent_l1(rows, pred[task])
                        assert abs(value - saved_metrics[task]['macro_f1']) < 1e-12
                    else:
                        assert [r['doc_id'] for r in stored] == [d['doc_id'] for d in docs[split]]
                        value = V.independent_l2(docs[split], pred[task])
                        assert abs(value - saved_metrics[task]['brier']) < 1e-12
                    results.append(dict(condition=condition, state=state, epoch=expected_epochs[state],
                                        split=split, task=task, exact=True, independent_primary=value))
        del model
        torch.cuda.empty_cache()
    output = dict(status='passed', sets=results, scopes=scopes,
                  original_artifacts_unchanged=True, source_snapshots_exact=True, test_used=False)
    write_json(M.ROOT / 'verification.json', output)
    return dict(status='passed', prediction_sets=len(results), scopes=scopes, test_used=False)


if __name__ == '__main__':
    result = dict(status='failed')
    try:
        result = main()
    finally:
        M.account('verification', STARTED, result)

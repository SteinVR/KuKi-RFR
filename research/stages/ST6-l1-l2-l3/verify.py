"""Fresh-process replay of joint and both reused baselines; independent primary scores."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.l1 import examples
from research.kuki.train_l3 import configure, encode_documents

M = importlib.import_module('research.stages.ST6-l1-l2-l3.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def independent_l1(rows, p):
    matrix = np.zeros((3, 3))
    for r, value in zip(rows, p, strict=True):
        matrix[r['target'], int(np.argmax(value))] += r['weight']
    denom = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * np.diag(matrix), denom, out=np.zeros(3), where=denom > 0).mean())


def independent_l2(docs, p):
    return float(np.mean([np.mean([(np.asarray(a['l2']) - row) ** 2 for a in d['annotations']])
                          for d, row in zip(docs, p, strict=True)]))


def main():
    M.reserve(300)
    cfg = M.read(M.ROOT / 'configs/joint.json')
    docs, enc, prov = M.inputs(cfg)
    run = M.ROOT / 'experiments' / cfg['run_id']
    assert M.read(run / 'metadata.json')['provenance'] == prov
    for source, expected in prov['code_sha256'].items():
        snap = run / 'code-tree' / M.Path(source).resolve().relative_to(M.Path.cwd())
        assert digest(snap) == expected
    configure(42)
    encoded = encode_documents(docs['train'] + docs['dev'], enc, cfg)
    model = J.JointRoles(enc, cfg).cuda()
    result = []
    for state, splits in [('best', ['train', 'dev']), ('last', ['dev'])]:
        payload = torch.load(run / 'checkpoints' / f'{state}.pt', map_location='cpu', weights_only=False)
        assert payload['provenance'] == prov
        M.set_weights(model, payload['trainable'])
        for split in splits:
            pred = M.predict(model, docs[split], encoded)
            scores = M.read(run / state / f'{split}-metrics.json')
            for task in ['l1', 'l2']:
                saved = M.jsonl(run / state / f'{split}-{task}.jsonl')
                assert np.array_equal(pred[task], np.asarray([r['probabilities'] for r in saved]))
                if task == 'l1':
                    rows = examples(docs[split])
                    keys = ['doc_id', 'annotation_index', 'start', 'end', 'target', 'weight']
                    assert [[r[k] for k in keys] for r in rows] == [[r[k] for k in keys] for r in saved]
                    value = independent_l1(rows, pred[task])
                    assert abs(value - scores['l1']['macro_f1']) < 1e-12
                else:
                    assert [r['doc_id'] for r in saved] == [d['doc_id'] for d in docs[split]]
                    value = independent_l2(docs[split], pred[task])
                    assert abs(value - scores['l2']['brier']) < 1e-12
                result.append(dict(model='joint', state=state, split=split, task=task, exact=True,
                                   independent_primary=value))
    # L2 is invariant to all L1-head parameters and to the positions supplied for L1.
    before = M.predict(model, docs['dev'][:2], encoded)['l2']
    with torch.no_grad():
        for n, p in model.head.named_parameters():
            if not n.startswith('l2.'):
                p.add_(1)
    assert np.array_equal(before, M.predict(model, docs['dev'][:2], encoded)['l2'])
    del model
    torch.cuda.empty_cache()
    for task, path in [('l1', M.L1), ('l2', M.L2)]:
        control_cfg = dict(cfg, l1_weight=int(task == 'l1'), l2_weight=int(task == 'l2'))
        configure(42)
        model = J.JointRoles(enc, control_cfg).cuda()
        payload = torch.load(path / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
        assert payload['provenance']['data_hashes'] == prov['data_hashes']
        M.set_weights(model, payload['trainable'])
        pred = M.predict(model, docs['dev'], encoded)[task]
        stored = path / 'best' / ('dev-predictions.jsonl' if task == 'l1' else 'dev-l2.jsonl')
        saved = M.jsonl(stored)
        assert np.array_equal(pred, np.asarray([r['probabilities'] for r in saved]))
        value = independent_l1(examples(docs['dev']), pred) if task == 'l1' else independent_l2(docs['dev'], pred)
        ref = M.read(M.ROOT / 'references.json')
        assert abs(value - ref['l1_f1' if task == 'l1' else 'l2_brier']) < 1e-12
        result.append(dict(model='historical', task=task, state='best', split='dev', exact=True,
                           independent_primary=value))
        del model
        torch.cuda.empty_cache()
    output = dict(status='passed', sets=result, no_l1_head_to_l2_path=True,
                  source_snapshots_exact=True, test_used=False)
    write_json(M.ROOT / 'verification.json', output)
    return dict(status='passed', prediction_sets=len(result), test_used=False)


if __name__ == '__main__':
    result = dict(status='failed')
    try:
        result = main()
    finally:
        M.account('verification', STARTED, result)

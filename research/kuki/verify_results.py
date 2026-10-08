"""Verify published data and independently recompute final metrics on a CPU.

Run from the repository root. This command uses NumPy and the standard library,
reads the saved predictions, and never trains models or selects thresholds.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import FAMILIES, ROLES, digest, prepare_document

FINAL = Path('research/stages/final')
DATA_CONFIG = Path('research/configs/data.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probabilities(value, width, categorical=False):
    result = np.asarray(value, dtype=float)
    require(result.shape == (width,), 'Unexpected prediction dimensions')
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(),
            'Invalid probabilities')
    if categorical:
        require(np.isclose(result.sum(), 1, atol=1e-6), 'Role probabilities do not sum to one')
    return result


def corpus():
    cfg = read(DATA_CONFIG)
    split = read(cfg['split_manifest'])
    require(digest(Path(cfg['split_manifest'])) == cfg['split_sha256'], 'Split hash mismatch')
    for field, hashfield in [('source', 'source_sha256'), ('prepared_path', 'prepared_sha256')]:
        require(digest(Path(split[field])) == split[hashfield], f'{field} hash mismatch')
    docs = jsonl(cfg['prepared_path'])
    raw = sorted(jsonl(split['source']), key=lambda d: d['doc_id'])
    require(docs == [prepare_document(d)[0] for d in raw], 'Prepared data differ from source policy')
    require(len(docs) == 207, 'Unexpected corpus size')
    require(sum(len(d['annotations']) for d in docs) == 301, 'Unexpected annotation count')
    index = {r['doc_id']: r for r in split['rows']}
    require(len(index) == len(docs) and set(index) == {d['doc_id'] for d in docs},
            'Incomplete or duplicated split IDs')
    require(Counter(r['split'] for r in index.values()) == {'train': 145, 'dev': 31, 'test': 31},
            'Unexpected split sizes')
    groups = {}
    for d in docs:
        row = index[d['doc_id']]
        require(d['text_sha256'] == row['text_sha256'], 'Text hash mismatch')
        require(groups.setdefault(row['group_id'], row['split']) == row['split'],
                'Document group crosses split boundaries')
        for answer in d['annotations']:
            for layer in ['l1_roles', 'l3_persuasion', 'l4_coded']:
                for span in answer[layer]:
                    require(d['text'][span['start']:span['end']] == span['text'],
                            'Span coordinates do not match the text')
    return sorted((d for d in docs if index[d['doc_id']]['split'] == 'test'),
                  key=lambda d: d['doc_id'])


def l1(docs, path):
    expected = []
    for d in docs:
        active = sum(bool(a['l1_roles']) for a in d['annotations'])
        for ai, a in enumerate(d['annotations']):
            labels = {(s['start'], s['end']): ROLES.index(s['label']) for s in a['l1_roles']}
            for (start, end), target in sorted(labels.items()):
                expected.append((d['doc_id'], ai, start, end, target, 1 / (active * len(labels))))
    matrices = {}
    for (did, ai, start, end, target, weight), pred in zip(expected, jsonl(path), strict=True):
        require((did, ai, start, end, target) == tuple(pred[k] for k in
                ['doc_id', 'annotation_index', 'start', 'end', 'target']), 'L1 reference mismatch')
        require(weight == pred['weight'], 'L1 weight mismatch')
        prob = probabilities(pred['probabilities'], 3, categorical=True)
        matrix = matrices.setdefault(did, np.zeros((3, 3)))
        matrix[target, prob.argmax()] += weight
    return matrices


def l2(docs, path):
    values = {}
    for d, pred in zip(docs, jsonl(path), strict=True):
        require(d['doc_id'] == pred['doc_id'], 'L2 article mismatch')
        prob = probabilities(pred['probabilities'], 14)
        values[d['doc_id']] = float(np.mean([(np.array(a['l2']) - prob) ** 2
                                             for a in d['annotations']]))
    return values


def conditional(docs, path, threshold):
    expected = []
    for d in docs:
        for ai, a in enumerate(d['annotations']):
            targets = {}
            for s in a['l3_persuasion']:
                target = targets.setdefault((s['start'], s['end']), np.zeros(6, dtype=int))
                target[FAMILIES.index(s['label'])] = 1
            for (start, end), target in sorted(targets.items()):
                expected.append((d['doc_id'], ai, start, end, target))
    counts = {}
    for (did, ai, start, end, target), pred in zip(expected, jsonl(path), strict=True):
        require((did, ai, start, end) == tuple(pred[k] for k in
                ['doc_id', 'annotation_index', 'start', 'end']), 'L3 reference mismatch')
        prob = probabilities(pred['probabilities'], 6)
        selected = prob >= threshold
        selected[prob.argmax()] = True
        counts.setdefault((did, ai), np.zeros(3))[:] += [
            (selected & target.astype(bool)).sum(), selected.sum(), target.sum()]
    by_article = {}
    for (did, _), (tp, predicted, gold) in counts.items():
        by_article.setdefault(did, []).append(2 * tp / (predicted + gold))
    return {did: float(np.mean(v)) for did, v in by_article.items()}


def automatic(docs, path, decoder):
    values = {}
    for d, pred in zip(docs, jsonl(path), strict=True):
        require(d['doc_id'] == pred['doc_id'], 'Automatic L3 article mismatch')
        prob = np.array([probabilities(p, 6) for p in pred['probabilities']])
        require(len(prob) == len(pred['pairs']), 'Candidate count mismatch')
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        scores = np.log(prob / (1 - prob)) + decoder['alpha'] * np.array(
            [v for _, _, v in pred['pairs']])[:, None]
        ranked = sorted([(float(scores[i, j]), start, end, family)
                         for i, (start, end, _) in enumerate(pred['pairs'])
                         for j, family in enumerate(FAMILIES)],
                        key=lambda v: (-v[0], v[1], v[2], v[3]))
        predicted = {(start, end, family) for value, start, end, family
                     in ranked[:decoder['top_outputs']] if value >= decoder['threshold']}
        answers = []
        for a in d['annotations']:
            gold = {(s['start'], s['end'], s['label']) for s in a['l3_persuasion']}
            answers.append(2 * len(gold & predicted) / max(1, len(gold) + len(predicted)))
        values[d['doc_id']] = float(np.mean(answers))
    return values


def macro(matrix):
    denominator = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0).mean(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optionally save the verification log')
    args = parser.parse_args()
    docs = corpus()
    expected = read(FINAL / 'results.json')
    manifest = read(FINAL / 'manifest.json')
    stats = {}
    for name, entry in manifest['systems'].items():
        path = FINAL / 'predictions/test' / name
        record = {}
        if (path / 'test-l1.jsonl').exists():
            record['l1_f1'] = l1(docs, path / 'test-l1.jsonl')
        if (path / 'test-l2.jsonl').exists():
            record['l2_brier'] = l2(docs, path / 'test-l2.jsonl')
        if (path / 'test-conditional.jsonl').exists():
            record['l3_conditional'] = conditional(docs, path / 'test-conditional.jsonl',
                                                   entry['conditional_threshold'])
            record['l3_exact'] = automatic(docs, path / 'test-chain.jsonl', manifest['decoder'])
        stats[name] = record
    for name in ['st4_control', 'st4_families']:
        for metric in ['l3_conditional', 'l3_exact']:
            stats[name][metric] = stats['st3'][metric]
    draws, metrics = {}, {}
    for name, record in stats.items():
        draws[name], metrics[name] = {}, {}
        for metric, values in record.items():
            keys = sorted(values)
            array = np.asarray([values[k] for k in keys])
            indices = np.random.default_rng(42).integers(0, len(keys), (10000, len(keys)))
            if metric == 'l1_f1':
                point, samples = float(macro(array.sum(0))), macro(array[indices].sum(1))
            else:
                point, samples = float(array.mean()), array[indices].mean(1)
            reference = expected['metrics'][name][metric]
            require(abs(point - reference['value']) < 1e-12, f'Metric mismatch: {name}/{metric}')
            require(len(keys) == reference['n_articles'], 'Active article count mismatch')
            require(np.allclose(np.quantile(samples, [.025, .975]), reference['ci95'],
                                atol=1e-12, rtol=0), f'Interval mismatch: {name}/{metric}')
            draws[name][metric] = (keys, samples)
            metrics[name][metric] = point
    for comparison in expected['comparisons']:
        metric = comparison['metric']
        # The study stores model names in first/second fields.
        first, second = comparison['first'], comparison['second']
        keys, a = draws[first][metric]
        other_keys, b = draws[second][metric]
        require(keys == other_keys, 'Paired bootstrap article mismatch')
        require(np.allclose(np.quantile(a - b, [.025, .975]), comparison['ci95'],
                            atol=1e-12, rtol=0), 'Paired interval mismatch')
        require(abs(metrics[first][metric] - metrics[second][metric] - comparison['delta'])
                < 1e-12, 'Paired difference mismatch')
    log = dict(status='passed', articles=207, annotations=301, split=[145, 31, 31],
               test_articles=len(docs), systems=len(metrics), metrics=metrics,
               paired_comparisons=len(expected['comparisons']), bootstrap_replicates=10000,
               absolute_tolerance=1e-12)
    if args.output:
        args.output.write_text(json.dumps(log, indent=2, sort_keys=True) + '\n')
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

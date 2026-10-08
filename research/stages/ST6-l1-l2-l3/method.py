"""ST6-A provenance, immutable historical references and independent predictions."""

import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json, write_jsonl
from research.kuki.evaluation import save_predictions
from research.kuki.evaluation import score as score_l2
from research.kuki.experiment import initialize_run, load_development, provenance
from research.kuki.l1 import examples
from research.kuki.l1 import score as score_l1

ROOT = Path(__file__).parent
L1 = Path('research/stages/ST5-l1-roles/search/experiments/20260929-l1-search-dropout-s42-01')
L2 = Path('research/stages/ST3-l2-l3-joint/experiments/20260929-l2-only-s42-01')


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def inputs(cfg):
    dc, enc = read(cfg['data_config']), read(cfg['encoder_config'])
    enc['gradient_checkpointing'] = cfg['gradient_checkpointing']
    docs, hashes = load_development(dc)
    prov = provenance(cfg, dc, hashes, enc)
    for name in ['model.py', 'method.py', 'train.py', 'preflight.py', 'verify.py']:
        p = ROOT / name
        prov['code_sha256'][str(p)] = digest(p)
    prov['protocol_sha256'] = digest(ROOT / 'protocol.md')
    prov['references_sha256'] = digest(ROOT / 'references.json')
    prov['reference_files'] = read(ROOT / 'references.json')['files']
    for path, expected in prov['reference_files'].items():
        assert digest(Path(path)) == expected, path
    prov['parent_checkpoint'] = None
    prov['adaptation_history'] = 'pinned pretrained base; reused zero-update random initializations only'
    assert len(docs['train']) == 145 and len(docs['dev']) == 31
    assert sum(any(a['l1_roles'] for a in d['annotations']) for d in docs['train']) == 135
    assert cfg['l1_normalization'] == 145 / 135
    return docs, enc, prov


def initialize(out, cfg, prov):
    initialize_run(out, cfg, prov)
    for source, expected in prov['code_sha256'].items():
        dest = out / 'code-tree' / Path(source).resolve().relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert digest(dest) == expected
    for name in ['protocol.md', 'references.json']:
        shutil.copy2(ROOT / name, out / name)


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])


def initialize_weights(model):
    one = torch.load(L1 / 'checkpoints/start.pt', map_location='cpu', weights_only=False)
    two = torch.load(L2 / 'checkpoints/start.pt', map_location='cpu', weights_only=False)
    assert one['progress']['updates'] == two['progress']['updates'] == 0
    state = dict(one['trainable'])
    for name, value in two['trainable'].items():
        if name.startswith('encoder.'):
            assert torch.equal(state[name], value), name
        elif name.startswith('head.l2.'):
            state[name] = value
        else:
            raise AssertionError(name)
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad or
                          (n.startswith('head.') and not n.startswith('head.boundary.'))}
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
    restore_rng(two['rng'])
    return state_hash(state)


def state_hash(state):
    h = hashlib.sha256()
    for n, p in sorted(state.items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def set_weights(model, state):
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def predict(model, docs, encoded):
    rows = examples(docs)
    pairs_by_doc = {}
    for r in rows:
        pairs_by_doc.setdefault(r['doc_id'], set()).add((r['start'], r['end']))
    lookup, l2 = {}, []
    model.eval()
    with torch.inference_mode():
        for d in docs:
            did = d['doc_id']
            item = encoded[did]
            pooled, features, prefix = model.encode_shared(item, model.l1_weight > 0)
            if model.l2_weight:
                l2.append(model.head['l2'](pooled).sigmoid().cpu().numpy())
            pairs = sorted(pairs_by_doc.get(did, set()))
            if model.l1_weight and pairs:
                values = model.roles_from_features(item, features, prefix, pairs).softmax(-1).cpu().numpy()
                lookup.update({(did, *p): v for p, v in zip(pairs, values, strict=True)})
    return dict(l1=np.asarray([lookup[r['doc_id'], r['start'], r['end']] for r in rows])
                if model.l1_weight else None,
                l2=np.asarray(l2) if model.l2_weight else None)


def metrics(docs, predictions):
    return dict(l1=score_l1(examples(docs), predictions['l1']) if predictions['l1'] is not None else None,
                l2=score_l2(docs, predictions['l2']) if predictions['l2'] is not None else None)


def persist(out, split, docs, predictions):
    if predictions['l1'] is not None:
        write_jsonl(out / f'{split}-l1.jsonl', [
            dict(**{k: v for k, v in r.items() if k != 'text'}, probabilities=p.tolist())
            for r, p in zip(examples(docs), predictions['l1'], strict=True)])
    if predictions['l2'] is not None:
        save_predictions(out / f'{split}-l2.jsonl', docs, predictions['l2'], split)
    result = metrics(docs, predictions)
    write_json(out / f'{split}-metrics.json', result)
    return result


def account(kind, started, result):
    seconds = time.perf_counter() - started
    ledger = read(ROOT / 'budget.json')
    ledger['seconds'] += seconds
    ledger['runs'].append(dict(kind=kind, seconds=seconds, **result))
    write_json(ROOT / 'budget.json', ledger)
    print(json.dumps(dict(kind=kind, seconds=seconds, **result)), flush=True)


def reserve(seconds):
    ledger = read(ROOT / 'budget.json')
    if ledger['seconds'] + seconds > ledger['limit_seconds']:
        raise RuntimeError('ST6-A series budget exhausted')

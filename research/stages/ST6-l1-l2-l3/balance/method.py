"""Isolated two-run balance series; immutable historical artifacts and initialization."""

import importlib
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import digest, write_json
from research.kuki.l2 import trainable_state

BASE = importlib.import_module('research.stages.ST6-l1-l2-l3.method')
ROOT = Path(__file__).parent
read, jsonl = BASE.read, BASE.jsonl
predict, metrics, persist = BASE.predict, BASE.metrics, BASE.persist
set_weights, state_hash = BASE.set_weights, BASE.state_hash
CONDITIONS = {'roles': (1.6, .4), 'frames': (.4, 1.6)}


def validate_config(cfg):
    assert (cfg['l1_weight'], cfg['l2_weight']) == CONDITIONS[cfg['condition']]
    assert math.isclose(cfg['l1_weight'] + cfg['l2_weight'], 2)
    original = read(BASE.ROOT / 'configs/joint.json')
    allowed = {'l1_weight', 'l2_weight', 'condition', 'run_id', 'run_budget_seconds'}
    assert cfg.keys() == original.keys()
    assert all(cfg[k] == v for k, v in original.items() if k not in allowed)
    assert cfg['run_id'] == f"20260929-st6-balance-{cfg['condition']}-s42-01"
    assert cfg['run_budget_seconds'] == 750


def inputs(cfg):
    validate_config(cfg)
    docs, enc, prov = BASE.inputs(cfg)
    prov['original_protocol_sha256'] = prov['protocol_sha256']
    prov['protocol_sha256'] = digest(ROOT / 'protocol.md')
    prov['references_sha256'] = digest(ROOT / 'references.json')
    prov['reference_files'] = read(ROOT / 'references.json')['files']
    for path, expected in prov['reference_files'].items():
        assert digest(Path(path)) == expected, path
    for name in ['method.py', 'train.py', 'preflight.py', 'verify.py']:
        p = ROOT / name
        prov['code_sha256'][str(p)] = digest(p)
    return docs, enc, prov


def initialize(out, cfg, prov):
    BASE.initialize(out, cfg, prov)
    for name in ['protocol.md', 'references.json']:
        shutil.copy2(ROOT / name, out / name)


def initialize_weights(model):
    result = BASE.initialize_weights(model)
    original = torch.load(BASE.ROOT / 'experiments/20260929-st6a-joint-s42-01/checkpoints/start.pt',
                          map_location='cpu', weights_only=False)
    assert original['progress']['updates'] == 0
    assert result == state_hash(original['trainable'])
    assert all(torch.equal(v, original['trainable'][n]) for n, v in trainable_state(model).items())
    import random
    assert random.getstate() == original['rng']['python']
    now = np.random.get_state()
    expected = original['rng']['numpy']
    assert now[0] == expected[0] and np.array_equal(now[1], expected[1]) and now[2:] == expected[2:]
    assert torch.equal(torch.get_rng_state(), original['rng']['torch'])
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), original['rng']['cuda'], strict=True))
    return result


def account(kind, started, result):
    elapsed = time.perf_counter() - started
    ledger = read(ROOT / 'budget.json')
    ledger['seconds'] += elapsed
    ledger['runs'].append(dict(kind=kind, seconds=elapsed, **result))
    write_json(ROOT / 'budget.json', ledger)
    print(dict(seconds=elapsed, **result), flush=True)


def reserve(seconds):
    ledger = read(ROOT / 'budget.json')
    if seconds < 0 or ledger['seconds'] + seconds > ledger['limit_seconds']:
        raise RuntimeError('ST6 balance series budget exhausted')


def remaining():
    ledger = read(ROOT / 'budget.json')
    return ledger['limit_seconds'] - ledger['seconds']

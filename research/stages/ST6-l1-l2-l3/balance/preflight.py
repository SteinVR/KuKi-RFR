"""Both weighted variants: gradients, immutable initialization and exact next-update replay."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.train_l2 import make_optimizer
from research.kuki.train_l3 import configure, encode_documents, update

M = importlib.import_module('research.stages.ST6-l1-l2-l3.balance.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def main():
    M.reserve(120)
    checks = []
    for condition in M.CONDITIONS:
        cfg = M.read(M.ROOT / 'configs' / f'{condition}.json')
        docs, enc, prov = M.inputs(cfg)
        out = M.ROOT / 'experiments' / f'20260929-st6-balance-preflight-{condition}-s42-01'
        M.initialize(out, cfg, prov)
        ordered = sorted(docs['train'], key=lambda d: (len(d['text']), d['doc_id']))
        chosen = [ordered[i] for i in np.linspace(0, len(ordered)-1, 8, dtype=int)]
        configure(42)
        encoded = encode_documents(chosen, enc, cfg)
        model = J.JointRoles(enc, cfg).cuda()
        initial_hash = M.initialize_weights(model)
        initial = trainable_state(model)
        frozen = frozen_digest(model)
        opt, sch = make_optimizer(model, cfg)
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        update(model, opt, sch, chosen, encoded, cfg)
        duration = time.perf_counter() - tick
        progress = dict(updates=1, epoch=0, next_position=8, article_order=[d['doc_id'] for d in chosen],
                        best_l1_key=[.5, -1.], best_l1_epoch=0, best_l2_key=[-.2, .4], best_l2_epoch=0)
        save_checkpoint(out / 'checkpoints/resume.pt', model, opt, sch, progress, prov)
        before = M.predict(model, chosen, encoded)
        update(model, opt, sch, chosen, encoded, cfg)
        expected, expected_opt = trainable_state(model), opt.state_dict()
        for part in ['lora_A', 'lora_B', 'head.span', 'head.token', 'head.char', 'head.boundary_features', 'head.l2']:
            assert any(not torch.equal(v, initial[n]) for n, v in expected.items() if part in n), part
        del model, opt, sch
        torch.cuda.empty_cache()
        configure(42)
        model = J.JointRoles(enc, cfg).cuda()
        assert M.initialize_weights(model) == initial_hash
        opt, sch = make_optimizer(model, cfg)
        assert load_checkpoint(out / 'checkpoints/resume.pt', model, opt, sch, prov) == progress
        after = M.predict(model, chosen, encoded)
        assert all(np.array_equal(before[k], after[k]) for k in before)
        update(model, opt, sch, chosen, encoded, cfg)
        assert all(torch.equal(v, expected[n]) for n, v in trainable_state(model).items())
        for i, values in expected_opt['state'].items():
            for k, v in values.items():
                actual = opt.state_dict()['state'][i][k]
                assert torch.equal(v, actual) if torch.is_tensor(v) else v == actual
        assert sch.last_epoch == 2 and frozen_digest(model) == frozen
        result = dict(condition=condition, status='passed', initialization_and_rng_equal_original=True,
                      initial_sha256=initial_hash, predictions_exact=True, next_update_exact=True,
                      optimizer_exact=True, base_unchanged=True, all_active_heads_and_lora_changed=True,
                      technical_optimizer_steps=3, profile_update_seconds=duration,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), test_used=False)
        write_json(out / 'verification.json', result)
        checks.append(result)
        del model, opt, sch
        torch.cuda.empty_cache()
    write_json(M.ROOT / 'preflight.json', dict(status='passed', checks=checks))
    return dict(status='passed', technical_optimizer_steps=6)


if __name__ == '__main__':
    result = dict(status='failed')
    try:
        result = main()
    finally:
        M.account('preflight', STARTED, result)

"""Train-only GPU checks, historical L2 equivalence, and exact restart."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.joint import JointModel
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.train_l2 import epoch_order, make_optimizer
from research.kuki.train_l3 import configure, encode_documents, update

M = importlib.import_module('research.stages.ST6-l1-l2-l3.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def main():
    M.reserve(300)
    cfg = M.read(M.ROOT / 'configs/joint.json')
    docs, enc, prov = M.inputs(cfg)
    out = M.ROOT / 'experiments/20260929-st6a-preflight-s42-01'
    M.initialize(out, cfg, prov)
    ordered = sorted(docs['train'], key=lambda d: (len(d['text']), d['doc_id']))
    selected = [ordered[i] for i in np.linspace(0, len(ordered)-1, 8, dtype=int)]
    first_batch = [docs['train'][i] for i in epoch_order(145, 42, 0)[:8]]
    empty = next(d for d in docs['train'] if not any(a['l1_roles'] for a in d['annotations']))
    configure(42)
    encoded = encode_documents(list({d['doc_id']: d for d in selected + first_batch + [empty]}.values()), enc, cfg)
    # Exact old/new L2 update, including historical RNG and optimizer schedule.
    old_cfg = M.read(M.L2 / 'config.json')['config']
    old = JointModel(enc, old_cfg).cuda()
    payload = torch.load(M.L2 / 'checkpoints/start.pt', weights_only=False, map_location='cpu')
    M.set_weights(old, payload['trainable'])
    opt, sch = make_optimizer(old, old_cfg)
    M.restore_rng(payload['rng'])
    update(old, opt, sch, first_batch, encoded, old_cfg)
    expected = trainable_state(old)
    del old, opt, sch
    torch.cuda.empty_cache()
    control_cfg = dict(cfg, l1_weight=0)
    configure(42)
    control = J.JointRoles(enc, control_cfg).cuda()
    M.initialize_weights(control)
    opt, sch = make_optimizer(control, cfg)
    update(control, opt, sch, first_batch, encoded, cfg)
    actual = trainable_state(control)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(actual[n], v) for n, v in expected.items())
    del control, opt, sch
    torch.cuda.empty_cache()
    # Full joint gradients and continuation.
    configure(42)
    model = J.JointRoles(enc, cfg).cuda()
    initial_hash = M.initialize_weights(model)
    initial = trainable_state(model)
    frozen = frozen_digest(model)
    model.eval()
    with torch.inference_mode():
        for doc in selected:
            pairs = sorted({(s['start'], s['end']) for a in doc['annotations'] for s in a['l1_roles']})
            if pairs:
                item = encoded[doc['doc_id']]
                _, features, prefix = model.encode_shared(item, True)
                assert torch.equal(model.roles_from_features(item, features, prefix, pairs),
                                   model.role_logits(item, pairs))
    # Empty observed L1 gives zero gradients even though L2 still learns.
    model.train()
    loss, detail = model.training_loss(encoded[empty['doc_id']], empty['annotations'])
    assert detail['l1'] == 0
    loss.backward()
    for n, p in model.head.named_parameters():
        if p.requires_grad and not n.startswith('l2.'):
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0, n
    model.zero_grad(set_to_none=True)
    M.initialize_weights(model)
    opt, sch = make_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    update(model, opt, sch, selected, encoded, cfg)
    elapsed = time.perf_counter() - tick
    progress = dict(updates=1, next_position=8)
    save_checkpoint(out / 'checkpoints/resume.pt', model, opt, sch, progress, prov)
    before = M.predict(model, selected, encoded)
    update(model, opt, sch, selected, encoded, cfg)
    continuous, optimizer_state = trainable_state(model), opt.state_dict()
    for part in ['lora_A', 'lora_B', 'head.span', 'head.token', 'head.char', 'head.boundary_features', 'head.l2']:
        assert any(not torch.equal(v, initial[n]) for n, v in continuous.items() if part in n), part
    del model, opt, sch
    torch.cuda.empty_cache()
    configure(42)
    model = J.JointRoles(enc, cfg).cuda()
    assert M.initialize_weights(model) == initial_hash
    opt, sch = make_optimizer(model, cfg)
    assert load_checkpoint(out / 'checkpoints/resume.pt', model, opt, sch, prov) == progress
    replay = M.predict(model, selected, encoded)
    assert all(np.array_equal(before[k], replay[k]) for k in before)
    update(model, opt, sch, selected, encoded, cfg)
    assert all(torch.equal(v, continuous[n]) for n, v in trainable_state(model).items())
    for i, values in optimizer_state['state'].items():
        for k, v in values.items():
            actual = opt.state_dict()['state'][i][k]
            assert torch.equal(v, actual) if torch.is_tensor(v) else v == actual
    assert sch.last_epoch == 2 and frozen_digest(model) == frozen
    result = dict(status='passed', historical_l2_update_exact=True, l1_architecture_exact=True,
                  next_update_exact=True, optimizer_exact=True, predictions_exact=True,
                  base_unchanged=True, empty_l1_masked=True, all_active_heads_and_lora_changed=True,
                  technical_optimizer_steps=5, trainable_sha256=initial_hash,
                  profile_batch_ids=[d['doc_id'] for d in selected], update_seconds=elapsed,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), test_used=False)
    write_json(out / 'verification.json', result)
    return result


if __name__ == '__main__':
    result = dict(status='failed')
    try:
        result = main()
    finally:
        M.account('preflight', STARTED, result)

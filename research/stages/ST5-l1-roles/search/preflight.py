"""L1 technical checks on training data, including exact next-update replay."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import importlib

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.l1 import examples, predict
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.train_l2 import make_optimizer
from research.kuki.train_l3 import configure, encode_documents, update

M = importlib.import_module("research.stages.ST5-l1-roles.search.method")
RoleClassifier = importlib.import_module("research.stages.ST5-l1-roles.search.model").TunedRoleClassifier


def main():
    M.reserve(120)
    cfg = M.read(M.ROOT / "configs/smooth.json")
    docs, encoder, prov = M.inputs(cfg)
    out = M.ROOT / "experiments/20260929-l1-search-preflight-s42-02"
    M.initialize(out, cfg, prov)
    ordered = sorted(M.active(docs["train"]), key=lambda d: (len(d["text"]), d["doc_id"]))
    selected = [ordered[i] for i in np.linspace(0, len(ordered)-1, 8, dtype=int)]
    configure(42)
    encoded = encode_documents(selected, encoder, cfg)
    model = RoleClassifier(encoder, cfg).cuda()
    initial = trainable_state(model)
    initial_hash = M.initialization(model)
    original = M.ROOT.parent / "experiments/20260929-l1-base-s42-01/initialization.json"
    assert initial_hash == M.read(original)["trainable_sha256"]
    frozen = frozen_digest(model)
    optimizer, scheduler = make_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    update(model, optimizer, scheduler, selected, encoded, cfg)
    seconds = time.perf_counter() - tick
    state = dict(epoch=0, next_position=8, article_order=[d["doc_id"] for d in selected], updates=1)
    save_checkpoint(out / "checkpoints/resume.pt", model, optimizer, scheduler, state, prov)
    expected_prob = predict(model, selected, encoded, examples(selected))
    update(model, optimizer, scheduler, selected, encoded, cfg)
    continuous = trainable_state(model)
    expected_opt = optimizer.state_dict()
    for part in ["lora_A", "lora_B", "head.span", "head.token", "head.char", "head.boundary_features"]:
        assert any(not torch.equal(v, initial[n]) for n, v in continuous.items() if part in n), part
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    configure(42)
    resumed = RoleClassifier(encoder, cfg).cuda()
    assert M.initialization(resumed) == initial_hash
    opt, sch = make_optimizer(resumed, cfg)
    assert load_checkpoint(out / "checkpoints/resume.pt", resumed, opt, sch, prov) == state
    assert np.array_equal(expected_prob, predict(resumed, selected, encoded, examples(selected)))
    update(resumed, opt, sch, selected, encoded, cfg)
    assert all(torch.equal(v, continuous[n]) for n, v in trainable_state(resumed).items())
    assert sch.last_epoch == 2
    for i, values in expected_opt["state"].items():
        for key, value in values.items():
            actual = opt.state_dict()["state"][i][key]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    assert frozen_digest(resumed) == frozen
    result = dict(status="passed", technical_optimizer_steps=3, trainable_sha256=initial_hash,
                  next_update_exact=True, optimizer_exact=True, predictions_exact=True,
                  base_unchanged=True, lora_A_B_and_all_active_heads_changed=True,
                  profile_batch_ids=[d["doc_id"] for d in selected],
                  profile_char_lengths=[len(d["text"]) for d in selected],
                  update_seconds=seconds, rough_20_epoch_train_seconds=seconds*340,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), test_used=False)
    write_json(out / "verification.json", result)
    return result


if __name__ == "__main__":
    result = dict(status="failed")
    try:
        result = main()
    finally:
        M.account("preflight", STARTED, result)

"""Train/dev representation audit and bounded train-only GPU profile/restart check."""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import FAMILIES, write_json
from research.kuki.experiment import initialize_run, load_development, provenance
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.l3 import L3Model, span_key
from research.kuki.l3_evaluation import empty_outputs, evaluate, keys, matched
from research.kuki.train_l2 import make_optimizer
from research.kuki.train_l3 import configure, encode_documents, update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begin = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads(Path(cfg["data_config"]).read_text())
    encoder = json.loads(Path(cfg["encoder_config"]).read_text())
    encoder["gradient_checkpointing"] = cfg["gradient_checkpointing"]
    docs, hashes = load_development(data_cfg)
    prov = provenance(cfg, data_cfg, hashes, encoder)
    initialize_run(args.output, cfg, prov)
    configure(cfg["seed"])
    encoded = encode_documents(docs["train"] + docs["dev"], encoder, cfg)
    audit = {}
    for split, part in docs.items():
        counts = Counter()
        family = Counter()
        positives, negatives, lengths, spans_per_answer, union_sizes = [], [], [], [], []
        for d in part:
            enc = encoded[d["doc_id"]]["encoded"]
            starts = {s for s, _ in enc["offsets"]}
            ends = {e for _, e in enc["offsets"]}
            n = len(d["text"])
            counts["all_character_pair_family_decisions"] += n * (n + 1) // 2 * len(FAMILIES)
            union_sizes.append(
                len({span_key(s) for a in d["annotations"] for s in a["l3_persuasion"]})
            )
            for a in d["annotations"]:
                spans = a["l3_persuasion"]
                spans_per_answer.append(len(spans))
                positives.append(len(spans))
                negatives.append(n * (n + 1) // 2 * len(FAMILIES) - len(spans))
                counts["answers"] += 1
                counts["empty_answers"] += not spans
                assert matched(spans, spans)[0] == len(keys(spans))
                for index, span in enumerate(spans):
                    s, e, family_index = span_key(span)
                    assert 0 <= s < e <= n
                    assert d["text"][s:e] == span["text"]
                    assert FAMILIES[family_index] == span["label"]
                    counts["spans"] += 1
                    counts["exact_roundtrip"] += 1
                    counts["not_token_aligned"] += s not in starts or e not in ends
                    counts["inside_word_boundary"] += any(
                        0 < p < n and d["text"][p - 1].isalnum() and d["text"][p].isalnum()
                        for p in [s, e]
                    )
                    counts["whitespace_edge"] += (
                        d["text"][s].isspace() or d["text"][e - 1].isspace()
                    )
                    contained = any(
                        enc["offsets"][w["start"]][0] <= s and enc["offsets"][w["end"] - 1][1] >= e
                        for w in enc["windows"]
                    )
                    counts["not_contained_in_one_window"] += not contained
                    family[span["label"]] += 1
                    lengths.append(e - s)
                    for t in spans[:index]:
                        if min(e, t["end"]) > max(s, t["start"]):
                            counts["overlapping_pairs"] += 1
                            counts["same_family_overlapping_pairs"] += span["label"] == t["label"]
        audit[split] = dict(
            counts=dict(counts),
            families=dict(family),
            span_length_quantiles=np.percentile(lengths, [0, 50, 95, 100]).tolist(),
            maximum_spans_per_answer=max(spans_per_answer),
            maximum_union_spans=max(union_sizes),
            positive_decisions_total=sum(positives),
            full_universe_negative_decisions_total=sum(negatives),
            empty_baseline=evaluate(part, empty_outputs(part), 0.5)[0],
        )
    write_json(args.output / "representation.json", audit)
    train = sorted(
        docs["train"], key=lambda d: (encoded[d["doc_id"]]["encoded"]["tokens"], d["doc_id"])
    )
    probe = [train[len(train) // 2], train[-1]]
    model = L3Model(encoder, cfg).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    initial = trainable_state(model)
    frozen = frozen_digest(model)
    timings = []
    details = []
    torch.cuda.reset_peak_memory_stats()
    for d in probe * 2:
        torch.cuda.synchronize()
        start = time.perf_counter()
        detail = update(model, optimizer, scheduler, [d], encoded, cfg)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        details.append(
            dict(doc_id=d["doc_id"], tokens=encoded[d["doc_id"]]["encoded"]["tokens"], **detail[0])
        )
    assert frozen_digest(model) == frozen
    changed = sum(not torch.equal(v, initial[k]) for k, v in trainable_state(model).items())
    checkpoint = args.output / "checkpoints/resume.pt"
    progress = dict(updates=4, position=0, order=[d["doc_id"] for d in probe])
    save_checkpoint(checkpoint, model, optimizer, scheduler, progress, prov)
    continuous = update(model, optimizer, scheduler, probe, encoded, cfg)
    expected = trainable_state(model)
    expected_lr = scheduler.get_last_lr()
    peak = torch.cuda.max_memory_reserved()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model = L3Model(encoder, cfg).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    assert load_checkpoint(checkpoint, model, optimizer, scheduler, prov) == progress
    resumed = update(model, optimizer, scheduler, probe, encoded, cfg)
    difference = max(
        float((v - expected[k]).abs().max()) for k, v in trainable_state(model).items()
    )
    assert difference == 0 and continuous == resumed and expected_lr == scheduler.get_last_lr()
    metrics = dict(
        status="passed",
        timings_seconds=timings,
        profiles=details,
        peak_reserved_bytes=peak,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        changed_tensors=changed,
        trainable_tensors=len(initial),
        base_unchanged=True,
        resume_max_parameter_difference=difference,
        optimizer_updates_including_repeat=6,
        elapsed_seconds=time.perf_counter() - begin,
    )
    assert changed == len(initial)
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "profiles"}))


if __name__ == "__main__":
    main()

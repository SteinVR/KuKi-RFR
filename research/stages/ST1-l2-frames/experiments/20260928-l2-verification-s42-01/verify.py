"""Independent artifact checks plus fresh-process GPU inference/restart verification."""

import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from research.kuki.baselines import predict_linear
from research.kuki.data import digest, write_json
from research.kuki.experiment import load_development
from research.kuki.l2 import L2Model, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.train_l2 import make_optimizer, predict, train_update
from research.kuki.windows import tokenize_document

STAGE = Path('research/stages/ST1-l2-frames')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def independent_scores(docs, predictions, expected):
    assert [d['doc_id'] for d in docs] == [p['doc_id'] for p in predictions]
    errors, targets, predicted = [], [], []
    for d, p in zip(docs, predictions, strict=True):
        answers = np.asarray([a['l2'] for a in d['annotations']], dtype=float)
        probability = np.asarray(p['probabilities'])
        errors.append(np.square(answers - probability).mean())
        targets.append(answers.mean(0) > 0.5)
        predicted.append(probability >= 0.5)
    targets, predicted = np.asarray(targets), np.asarray(predicted)
    tp = (targets & predicted).sum(0)
    fp = (~targets & predicted).sum(0)
    fn = (targets & ~predicted).sum(0)
    denominator = 2 * tp + fp + fn
    macro = np.divide(2 * tp, denominator, out=np.zeros(14), where=denominator != 0).mean()
    micro = 2 * tp.sum() / denominator.sum() if denominator.sum() else 0
    assert abs(np.mean(errors) - expected['brier']) < 1e-12
    assert abs(macro - expected['macro_f1']) < 1e-12
    assert abs(micro - expected['micro_f1']) < 1e-12


def main():
    started = time.perf_counter()
    docs, _ = load_development(read(Path('research/configs/data.json')))
    baseline = STAGE / 'experiments/20260928-l2-baselines-s42-01'
    neural = STAGE / 'experiments/20260928-l2-pilot-s42-01'
    for run in [baseline, neural]:
        prov = read(run / 'metadata.json')['provenance']
        for path, checksum in prov['code_sha256'].items():
            assert digest(run / 'code' / Path(path).name) == checksum
            assert digest(Path(path)) == checksum
        assert digest(run / 'uv.lock') == prov['uv_lock_sha256'] == digest(Path('uv.lock'))
    bm, nm = read(baseline / 'metrics.json'), read(neural / 'metrics.json')
    for split, part in docs.items():
        independent_scores(part, rows(neural / f'{split}-predictions.jsonl'), nm[split])
        for name in ['frequency', 'tfidf_logistic']:
            independent_scores(part, rows(baseline / f'{name}-{split}-predictions.jsonl'), bm[name][split])
    with (baseline / 'checkpoints/baselines.pkl').open('rb') as f:
        saved_baseline = pickle.load(f)  # Trusted local output from this run.
    linear = predict_linear(saved_baseline['features'], saved_baseline['models'],
                            [d['text'] for d in docs['dev']])
    assert np.array_equal(linear, [r['probabilities'] for r in rows(baseline / 'tfidf_logistic-dev-predictions.jsonl')])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    prov = read(neural / 'metadata.json')['provenance']
    cfg, encoder = prov['config'], prov['encoder']
    tok = AutoTokenizer.from_pretrained(encoder['model_id'], revision=encoder['revision'])
    required = docs['dev'] + docs['train'][:9]
    encoded = {d['doc_id']: tokenize_document(tok, d['text'], cfg['window_length'], cfg['overlap'])
               for d in required}
    model = L2Model(encoder).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    state = load_checkpoint(neural / 'checkpoints/best.pt', model, optimizer, scheduler, prov)
    reloaded = predict(model, docs['dev'], encoded)
    stored = np.asarray([r['probabilities'] for r in rows(neural / 'dev-predictions.jsonl')])
    maximum = float(np.max(np.abs(reloaded - stored)))
    assert maximum == 0
    best_updates = state['updates']
    assert state['best_epoch'] == nm['selected_epoch'] == 7 and best_updates == 133
    last = torch.load(neural / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    assert last['progress']['status'] == 'completed'
    assert last['progress']['updates'] == nm['total_updates'] == 152
    assert last['progress']['articles_seen'] == 8 * len(docs['train']) == 1160
    batch = [(encoded[d['doc_id']], [a['l2'] for a in d['annotations']]) for d in docs['train'][:9]]
    train_update(model, optimizer, scheduler, batch[:8], cfg['gradient_clip_norm'])
    technical = STAGE / 'experiments/20260928-l2-verification-s42-01'
    technical.mkdir(exist_ok=False)
    (technical / 'checkpoints').mkdir()
    scratch = technical / 'checkpoints/resume.pt'
    technical_progress = dict(purpose='technical_train_only_no_model_selection',
                              parent_run=neural.name, parent_updates=best_updates,
                              additional_updates=1, next_position=8)
    save_checkpoint(scratch, model, optimizer, scheduler, technical_progress, prov)
    continuous = train_update(model, optimizer, scheduler, batch[8:], cfg['gradient_clip_norm'])
    expected = trainable_state(model)
    expected_lr = scheduler.get_last_lr()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model = L2Model(encoder).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    restored = load_checkpoint(scratch, model, optimizer, scheduler, prov)
    assert restored == technical_progress
    resumed = train_update(model, optimizer, scheduler, batch[8:], cfg['gradient_clip_norm'])
    difference = max(float((value - expected[n]).abs().max()) for n, value in trainable_state(model).items())
    assert difference == 0 and continuous == resumed and expected_lr == scheduler.get_last_lr()
    result = dict(status='passed', independent_metric_recalculation=True,
                  snapshot_and_current_code_hashes_match=True, uv_lock_hashes_match=True,
                  split_source_prepared_hashes_match=True, test_predictions_absent=True,
                  baseline_pickle_predictions_exact=True,
                  fresh_gpu_checkpoint_dev_predictions_max_difference=maximum,
                  gpu_resume=dict(max_parameter_difference=difference, losses_equal=continuous == resumed,
                                  learning_rates_equal=expected_lr == scheduler.get_last_lr(),
                                  continuous_losses=continuous, resumed_losses=resumed),
                  training_counts_correct=True, technical_optimizer_updates_including_repeat=3,
                  parent_run=neural.name, parent_epoch=7, parent_updates=133,
                  verification_script_sha256=digest(Path(__file__)),
                  elapsed_seconds=time.perf_counter() - started)
    for path in STAGE.rglob('*predictions.jsonl'):
        assert all(r['split'] in ['train', 'dev'] for r in rows(path))
    write_json(technical / 'metrics.json', result)
    write_json(technical / 'config.json', dict(parent_run=neural.name, purpose=technical_progress['purpose'],
                                             train_ids=[d['doc_id'] for d in docs['train'][:9]],
                                             accumulation=[8, 1], repeat_final_update=True))
    write_json(technical / 'metadata.json', dict(provenance=prov, verification_script_sha256=digest(Path(__file__))))
    write_json(STAGE / 'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

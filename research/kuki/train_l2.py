"""Bounded, resumable ST1 training; the held-out test is inaccessible here."""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import write_json, write_jsonl
from .evaluation import save_predictions, score
from .experiment import initialize_run, load_development, provenance
from .l2 import (
    L2Model,
    article_loss,
    frozen_digest,
    load_checkpoint,
    save_checkpoint,
    seed_all,
    trainable_state,
)
from .windows import tokenize_document


def learning_rate_multiplier(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def make_optimizer(model, cfg):
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith('head.')], 'lr': cfg['adapter_lr']},
        {'params': model.head.parameters(), 'lr': cfg['head_lr']},
    ], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_multiplier(
            step, cfg['warmup_updates'], cfg['max_updates']))
    return optimizer, scheduler


def train_update(model, optimizer, scheduler, batches, clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for encoded, answers in batches:
        loss = article_loss(model(encoded), answers)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        (loss / len(batches)).backward()
        losses.append(float(loss.detach()))
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise ValueError('Missing or non-finite gradient')
    norm = torch.nn.utils.clip_grad_norm_(params, clip)
    if not torch.isfinite(norm):
        raise ValueError('Non-finite gradient norm')
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return losses


def predict(model, docs, encoded):
    model.eval()
    with torch.inference_mode():
        return np.asarray([model(encoded[d['doc_id']]).float().sigmoid().cpu().numpy() for d in docs])


def epoch_order(n, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(n).tolist()


def initial_progress():
    return dict(epoch=0, next_position=0, article_order=[], updates=0, articles_seen=0,
                epoch_loss_sum=0.0, epoch_articles=0, epoch_train_seconds=0.0,
                best_brier=None, best_epoch=None, patience_anchor=None, bad_epochs=0,
                history=[], active_seconds=0.0, status='running')


def finish_epoch(progress, metrics, cfg):
    value = metrics['brier']
    best = progress['best_brier'] is None or value < progress['best_brier']
    if best:
        progress['best_brier'] = value
        progress['best_epoch'] = progress['epoch'] + 1
    anchor = progress['patience_anchor']
    if anchor is None or value < anchor - cfg['early_stopping_min_delta']:
        progress['patience_anchor'] = value
        progress['bad_epochs'] = 0
    else:
        progress['bad_epochs'] += 1
    progress['history'].append(dict(
        epoch=progress['epoch'] + 1, updates=progress['updates'],
        articles_seen=progress['articles_seen'],
        train_loss=progress['epoch_loss_sum'] / progress['epoch_articles'],
        train_seconds=progress['epoch_train_seconds'], dev=metrics))
    progress['epoch'] += 1
    progress['next_position'] = 0
    progress['article_order'] = []
    progress['epoch_loss_sum'] = 0.0
    progress['epoch_articles'] = 0
    progress['epoch_train_seconds'] = 0.0
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int, help='Technical pause, not early stopping')
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads(Path(cfg['data_config']).read_text())
    encoder_cfg = json.loads(Path(cfg['encoder_config']).read_text())
    encoder_cfg['gradient_checkpointing'] = cfg['gradient_checkpointing']
    docs, hashes = load_development(data_cfg)
    prov = provenance(cfg, data_cfg, hashes, encoder_cfg)
    expected_updates = math.ceil(len(docs['train']) / cfg['gradient_accumulation_articles'])
    if cfg['max_updates'] != expected_updates * cfg['max_epochs']:
        raise ValueError('This trainer requires an integral maximum epoch budget')
    if not args.resume:
        initialize_run(args.output, cfg, prov)
    elif json.loads((args.output / 'metadata.json').read_text())['provenance'] != prov:
        raise ValueError('Resume provenance changed')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(cfg['deterministic'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 device required')
    seed_all(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(encoder_cfg['model_id'], revision=encoder_cfg['revision'])
    encoded = {d['doc_id']: tokenize_document(tokenizer, d['text'], cfg['window_length'], cfg['overlap'])
               for part in docs.values() for d in part}
    model = L2Model(encoder_cfg).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    progress = initial_progress()
    checkpoints = args.output / 'checkpoints'
    if args.resume:
        progress = load_checkpoint(checkpoints / 'last.pt', model, optimizer, scheduler, prov)
        if progress['status'] == 'completed':
            raise ValueError('Already completed; refusing to run again')
    previous_seconds = progress['active_seconds']
    frozen_before = frozen_digest(model)
    initial = trainable_state(model)
    torch.cuda.reset_peak_memory_stats()
    stop_requested = False

    def request_stop(_signal, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save(name):
        progress['active_seconds'] = previous_seconds + time.perf_counter() - started
        save_checkpoint(checkpoints / name, model, optimizer, scheduler, progress, prov)

    while progress['epoch'] < cfg['max_epochs']:
        if not progress['article_order']:
            progress['article_order'] = epoch_order(len(docs['train']), cfg['seed'], progress['epoch'])
        order = progress['article_order']
        while progress['next_position'] < len(order):
            if (stop_requested or time.perf_counter() - started >= cfg['session_save_stop_minutes'] * 60
                    or (args.stop_after_updates is not None
                        and progress['updates'] >= args.stop_after_updates)):
                progress['status'] = 'paused'
                save('last.pt')
                write_json(args.output / 'progress.json', progress)
                print(json.dumps(dict(status='paused', updates=progress['updates'])), flush=True)
                return
            pos = progress['next_position']
            indices = order[pos:pos + cfg['gradient_accumulation_articles']]
            batch = [(encoded[docs['train'][i]['doc_id']],
                      [a['l2'] for a in docs['train'][i]['annotations']]) for i in indices]
            torch.cuda.synchronize()
            begin = time.perf_counter()
            losses = train_update(model, optimizer, scheduler, batch, cfg['gradient_clip_norm'])
            torch.cuda.synchronize()
            progress['epoch_train_seconds'] += time.perf_counter() - begin
            progress['updates'] += 1
            progress['articles_seen'] += len(indices)
            progress['epoch_articles'] += len(indices)
            progress['epoch_loss_sum'] += sum(losses)
            progress['next_position'] += len(indices)
            progress['status'] = 'running'
            save('last.pt')
        dev = score(docs['dev'], predict(model, docs['dev'], encoded), cfg['prediction_threshold'])
        best = finish_epoch(progress, dev, cfg)
        if best:
            save('best.pt')
        # Epoch completion is committed before honoring a technical pause. It never
        # skips dev evaluation when pausing at the final update of an epoch.
        save('last.pt')
        write_json(args.output / 'progress.json', progress)
        write_jsonl(args.output / 'logs' / 'epochs.jsonl', progress['history'])
        print(json.dumps(progress['history'][-1]), flush=True)
        if progress['bad_epochs'] >= cfg['early_stopping_patience_epochs']:
            break
    progress['status'] = 'completed'
    save('last.pt')
    base_unchanged = frozen_before == frozen_digest(model)
    changed = {n: float((p - initial[n]).abs().max()) for n, p in trainable_state(model).items()}
    if not base_unchanged:
        raise ValueError('Frozen base changed')
    # Evaluate exactly the selected best state, not the last optimization state.
    selected = load_checkpoint(checkpoints / 'best.pt', model, optimizer, scheduler, prov)
    metrics = dict(status='completed', selected_epoch=selected['best_epoch'],
                   total_updates=progress['updates'], articles_seen=progress['articles_seen'],
                   history=progress['history'], base_unchanged=base_unchanged,
                   trainable_tensors_changed_since_session_start=sum(v > 0 for v in changed.values()),
                   trainable_tensors=len(changed),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    for split, part in docs.items():
        p = predict(model, part, encoded)
        metrics[split] = score(part, p, cfg['prediction_threshold'])
        save_predictions(args.output / f'{split}-predictions.jsonl', part, p, split,
                         cfg['prediction_threshold'])
    metrics['elapsed_seconds'] = previous_seconds + time.perf_counter() - started
    write_json(args.output / 'metrics.json', metrics)
    write_json(args.output / 'progress.json', progress)
    print(json.dumps(dict(status='completed', selected_epoch=metrics['selected_epoch'],
                          dev_brier=metrics['dev']['brier'], seconds=metrics['elapsed_seconds'])),
          flush=True)


if __name__ == '__main__':
    main()

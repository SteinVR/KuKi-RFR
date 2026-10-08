"""One bounded joint fit, one checkpoint criterion, complete update-boundary resume."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import argparse
import importlib
import json
import signal

import torch

from research.kuki.data import write_json, write_jsonl
from research.kuki.l2 import frozen_digest, load_checkpoint, save_checkpoint, trainable_state
from research.kuki.train_l2 import epoch_order, make_optimizer
from research.kuki.train_l3 import configure, encode_documents, update

M = importlib.import_module('research.stages.ST6-l1-l2-l3.method')
J = importlib.import_module('research.stages.ST6-l1-l2-l3.model')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int)
    args = parser.parse_args()
    cfg = M.read(M.ROOT / 'configs/joint.json')
    docs, enc, prov = M.inputs(cfg)
    refs = M.read(M.ROOT / 'references.json')
    train = docs['train']
    assert cfg['max_updates'] == cfg['max_epochs'] * ((len(train) + 7) // 8)
    out = M.ROOT / 'experiments' / cfg['run_id']
    M.reserve(cfg['run_budget_seconds'])
    if not args.resume:
        M.initialize(out, cfg, prov)
    configure(cfg['seed'])
    encoded = encode_documents(train + docs['dev'], enc, cfg)
    model = J.JointRoles(enc, cfg).cuda()
    initial = M.initialize_weights(model)
    frozen = frozen_digest(model)
    optimizer, scheduler = make_optimizer(model, cfg)
    progress = dict(epoch=0, next_position=0, article_order=[], updates=0, articles_seen=0,
                    l1_articles_seen=0, history=[], best_key=None, best_epoch=None,
                    epoch_loss_sum=0., epoch_l1_sum=0., epoch_l2_sum=0.,
                    active_seconds=0., status='running')
    if args.resume:
        progress = load_checkpoint(out / 'checkpoints/last.pt', model, optimizer, scheduler, prov)
        assert progress['status'] == 'paused'
    previous = progress['active_seconds']
    stop = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.cuda.reset_peak_memory_stats()

    def save(name):
        progress['active_seconds'] = previous + time.perf_counter() - STARTED
        save_checkpoint(out / 'checkpoints' / name, model, optimizer, scheduler, progress, prov)

    if not args.resume:
        write_json(out / 'initialization.json', dict(trainable_sha256=initial))
        save('start.pt')
    timed_out = False
    while progress['epoch'] < cfg['max_epochs']:
        if not progress['article_order']:
            progress['article_order'] = epoch_order(len(train), cfg['seed'], progress['epoch'])
        order = progress['article_order']
        while progress['next_position'] < len(order):
            if stop or (args.stop_after_updates and progress['updates'] >= args.stop_after_updates):
                progress['status'] = 'paused'
                save('last.pt')
                write_json(out / 'progress.json', progress)
                return dict(status='paused', updates=progress['updates'])
            if previous + time.perf_counter() - STARTED > cfg['run_budget_seconds'] - 70:
                timed_out = True
                break
            pos = progress['next_position']
            batch = [train[i] for i in order[pos:pos + cfg['gradient_accumulation_articles']]]
            detail = update(model, optimizer, scheduler, batch, encoded, cfg)
            for key in ['loss', 'l1', 'l2']:
                progress[f'epoch_{key}_sum'] += sum(x[key] for x in detail)
            progress['updates'] += 1
            progress['articles_seen'] += len(batch)
            progress['l1_articles_seen'] += sum(any(a['l1_roles'] for a in d['annotations']) for d in batch)
            progress['next_position'] += len(batch)
            save('last.pt')
        if timed_out:
            break
        metrics = M.metrics(docs['dev'], M.predict(model, docs['dev'], encoded))
        key = J.selection_key(metrics, refs)
        improved = progress['best_key'] is None or tuple(key) > tuple(progress['best_key'])
        progress['epoch'] += 1
        history = dict(epoch=progress['epoch'], updates=progress['updates'],
                       articles_seen=progress['articles_seen'], l1_articles_seen=progress['l1_articles_seen'],
                       dev=metrics, selection_key=key,
                       train_loss=progress['epoch_loss_sum'] / len(train),
                       train_l1=progress['epoch_l1_sum'] / 135,
                       train_l2=progress['epoch_l2_sum'] / len(train),
                       elapsed_seconds=previous + time.perf_counter() - STARTED)
        progress['history'].append(history)
        progress.update(article_order=[], next_position=0, epoch_loss_sum=0., epoch_l1_sum=0., epoch_l2_sum=0.)
        if improved:
            progress.update(best_key=key, best_epoch=progress['epoch'])
            save('best.pt')
        save('last.pt')
        write_json(out / 'progress.json', progress)
        write_jsonl(out / 'logs/epochs.jsonl', progress['history'])
        print(json.dumps(dict(epoch=progress['epoch'], best=progress['best_epoch'],
                              l1_f1=metrics['l1']['macro_f1'], l2_brier=metrics['l2']['brier'],
                              seconds=history['elapsed_seconds'])), flush=True)
    assert frozen_digest(model) == frozen
    progress['status'] = 'completed'
    progress['stop_reason'] = 'time_limit' if timed_out else 'epoch_limit'
    save('last.pt')
    if progress['best_epoch'] is None:
        raise RuntimeError('No complete development evaluation within the budget')
    for state, splits in [('last', ['dev']), ('best', ['dev', 'train'])]:
        payload = torch.load(out / 'checkpoints' / f'{state}.pt', weights_only=False, map_location='cpu')
        M.set_weights(model, payload['trainable'])
        for split in splits:
            M.persist(out / state, split, docs[split], M.predict(model, docs[split], encoded))
    write_json(out / 'progress.json', progress)
    result = dict(status='completed', run_id=cfg['run_id'], updates=progress['updates'],
                  article_presentations=progress['articles_seen'],
                  l1_labeled_presentations=progress['l1_articles_seen'],
                  best_epoch=progress['best_epoch'], stop_reason=progress['stop_reason'],
                  base_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  trainable_parameters=sum(p.numel() for p in trainable_state(model).values()))
    write_json(out / 'runtime.json', dict(seconds=time.perf_counter() - STARTED + previous, **result))
    return result


if __name__ == '__main__':
    result = dict(status='failed')
    try:
        result = main()
    finally:
        M.account('fit', STARTED, result)

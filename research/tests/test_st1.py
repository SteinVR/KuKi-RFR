import copy
import json

import numpy as np
import pytest
import torch

from research.kuki.data import FRAMES, digest, write_json, write_jsonl
from research.kuki.evaluation import article_brier, score, votes
from research.kuki.experiment import load_development
from research.kuki.l2 import load_checkpoint, save_checkpoint, seed_all, trainable_state
from research.kuki.train_l2 import (
    epoch_order,
    finish_epoch,
    initial_progress,
    learning_rate_multiplier,
    make_optimizer,
    train_update,
)


def document(answers):
    return {'annotations': [{'l2': row} for row in answers]}


def test_brier_individual_answers_equal_article_weights_and_decomposition():
    docs = [document([[1] * 14]), document([[0] * 14] * 3)]
    p = np.full((2, 14), 0.2)
    assert article_brier(docs, p).mean() == pytest.approx((0.64 + 0.04) / 2)
    mixed = [document([[1] * 14, [0] * 14]), document([[1] * 14])]
    result = score(mixed, p)
    assert result['brier'] == pytest.approx(
        result['vote_fraction_mse'] + result['observed_disagreement_floor'])
    assert result['per_frame'][0]['majority_support'] == 1  # ties are negative
    assert result['per_frame'][0]['ties'] == 1
    assert result['micro_f1'] == 0
    assert np.array_equal(votes(docs), votes([docs[0], document([[0] * 14] * 30)]))
    with pytest.raises(ValueError):
        score(docs, np.full((2, 14), np.nan))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.25)
        self.head = torch.nn.Linear(3, 14)

    def forward(self, encoded):
        return self.head(self.dropout(self.adapter(encoded)))


def test_training_resume_matches_across_shuffle_and_partial_batch(tmp_path):
    cfg = dict(adapter_lr=0.002, head_lr=0.003, weight_decay=0.01,
               warmup_updates=2, max_updates=6)
    seed_all(42)
    model = ToyModel()
    opt, sched = make_optimizer(model, cfg)
    data = [(torch.tensor([i / 5, 0.2, 0.5]), [[i % 2] * 14]) for i in range(5)]
    batches = []
    for epoch in range(2):
        order = epoch_order(5, 42, epoch)
        batches.extend([[data[i] for i in order[k:k + 2]] for k in range(0, 5, 2)])
    train_update(model, opt, sched, batches[0], 1.0)
    path = tmp_path / 'state.pt'
    progress = {'epoch': 0, 'article_order': epoch_order(5, 42, 0), 'next_position': 2}
    save_checkpoint(path, model, opt, sched, progress, {'config': cfg})
    continuous = [train_update(model, opt, sched, b, 1.0) for b in batches[1:]]
    expected = trainable_state(model)
    expected_opt = copy.deepcopy(opt.state_dict())
    seed_all(99)
    restored = ToyModel()
    ropt, rsched = make_optimizer(restored, cfg)
    assert load_checkpoint(path, restored, ropt, rsched, {'config': cfg}) == progress
    resumed = [train_update(restored, ropt, rsched, b, 1.0) for b in batches[1:]]
    assert resumed == continuous
    assert sched.get_last_lr() == rsched.get_last_lr()
    for name, value in trainable_state(restored).items():
        assert torch.equal(value, expected[name])
    for key, value in ropt.state_dict()['state'].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, expected_opt['state'][key][field])


def test_partial_accumulation_uses_actual_count():
    seed_all(42)
    model = ToyModel()
    model.dropout.p = 0
    other = copy.deepcopy(model)
    cfg = dict(adapter_lr=0.002, head_lr=0.003, weight_decay=0.01,
               warmup_updates=2, max_updates=6)
    opt, sched = make_optimizer(model, cfg)
    opt2, sched2 = make_optimizer(other, cfg)
    batch = [(torch.ones(3), [[1] * 14])]
    train_update(model, opt, sched, batch, 1.0)
    train_update(other, opt2, sched2, batch * 3, 1.0)
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-7, rtol=0)


def test_best_selection_differs_from_patience_delta():
    p = initial_progress()
    cfg = dict(early_stopping_min_delta=0.01)
    for brier in [0.2, 0.199, 0.198]:
        p['epoch_articles'] = 1
        assert finish_epoch(p, {'brier': brier}, cfg)
    assert p['best_epoch'] == 3
    assert p['bad_epochs'] == 2
    assert learning_rate_multiplier(0, 15, 152) == pytest.approx(1 / 15)
    assert learning_rate_multiplier(14, 15, 152) == 1
    assert learning_rate_multiplier(152, 15, 152) == 0


def test_loader_excludes_test_and_rejects_group_leakage(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    prepared = tmp_path / 'prepared.jsonl'
    rows = [dict(doc_id=s, text_sha256=s, split=s, group_id=s) for s in ['train', 'dev', 'test']]
    write_jsonl(prepared, [dict(doc_id=r['doc_id'], text_sha256=r['text_sha256']) for r in rows])
    manifest = dict(prepared_path=str(prepared), prepared_sha256=digest(prepared),
                    source=str(source), source_sha256=digest(source), labels={'l2': FRAMES}, rows=rows)
    split = tmp_path / 'split.json'
    write_json(split, manifest)
    cfg = dict(split_manifest=str(split), split_sha256=digest(split), prepared_path=str(prepared))
    docs, _ = load_development(cfg)
    assert set(docs) == {'train', 'dev'}
    assert [d['doc_id'] for part in docs.values() for d in part] == ['train', 'dev']
    manifest['rows'][2]['group_id'] = 'train'
    write_json(split, manifest)
    cfg['split_sha256'] = digest(split)
    with pytest.raises(ValueError, match='Group crosses'):
        load_development(cfg)
    manifest['rows'][2]['group_id'] = 'test'
    split.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Split hash'):
        load_development(cfg)


def test_linear_baseline_fractional_votes_equal_explicit_annotators():
    from sklearn.linear_model import LogisticRegression

    from research.kuki.baselines import fit_linear, predict_linear

    texts = ['cat cat red', 'cat red green', 'dog blue blue', 'dog green blue']
    answers = [[1], [0, 1], [0, 0, 1], [0]]
    targets = np.tile(np.array([np.mean(a) for a in answers])[:, None], (1, 14))
    cfg = dict(word_max_features=100, char_max_features=100, C=10,
               max_iter=1000, tolerance=1e-10, seed=42)
    features, models = fit_linear(texts, targets, cfg)
    explicit_texts = [text for text, a in zip(texts, answers, strict=True) for _ in a]
    explicit_labels = [value for a in answers for value in a]
    explicit_weights = [1 / len(a) for a in answers for _ in a]
    explicit = LogisticRegression(C=10, max_iter=1000, tol=1e-10, solver='lbfgs')
    explicit.fit(features.transform(explicit_texts), explicit_labels, sample_weight=explicit_weights)
    expected = explicit.predict_proba(features.transform(texts))[:, 1]
    actual = predict_linear(features, models, texts)[:, 0]
    assert np.allclose(actual, expected, atol=1e-7, rtol=0)
    before = features.get_feature_names_out().tolist()
    predict_linear(features, models, ['unseen vocabulary never fitted'])
    assert features.get_feature_names_out().tolist() == before

"""Train-only frequency and TF-IDF logistic baselines with article weighting."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .data import write_json
from .evaluation import save_predictions, score, votes
from .experiment import initialize_run, load_development, provenance


def fit_linear(texts, targets, cfg):
    features = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                max_features=cfg['word_max_features'])),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                                sublinear_tf=True, max_features=cfg['char_max_features'])),
    ], transformer_weights={'word': 2 ** -0.5, 'char': 2 ** -0.5})
    x = features.fit_transform(texts)
    doubled = vstack([x, x], format='csr')
    y = np.r_[np.ones(len(texts)), np.zeros(len(texts))]
    models = []
    for column in targets.T:
        if np.all(column == 0) or np.all(column == 1):
            models.append(float(column[0]))
            continue
        model = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter'], solver='lbfgs',
                                   tol=cfg['tolerance'], random_state=cfg['seed'])
        # Each article contributes total weight 1: fractional positives + negatives.
        model.fit(doubled, y, sample_weight=np.r_[column, 1 - column])
        if model.n_iter_.max() >= cfg['max_iter']:
            raise ValueError('Linear baseline did not converge')
        models.append(model)
    return features, models


def predict_linear(features, models, texts):
    x = features.transform(texts)
    return np.stack([np.full(len(texts), m) if isinstance(m, float)
                     else m.predict_proba(x)[:, 1] for m in models], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    begin = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads(Path(cfg['data_config']).read_text())
    docs, hashes = load_development(data_cfg)
    initialize_run(args.output, cfg, provenance(cfg, data_cfg, hashes))
    fractions = votes(docs['train'])
    frequency = fractions.mean(axis=0)
    features, models = fit_linear([d['text'] for d in docs['train']], fractions, cfg)
    with (args.output / 'checkpoints' / 'baselines.pkl').open('wb') as f:
        pickle.dump(dict(frequency=frequency, features=features, models=models), f)
    metrics = {}
    for name in ['frequency', 'tfidf_logistic']:
        metrics[name] = {}
        for split, part in docs.items():
            p = (np.tile(frequency, (len(part), 1)) if name == 'frequency' else
                 predict_linear(features, models, [d['text'] for d in part]))
            metrics[name][split] = score(part, p, cfg['prediction_threshold'])
            save_predictions(args.output / f'{name}-{split}-predictions.jsonl', part, p, split)
    metrics.update(elapsed_seconds=time.perf_counter() - begin,
                   linear_feature_count=len(features.get_feature_names_out()), status='completed')
    write_json(args.output / 'metrics.json', metrics)
    print(json.dumps({n: metrics[n]['dev']['brier'] for n in ['frequency', 'tfidf_logistic']}))


if __name__ == '__main__':
    main()

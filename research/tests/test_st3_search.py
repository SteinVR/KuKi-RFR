"""Prevent selecting incompatible L2/L3 checkpoints in the search objective."""
import importlib

S = importlib.import_module('research.stages.ST3-l2-l3-joint.search.run_search')


def test_objectives_share_selected_checkpoint():
    progress = dict(best_epoch=2, history=[
        dict(epoch=1, dev=dict(l2=dict(brier=0.2), conditional=dict(article_mean_f1=0.9))),
        dict(epoch=2, dev=dict(l2=dict(brier=0.1), conditional=dict(article_mean_f1=0.7))),
    ])
    assert S.objective_values(progress) == (0.1, 0.7)

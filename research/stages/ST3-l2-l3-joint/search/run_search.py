"""Bounded, sequential Optuna grid; each objective is one shared checkpoint."""

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import optuna

from research.kuki.data import write_json

ROOT = Path(__file__).parent
M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
RESERVE_SECONDS = 40


def objective_values(progress):
    """Use both outputs at the predeclared L2-selected epoch, not separate maxima."""
    epoch = progress["best_epoch"]
    chosen = next(h for h in progress["history"] if h["epoch"] == epoch)
    return chosen["dev"]["l2"]["brier"], chosen["dev"]["conditional"]["article_mean_f1"]


def objective(trial):
    weight = trial.suggest_float("l3_weight", 0.25, 4.0, log=True)
    name = {2.0: "joint200", 4.0: "joint400"}[weight]
    cfg = M.read(ROOT / "configs" / f"{name}.json")
    ledger = M.read(ROOT / "budget.json")
    available = ledger["limit_seconds"] - ledger["seconds"] - RESERVE_SECONDS
    if available < cfg["run_budget_seconds"]:
        raise RuntimeError("Insufficient remaining budget for a complete fit plus verification")
    out = ROOT / "experiments" / cfg["run_id"]
    log = ROOT / f"{name}-stdout.log"
    started = time.perf_counter()
    status = "failed"
    try:
        with log.open("w") as stream:
            subprocess.run(
                [sys.executable, "-m", "research.stages.ST3-l2-l3-joint.search.train_search",
                 "--condition", name],
                stdout=stream, stderr=subprocess.STDOUT,
                timeout=cfg["run_budget_seconds"], check=True,
            )
        progress = M.read(out / "progress.json")
        status = progress["status"]
        if status != "completed":
            raise RuntimeError(f"Incomplete fit: {status}")
        trial.set_user_attr("run_id", cfg["run_id"])
        trial.set_user_attr("selected_epoch", progress["best_epoch"])
        trial.set_user_attr("selection", "minimum_dev_L2_Brier_shared_checkpoint")
        return objective_values(progress)
    finally:
        elapsed = time.perf_counter() - started
        ledger["seconds"] += elapsed
        record = dict(kind="fit", run_id=cfg["run_id"], seconds=elapsed, status=status)
        ledger["runs"].append(record)
        write_json(ROOT / "budget.json", ledger)
        if out.exists():
            write_json(out / "execution.json", record)
            (out / "logs/stdout.log").write_bytes(log.read_bytes())
        trial.set_user_attr("gpu_process_wall_seconds", elapsed)
        print(json.dumps(record), flush=True)


def main():
    # New study only: rerunning cannot silently duplicate completed experiments.
    study = optuna.create_study(
        study_name="ST3-lambda-extension-20260929",
        storage=f"sqlite:///{ROOT / 'study.sqlite3'}",
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.GridSampler({"l3_weight": [2.0, 4.0]}, seed=42),
    )
    study.set_metric_names(["L2_dev_Brier", "L3_conditional_dev_F1"])
    study.set_user_attr("protocol", str(ROOT / "protocol.md"))
    study.set_user_attr("test_used", False)
    for weight in [2.0, 4.0]:
        study.enqueue_trial({"l3_weight": weight})
    study.optimize(objective, n_trials=2)
    write_json(ROOT / "study.json", dict(
        optuna_version=optuna.__version__, sampler="GridSampler", directions=["minimize", "maximize"],
        trials=[dict(number=t.number, state=t.state.name, params=t.params,
                     values=t.values, attrs=t.user_attrs) for t in study.trials],
        pareto_trial_numbers=[t.number for t in study.best_trials],
    ))


if __name__ == "__main__":
    main()

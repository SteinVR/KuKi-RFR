"""Serial bounded dispatch only; exits at the first incomplete or failed run."""

import importlib
import subprocess
import sys

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
A = importlib.import_module("research.stages.ST4-l3-to-l2.artifacts")


def run(module, args, log):
    if M.remaining() < 180:
        raise RuntimeError("Insufficient global budget; no extra run started")
    with log.open("w") as output:
        subprocess.run(
            [sys.executable, "-m", module, *args],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=M.remaining(),
        )


def main():
    for i in range(3):
        out = M.ROOT / "oof" / f"20260929-oof{i}-s42-01"
        if not out.exists():
            run(
                "research.stages.ST4-l3-to-l2.oof_train",
                ["--fold", str(i)],
                M.ROOT / "oof" / f"fold{i}.log",
            )
        assert M.read(out / "progress.json")["status"] == "completed"
        print(
            dict(completed=f"oof{i}", seconds=M.read(M.ROOT / "budget.json")["seconds"]), flush=True
        )
    A.assemble()
    for mode in ["continue", "self", "hidden", "uniform", "confidence", "contextual"]:
        cfg = M.read(M.ROOT / "configs" / f"{mode}.json")
        out = M.ROOT / "experiments" / cfg["run_id"]
        if not out.exists():
            run("research.stages.ST4-l3-to-l2.train", ["--condition", mode], M.ROOT / f"{mode}.log")
        assert M.read(out / "progress.json")["status"] == "completed"
        A.snapshot(out)
        print(dict(completed=mode, seconds=M.read(M.ROOT / "budget.json")["seconds"]), flush=True)


if __name__ == "__main__":
    main()

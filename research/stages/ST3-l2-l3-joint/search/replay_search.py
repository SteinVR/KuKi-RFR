"""Hard-budgeted fresh checkpoint replay, charged even when interrupted."""

import importlib
import subprocess
import sys
import time
from pathlib import Path

from research.kuki.data import write_json

ROOT = Path(__file__).parent
M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")


def main():
    ledger = M.read(ROOT / "budget.json")
    available = min(40, ledger["limit_seconds"] - ledger["seconds"])
    assert available > 0
    status = "failed"
    started = time.perf_counter()
    try:
        subprocess.run(
            [sys.executable, "-m", "research.stages.ST3-l2-l3-joint.search.verify_search_gpu"],
            timeout=available,
            check=True,
        )
        status = "passed"
    finally:
        elapsed = time.perf_counter() - started
        ledger["seconds"] += elapsed
        ledger["runs"].append(
            dict(kind="fresh_reload_verification", status=status, seconds=elapsed)
        )
        write_json(ROOT / "budget.json", ledger)


if __name__ == "__main__":
    main()

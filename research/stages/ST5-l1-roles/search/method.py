"""Isolated ST5 search artifacts and resource budget; preserve original series."""

import importlib
import shutil
import time
from pathlib import Path

from research.kuki.data import digest, write_json
from research.kuki.experiment import initialize_run

BASE = importlib.import_module("research.stages.ST5-l1-roles.method")
ROOT = Path(__file__).parent
read = BASE.read
active = BASE.active
initialization = BASE.initialization
set_weights = BASE.set_weights
persist = BASE.persist


def inputs(cfg):
    docs, encoder, prov = BASE.inputs(cfg)
    prov["original_protocol_sha256"] = prov["protocol_sha256"]
    prov["protocol_sha256"] = digest(ROOT / "protocol.md")
    prov["code_sha256"].update({str(p): digest(p) for p in sorted(ROOT.glob("*.py"))})
    return docs, encoder, prov


def initialize(out, cfg, prov):
    initialize_run(out, cfg, prov)
    # The flat legacy snapshot has filename collisions across stages. This tree
    # is authoritative and preserves each source at its provenance path.
    for source in prov["code_sha256"]:
        relative = Path(source).resolve().relative_to(Path.cwd())
        dest = out / "code-tree" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert digest(dest) == prov["code_sha256"][source]


def account(kind, started, result):
    seconds = time.perf_counter() - started
    ledger = read(ROOT / "budget.json")
    ledger["seconds"] += seconds
    ledger["runs"].append(dict(kind=kind, seconds=seconds, **result))
    write_json(ROOT / "budget.json", ledger)
    print(dict(seconds=seconds, **result), flush=True)


def reserve(seconds):
    ledger = read(ROOT / "budget.json")
    if ledger["seconds"] + seconds > ledger["limit_seconds"]:
        raise RuntimeError("ST5 search budget exhausted")

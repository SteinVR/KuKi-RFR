"""Finalize local evidence, link checks and hashes without touching prior ST3."""

import importlib
import re
from pathlib import Path

from research.kuki.data import digest, write_json

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path(__file__).parent


def main():
    prior = M.read(ROOT / "prior-artifact-hashes.json")
    historical = M.read(ROOT.parent / "prior-artifact-hashes.json")
    for paths in [prior, historical]:
        assert all(digest(Path(p)) == h for p, h in paths.items())
    v = M.read(ROOT / "verification.json")
    assert v["status"] == "passed"
    a = M.read(ROOT / "analysis.json")
    assert a["selected_joint"] == "joint200"
    assert not a["feasible_joint"]
    b = M.read(ROOT / "budget.json")
    assert b["seconds"] + b["original_seconds"] <= 3600
    missing = []
    reports = [ROOT / "README.md", ROOT / "report.md", *ROOT.glob("experiments/*/report.md")]
    for p in reports:
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if link.startswith(("http:", "https:", "#")):
                continue
            target = (p.parent / link.split("#")[0]).resolve()
            if not target.exists():
                missing.append(dict(source=str(p), target=str(target)))
    assert not missing, missing
    checks = M.read(ROOT / "checks.json")
    checks.update(
        self_review=dict(
            completed=True,
            findings_addressed=[
                "Macro-F1 L2 fell despite Brier gain; disclosed in reports and shared status docs.",
                "Brier difference vs joint100 has descriptive CI crossing zero; no robustness claim.",
                "GridSampler is not presented as adaptive TPE or proof of a global optimum.",
                "Both objectives use one L2-selected checkpoint; no per-output checkpoint mixing.",
                "Two of three development guards pass; no replacement of both standalone models.",
            ],
            prior_ST3_files_unchanged=len(prior),
            historical_ST1_ST2_files_unchanged=len(historical),
            report_links_checked=len(reports),
            missing_links=missing,
            chart_visually_reviewed=True,
            test_used=False,
            ST4_started=False,
        )
    )
    write_json(ROOT / "checks.json", checks)
    files = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.name != "manifest.json"
    }
    write_json(
        ROOT / "manifest.json",
        dict(
            files=files,
            count=len(files),
            external_sources={
                str(p): digest(p)
                for p in [
                    Path("pyproject.toml"),
                    Path("uv.lock"),
                    Path("research/tests/test_st3_search.py"),
                ]
            },
            status="completed",
            total_GPU_process_seconds=b["seconds"] + b["original_seconds"],
        ),
    )
    print(
        dict(
            status="completed",
            artifacts=len(files),
            total_seconds=b["seconds"] + b["original_seconds"],
            prior_files_unchanged=len(prior) + len(historical),
            missing_links=missing,
        )
    )


if __name__ == "__main__":
    main()

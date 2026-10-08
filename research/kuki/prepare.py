"""Deterministic RU preparation; refuses to overwrite a prior run."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from .data import (
    FAMILIES,
    FRAMES,
    ROLES,
    digest,
    distributions,
    group_duplicates,
    prepare_document,
    split_groups,
    write_json,
    write_jsonl,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(cfg["source"])
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    assert len({d["doc_id"] for d in rows}) == len(rows)
    docs, events = [], []
    for raw in sorted(rows, key=lambda d: d["doc_id"]):
        doc, log = prepare_document(raw)
        docs.append(doc)
        events.extend(log)
    groups, pairs, nearest = group_duplicates(
        docs, cfg["near_duplicate_jaccard"], cfg.get("must_link", [])
    )
    assignment = split_groups(docs, groups, cfg["split_seed"])
    prepared = Path(cfg["prepared_path"])
    # Only the reproducible local derivative may be replaced; source never changes.
    write_jsonl(prepared, docs)
    write_jsonl(args.output / "transformations.jsonl", events)
    dist = distributions(docs, assignment)
    labels = (
        ["l2:" + f for f in FRAMES]
        + ["l1_roles:" + r for r in ROLES]
        + ["l3_persuasion:" + f for f in FAMILIES]
    )
    manifest = dict(
        schema_version=1,
        source=str(source),
        source_sha256=digest(source),
        prepared_path=str(prepared),
        prepared_sha256=digest(prepared),
        config_sha256=digest(args.config),
        seed=cfg["split_seed"],
        protocol="internal_70_15_15",
        labels={"l2": FRAMES, "l1": ROLES, "l3": FAMILIES},
        rows=[
            dict(
                doc_id=d["doc_id"],
                split=assignment[d["doc_id"]][0],
                group_id=assignment[d["doc_id"]][1],
                text_sha256=d["text_sha256"],
            )
            for d in docs
        ],
    )
    write_json(args.output / "split.json", manifest)
    write_json(args.output / "config.json", cfg)
    write_json(
        args.output / "audit.json",
        dict(
            documents=len(docs),
            annotations=sum(len(d["annotations"]) for d in docs),
            events=dict(Counter(e["kind"] for e in events)),
            groups=len(groups),
            duplicate_pairs=pairs,
            nearest_pairs=nearest,
            distributions=dist,
            absent_labels={
                s: [label for label in labels if dist[s]["document_presence"].get(label, 0) == 0]
                for s in dist
            },
            source_unchanged=digest(source) == manifest["source_sha256"],
        ),
    )
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "research/kuki"]).decode().split("\0")
    )
    code = {str(p): digest(p) for p in sorted(Path("research/kuki").glob("*.py"))}
    write_json(
        args.output / "metadata.json",
        dict(
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            code_sha256=code,
            uv_lock_sha256=digest(Path("uv.lock")),
            split_sha256=digest(args.output / "split.json"),
            code_tracked={p: p in tracked for p in code},
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sizes": {s: v["documents"] for s, v in dist.items()},
                "events": dict(Counter(e["kind"] for e in events)),
            }
        )
    )


if __name__ == "__main__":
    main()

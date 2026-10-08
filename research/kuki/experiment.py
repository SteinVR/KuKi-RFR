"""Shared provenance and train/dev-only loading for ST1."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

from .data import FRAMES, digest, write_json


def load_development(data_cfg):
    split_path = Path(data_cfg['split_manifest'])
    if digest(split_path) != data_cfg['split_sha256']:
        raise ValueError('Split hash mismatch')
    manifest = json.loads(split_path.read_text())
    prepared = Path(data_cfg['prepared_path'])
    if str(prepared) != manifest['prepared_path'] or digest(prepared) != manifest['prepared_sha256']:
        raise ValueError('Prepared data hash mismatch')
    if digest(Path(manifest['source'])) != manifest['source_sha256']:
        raise ValueError('Source hash mismatch')
    if manifest['labels']['l2'] != FRAMES:
        raise ValueError('Frame order mismatch')
    rows = {r['doc_id']: r for r in manifest['rows']}
    if len(rows) != len(manifest['rows']):
        raise ValueError('Duplicate manifest IDs')
    groups = {}
    for row in rows.values():
        if groups.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('Group crosses split boundary')
    docs = {'train': [], 'dev': []}
    seen = set()
    # The combined derivative is hashed/read for integrity; test records are never
    # returned to tokenizers, feature fitting, training, scoring, or diagnostics.
    for line in prepared.read_text().splitlines():
        d = json.loads(line)
        row = rows[d['doc_id']]
        if d['doc_id'] in seen or d['text_sha256'] != row['text_sha256']:
            raise ValueError('Duplicate ID or text mismatch')
        seen.add(d['doc_id'])
        if row['split'] in docs:
            d['group_id'] = row['group_id']
            docs[row['split']].append(d)
    if seen != set(rows) or not all(docs.values()):
        raise ValueError('Incomplete prepared corpus')
    for part in docs.values():
        part.sort(key=lambda d: d['doc_id'])
    return docs, dict(split_sha256=digest(split_path), prepared_sha256=digest(prepared),
                      source_sha256=manifest['source_sha256'])


def provenance(cfg, data_cfg, data_hashes, encoder=None):
    return dict(config=cfg, data=data_cfg, data_hashes=data_hashes, encoder=encoder,
                uv_lock_sha256=digest(Path('uv.lock')),
                code_sha256={str(p): digest(p) for p in sorted(Path('research/kuki').glob('*.py'))})


def initialize_run(output, cfg, prov):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    (output / 'logs').mkdir()
    (output / 'code').mkdir()
    for source in prov['code_sha256']:
        shutil.copy2(source, output / 'code' / Path(source).name)
    shutil.copy2('uv.lock', output / 'uv.lock')
    shutil.copy2('pyproject.toml', output / 'pyproject.toml')
    write_json(output / 'config.json', dict(config=cfg, data=prov['data'], encoder=prov['encoder']))
    write_json(output / 'metadata.json', dict(
        provenance=prov, python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in
                  ['torch', 'transformers', 'peft', 'numpy', 'scipy', 'scikit-learn', 'kernels']},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        test_used_for_training_selection_or_scoring=False,
    ))

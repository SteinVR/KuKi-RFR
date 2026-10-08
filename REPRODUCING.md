# Reproducing KuKi-RFR

Run commands from the repository root. The corpus, configurations, epoch logs, and predictions are included. The pretrained encoder and attention kernel are identified by immutable revisions in [encoder.json](research/configs/encoder.json).

## Verify the published results

```bash
uv run --no-project --python 3.14 --with numpy==2.5.3 python -m research.kuki.verify_results \
  --output verification.json
uv run --no-project --python 3.14 python -m research.kuki.check_repository
```

The first command checks corpus hashes, reconstructs preparation, validates the split and span coordinates, and recomputes the primary metrics and bootstrap intervals. It reads the saved test predictions and requires agreement within `1e-12`. The second checks Python syntax, internal imports, documentation links, configuration paths, and the completeness of saved prediction sets. GitHub Actions runs both checks.

Each article has equal evaluation weight. Individual responses stay together during resampling. L1 macro-F1 is recomputed from the summed weighted confusion matrix in each replicate. L2 Brier averages squared errors over frames, responses, and articles. L3 F1 averages over responses and then articles. The [final protocol](research/stages/final/protocol.md) defines the comparisons and bootstrap settings.

## Prepare data

The [source aggregate](research/external/science-work-ref/kuki_ru_aggregate.jsonl) contains 207 articles and 301 annotations. The [prepared corpus](research/data/kuki-ru-v1.jsonl) applies the [recorded policy](research/stages/ST0-preparation/configs/data-v2.json): duplicate marks are merged, invalid layer–label combinations are quarantined, and conflicting role labels are masked within the affected response. Two empty frame answers are retained as all-zero targets; their annotation status is unknown. L4 remains in the data for provenance.

The split is fixed by article and duplicate group. The manifest records document assignments, label order, and source hashes. To run preparation into a new directory:

```bash
uv run --locked python -m research.kuki.prepare \
  --config research/stages/ST0-preparation/configs/data-v2.json \
  --output research/stages/ST0-preparation/experiments/reproduction-data
```

This command also writes the prepared corpus at the path specified by the policy. Use a separate checkout for experiments that change data or configuration.

## Training environment

The recorded stack targets Linux and an NVIDIA GPU:

```bash
uv python install
uv sync --locked --managed-python
uv run --locked pytest -q
```

[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) pin Python 3.14, PyTorch 2.12.1 with CUDA 13.0, Transformers 5.17, PEFT 0.21, and the remaining dependencies. The study used Python 3.14.7 and an RTX 4060 Laptop GPU with 8 GB VRAM. The encoder and FlashAttention kernel download from Hugging Face on first use. The tokenizer test uses the local cache; populate it with the pinned encoder revision to run that test.

## Experiment dependencies

| Step | Entry point | Configuration or condition | Required predecessor |
|:---|:---|:---|:---|
| Standalone frame pilot | `research.kuki.train_l2` | `ST1-l2-frames/configs/pilot-v1.json` | Prepared corpus and split |
| Rhetoric detector and family classifier | `research.stages.ST2-l3-rhetoric.modular.train` | `ST2-l3-rhetoric/modular/E1-classification/{base,low-lr}.json`, `E3-detector/{base,low-lr}.json` | Prepared corpus and split |
| Standalone and joint frame–rhetoric models | `research.stages.ST3-l2-l3-joint.train` | `--condition l2-only`, `l3-only`, `joint025`, or `joint100` | Fixed detector and its train/dev candidates |
| Extended frame–rhetoric weights | `research.stages.ST3-l2-l3-joint.search.train_search` | `--condition joint200` or `joint400` | ST3 inputs |
| Out-of-fold rhetoric candidates | `research.stages.ST4-l3-to-l2.oof_train` | `--fold 0`, `1`, or `2` | Detector settings and fold assignment |
| Frozen transfer features | `research.stages.ST4-l3-to-l2.frozen.extract` | `ST4-l3-to-l2/frozen/config.json` | Selected joint200 checkpoint and OOF candidates |
| Residual frame models | `research.stages.ST4-l3-to-l2.frozen.fit` | `control` and `families` | Frozen feature cache |
| Standalone role controls | `research.stages.ST5-l1-roles.train` | `--condition base` and `low-lr` | Prepared corpus and split |
| Role search | `research.stages.ST5-l1-roles.search.train` | `--condition dropout`, `head-slow`, `smooth`, or `head-slow-smooth` | ST5 configuration |
| Joint role–frame models | `research.stages.ST6-l1-l2-l3.train`, `research.stages.ST6-l1-l2-l3.balance.train` | Base run, then `--condition roles` and `frames` | Standalone L1/L2 initial states and reference artifacts |
| Final GPU inference | `research.stages.final.evaluate` | Seven systems in the final manifest | Selected checkpoints and fixed decoder |

Stage paths in the configuration column are relative to `research/stages/`. Invoke entry points with `uv run --locked python -m MODULE`; use `--help` to inspect accepted arguments.

The scripts preserve the original study's run IDs, resource budgets, and reference hashes. Prepare a separate checkout for a new training series: clear the corresponding recorded output directories, reset each series' `budget.json`, and rebuild predecessor artifacts before dependent runs. Refresh reference manifests for the new artifacts, including code and initialization states. Historical manifests include hashes of checkpoints and archived snapshots from the original workspace; those files are absent from this repository. Keep new manifests and results with the new series.

The published predictions support CPU verification of the reported results. Full GPU training and inference require regenerated checkpoints and caches; an end-to-end GPU rerun of this repository has not been verified.

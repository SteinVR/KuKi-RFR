<div align="center">

# KuKi-RFR

### Modeling Participant Roles, Frames, and Rhetoric in Russian-Language News

**Three annotation layers · Seven evaluated systems · One reproducible study**

[![Corpus](https://img.shields.io/badge/corpus-207_articles-243E64?style=flat-square)](#data)
[![Annotations](https://img.shields.io/badge/annotations-301-476B85?style=flat-square)](#data)
[![Tasks](https://img.shields.io/badge/tasks-Roles_%C2%B7_Frames_%C2%B7_Rhetoric-617B69?style=flat-square)](#tasks)
[![Python](https://img.shields.io/badge/Python-3.14-7A617E?style=flat-square)](#reproduce-the-results)

[Read the paper](paper/paper.pdf) · [Browse the manuscript](paper/paper.md) · [Inspect the results](research/stages/final/results.json) · [Verify the results](#reproduce-the-results)

</div>

---

KuKi-RFR studies how Russian-language news portrays participants, emphasizes aspects of events, and uses rhetorical techniques. It compares standalone models, joint training of two task pairs, and the transfer of rhetorical-family predictions to frame recognition.

The implementation uses **RuModernBERT-base with LoRA**, overlapping text windows, and a character-level rhetorical-span detector. Individual annotator responses are retained throughout preparation, training, and evaluation.

**Main finding.** Joint frame–rhetoric training improved automatic rhetorical annotation on held-out articles, with exact-F1 rising from **0.0561 to 0.0764**. The paired 95% interval for this increase is **[0.0084, 0.0323]**. Frame improvements and the additional benefit of transferring rhetorical probabilities remain uncertain. The selected joint role–frame model scored below the standalone models on both tasks.

## Tasks

| Layer | Question | Prediction | Evaluation input |
|:---|:---|:---|:---|
| **L1 · Roles** | How is a participant portrayed? | PROTAGONIST, ANTAGONIST, or INNOCENT | Article and annotator-selected mention boundaries |
| **L2 · Frames** | Which aspects of the event are emphasized? | Probabilities for 14 frames | Article |
| **L3 · Rhetoric** | Which passages contain rhetorical techniques? | Character spans labeled with six rhetorical families | Article; supplied spans for conditional evaluation |

The six rhetorical families are **Attack on Reputation**, **Justification**, **Simplification**, **Distraction**, **Call**, and **Manipulative Wording**. Spans may overlap and receive multiple labels.

L1 measures classification of supplied mentions. Conditional L3 measures classification of supplied spans. Automatic L3 measures the complete detection, classification, and selection pipeline. These conditions are reported separately.

## Results

Seven systems were fixed before final evaluation on **31 test articles with 46 individual annotations**.

| System | Role macro-F1 ↑ | Frame Brier ↓ | Conditional rhetoric F1 ↑ | Automatic rhetoric exact-F1 ↑ |
|:---|---:|---:|---:|---:|
| L1-only | **0.7067** | — | — | — |
| L2-only | — | 0.1362 | — | — |
| L3-only | — | — | 0.6934 | 0.0561 |
| Joint L2–L3, λ = 2 | — | 0.1328 | **0.7004** | **0.0764** |
| Transfer −families | — | 0.1343 | 0.7004¹ | 0.0764¹ |
| Transfer +families | — | **0.1326** | 0.7004¹ | 0.0764¹ |
| Joint L1–L2, 4:1 | 0.6935 | 0.1379 | — | — |

¹ Both transfer systems inherit the rhetorical predictions of Joint L2–L3. Bold indicates the best point estimate for each column. An em dash indicates an unevaluated output.

Each article has equal weight in evaluation; individual annotations are averaged within articles. Role macro-F1 uses 30 active test articles and conditional rhetoric F1 uses 29. Frame Brier and automatic rhetoric exact-F1 use all 31. Brier measures the squared error against individual frame annotations.

The frame–rhetoric model's Brier improvement over L2-only is **−0.0034**, with paired interval **[−0.0092, +0.0023]**. Adding rhetorical-family probabilities to the transfer module changes Brier by **−0.0017**, with paired interval **[−0.0084, +0.0044]**. Both intervals include zero.

The study uses one training seed and a small test set drawn from the same four outlets as training. Bootstrap intervals describe article-sampling variability for the trained models, with no adjustment for multiple comparisons. The paper discusses annotation disagreement, incomplete span coverage, and the role models' differing update schedules.

[Full metrics and intervals](research/stages/final/results.json) · [Per-article statistics](research/stages/final/article-statistics.json) · [Test predictions](research/stages/final/predictions/test/) · [Prespecified final evaluation](research/stages/final/protocol.md)

## Method

- **Text representation:** a pinned RuModernBERT-base revision, a frozen BF16 backbone, FP32 LoRA adapters, and overlapping 1,024-token windows.
- **Roles and frames:** mention and article representations feed task-specific output modules.
- **Rhetoric:** a character-level Global Pointer detector proposes boundaries; a separate family classifier and a fixed decoder produce span–family labels.
- **Joint training:** a shared encoder is optimized for the frame–rhetoric and role–frame pairs, with task-specific losses and outputs.
- **Prediction transfer:** a frozen joint frame–rhetoric model supplies representations and candidates to a trainable residual frame module. Matched variants include or zero the family probabilities.

<details>
<summary><strong>View the frozen rhetoric-to-frame transfer architecture</strong></summary>

![Frozen rhetoric-to-frame transfer](paper/figures/R1f-frozen-transfer.png)

Blue denotes trainable modules; gray denotes fixed components. The transfer comparison controls the text representations, candidates, architecture, and initialization.

</details>

## Data

The Russian-language KuKi sample contains **207 articles and 301 individual annotations** from three annotators. The sources are RIA Novosti, Nezavisimaya Gazeta, Kholod, and The Insider. The author's 120 annotated articles are included in this aggregate.

| Split | Articles | Individual annotations |
|:---|---:|---:|
| Train | 145 | 215 |
| Development | 31 | 40 |
| Test | 31 | 46 |
| **Total** | **207** | **301** |

Splitting is performed by article with seed 42. All annotations of an article remain together; two articles sharing a substantial passage belong to the same training group. Text is preserved in its original Unicode coordinate space, using half-open character offsets `[start, end)`.

Preparation merges duplicate marks, quarantines invalid layer–label combinations, and masks conflicting roles within the affected annotator's answer. Two empty frame answers are retained as all-zero targets; the export does not distinguish intentional empty answers from an unannotated layer. L4 annotations remain in the data for provenance and fall outside the modeled tasks.

| File | Contents |
|:---|:---|
| [Source corpus](research/external/science-work-ref/kuki_ru_aggregate.jsonl) | Original Russian aggregate with individual responses |
| [Prepared corpus](research/data/kuki-ru-v1.jsonl) | Exactly reconstructed study input |
| [Split manifest](research/stages/ST0-preparation/experiments/20260928-data-s42-02/split.json) | Document assignments, groups, label order, and SHA-256 hashes |
| [Transformation log](research/stages/ST0-preparation/experiments/20260928-data-s42-02/transformations.jsonl) | Record of annotation-policy decisions |
| [Preparation policy](research/stages/ST0-preparation/configs/data-v2.json) | Frozen preprocessing and grouping settings |

## Reproduce the results

Run commands from the repository root. [uv](https://docs.astral.sh/uv/) manages Python and dependencies.

### CPU verification

The published predictions suffice to independently reproduce the main results. This command needs **NumPy**, reads the corpus and saved predictions, and completes without model weights or a GPU:

```bash
uv run --no-project --python 3.14 --with numpy python -m research.kuki.verify_results
```

It checks source and prepared-data hashes, reconstructs preparation, validates the split and character offsets, recomputes all seven systems' primary metrics, and reproduces the 10,000-replicate bootstrap intervals and eight paired comparisons. Agreement is required within an absolute tolerance of `1e-12`.

An optional verification log can be written to a chosen output path:

```bash
uv run --no-project --python 3.14 --with numpy python -m research.kuki.verify_results \
  --output verification.json
```

### Research environment

The training environment is pinned by [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), and [.python-version](.python-version). The recorded hardware was an **RTX 4060 Laptop GPU with 8 GB VRAM**.

```bash
uv python install
uv sync --locked --managed-python
uv run --locked pytest -q
```

The training dependency group targets Linux/NVIDIA: Python 3.14.7, PyTorch 2.12.1 with CUDA 13.0, Transformers 5.17, and PEFT 0.21. The pinned FlashAttention Hub revision appears in [encoder.json](research/configs/encoder.json).

The repository contains the training and diagnostic code, frozen configurations, epoch logs, model-selection records, and final predictions. Model checkpoints and local feature caches were excluded from the source repository and are unavailable in this release. Training the pipeline requires rebuilding these dependencies in experiment order. Recorded run directories are retained as evidence of the original experiments; use fresh run IDs and resource ledgers for new fits.

### Experiment map

| Component | Implementation | Recorded outputs |
|:---|:---|:---|
| Preparation | [prepare.py](research/kuki/prepare.py), [data.py](research/kuki/data.py) | [ST0](research/stages/ST0-preparation/) |
| Standalone frames | [train_l2.py](research/kuki/train_l2.py) | [ST1](research/stages/ST1-l2-frames/) |
| Rhetoric detector and classifier | [modular/train.py](research/stages/ST2-l3-rhetoric/modular/train.py) | [ST2](research/stages/ST2-l3-rhetoric/) |
| Joint frames and rhetoric | [train.py](research/stages/ST3-l2-l3-joint/train.py), [train_search.py](research/stages/ST3-l2-l3-joint/search/train_search.py) | [ST3](research/stages/ST3-l2-l3-joint/) |
| Rhetoric-to-frame transfer | [frozen/fit.py](research/stages/ST4-l3-to-l2/frozen/fit.py) | [ST4](research/stages/ST4-l3-to-l2/) |
| Standalone roles | [search/train.py](research/stages/ST5-l1-roles/search/train.py) | [ST5](research/stages/ST5-l1-roles/) |
| Joint roles and frames | [balance/train.py](research/stages/ST6-l1-l2-l3/balance/train.py) | [ST6](research/stages/ST6-l1-l2-l3/) |
| Final evaluation | [evaluate.py](research/stages/final/evaluate.py), [verify_results.py](research/kuki/verify_results.py) | [Final results](research/stages/final/) |

## Repository

```text
KuKi-RFR/
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── paper/
│   ├── paper.pdf                 Article, references, and appendices
│   ├── paper.md                  Readable manuscript source
│   └── figures/                  Figures cited in the article
└── research/
    ├── kuki/                     Data, models, training, evaluation
    ├── tests/                    Scientific-code regression tests
    ├── configs/                  Pinned encoder and data references
    ├── data/                     Prepared Russian corpus
    ├── external/science-work-ref/ Original Russian aggregate
    └── stages/                   Experiment code and structured results
        ├── ST0-preparation/
        ├── ST1-l2-frames/
        ├── ST2-l3-rhetoric/
        ├── ST3-l2-l3-joint/
        ├── ST4-l3-to-l2/
        ├── ST5-l1-roles/
        ├── ST6-l1-l2-l3/
        └── final/                Predictions, metrics, intervals, hashes
```

Scientific code paths and historical run IDs are retained for traceability. The release contains the article body, references, and appendices. Title-page and academic-integrity declaration material is excluded. Research-planning drafts, annotation tooling, editorial reviews, duplicated code snapshots, environment-audit runs, and the interrupted regularization series are excluded.

## Paper and attribution

**Modeling Participant Roles, Frames, and Rhetoric in Russian-Language News**

[PDF](paper/paper.pdf) · [Markdown](paper/paper.md)

The article provides the study's interpretation, related work, experimental settings, and complete references. RuModernBERT, Global Pointer, LoRA, and the SemEval task schemes are credited there. The original corpus includes work by three annotators and news texts from the named publishers. This repository contains no blanket license for those source materials.

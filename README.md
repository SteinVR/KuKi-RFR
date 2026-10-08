<div align="center">

# KuKi-RFR

### Modeling Participant Roles, Frames, and Rhetoric in Russian-Language News

[![Corpus](https://img.shields.io/badge/corpus-207_articles-243E64?style=flat-square)](#data)
[![Annotations](https://img.shields.io/badge/annotations-301-476B85?style=flat-square)](#data)
[![Verification](https://github.com/SteinVR/KuKi-RFR/actions/workflows/verify.yml/badge.svg)](https://github.com/SteinVR/KuKi-RFR/actions/workflows/verify.yml)

[Paper · PDF](paper/paper.pdf) · [Paper · Markdown](paper/paper.md) · [Results](research/stages/final/results.json) · [Reproduce](#reproduce-the-results)

</div>

---

KuKi-RFR studies participant roles, event frames, and rhetorical techniques in Russian-language news. It compares standalone models, joint training of task pairs, and the transfer of rhetorical predictions to frame recognition. The repository contains the study's code, corpus, experiment results, and paper.

The models use **RuModernBERT-base with LoRA**, overlapping text windows, and a character-level rhetorical-span detector. Training and evaluation retain individual annotator responses.

## Tasks

| Layer | Prediction | Input |
|:---|:---|:---|
| **L1 · Roles** | PROTAGONIST, ANTAGONIST, or INNOCENT | Article and supplied participant mentions |
| **L2 · Frames** | Probabilities for 14 event frames | Article |
| **L3 · Rhetoric** | Character spans with six rhetorical-family labels | Article; supplied spans for conditional evaluation |

The rhetorical families are **Attack on Reputation**, **Justification**, **Simplification**, **Distraction**, **Call**, and **Manipulative Wording**. Spans may overlap and receive multiple labels. Conditional L3 evaluates classification at supplied boundaries; automatic L3 evaluates detection, classification, and selection together.

## Results

Seven systems were evaluated on **31 held-out articles with 46 individual annotations**.

| System | Role macro-F1 ↑ | Frame Brier ↓ | Conditional rhetoric F1 ↑ | Automatic rhetoric exact-F1 ↑ |
|:---|---:|---:|---:|---:|
| L1-only | **0.7067** | — | — | — |
| L2-only | — | 0.1362 | — | — |
| L3-only | — | — | 0.6934 | 0.0561 |
| Joint L2–L3, λ = 2 | — | 0.1328 | **0.7004** | **0.0764** |
| Transfer −families | — | 0.1343 | 0.7004¹ | 0.0764¹ |
| Transfer +families | — | **0.1326** | 0.7004¹ | 0.0764¹ |
| Joint L1–L2, 4:1 | 0.6935 | 0.1379 | — | — |

¹ Transfer systems inherit the Joint L2–L3 rhetorical predictions. Bold marks the best point estimate. Each article has equal evaluation weight; responses are averaged within articles. Role F1 uses 30 articles and conditional rhetoric F1 uses 29 with annotations for those tasks. Brier is the mean squared error against individual frame labels.

**Joint frame–rhetoric training improved automatic rhetoric exact-F1 from 0.0561 to 0.0764**, with a paired 95% interval of **[0.0084, 0.0323]** for the increase. The frame Brier change was −0.0034, with interval [−0.0092, +0.0023]. Adding family probabilities to the transfer model changed Brier by −0.0017, with interval [−0.0084, +0.0044]. Both frame intervals include zero. Joint role–frame training scored below the standalone models on both tasks.

The study uses one training seed and a test sample from the same four outlets as training. Bootstrap intervals describe variation across articles and have no multiple-comparison adjustment.

[Full metrics and intervals](research/stages/final/results.json) · [Article statistics](research/stages/final/article-statistics.json) · [Test predictions](research/stages/final/predictions/test/)

## Models

- **Encoder:** pinned RuModernBERT-base revision, a frozen BF16 backbone, FP32 LoRA adapters, and overlapping 1,024-token windows.
- **Roles and frames:** task-specific classifiers use mention and article representations.
- **Rhetoric:** a Global Pointer detector proposes character boundaries; a family classifier and fixed decoder produce labeled spans.
- **Joint training:** the frame–rhetoric and role–frame pairs share an encoder and use separate task losses.
- **Prediction transfer:** a frozen joint frame–rhetoric model supplies features to a residual frame module. Matched variants receive or zero the family probabilities.

<details>
<summary><strong>Rhetoric-to-frame transfer architecture</strong></summary>

![Rhetoric-to-frame transfer](paper/figures/R1f-frozen-transfer.png)

Blue modules are trainable; gray components are fixed.

</details>

## Data

The Russian-language KuKi sample contains **207 articles and 301 individual annotations** from three annotators. The articles come from RIA Novosti, Nezavisimaya Gazeta, Kholod, and The Insider.

| Split | Articles | Annotations |
|:---|---:|---:|
| Train | 145 | 215 |
| Development | 31 | 40 |
| Test | 31 | 46 |
| **Total** | **207** | **301** |

The split uses seed 42 and keeps each article's annotations together. Two articles sharing a substantial passage are grouped in training. Text retains its original Unicode coordinates; spans use half-open character offsets `[start, end)`.

[Source corpus](research/external/science-work-ref/kuki_ru_aggregate.jsonl) · [Prepared corpus](research/data/kuki-ru-v1.jsonl) · [Split and hashes](research/stages/ST0-preparation/experiments/20260928-data-s42-02/split.json) · [Preparation policy](research/stages/ST0-preparation/configs/data-v2.json) · [Transformation log](research/stages/ST0-preparation/experiments/20260928-data-s42-02/transformations.jsonl)

## Reproduce the results

Run from the repository root using [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --python 3.14 --with numpy==2.5.3 python -m research.kuki.verify_results
```

This CPU check reconstructs the prepared corpus, checks data hashes and span offsets, and recomputes the seven systems' metrics, 10,000-replicate bootstrap intervals, and eight paired comparisons. Required agreement is `1e-12`. Add `--output verification.json` to save the verification log.

Check repository files, links, and Python imports:

```bash
uv run --no-project --python 3.14 python -m research.kuki.check_repository
```

For training and the full test suite on Linux/NVIDIA:

```bash
uv python install
uv sync --locked --managed-python
uv run --locked pytest -q
```

The recorded environment used Python 3.14.7, PyTorch 2.12.1 with CUDA 13.0, Transformers 5.17, PEFT 0.21, and an RTX 4060 Laptop GPU with 8 GB VRAM. [uv.lock](uv.lock) pins dependencies; [encoder.json](research/configs/encoder.json) pins the model and attention kernel.

Model checkpoints and feature caches are absent. Repeat training in experiment order to generate them. [Reproduction guide](REPRODUCING.md) describes dependencies, entry points, and preparation for new runs.

## Code and experiment outputs

| Component | Code | Results |
|:---|:---|:---|
| Data preparation | [prepare.py](research/kuki/prepare.py) | [ST0](research/stages/ST0-preparation/) |
| Standalone frames | [train_l2.py](research/kuki/train_l2.py) | [ST1](research/stages/ST1-l2-frames/) |
| Rhetoric detector and classifier | [modular/train.py](research/stages/ST2-l3-rhetoric/modular/train.py) | [ST2](research/stages/ST2-l3-rhetoric/) |
| Joint frames and rhetoric | [train.py](research/stages/ST3-l2-l3-joint/train.py), [train_search.py](research/stages/ST3-l2-l3-joint/search/train_search.py) | [ST3](research/stages/ST3-l2-l3-joint/) |
| Rhetoric-to-frame transfer | [frozen/fit.py](research/stages/ST4-l3-to-l2/frozen/fit.py) | [ST4](research/stages/ST4-l3-to-l2/) |
| Standalone roles | [search/train.py](research/stages/ST5-l1-roles/search/train.py) | [ST5](research/stages/ST5-l1-roles/) |
| Joint roles and frames | [balance/train.py](research/stages/ST6-l1-l2-l3/balance/train.py) | [ST6](research/stages/ST6-l1-l2-l3/) |
| Final evaluation | [evaluate.py](research/stages/final/evaluate.py), [verify_results.py](research/kuki/verify_results.py) | [Final](research/stages/final/) |

The [paper](paper/paper.pdf) contains the experimental settings, interpretation, references, and appendices. It credits RuModernBERT, Global Pointer, LoRA, and the SemEval annotation schemes. Source news texts and corpus annotations retain their original authorship.

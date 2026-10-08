"""Deterministically selected qualitative examples; no adjudication of human labels."""

import json
from pathlib import Path

import numpy as np

from research.kuki.data import FAMILIES, write_json
from research.kuki.experiment import load_development

ROOT = Path(__file__).parent


def rows(p):
    return [json.loads(x) for x in p.read_text().split("\n## Иллюстрации")[0].splitlines()]


def main():
    chain = json.loads((ROOT / "E3-detector/chain.json").read_text())
    docs, _ = load_development(json.loads(Path("research/configs/data.json").read_text()))
    lookup = {d["doc_id"]: d for d in docs["dev"]}
    run = Path(chain["selected_runs"]["classifier"])
    metric = json.loads((run / "metrics.json").read_text())
    examples = []
    for r in rows(run / "dev-predictions.jsonl"):
        prob = np.array(r["probabilities"])
        actual = np.array(r["target"], dtype=bool)
        predicted = prob >= metric["dev"]["threshold"]
        predicted[prob.argmax()] = True
        false = predicted & ~actual
        if not false.any():
            continue
        d = lookup[r["doc_id"]]
        examples.append(
            dict(
                doc_id=r["doc_id"],
                annotator_id=r["annotator_id"],
                start=r["start"],
                end=r["end"],
                text=d["text"][r["start"] : r["end"]],
                human_labels=[f for f, b in zip(FAMILIES, actual, strict=True) if b],
                predicted_labels=[f for f, b in zip(FAMILIES, predicted, strict=True) if b],
                largest_false_label_probability=float(prob[false].max()),
            )
        )
    examples.sort(
        key=lambda x: (
            -x["largest_false_label_probability"],
            x["doc_id"],
            x["start"],
            x["annotator_id"],
        )
    )
    write_json(
        ROOT / "E1-classification/error-examples.json",
        dict(
            selection="top5 false-label probabilities; may repeat span across annotators; descriptive, not adjudicated",
            examples=examples[:5],
        ),
    )
    for directory, records in [("E1-classification", examples[:5])]:
        p = ROOT / directory / "report.md"
        text = (
            p.read_text().split("\n## Иллюстрации")[0]
            + "\n## Иллюстрации ошибок\n\nПервые пять ошибок по максимальной вероятности неверно выданного семейства; tie-break по статье/позиции/разметчику. Это воспроизводимый срез уверенных ошибок, не оценка их распространённости и не пересмотр разметки. Полные тексты и границы — [error-examples.json](error-examples.json).\n\n"
        )
        for r in records:
            sample = r["text"].replace("\n", " ")[:180]
            text += f"* `{r['doc_id']}`, ответ `{r['annotator_id']}`, [{r['start']}:{r['end']}]: «{sample}». В этом ответе: {', '.join(r['human_labels'])}; модель: {', '.join(r['predicted_labels'])}.\n"
        p.write_text(text)
    proposals = rows(ROOT / "E3-detector/chain-dev-scores.jsonl")
    misses = []
    for p in proposals:
        d = lookup[p["doc_id"]]
        pairs = {(s, e) for s, e, _ in p["pairs"]}
        for a in d["annotations"]:
            for s in a["l3_persuasion"]:
                if (s["start"], s["end"]) in pairs:
                    continue
                best = max(
                    max(0, min(e, s["end"]) - max(b, s["start"]))
                    / (max(e, s["end"]) - min(b, s["start"]))
                    for b, e in pairs
                )
                misses.append(
                    dict(
                        s,
                        doc_id=d["doc_id"],
                        annotator_id=a["annotator_id"],
                        text=d["text"][s["start"] : s["end"]],
                        best_candidate_iou=best,
                    )
                )
    misses.sort(key=lambda r: (r["best_candidate_iou"], r["doc_id"], r["start"], r["label"]))
    write_json(
        ROOT / "E3-detector/error-examples.json",
        dict(
            selection="top5 exact proposal misses with smallest maximum candidate IoU; deterministic ties",
            examples=misses[:5],
        ),
    )
    p = ROOT / "E3-detector/report.md"
    text = (
        p.read_text().split("\n## Иллюстрации")[0]
        + "\n## Иллюстрации пропусков детектора\n\nПять отсутствующих точных фрагментов с наименьшим лучшим IoU среди top500; это специально выбранные тяжёлые пропуски, не случайная выборка. [Полный машиночитаемый срез](error-examples.json).\n\n"
    )
    for r in misses[:5]:
        sample = r["text"].replace("\n", " ")[:180]
        text += f"* `{r['doc_id']}`, [{r['start']}:{r['end']}], {r['label']}: «{sample}». Максимальный IoU кандидата: {r['best_candidate_iou']:.3f}.\n"
    p.write_text(text)


if __name__ == "__main__":
    main()

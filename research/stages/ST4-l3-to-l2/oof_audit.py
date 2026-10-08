"""Descriptive candidate coverage; no tuning or neural execution."""

import importlib

from research.kuki.data import write_json

M = importlib.import_module("research.stages.ST4-l3-to-l2.method")
MOD = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.train")


def main():
    cfg = M.read(M.ROOT / "configs/continue.json")
    docs, _, _ = M.inputs(cfg)
    manifest = M.read(M.ROOT / "oof/manifest.json")
    results = {}
    for name, part, pred in [
        ("oof_train", docs["train"], M.jsonl(M.ROOT / "oof/train-predictions.jsonl")),
        (
            "historical_in_sample_train",
            docs["train"],
            M.jsonl(M.OLD.DETECTOR / "train-predictions.jsonl"),
        ),
        ("historical_dev", docs["dev"], M.jsonl(M.OLD.DETECTOR / "dev-predictions.jsonl")),
    ]:
        m = MOD.detector_score(part, pred, 0)
        results[name] = dict(
            recall=m["recall"],
            articles=len(part),
            nonempty_articles=sum(r["recall"]["500"] is not None for r in m["articles"]),
        )
    for fold in manifest["folds"]:
        i = fold["fold"]
        out = M.ROOT / "oof" / f"20260929-oof{i}-s42-01"
        part = [d for d in docs["train"] if d["doc_id"] in fold["prediction_ids"]]
        rr = M.jsonl(out / "predictions.jsonl")
        m = MOD.detector_score(part, rr, 0)
        results[f"fold{i}"] = dict(recall=m["recall"], articles=len(part))
        write_json(out / "coverage.json", results[f"fold{i}"])
        write_json(
            out / "config-interpretation.json",
            dict(
                actual_selection="fixed final epoch13, no dev or heldout tuning",
                unused_inherited_fields=[
                    "selection",
                    "threshold_grid",
                    "early_stopping_patience_epochs",
                    "early_stopping_min_delta",
                ],
                note="Original copied E3 descriptive fields are unused in oof_train; metadata and snapshots preserve exact executed configuration.",
            ),
        )
        (out / "report.md").write_text(f"""# ST4: OOF-детектор {i}

Выполнено13эпох, {fold["updates"]} updates, {fold["articles_seen"]} предъявлений.
Чистая закреплённая база, новые LoRA и головы; ST3/E3-веса не загружались.
Источник обучен на {len(fold["train_ids"])} статьях; прогнозы получены для {len(part)} исключённых статей. Группы не пересекаются, по исключённым статьям и dev источник не выбирался и не настраивался. Список исключений — metadata.json и [общий manifest](../manifest.json).

Выбран последний checkpoint эпохи13. Поля selection/threshold_grid/patience унаследованы конфигурацией из E3, но не используются этим trainer: фактическое правило разъяснено в config-interpretation.json; исходная конфигурация сохранена без переписывания.

Охват исключённых непустых ответов (сначала ответы, затем статьи): {m["recall"]}.
Это диагностика, она не меняет исходный выбор. Не приписывать различие с full-train исключительно переобучению: меняются объём обучения и расписание.

Веса и RNG/optimizer/scheduler/progress — checkpoints/last.pt; точный снимок исходников с исходными путями — code-tree/. Воспроизведение прогнозов на двух исключённых статьях проверяется общей GPU-верификацией.
""")
    write_json(M.ROOT / "oof/coverage.json", results)
    (M.ROOT / "oof/report.md").write_text(
        "# Генерация кандидатов без обучения на целевой группе\n\n"
        + "\n".join(
            f"- {k}: {v['articles']} статей, recall@500={v['recall']['500']:.4f}."
            for k, v in results.items()
        )
        + "\n\nКаждая train-статья покрыта ровно одним исключающим её группу источником. OOF — подготовка сообщений, а не новый независимый тест. Исторический in-sample результат приведён для сопоставления; меньшие train-fold и другое расписание также влияют на охват.\n"
    )
    print(results)


if __name__ == "__main__":
    main()

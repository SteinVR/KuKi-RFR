"""Finalize navigation, per-run reports and provenance after verified experiments."""

import hashlib
import json
import shutil
from pathlib import Path

from research.kuki.data import write_json

ROOT = Path(__file__).parent
STAGE = ROOT.parent


def read(p):
    return json.loads(p.read_text())


def main():
    verification = read(ROOT / "verification.json")
    assert verification["passed"]
    chain = read(ROOT / "E3-detector/chain.json")
    extended = read(ROOT / "E3-detector/calibration-extension.json")
    ledger = read(ROOT / "budget.json")
    classifier = Path(chain["selected_runs"]["classifier"])
    detector = Path(chain["selected_runs"]["detector"])
    cm = read(classifier / "metrics.json")
    dm = read(detector / "metrics.json")
    results = []
    for kind, folder, prefix in [
        ("classifier", "E1-classification", "e1"),
        ("detector", "E3-detector", "e3"),
    ]:
        for variant in ["base", "low-lr"]:
            run = STAGE / "experiments" / f"20260928-{kind}-{variant}-s42-01"
            m = read(run / "metrics.json")
            shutil.copy2(Path(f"/tmp/kuki-{prefix}-{variant}.log"), run / "logs/stdout.log")
            (run / "report.md").write_text(f"""# {run.name}

Самостоятельный запуск в составе [{folder}](../../modular/{folder}/report.md).

Статус: {m["status"]}; выбрана эпоха {m["selected_epoch"]}; выполнено {m["updates"]} обновлений и {m["article_presentations"]} предъявлений статей. Время процесса: {m["elapsed_seconds"]:.2f} с. Frozen base не изменена; test не использован.

Полные [метрики](metrics.json), [кривые](history.json), [конфигурация](config.json), [происхождение](metadata.json), [журнал](logs/stdout.log). Интерпретация, контроли и ограничения изложены в общем отчёте эксперимента по ссылке выше. Понятие «completed» означает завершение запуска в его пределах, а не подтверждение гипотезы.
""")
            results.append(
                dict(
                    run_id=run.name,
                    **{
                        k: m[k]
                        for k in [
                            "updates",
                            "article_presentations",
                            "elapsed_seconds",
                            "status",
                            "selected_epoch",
                        ]
                    },
                )
            )
    summary = dict(
        status="completed_development_experiments",
        selected_classifier=str(classifier),
        selected_detector=str(detector),
        conditional_classifier_dev_f1=cm["dev"]["article_mean_f1"],
        conditional_classifier_dev_macro_f1=cm["dev"]["macro_family_f1"],
        detector_dev_recall=dm["dev"]["recall"],
        chain_dev=extended["dev"],
        primary_chain_dev=chain["dev"],
        decoder_selected_post_hoc=True,
        gpu_seconds=ledger["seconds"],
        test_used=False,
        runs=results,
    )
    write_json(ROOT / "summary.json", summary)
    write_json(
        ROOT / "selection.json",
        dict(
            classifier=str(classifier),
            detector=str(detector),
            decoder="logit_family_plus_alpha_detector",
            alpha=extended["selection"]["alpha"],
            threshold=extended["selection"]["threshold"],
            selection_source="post_hoc_dev_calibration_extension",
            test_used=False,
        ),
    )
    (ROOT / "README.md").write_text(f"""# Независимые генерация фрагментов и классификация L3

Три отдельных эксперимента завершены на train/dev. [Протокол с уточнениями](protocol.md), [самопроверка](verification.json), [бюджет](budget.json), [сводка](summary.json).

| Эксперимент | Основной результат dev | Подробный отчёт |
|---|---|---|
| E1 — классификация предоставленных фрагментов | F1 {cm["dev"]["article_mean_f1"]:.4f}, macro-F1 {cm["dev"]["macro_family_f1"]:.4f} | [E1](E1-classification/report.md) |
| E2 — необучаемые генераторы | полный токенный набор: 749700 кандидатов в медиане, точный охват 93.15%; многомасштабный: 10081 и 71.46% | [E2](E2-proposals/report.md) |
| E3 — GlobalPointer и замороженная цепочка | top500 охват {dm["dev"]["recall"]["500"]:.4f}; typed exact-F1 цепочки {extended["dev"]["exact_f1"]:.4f} (дополнительная dev-калибровка) | [E3](E3-detector/report.md) |

E1 — условная классификация, E2 — ограничения кандидатов, E3 — автоматическая локализация и итоговая выдача. Их числа не являются одной взаимозаменяемой метрикой. Два E1 fit, два E3 fit, профиль и независимая загрузка/композиция заняли {ledger["seconds"]:.2f} с ({ledger["seconds"] / 60:.2f} мин) из предела 40 минут. CPU-аудиты и подготовка отчётов сюда не входят.

Исторические ST2-прогоны сохранены; ST3–ST6 не запускались. Seed42, неизменный split145/31/31; test закрыт. Это результаты разработки, не независимая итоговая оценка.

Воспроизведение отчётов по сохранённым запускам:

```bash
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.report_models
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.report_proposals
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.report_detector
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.error_examples
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.verify
```

`chain` — отдельная GPU-проверка, повторно расходующая ledger; для обычного пересчёта метрик достаточно CPU `verify`. Независимый повтор обучения требует нового Run ID и отдельного ledger; существующие архивы не удалять. Точное возобновление modular-fit посреди эпохи не реализовано. Исторический trainer ST2 с его resume не изменён.
""")
    p = STAGE / "README.md"
    s = p.read_text()
    marker = "\n## Модульные эксперименты E1–E3\n"
    s = (
        s.split(marker)[0]
        + marker
        + "\nЗавершена отдельная [серия с независимым генератором и классификатором](modular/README.md): три подробных отчёта, два классификатора, два GlobalPointer-детектора и замороженная цепочка. Исторические результаты ниже относятся к исходному ST2; новая серия не перезаписывает их. Test закрыт, ST3 не запускался.\n"
    )
    p.write_text(s)
    p = STAGE.parent / "experiments.md"
    s = p.read_text()
    for r in results:
        rows = s.splitlines()
        for i, line in enumerate(rows):
            if line.startswith("| " + r["run_id"] + " |"):
                rows[i] = (
                    line.rsplit("|", 2)[0]
                    + f"| completed; {r['status']}, best {r['selected_epoch']}, {r['updates']} updates, {r['elapsed_seconds']:.2f}с |"
                )
        s = "\n".join(rows) + "\n"
    s = s.replace(
        "Ближайшая работа — ST3 в пределах [экспериментального плана](../docs/experimental-plan.md).",
        "Дополнительно завершена [модульная серия ST2 E1–E3](ST2-l3-rhetoric/modular/README.md); ST3–ST6 ещё не запускались.",
    )
    s = s.replace(
        "По поручению пользователя выполняются три отдельных эксперимента; ST3 не начинается.",
        "По поручению пользователя выполнены три отдельных эксперимента; ST3 не запускался.",
    )
    s = s.replace("E2 выполняется на CPU.", "E2 выполнен на CPU.")
    s = s.replace(
        "факт и остановки будут дополнены после запусков.",
        "факт и остановки приведены в таблице. Лимиты проверяются между блоками обучения; измеренное время также включает заключительную оценку и сохранение.",
    )
    marker = "\nИтог модульной серии:"
    s = (
        s.split(marker)[0]
        + f"\nИтог модульной серии: {ledger['seconds']:.2f}с / {ledger['seconds'] / 60:.2f}мин GPU-процессов, включая профиль и цепочку/проверку. Все четыре fit завершены в объявленных пределах; один E1 остановлен по времени в неполной последней эпохе. [Три отчёта и ограничения](ST2-l3-rhetoric/modular/README.md).\n"
    )
    p.write_text(s)
    files = list(ROOT.rglob("*"))
    for r in results:
        files += list((STAGE / "experiments" / r["run_id"]).rglob("*"))
    files += [
        Path("research/kuki/global_pointer.py"),
        Path("research/kuki/span_classifier.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    manifest = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file() and "__pycache__" not in p.parts and p.name != "manifest.json"
    }
    write_json(ROOT / "manifest.json", dict(files=manifest, test_used=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

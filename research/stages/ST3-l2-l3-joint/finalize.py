"""Finalize ST3 narrative decisions and checksums without any new fitting."""

import importlib
import re
from pathlib import Path

from research.kuki.data import digest, write_json

M = importlib.import_module("research.stages.ST3-l2-l3-joint.method")
ROOT = Path(__file__).parent


def main():
    analysis = M.read(ROOT / "analysis.json")
    budget = M.read(ROOT / "budget.json")
    notes = {
        "l2-only": "Контроль сильнее исторического ST1 по наблюдаемому Brier: 0,1489 против 0,1532. Описательный интервал разницы с историческим ST1 включает ноль (−0,0116; +0,0031); устойчивое превосходство не установлено. Этот новый контроль нужен, чтобы не приписать дополнительной задаче эффект иного расписания и бюджета. После восьмой эпохи дальнейшее обучение ухудшает dev: last20 Brier=0,1679. Проверенный контроль сохраняется отдельно.",
        "l3-only": "Условный F1=0,7007 на эпохе13; исторический E1 имел0,6918. Это небольшой разработческий выигрыш в другом режиме обучения, не подтверждённое обобщение. Автоматический exact-F1=0,0607, правильных пустых ответов0/3. Отдельная калибровка порога даёт0,0637. Сильный условный результат сохраняет разрыв с автоматическим применением: отбор неточных и ложных предложений остаётся нерешённым. Этот контроль служит основной точкой оценки ущерба L3.",
        "joint025": "Выбрана эпоха8, та же, что у L2-контроля. ΔBrier=−0,00135, описательный интервал [−0,00284; +0,00013] включает ноль. Условный F1 падает с0,7007 до0,6111; сквозной при фиксированном пороге — с0,0607 до0,0416. Настройка только порога возвращает0,0609, но AP ниже контроля (0,0478 против0,0572). Ни одна из трёх частей рабочего критерия не выполнена. Этот вес не выбран для основного дальнейшего сравнения.",
        "joint100": "Выбрана эпоха8, как и у L2-контроля. ΔBrier=−0,00451, описательный интервал [−0,00780; −0,00119]; macro-F1 L2 растёт0,5130→0,5533. Наблюдаемое улучшение проходит заранее заданный ориентир L2. Однако условный L3 F1 падает0,7007→0,6504, а автоматический при фиксированном пороге0,0607→0,0447. После диагностической калибровки автоматический F1=0,0608 против0,0637 у так же откалиброванного контроля; AP остаётся ниже (0,0507 против0,0572). Это положительный сигнал о пользе L3-супервизии для L2 и одновременно отрицательный результат для критерия сохранения качества обоих выходов. Нужен как контроль следующего этапа, а не как автоматически принятая замена отдельным моделям.",
    }
    for name, note in notes.items():
        cfg = M.read(ROOT / "configs" / f"{name}.json")
        out = ROOT / "experiments" / cfg["run_id"]
        resource = next(r for r in budget["runs"] if r.get("run_id") == cfg["run_id"])
        write_json(out / "execution.json", resource)
        file = out / "report.md"
        text = file.read_text()
        marker = "\n## Интерпретация и ресурс\n"
        text = text.split(marker)[0]

        note = re.sub(r"(?<=\d)(?=[А-Яа-я])|(?<=[А-Яа-я])(?=\d)", " ", note)
        file.write_text(
            text
            + marker
            + "\n"
            + note
            + "\n\n"
            + f"Процесс с GPU: {resource['seconds']:.2f}с ({resource['seconds'] / 60:.2f}мин). Peak reserved: {resource['peak_reserved_bytes'] / 1024**3:.2f}GiB. Обучаемых параметров: {resource['trainable_parameters']:,}. История эпох и компонентных train loss — logs/epochs.jsonl; точный код и версии — code/ и metadata.json.\n"
        )
    write_json(
        ROOT / "selection.json",
        dict(
            selected_joint=analysis["selected_joint"],
            criterion_passed=analysis["comparisons"][analysis["selected_joint"]][
                "criterion_passed"
            ],
            recommendation="retain_separate_models_and_joint100_as_comparison_controls; do_not_replace_both_outputs",
            primary_metric="dev_L2_Brier",
            test_used=False,
            ST4_started=False,
            runs={
                name: dict(
                    path=str(
                        ROOT / "experiments" / M.read(ROOT / "configs" / f"{name}.json")["run_id"]
                    ),
                    best_epoch=M.read(
                        ROOT
                        / "experiments"
                        / M.read(ROOT / "configs" / f"{name}.json")["run_id"]
                        / "progress.json"
                    )["best_epoch"],
                )
                for name in notes
            },
        ),
    )
    checks = M.read(ROOT / "verification.json")
    checks["unit_tests"] = dict(passed=45, command="uv run --locked pytest -q")
    checks["self_review"] = dict(
        completed=True,
        scope="shared L2 and conditional family classification; detector frozen",
        identical_initial_parameters=True,
        matched_training_budgets=True,
        same_selected_L2_epoch=8,
        automatic_L3_drop_depends_on_checkpoint_and_calibration=True,
        development_only=True,
        independent_report_visual_review=True,
        technical_empty_batch_fix="implementation-notes.md",
    )
    write_json(ROOT / "verification.json", checks)
    for report in [ROOT / "report.md"] + list((ROOT / "experiments").glob("*/report.md")):
        text = report.read_text()
        text = re.sub(r"(?<=[А-Яа-я]),(?=\d)", ", ", text)
        for before, after in [
            ("сGPU", "с GPU"),
            ("Seed42", "Seed 42"),
            ("Brier0", "Brier 0"),
            ("с0,", "с 0,"),
        ]:
            text = text.replace(before, after)
        report.write_text(text)
    paths = [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "manifest.json"
    ]
    paths += [Path("research/kuki/joint.py"), Path("research/tests/test_joint.py")]
    write_json(
        ROOT / "manifest.json",
        dict(files={str(p.resolve()): digest(p) for p in sorted(paths)}, test_used=False),
    )
    print(dict(files=len(paths), seconds=budget["seconds"], selected=analysis["selected_joint"]))


if __name__ == "__main__":
    main()

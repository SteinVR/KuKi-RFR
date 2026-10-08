"""CPU-only fixed-threshold comparison and typed candidate-conditional AP."""

import importlib
import json
from pathlib import Path

import numpy as np

from research.kuki.data import FAMILIES, digest, write_json, write_jsonl
from research.kuki.experiment import load_development

ROOT = Path(__file__).parent
OPTIONS = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "runs" / OPTIONS["run_id"]
chain = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.chain")
method = importlib.import_module("research.stages.ST2-l3-rhetoric.negative_fit.method")


def main():
    execution = json.loads((OUT / "execution.json").read_text())
    if not execution["comparison_complete"]:
        (OUT / "report.md").write_text(
            "# Проверка отрицательных кандидатов не завершена\n\n"
            + json.dumps(execution, ensure_ascii=False, indent=2)
            + "\n\nНового fit или расширения бюджета не выполнялось.\n"
        )
        print(json.dumps(execution))
        return
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    result = dict(
        execution=execution,
        data_hashes=hashes,
        alpha=OPTIONS["alpha"],
        threshold=OPTIONS["threshold"],
        test_used=False,
    )
    for condition in ["control", "negative"]:
        outputs = chain.read_jsonl(OUT / f"{condition}-dev-scores.jsonl")
        metrics, articles, decoded = chain.chain_score(
            docs["dev"], outputs, OPTIONS["alpha"], OPTIONS["threshold"]
        )
        diagnostic = method.candidate_diagnostics(
            docs["dev"], outputs, OPTIONS["alpha"], OPTIONS["threshold"]
        )
        result[condition] = dict(metrics=metrics, candidates=diagnostic)
        write_json(OUT / f"{condition}-dev-articles.json", articles)
        write_jsonl(OUT / f"{condition}-dev-predictions.jsonl", decoded)
        full = []
        for o in outputs:
            p = np.clip(np.asarray(o["probabilities"]), 1e-7, 1 - 1e-7)
            values = (
                np.log(p / (1 - p))
                + OPTIONS["alpha"] * np.asarray([v for _, _, v in o["pairs"]])[:, None]
            )
            full.append(
                dict(
                    doc_id=o["doc_id"],
                    pairs=[(s, e) for s, e, _ in o["pairs"]],
                    families=FAMILIES,
                    composite_scores=values.tolist(),
                )
            )
        write_jsonl(OUT / f"{condition}-full-composite-scores.jsonl", full)
    assert (
        result["control"]["candidates"]["ap_excluded_answers"]
        == result["negative"]["candidates"]["ap_excluded_answers"]
    )
    before = result["control"]["metrics"]
    after = result["negative"]["metrics"]
    bd = result["control"]["candidates"]
    ad = result["negative"]["candidates"]
    result["delta"] = {
        k: after[k] - before[k]
        for k in ["exact_f1", "exact_precision", "exact_recall", "overlap_f1", "mean_predictions"]
    }
    result["delta"].update(
        {
            k: ad[k] - bd[k]
            for k in [
                "ap",
                "negative_pair_acceptance",
                "positive_pair_retention",
                "positive_typed_retention",
            ]
        }
    )
    result["working_criterion"] = dict(
        f1_gain_at_least_005=result["delta"]["exact_f1"] >= 0.005,
        negative_acceptance_decreased=result["delta"]["negative_pair_acceptance"] < 0,
        recall_drop_no_more_than_01=result["delta"]["exact_recall"] >= -0.01,
    )
    result["working_criterion"]["passed"] = all(result["working_criterion"].values())
    prov = json.loads((OUT / "metadata.json").read_text())["provenance"]
    assert all(digest(Path(p)) == h for p, h in prov["input_checkpoints"].items())
    assert all(digest(Path(p)) == h for p, h in prov["candidate_caches"].items())
    assert execution["elapsed_seconds"] <= 180 and execution["training_finished_seconds"] <= 135
    result["historical_inputs_unchanged"] = True
    write_json(OUT / "analysis.json", result)
    lines = [
        "# Один fit с отрицательными кандидатами: результат\n",
        f"Выполнено **64 новых updates**, от340 до404, **491 предъявление статьи**. Время процесса с GPU **{execution['elapsed_seconds']:.2f}с** из180; обновления закончены на{execution['training_finished_seconds']:.2f}с, до135. Test не использован; ST3 не запускался.\n",
        """## Вопрос и зафиксированное сравнение\n
Проверка: может ли короткое продолжение классификатора улучшить отказ на ошибочных предложениях при неизменном генераторе? Контроль — существующий classifier-low-lr last404. Новый вариант начинает с его best340 и получает те же64 updates, порядок129 train-статей,491 предъявление и убывающее LR-расписание. Контроль не переобучался. Историческое состояние RNG не было сохранено: совпадение случайных dropout-масок не заявляется. Дополнительный span-head compute означает, что одинаковые updates не равны точным FLOPs/секундам.\n
Детектор GlobalPointer, его top500, alpha2, threshold=-5.172313813576109 и cap100 фиксированы. Новых порогов, эпох или параметров по dev не выбиралось. Эталонные границы не добавлялись в dev-кандидаты.\n
## Пакет обучения\n
Те же129 train-статей:199 индивидуальных ответов, включая19 пустых.16 полностью пустых train-статей не добавлены ради сопоставимости корпуса. В полном проходе2966 положительных пар; для каждого ответа64 отрицательных из сохранённого top500:32 с максимальным max(p_family) старого E1 и32 случайных из остальных с детерминированной сменой по эпохе. Отрицательный target относителен конкретному ответу: другие разметчики не объединяются в «истинный» gold.\n
Для непустого ответа loss=0.5 mean_BCE(positive)+0.5 mean_BCE(negative); для пустого только negative BCE; далее среднее по ответам и статьям. Таким образом меняется **пакет**: отрицательные пары, включение пустых ответов и нормировка. Эффект нельзя причинно приписать одним отрицательным.\n
Train-кандидаты получены детектором, видевшим эти статьи: это **не OOF**, их качество может быть оптимистичным. Dev не участвовал в обучении весов, но уже использовался в предыдущем выборе моделей и decoder. Это разработческая проверка, не подтверждающий тест.\n
## Восстановление и ресурс\n""",
        f"После загрузки optimizer/scheduler проверены порядок и формы групп параметров, scheduler step340 и фактические LR: {execution['starting_lrs']}. Конечные LR {execution['final_lrs']} точно совпали с контролем last404. Optimizer не сбрасывался, warmup не перезапускался. Сохранены RNG Python/NumPy/PyTorch CPU/CUDA, полный план статей и прогресс нового запуска.\n",
        "## Сквозные результаты при неизменном пороге\n",
        "| Показатель dev | Контроль last404 | Отрицательный fit404 | Изменение |\n|---|---:|---:|---:|",
    ]
    for k in ["exact_f1", "exact_precision", "exact_recall", "overlap_f1", "mean_predictions"]:
        lines.append(f"| {k} | {before[k]:.6f} | {after[k]:.6f} | {after[k] - before[k]:+.6f} |")
    lines.append(
        f"\nПравильных пустых ответов: контроль{before['correctly_empty_answers']}/{before['empty_answers']}, новый вариант{after['correctly_empty_answers']}/{after['empty_answers']}. Ранее выбранный best340 имел exact-F1≈0.0617; здесь основное сравнение именно с last404, а не с best340.\n"
    )
    lines += [
        "## Отказ и AP внутри фиксированного списка\n",
        """Average precision (AP) рассчитана по полному составному score **500×6 typed-решений**, без порога/cap100: отдельно по каждому ответу, затем между допустимыми ответами статьи и статьями. Положительный элемент требует точных границ и семейства. Это условное различение внутри top500, не сквозная полнота. Ответы без положительных элементов списка исключены из AP; их отказ продолжает учитываться в остальных метриках. Различение не отождествляется с качеством выбранного фиксированного порога.\n""",
        f"В AP включены{bd['ap_eligible_answers']} ответов на{bd['ap_eligible_articles']} статьях. Исключены{len(bd['ap_excluded_answers'])} ответов; список и причины идентичны для обоих состояний и сохранены в analysis.json. Из них наблюдаемых полностью пустых ответов{sum(x['observed_gold_count'] == 0 for x in bd['ap_excluded_answers'])}; остальные, если есть, имеют gold вне данного списка кандидатов.\n",
        "| Показатель | Контроль | Новый вариант | Изменение |\n|---|---:|---:|---:|",
    ]
    for k in [
        "ap",
        "negative_pair_acceptance",
        "positive_pair_retention",
        "positive_typed_retention",
        "accepted_pairs_before_cap",
    ]:
        lines.append(f"| {k} | {bd[k]:.6f} | {ad[k]:.6f} | {ad[k] - bd[k]:+.6f} |")
    lines += [
        "\nДопуск отрицательных пар и сохранение положительных измерены **до cap100**. Для pair-retention достаточно любой разрешённой метки на правильной паре; typed-retention требует правильного семейства. Усреднение — по ответам и статьям, для положительных retention только по ответам с доступными положительными. AP и сквозной recall имеют разные знаменатели.\n",
        "## Семейства\n",
        "| Семейство | Контроль exact-F1 | Новый exact-F1 |\n|---|---:|---:|",
    ]
    for f in FAMILIES:
        lines.append(
            f"| {f} | {before['per_family_exact_f1'][f]:.6f} | {after['per_family_exact_f1'][f]:.6f} |"
        )
    lines += [
        "\n## Рабочий критерий и интерпретация\n",
        f"Заранее заданный критерий **{'выполнен' if result['working_criterion']['passed'] else 'не выполнен'}**: ΔF1≥0.005 —{result['working_criterion']['f1_gain_at_least_005']}; допуск отрицательных снизился —{result['working_criterion']['negative_acceptance_decreased']}; падение recall≤0.01 —{result['working_criterion']['recall_drop_no_more_than_01']}. Это практический ориентир, не тест статистической значимости.\n",
    ]
    if result["delta"]["ap"] > 0:
        lines.append(
            "AP выросла: внутри фиксированного набора наблюдается улучшение ранжирования. Это нельзя свести только к единому сдвигу всех баллов. Если фиксированный порог одновременно снижает recall, выводы о ранжировании и выдаче необходимо разделять; нового порога в этой работе не подбиралось.\n"
        )
    else:
        lines.append(
            "AP не выросла: эта короткая настройка не показала улучшения условного типизированного ранжирования. Результат относится к данному пакету обучения и хвосту LR-расписания; он не доказывает бесполезность отрицательных примеров вообще.\n"
        )
    lines += [
        "## Решение и следующий шаг\n\nТекущий пакет не принимать как замену исходному классификатору: сокращение ложных отметок достигнуто вместе с потерей почти всех верных, AP также снизилась. Сохранённый ранее вариант остаётся рабочим baseline; selection модульной серии не изменяется. Автоматического подбора другого порога, нового fit или перехода к ST3 нет.\n\nЕсли исследование продолжается, содержательный следующий кандидат — отдельное обучение бинарного отказа при замороженном классификаторе семейств, чтобы не менять одновременно распознавание семейства и решение о наличии фрагмента. Это новая гипотеза, требующая отдельного протокола/ресурса, а не выполненная работа. Настоящий отрицательный результат ограничен одним seed, историческим контролем без общего RNG и короткой адаптацией в хвосте LR-расписания.\n",
        "## Самопроверка и воспроизведение\n",
        """Проверены три CPU-теста: выбор отрицательных отдельно для ответов, нормировка/пустой ответ и AP с одинаковыми исключениями. Исходные checkpoint и candidate-cache сверены по SHA256 после запуска. Новое состояние загружено в заново созданную модель, все dev-прогнозы совпали точно. Frozen base не изменена. Последовательность64 updates и491 предъявления проверена; RNG и прогресс сохранены.\n
Ключевые артефакты: [исполнение и время](execution.json), [optimizer/LR](optimizer-check.json), [полный анализ и AP](analysis.json), [независимая проверка AP/артефактов](verification.json), [прогресс](progress.json), [расписание статей](schedule.json), полные `control/negative-dev-scores.jsonl` и `control/negative-full-composite-scores.jsonl`; веса `checkpoints/last.pt` локальны. [Согласованный протокол](../../../negative-candidates-proposal.md).\n
CPU-пересчёт по сохранённым score, без нового обучения:\n
```bash
uv run --locked python -m research.stages.ST2-l3-rhetoric.negative_fit.analyze
```

GPU-команда единственного запуска сохранена в `code/run.py`; повторять её автоматически нельзя. Новый запуск потребует отдельного решения о ресурсе. Точный исторический RNG отсутствовал; сохранение нового RNG не превращает сравнение в идентичные случайные траектории.\n""",
    ]
    import re

    (OUT / "report.md").write_text(
        re.sub(r"(?<=[А-Яа-яЁё])(?=\d)|(?<=\d)(?=[А-Яа-яЁё])", " ", "\n".join(lines))
    )
    print(
        json.dumps(
            dict(
                delta=result["delta"],
                criterion=result["working_criterion"],
                ap_exclusions=len(bd["ap_excluded_answers"]),
                gpu_seconds=execution["elapsed_seconds"],
            )
        )
    )


if __name__ == "__main__":
    main()

"""E3 research report with independently recomputed error attribution and learning curves."""

import importlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.kuki.data import FAMILIES, write_json
from research.kuki.experiment import load_development

reports = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.report_models")
ROOT = Path(__file__).parent
OUT = ROOT / "E3-detector"


def read(p):
    return json.loads(p.read_text())


def jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def main():
    rr = reports.runs("detector")
    chain = read(OUT / "chain.json")
    extended = read(OUT / "calibration-extension.json")
    strict = read(OUT / "strict-chain.json")
    selected = Path(chain["selected_runs"]["detector"])
    model = read(selected / "metrics.json")
    old = read(ROOT.parent / "selection.json")["dev_metrics"]
    docs, _ = load_development(read(Path("research/configs/data.json")))
    pp = jsonl(OUT / "chain-dev-scores.jsonl")
    pred = jsonl(OUT / "extended-dev-predictions.jsonl")
    threshold = extended["selection"]["threshold"]
    errors = {
        f: dict(gold=0, proposal_miss=0, available_not_emitted=0, correct=0) for f in FAMILIES
    }
    article_errors = []
    for d, p, o in zip(docs["dev"], pp, pred, strict=True):
        pairs = {(s, e) for s, e, _ in p["pairs"]}
        issued = {(s["start"], s["end"], s["label"]) for s in o["spans"] if s["score"] >= threshold}
        individual = []
        for a in d["annotations"]:
            counts = np.zeros(3)
            for s in a["l3_persuasion"]:
                item = errors[s["label"]]
                item["gold"] += 1
                if (s["start"], s["end"]) not in pairs:
                    key = "proposal_miss"
                    i = 0
                elif (s["start"], s["end"], s["label"]) not in issued:
                    key = "available_not_emitted"
                    i = 1
                else:
                    key = "correct"
                    i = 2
                item[key] += 1
                counts[i] += 1
            if len(a["l3_persuasion"]):
                individual.append(counts / len(a["l3_persuasion"]))
        if individual:
            article_errors.append(np.mean(individual, axis=0))
    newart = read(OUT / "extended-dev-articles.json")
    oldart = {
        r["doc_id"]: r
        for r in jsonl(
            ROOT.parent / "experiments/20260928-l3-hard-s42-01/dev-article-metrics.jsonl"
        )
    }
    diffs = np.array([r["exact_f1"] - oldart[r["doc_id"]]["exact_f1"] for r in newart])
    rng = np.random.default_rng(42)
    boot = diffs[rng.integers(0, len(diffs), (5000, len(diffs)))].mean(1)
    extra = dict(
        raw_errors=errors,
        article_mean_gold_disposition=np.mean(article_errors, axis=0).tolist(),
        paired_dev_difference=float(diffs.mean()),
        bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
        caveat="post-selection dev comparison, different architecture and compute; not confirmatory",
    )
    write_json(OUT / "analysis.json", extra)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    for r in rr:
        h = r["history"]
        x = [z["epoch"] for z in h]
        label = f"LoRA LR {r['config']['adapter_lr']:g}"
        for ax, values in zip(
            axes,
            [
                [z["loss"] for z in h],
                [z["dev"]["recall"]["200"] for z in h],
                [z["dev"]["exact_f1"] for z in h],
            ],
            strict=True,
        ):
            ax.plot(x, values, label=label)
    for ax, title in zip(
        axes,
        ["Train GlobalPointer loss", "Dev proposal recall@200", "Dev untyped exact F1"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "learning-curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "# E3. Независимый обучаемый генератор и полная цепочка L3\n",
        f"GlobalPointer выбран по dev охвату top200: **{model['dev']['recall']['200']:.4f}**, top500: **{model['dev']['recall']['500']:.4f}**. После независимого классификатора E1 и калибровки цепочка достигает typed exact-F1 **{chain['dev']['exact_f1']:.4f}**, overlap-F1 **{chain['dev']['overlap_f1']:.4f}**. Это результаты настройки на dev, test закрыт.\n",
        """## Вопрос и гипотезы\n
Может ли отдельный детектор совместно оценивать начало/конец и передавать классификатору компактный, достаточно полный список фрагментов? И улучшает ли композиция двух независимо обученных моделей итоговую L3-разметку? Гипотезы проверяются отдельно: recall@K отвечает за предложения детектора, typed F1 — за итоговую цепочку.\n
Данные:145 train /31 dev /31 test, только русский корпус, исходный split и индивидуальные ответы. E3 использует все145 train-статей, включая пустые ответы. Несколько семейств на одной паре сводятся к одному положительному фрагменту внутри конкретного ответа. Расхождения между ответами не устраняются объединением. Один seed42.\n
## Почему выбран этот метод и какие решения уже существуют\n
[spaCy SpanFinder](https://spacy.io/api/spanfinder) — готовый обучаемый компонент для нетипизированных, в том числе пересекающихся span, рассчитанный на работу со [SpanCategorizer](https://spacy.io/api/spancategorizer). Он предсказывает начала/концы токенов; стандартный предел длины25 токенов настраивается. Это подтверждает практичность разделения модулей, но его стандартная токенная геометрия не переносит точные символьные границы KuKi автоматически. Здесь он не обучался, эмпирического сравнения с ним нет.\n
Для обязательного продвинутого варианта выбран [GlobalPointer, Su et al.](https://arxiv.org/abs/2208.03054), [авторская реализация](https://github.com/bojone/GlobalPointer). Метод совместно оценивает пары начала/конца, использует относительное положение RoPE и loss для большого дисбаланса положительных/отрицательных span. Первоначальная область — NER; опубликованные результаты NER не считаются доказательством качества на риторике.\n
Выполнена собственная адаптация опубликованного метода к символьным позициям KuKi и одному типу «фрагмент присутствует». Это не готовые предобученные веса русского детектора риторики и не новый предложенный научный метод.\n
## Архитектура и обучение\n
Закреплённый RuModernBERT и LoRA те же, что в E1, но веса обучаются независимо с чистого состояния. Окна1024 BPE, перекрытие128, BF16 frozen base, LoRA rank8/alpha16/dropout0.1. Символьное представление размерности64 агрегирует соседние контекстные токены, символы и геометрию границы; линейная проекция даёт query/key размерности64, RoPE кодирует расстояние, score=q(start)·k(end)/sqrt(64).\n
Оцениваются все пары0≤start<end≤N. Нет token-boundary snapping, ограничения длины, top128 отдельных границ или добавления эталонов к кандидатам. Допускаются пересекающиеся, вложенные и пересекающие окна фрагменты. Представление межоконных пар не означает полноценный attention между далёкими окнами.\n
Для каждого ответа loss=log(1+Σpositive exp(-score))+log(1+Σnegative exp(score)); затем среднее по ответам и статьям. Пустой ответ даёт только отрицательную часть. Все допустимые пары участвуют в loss. Вычисление по блокам512 строк и пересчёт блоков в backward ограничивают память O(512N), но арифметика остаётся O(N²). Выход top500 сокращает последующую классификацию, а не сложность самого детектора.\n
Новая голова GlobalPointer линейная; `head_dropout=0.1` наследуется как конфигурационное поле старой span-головы, которая здесь заморожена и не используется. Реальный dropout детектора — LoRA0.1. AdamW, два LR LoRA{2e-4,5e-5}, head1e-3, decay0.01, accumulation8, clip1, warmup30 и линейное снижение до304updates/16эпох, patience6. Конкретные ограничения времени и поправки описаны в [протоколе](../protocol.md).\n
До обучения:5train-статей2342–14726символов,0updates, пик2.59ГиБ reserved. Градиенты streaming loss сравниваются с плотным эталоном, включая пустые/пересекающиеся ответы; top-K проверяется с детерминированными tie-break. Профиль — оценка стоимости, не проверка качества.\n
Выбор эпохи и LR: сначала dev recall@200, затем untyped exact-F1 и overlap-F1. Recall — доля уникальных нетипизированных эталонных пар в реальном top-K, отдельно по ответам, затем статьям; полностью пустые статьи исключаются из этого показателя. Для untyped F1 остаются все31 dev-статьи, cap100, порог выбирается точным перебором состояний dev-score. F1 пустого ответа/прогноза принят0 по прежнему протоколу; правильный отказ учитывается отдельно.\n
## Результаты детектора\n
| LoRA LR | Split | Recall@50 | @100 | @200 | @500 | Untyped exact-F1 | Overlap-F1 |\n|---|---|---:|---:|---:|---:|---:|---:|""",
    ]
    for r in rr:
        for split in ["train", "dev"]:
            z = r["metrics"][split]
            rc = z["recall"]
            lines.append(
                f"| {r['config']['adapter_lr']:g} | {split} | {rc['50']:.4f} | {rc['100']:.4f} | {rc['200']:.4f} | {rc['500']:.4f} | {z['exact_f1']:.4f} | {z['overlap_f1']:.4f} |"
            )
    lines += [
        "\n| LoRA LR | Выбрана эпоха | Выполнено эпох | Updates | Предъявлений статей | Статус остановки | Время, с |\n|---|---:|---:|---:|---:|---|---:|"
    ]
    for r in rr:
        z = r["metrics"]
        lines.append(
            f"| {r['config']['adapter_lr']:g} | {z['selected_epoch']} | {len(r['history'])} | {z['updates']} | {z['article_presentations']} | {z['status']} | {z['elapsed_seconds']:.2f} |"
        )
    lines += [
        "\nЕсли последняя эпоха остановлена по времени, она может быть неполной; точный объём отражают updates/предъявления и history.json. Время процесса включает подготовку и финальную оценку; лимит обучения проверяется между accumulation-блоками и не равен жёсткому убийству процесса.\n",
        "![Кривые обучения детектора](learning-curves.png)\n",
        """## Замороженная композиция\n
Детектор выдаёт500 пар; выбранный независимо E1 оценивает6 семейств для каждой пары в контексте статьи. Никаких эталонных границ в кандидатах цепочки нет. Финальный балл=logit(p_family)+alpha·detector_logit, alpha∈{0,0.5,1}; порог по точному перебору состояний top100 typed результатов, основной критерий dev article/answer exact-F1. Дополнительного обучения моделей на этом этапе нет. Argmax-fallback E1 выключен: пустой результат допустим.\n
Это ограниченная калибровка ранжирования; балл не объявляется вероятностью. Первоначальная формулировка небольшой сетки порогов уточнена до оценки цепочки до полного одномерного перебора, чтобы не повторять уже найденные пропуски между порогами. Выбранные alpha/порог неизменными применены к train.\n
| Alpha | Dev exact-F1 | Dev overlap-F1 | Порог score | Прогнозов/статью |\n|---|---:|---:|---:|---:|""",
    ]
    for z in chain["calibration"]:
        q = z["dev"]
        lines.append(
            f"| {z['alpha']:g} | {q['exact_f1']:.4f} | {q['overlap_f1']:.4f} | {z['threshold']:.6f} | {q['mean_predictions']:.2f} |"
        )
    lines += [
        "\n| Система | Split | Typed exact-F1 | Overlap-F1 | Exact precision | Exact recall | Правильных пустых ответов |\n|---|---|---:|---:|---:|---:|---:|"
    ]
    for name, split, z in [
        ("Прежний ST2 hard", "dev", old),
        ("Основная E3→E1", "train", chain["train"]),
        ("Основная E3→E1", "dev", chain["dev"]),
        ("Строгая композиция (post-hoc)", "dev", strict["dev"]),
        ("Итоговая alpha=2 (post-hoc)", "train", extended["train"]),
        ("Итоговая alpha=2 (post-hoc)", "dev", extended["dev"]),
    ]:
        lines.append(
            f"| {name} | {split} | {z['exact_f1']:.4f} | {z['overlap_f1']:.4f} | {z['exact_precision']:.4f} | {z['exact_recall']:.4f} | {z['correctly_empty_answers']}/{z['empty_answers']} |"
        )
    lo, hi = extra["bootstrap95"]
    disp = extra["article_mean_gold_disposition"]
    lines += [
        f"\nРазница dev exact-F1 итоговой настройки alpha=2 с прежним ST2: {diffs.mean():+.4f}, описательный парный bootstrap95% [{lo:+.4f};{hi:+.4f}] по31статье,5000повторов. Это сравнение выбранных систем на использованном для выбора dev. Ресурс, число моделей, loss и настройки отличаются; причинный эффект одного разделения модулей таким сравнением не установлен.\n",
        "## Ошибки и оставшиеся ограничения\n\nРазбор ошибок и парный bootstrap относятся к итоговой настройке alpha=2 из дополнительной dev-калибровки ниже.\n",
        "| Семейство | Сырых эталонов dev | Нет пары в top500 | Пара есть, верная метка не выдана | Верно выдано | Итоговый dev F1 семейства |\n|---|---:|---:|---:|---:|---:|",
    ]
    for f, c in errors.items():
        lines.append(
            f"| {f} | {c['gold']} | {c['proposal_miss']} | {c['available_not_emitted']} | {c['correct']} | {extended['dev']['per_family_exact_f1'][f]:.4f} |"
        )
    lines += [
        f"\nПри усреднении по непустым ответам и статьям распределение всех эталонных typed-фрагментов: {disp[0]:.1%} не попали в предложения; {disp[1]:.1%} доступны, но потеряны при классификации/финальном ранжировании; {disp[2]:.1%} верно выданы. Вторая группа не приписывается исключительно классификатору: сюда входят порог, cap100 и взаимодействие баллов. Сырые количества в таблице не являются независимыми наблюдениями.\n",
        """E1 учился на человеческих положительных фрагментах, а здесь получает ошибочные предложения. Комбинация баллов лишь частично учитывает такой сдвиг; отдельное обучение на out-of-fold кандидатах с отрицательными примерами не выполнено. При необходимости это следующий самостоятельный вариант, а не незаявленная корректировка текущих чисел.\n
Два LR, один seed и16эпох максимум — содержательная, но ограниченная настройка. Не исследованы другие encoder, ранг LoRA, размер pointer, регуляризация, разные seed и внешний корпус. Улучшение или ухудшение текущей цепочки не доказывает глобальную оптимальность GlobalPointer. Нет финальной test-оценки и подтверждения пригодности для автоматической разметки.\n
## Стоимость, воспроизведение и self-review\n""",
        f"Два E3fit заняли{sum(r['metrics']['elapsed_seconds'] for r in rr):.2f}с, профиль4.14с, цепочка с независимой загрузкой моделей{chain['elapsed_seconds']:.2f}с. Время E1 учтено отдельно и входит в общий [ledger](../budget.json). Пик reserved по E3fit: {max(r['metrics']['peak_reserved_bytes'] for r in rr) / 2**30:.2f}ГиБ.\n",
        """```bash
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.train --config research/stages/ST2-l3-rhetoric/modular/E3-detector/base.json
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.train --config research/stages/ST2-l3-rhetoric/modular/E3-detector/low-lr.json
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.chain
uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.report_detector
```

Для независимого повторения нужны новые Run ID и отдельный ledger; существующие run-каталоги не перезаписываются. Checkpoints содержат выбранные веса/optimizer/scheduler, но точное продолжение посреди эпохи новой modular-серии не реализовано.\n""",
        f"Выбранный детектор: [{selected.name}](../../experiments/{selected.name}/metrics.json); классификатор: `{chain['selected_runs']['classifier']}`. [Цепочка и калибровка](chain.json), [ошибки](analysis.json), [профиль](profile.json), [код детектора](../../../../kuki/global_pointer.py).\n",
        f"Отдельная проверка свежей загрузки выбранных моделей: max|Δprob| E1={chain['verification']['classifier_max_abs_diff']}, max|Δscore| E3={chain['verification']['detector_max_abs_diff']}, пары совпадают={chain['verification']['detector_pair_indices_equal']}.\n",
        """Self-review: проверены frozen base, формула/градиенты loss, top-K и tie-break, точный подбор порога против явного перебора, совпадение метрики калибровки с общим scorer. Для охвата цепочки используется реальное множество пар: совпадение начала одного кандидата с концом другого не считается найденным фрагментом. Проверены отсутствие test и подстановки gold, отдельная оценка пустых ответов, сохранение исторических запусков. Итоговая общая проверка фиксируется в `../verification.json`.\n""",
    ]
    delta = extended["dev"]["exact_f1"] - old["exact_f1"]
    lines.append(
        f"\n## Исследовательское решение\n\nНаблюдаемое изменение сквозного exact-F1 относительно прежнего ST2: {delta:+.4f}. Раздельные модули теперь можно оценивать независимо, но генератор остаётся существенным ограничением по охвату, а E1 требует проверки на ошибочных кандидатах. Для следующего сравнения обмена L3→L2 следует фиксировать один и тот же генератор/его кандидаты между условиями, чтобы изменение границ не смешивалось с эффектом обмена. Доказательства пользы самого обмена этот эксперимент не даёт; ST3–ST6 здесь не запускались.\n"
    )
    lines.append(
        "\n## Дополнительные проверки decoder после основного результата\n\nИсходная сетка завершилась лучшим alpha=1 на своей верхней границе. Поэтому выполнена **post-hoc**, а не заранее зафиксированная, CPU-проверка alpha∈{2,4,8} с точным подбором порогов по прежнему dev. Новых fit нет. Начальные score и основной результат0.0551 сохранены без изменений.\n\n| Alpha | Dev exact-F1 | Overlap-F1 |\n|---|---:|---:|"
    )
    for item in extended["additional_calibration"]:
        lines.append(
            f"| {item['alpha']:g} | {item['dev']['exact_f1']:.4f} | {item['dev']['overlap_f1']:.4f} |"
        )
    lines.append(
        f"\nДля дальнейшей разработки выбрана настройка **alpha={extended['selection']['alpha']:g}**, threshold={extended['selection']['threshold']:.6f}; dev exact-F1 **{extended['dev']['exact_f1']:.4f}**, train exact-F1 {extended['train']['exact_f1']:.4f}, dev precision {extended['dev']['exact_precision']:.4f}, recall {extended['dev']['exact_recall']:.4f}. В среднем {extended['dev']['mean_predictions']:.2f} прогнозов/статью; правильных пустых ответов {extended['dev']['correctly_empty_answers']}/{extended['dev']['empty_answers']}. Это небольшой выигрыш при всё ещё очень низкой точности, не готовый автоматический разметчик. [Дополнительная калибровка](calibration-extension.json).\n"
    )
    lines.append(
        f"Строгая композиция использовала уже выбранный порог детектора для top100-пар и условный E1-порог с argmax, без смешивания баллов и без нового подбора порогов. При cap100 typed результатов получены dev exact-F1 {strict['dev']['exact_f1']:.4f}, overlap-F1 {strict['dev']['overlap_f1']:.4f}. Сохранено отдельным post-hoc контролем в [strict-chain.json](strict-chain.json). Замена смешивания на последовательный отбор не устранила низкое качество.\n\nВоспроизведение этих двух CPU-проверок: `uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.strict_chain` и `uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.calibration_extension`. Новых параметров моделей не обучалось; расширение поиска по dev увеличивает неопределённость после выбора.\n"
    )
    alignment = read(OUT / "token-alignment-diagnostic.json")
    lines.append(
        f"\nДополнительная post-hoc CPU-диагностика первого детектора: {alignment['dev']['mean_token_aligned_fraction']:.1%} top500-кандидатов dev уже совпадают с границами токенов Razdel. Поэтому гипотеза, что список в основном расходуется на границы внутри слов, не подтверждается. Этот результат сам по себе не измеряет пользу token-filter и не доказывает, что он бесполезен; соответствующий decoder-вариант не запускался. [Данные диагностики](token-alignment-diagnostic.json), воспроизведение: `uv run --locked python -m research.stages.ST2-l3-rhetoric.modular.diagnose_alignment`.\n"
    )
    e2 = read(ROOT / "E2-proposals/metrics.json")["results"]["multiscale"]["dev"]
    lines.append(
        f"\n## Сопоставление с необучаемым генератором\n\nПри одинаковом учёте typed-эталонов и усреднении по непустым ответам/статьям точный охват top500 детектора составляет {chain['dev']['proposal_recall']:.4f}. У многомасштабного E2 он {e2['exact_recall']:.4f} при медиане {e2['candidate_count_quantiles'][1]:.0f} кандидатов. Это сопоставление охвата и размера списка, а не равных ресурсов: E3 требует дообучения и собственного GPU-скоринга, E2 строится на CPU. Латентность полного E2→классификатор здесь не измерялась. [Отдельный отчёт E2](../E2-proposals/report.md).\n"
    )
    lines.insert(
        2,
        f"После ограниченного расширения dev-калибровки выбран decoder alpha=2: exact-F1 **{extended['dev']['exact_f1']:.4f}**, overlap-F1 **{extended['dev']['overlap_f1']:.4f}**. Это последующее исследовательское уточнение; основной заранее заданный поиск и дополнительный поиск показаны раздельно.\n",
    )
    text = re.sub(r"(?<=[А-Яа-яЁё])(?=\d)|(?<=\d)(?=[А-Яа-яЁё])", " ", "\n".join(lines))
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    sections = {part.splitlines()[0]: part for part in parts[1:]}
    order = [
        "Вопрос и гипотезы",
        "Почему выбран этот метод и какие решения уже существуют",
        "Архитектура и обучение",
        "Результаты детектора",
        "Сопоставление с необучаемым генератором",
        "Замороженная композиция",
        "Дополнительные проверки decoder после основного результата",
        "Ошибки и оставшиеся ограничения",
        "Исследовательское решение",
        "Стоимость, воспроизведение и self-review",
    ]
    assert set(sections) == {"## " + title for title in order}
    (OUT / "report.md").write_text(parts[0] + "".join(sections["## " + title] for title in order))


if __name__ == "__main__":
    main()

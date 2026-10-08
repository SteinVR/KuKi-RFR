"""Render a CPU report from the published metrics and paired comparisons."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from research.kuki.data import digest, write_json

ROOT = Path("research/stages/final")
NAMES = {
    "l1": "L1 отдельно",
    "l2": "L2 отдельно",
    "l3": "L3 отдельно",
    "st3": "Совместное L2–L3, λ = 2",
    "st4_control": "Передача без вероятностей семейств",
    "st4_families": "Передача с вероятностями семейств",
    "st6": "Совместное L1–L2, 4:1",
}
LABELS = {
    "l1_f1": "L1 macro-F1 ↑",
    "l2_brier": "L2 Brier ↓",
    "l3_conditional": "L3 условный F1 ↑",
    "l3_exact": "L3 автоматический exact-F1 ↑",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    data = read(ROOT / "results.json")
    metrics = data["metrics"]

    def comparison(first, second, metric):
        return next(c for c in data["comparisons"]
                    if (c["first"], c["second"], c["metric"]) == (first, second, metric))

    def interval(record):
        return f"[{record['ci95'][0]:+.4f}; {record['ci95'][1]:+.4f}]"

    report = (
        "# KuKi-RFR: результаты исследования\n\n"
        f"Корпус: 207 статей и 301 индивидуальная разметка. Разбиение: "
        f"145 train / 31 dev / 31 test. Финальная оценка: {data['test_articles']} статья "
        f"и {data['test_answers']} индивидуальных ответов. Семь систем закреплены "
        "до открытия test.\n\n"
        "## Метрики на test\n\n"
        "| Система | L1 macro-F1 ↑ | L2 Brier ↓ | L3 условный F1 ↑ | "
        "L3 автоматический exact-F1 ↑ |\n"
        "|:---|---:|---:|---:|---:|\n"
    )
    for name, label in NAMES.items():
        cells = [f"{metrics[name][metric]['value']:.4f}" if metric in metrics[name] else "—"
                 for metric in LABELS]
        report += f"| {label} | {' | '.join(cells)} |\n"
    report += (
        "\nМодели передачи наследуют риторические предсказания совместного L2–L3. "
        "Статьи имеют равный вес; ответы разметчиков усредняются внутри статьи. "
        "L1 оценивает роли при заданных границах упоминаний, условный L3 — "
        "семейства при заданных границах фрагментов. L1 использует 30 активных "
        "test-статей, условный L3 — 29; L2 и автоматический L3 — все 31.\n\n"
        "## Парные сравнения\n\n"
        "| Системы: первая − вторая | Метрика | Разность | 95% интервал |\n"
        "|:---|:---|---:|:---|\n"
    )
    for c in data["comparisons"]:
        report += (f"| {NAMES[c['first']]} − {NAMES[c['second']]} | {LABELS[c['metric']]} | "
                   f"{c['delta']:+.6f} | {interval(c)} |\n")
    automatic = comparison("st3", "l3", "l3_exact")
    frames = comparison("st3", "l2", "l2_brier")
    transfer = comparison("st4_families", "st4_control", "l2_brier")
    report += (
        "\n![Парные разности на test](test-differences.png)\n\n"
        "## Результат и условия оценки\n\n"
        f"Разность автоматического exact-F1 между совместным L2–L3 и отдельным L3: "
        f"{automatic['delta']:+.4f}, интервал {interval(automatic)}. "
        f"Разность Brier между совместным L2–L3 и отдельным L2: "
        f"{frames['delta']:+.4f}, интервал {interval(frames)}. "
        f"Добавление вероятностей семейств в модель передачи изменило Brier на "
        f"{transfer['delta']:+.4f}, интервал {interval(transfer)}.\n\n"
        "Bootstrap использует 10 000 повторов и seed 42; единица ресэмплирования — "
        "статья. Интервалы описывают вариативность по статьям для моделей, обученных "
        "с одним seed. Они не учитывают вариативность обучения и не скорректированы "
        "за несколько сравнений. Test представляет те же четыре издания, что и train.\n\n"
        "[Полные метрики](results.json) · [Статистики статей](article-statistics.json) · "
        "[Протокол](protocol.md) · [Манифест исходной оценки](manifest.json) · "
        "[Статья](../../../paper/paper.md)\n"
    )
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, is_brier in zip(axes, [True, False], strict=True):
        comparisons = [c for c in data["comparisons"]
                       if (c["metric"] == "l2_brier") == is_brier]
        for i, c in enumerate(comparisons):
            ax.plot(c["ci95"], [i, i], color="#177e89")
            ax.scatter([c["delta"]], [i], color="#177e89")
        ax.set_yticks(range(len(comparisons)),
                      [f"{c['first']} − {c['second']}\n{c['metric']}" for c in comparisons])
        ax.axvline(0, color="gray", linestyle="--")
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Difference: negative is better" if is_brier
                      else "Difference: positive is better")
        ax.set_title("Test Brier: paired 95% intervals" if is_brier
                     else "Test F1: paired 95% intervals")
    fig.savefig(ROOT / "test-differences.png", dpi=160)
    plt.close(fig)
    paths = [Path(__file__), ROOT / "results.json", ROOT / "report.md",
             ROOT / "test-differences.png"]
    write_json(ROOT / "report-provenance.json",
               dict(files={str(p): digest(p) for p in paths}))
    print("Generated report.md, test-differences.png, and report-provenance.json")


if __name__ == "__main__":
    main()

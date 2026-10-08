"""Post-hoc CPU sensitivity: primary optimum was at alpha=1 grid boundary."""

import importlib
import json
from pathlib import Path

from research.kuki.data import write_json, write_jsonl
from research.kuki.experiment import load_development

chain = importlib.import_module("research.stages.ST2-l3-rhetoric.modular.chain")
ROOT = Path(__file__).parent


def main():
    original = json.loads((ROOT / "E3-detector/chain.json").read_text())
    docs, hashes = load_development(json.loads(Path("research/configs/data.json").read_text()))
    predictions = {
        s: chain.read_jsonl(ROOT / "E3-detector" / f"chain-{s}-scores.jsonl") for s in docs
    }
    tested = []
    for alpha in [2.0, 4.0, 8.0]:
        t, value = chain.exact_threshold(docs["dev"], chain.ranked(predictions["dev"], alpha))
        metrics, _, _ = chain.chain_score(docs["dev"], predictions["dev"], alpha, t)
        assert abs(metrics["exact_f1"] - value) < 1e-12
        tested.append(dict(alpha=alpha, threshold=t, dev=metrics))
    choice = max(
        original["calibration"] + tested,
        key=lambda z: (z["dev"]["exact_f1"], z["threshold"], -z["alpha"]),
    )
    result = dict(
        post_hoc=True,
        reason="primary best alpha1 at upper grid boundary",
        new_training=False,
        data_hashes=hashes,
        test_used=False,
        additional_calibration=tested,
        selection=choice,
    )
    for split in docs:
        metrics, records, outputs = chain.chain_score(
            docs[split], predictions[split], choice["alpha"], choice["threshold"]
        )
        result[split] = metrics
        write_json(ROOT / "E3-detector" / f"extended-{split}-articles.json", records)
        write_jsonl(ROOT / "E3-detector" / f"extended-{split}-predictions.jsonl", outputs)
    write_json(ROOT / "E3-detector/calibration-extension.json", result)
    print(
        json.dumps(
            dict(
                tested=[(x["alpha"], x["dev"]["exact_f1"]) for x in tested],
                selected_alpha=choice["alpha"],
            )
        )
    )


if __name__ == "__main__":
    main()

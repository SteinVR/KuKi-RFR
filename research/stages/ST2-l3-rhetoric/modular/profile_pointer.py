import json
import time
from pathlib import Path

import numpy as np
import torch

from research.kuki.data import write_json
from research.kuki.experiment import load_development
from research.kuki.global_pointer import GlobalPointer
from research.kuki.train_l3 import configure, encode_documents

ROOT = Path(__file__).parent


def main():
    begin = time.perf_counter()
    configure(42)
    cfg = json.loads((ROOT.parent / "configs/hard-negative-v1.json").read_text())
    cfg.update(pointer_dimension=64, pair_block_size=512)
    enc = json.loads(Path(cfg["encoder_config"]).read_text())
    docs, hashes = load_development(json.loads(Path(cfg["data_config"]).read_text()))
    ordered = sorted(docs["train"], key=lambda d: len(d["text"]))
    probe = [
        ordered[i] for i in [0, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, -1]
    ]
    encoded = encode_documents(probe, enc, cfg)
    model = GlobalPointer(enc, cfg).cuda()
    result = []
    for d in probe:
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, _ = model.training_loss(encoded[d["doc_id"]], d["annotations"])
        loss.backward()
        torch.cuda.synchronize()
        training = time.perf_counter() - start
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        model.eval()
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.no_grad():
            pairs = model.infer_pairs(encoded[d["doc_id"]])
        torch.cuda.synchronize()
        inference = time.perf_counter() - start
        result.append(
            dict(
                doc_id=d["doc_id"],
                characters=len(d["text"]),
                tokens=encoded[d["doc_id"]]["encoded"]["tokens"],
                answers=len(d["annotations"]),
                loss=float(loss.detach()),
                training_seconds=training,
                inference_seconds=inference,
                proposals=len(pairs),
            )
        )
        print(json.dumps(result[-1]), flush=True)
    elapsed = time.perf_counter() - begin
    measured = np.array([r["training_seconds"] for r in result])
    dev = np.array([r["inference_seconds"] for r in result])
    report = dict(
        records=result,
        elapsed_seconds=elapsed,
        data_hashes=hashes,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        rough_epoch_seconds=float(np.median(measured) * 145 + np.median(dev) * 31),
        caveat="5 train documents, CPU/tokenization/setup separate; no optimizer steps; only cost estimate, not trained performance",
        config=cfg,
    )
    write_json(ROOT / "E3-detector/profile.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["records", "config", "data_hashes"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

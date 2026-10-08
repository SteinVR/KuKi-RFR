"""Inspect pinned tokenizer/base behavior across candidate environments."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import write_json
from .l2 import L2Model, frozen_digest, seed_all
from .windows import tokenize_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-fp32", action="store_true")
    parser.add_argument("--attention", default="sdpa")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    seed_all(42)
    cfg = json.loads(Path("research/configs/encoder.json").read_text())
    cfg["attention"] = args.attention
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    manifest = json.loads(
        Path(
            "research/stages/ST0-preparation/experiments/20260928-data-s42-02/split.json"
        ).read_text()
    )
    train = {r["doc_id"] for r in manifest["rows"] if r["split"] == "train"}
    docs = [json.loads(s) for s in Path(manifest["prepared_path"]).read_text().splitlines()]
    docs = [d for d in docs if d["doc_id"] in train]
    h = hashlib.sha256()
    for doc in docs:
        encoded = tokenize_document(tok, doc["text"])
        h.update(json.dumps(encoded, sort_keys=True).encode())
    model = L2Model(cfg).cuda().eval()
    weights_digest = frozen_digest(model)
    base_dtypes = sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad})
    if args.probe_fp32:
        model.encoder.float()
    inputs = tok(docs[0]["text"][:300], return_tensors="pt").to("cuda")
    with torch.no_grad():
        hidden = model.encoder(**inputs).last_hidden_state.float().mean(1)[0].cpu().tolist()
    write_json(
        args.output,
        dict(
            tokenized_train_sha256=h.hexdigest(),
            frozen_weights_sha256=weights_digest,
            base_dtypes=base_dtypes,
            probe_dtype="float32" if args.probe_fp32 else "bfloat16",
            attention=args.attention,
            trainable_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
            parameter_count=sum(p.numel() for p in model.parameters()),
            encoder_probe_mean_hidden=hidden,
            encoder_class=type(model.encoder.base_model.model).__name__,
            model_config=model.encoder.config.to_dict(),
        ),
    )
    print(json.dumps({"output": str(args.output), "tokenized_train_sha256": h.hexdigest()}))


if __name__ == "__main__":
    main()

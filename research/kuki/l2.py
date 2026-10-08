"""Minimal L2 pilot with article-level loss, LoRA, and restartable state."""

import hashlib
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class L2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_config = AutoConfig.from_pretrained(cfg["model_id"], revision=cfg["revision"])
        # The pinned checkpoint contains both local_attention=128 and an old,
        # unused sliding_window=128. Transformers 5 treats sliding_window as a
        # half-window setter, accidentally doubling local_attention on load.
        # Preserve the original model's total local window explicitly.
        model_config.local_attention = cfg.get("local_attention", 128)
        model_config.reference_compile = cfg.get("reference_compile", False)
        model_config.deterministic_flash_attn = cfg.get("deterministic_flash_attn", True)
        encoder = AutoModel.from_pretrained(
            cfg["model_id"],
            revision=cfg["revision"],
            config=model_config,
            dtype=torch.bfloat16,
            attn_implementation=cfg["attention"],
        )
        lora = cfg["lora"]
        self.encoder = get_peft_model(
            encoder,
            LoraConfig(
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"],
                bias="none",
            ),
        )
        if cfg.get("gradient_checkpointing", True):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.head = nn.Linear(encoder.config.hidden_size, 14)

    def forward(self, encoded):
        device = self.head.weight.device
        pooled = None
        for window in encoded["windows"]:
            ids = torch.tensor([window["input_ids"]], device=device)
            weights = torch.tensor(window["weights"], device=device)
            hidden = self.encoder(
                input_ids=ids, attention_mask=torch.ones_like(ids)
            ).last_hidden_state[0]
            value = (hidden.float() * weights[:, None]).sum(dim=0) / encoded["tokens"]
            pooled = value if pooled is None else pooled + value
        return self.head(pooled)


def article_loss(logits, answers):
    # Mean over 14 frames, then annotators. Each article contributes one scalar.
    target = torch.tensor(answers, device=logits.device, dtype=torch.float32)
    return nn.functional.binary_cross_entropy_with_logits(logits.expand_as(target), target)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def frozen_digest(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, model, optimizer, scheduler, progress, provenance):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Save only at an update boundary after zero_grad(set_to_none=True)")
    payload = dict(
        trainable=trainable_state(model),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        progress=progress,
        provenance=provenance,
        rng=dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        ),
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, scheduler, expected_provenance):
    state = torch.load(
        path, map_location="cpu", weights_only=False
    )  # Locally generated trusted state only.
    if state["provenance"] != expected_provenance:
        raise ValueError("Checkpoint provenance mismatch")
    current = dict(model.named_parameters())
    if set(state["trainable"]) != {n for n, p in current.items() if p.requires_grad}:
        raise ValueError("Trainable parameter set changed")
    with torch.no_grad():
        for n, value in state["trainable"].items():
            current[n].copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["progress"]

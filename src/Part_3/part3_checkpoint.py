"""Save / load Part 3 Transformer checkpoints."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PART3 = Path(__file__).resolve().parent
_P1 = _PART3.parent / "Part_1"
_P2 = _PART3.parent / "Part_2"
for _p in (_P1, _P2, _PART3):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from part3_model import TransformerTagger, build_transformer_tagger  # noqa: E402


def make_config(
    task: str,
    num_labels: int,
    *,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    max_seq: int,
    dropout: float,
    freeze_embedding: bool,
    random_embedding: bool,
    use_crf: bool,
) -> dict[str, Any]:
    return {
        "arch": "transformer",
        "task": task,
        "num_labels": num_labels,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "max_seq": max_seq,
        "dropout": dropout,
        "freeze_embedding": freeze_embedding,
        "random_embedding": random_embedding,
        "use_crf": use_crf,
    }


def save_checkpoint(path: str, model: torch.nn.Module, config: dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_tagger_for_eval(
    path: str,
    weights_np: np.ndarray,
    device: torch.device,
) -> tuple[TransformerTagger, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Expected Part 3 checkpoint with state_dict + config: {path}")
    cfg = ckpt["config"]
    if cfg.get("arch") != "transformer":
        raise ValueError(f"Not a transformer checkpoint: {path}")
    w = None if cfg.get("random_embedding") else weights_np
    model = build_transformer_tagger(
        cfg["num_labels"],
        w,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        max_seq=cfg["max_seq"],
        dropout=cfg.get("dropout", 0.1),
        freeze_embedding=cfg.get("freeze_embedding", True),
        random_embedding=cfg.get("random_embedding", False),
        use_crf=cfg.get("use_crf", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg

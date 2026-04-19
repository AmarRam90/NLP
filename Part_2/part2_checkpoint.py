"""Save/load BiLSTM checkpoints with architecture config (Part 2)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_P1 = str(Path(__file__).resolve().parent.parent / "Part_1")
_P2 = str(Path(__file__).resolve().parent)
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
if _P2 not in sys.path:
    sys.path.insert(0, _P2)

from part2_dataset import NER_TAGS, POS_TAGS
from part2_models import BiLSTMTagger, build_bilstm_tagger


def make_config(
    task: str,
    num_labels: int,
    frozen: bool,
    bidirectional: bool,
    use_dropout: bool,
    random_embedding: bool,
    use_crf: bool,
) -> dict[str, Any]:
    return {
        "task": task,
        "num_labels": num_labels,
        "frozen": frozen,
        "bidirectional": bidirectional,
        "use_dropout": use_dropout,
        "random_embedding": random_embedding,
        "use_crf": use_crf,
    }


def save_checkpoint(path: str, model: torch.nn.Module, config: dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def _extract_state_dict(raw: Any) -> dict[str, Any] | None:
    """If `raw` is a checkpoint dict with weights only, return the state_dict; else None."""
    if not isinstance(raw, dict):
        return None
    if "state_dict" in raw and "config" not in raw:
        return raw["state_dict"]
    if "embedding.weight" in raw and "emission.weight" in raw:
        return raw
    return None


def _infer_plain_checkpoint(
    sd: dict[str, Any],
    weights_np: np.ndarray,
) -> tuple[BiLSTMTagger, dict[str, Any]]:
    """Build tagger from a plain `state_dict` (no config): infer POS vs NER and CRF from tensors."""
    w = sd["emission.weight"]
    n_labels = int(w.shape[0])
    use_crf = any(str(k).startswith("crf.") for k in sd)
    if n_labels == len(POS_TAGS):
        cfg = make_config("pos", n_labels, True, True, True, False, False)
        model = build_bilstm_tagger(n_labels, weights_np, freeze=True, use_crf=False)
    elif n_labels == len(NER_TAGS):
        cfg = make_config("ner", n_labels, True, True, True, False, use_crf)
        model = build_bilstm_tagger(n_labels, weights_np, freeze=True, use_crf=use_crf)
    else:
        raise ValueError(
            f"Cannot infer task: emission has {n_labels} labels "
            f"(expected {len(POS_TAGS)} POS or {len(NER_TAGS)} NER)."
        )
    model.load_state_dict(sd)
    return model, cfg


def load_tagger_for_eval(
    path: str,
    weights_np: np.ndarray,
    device: torch.device,
    legacy: str | None = None,
) -> tuple[BiLSTMTagger, dict[str, Any]]:
    """
    Load model for evaluation. New checkpoints store `{state_dict, config}`.
    Plain `.pt` files without config: pass `legacy='pos'|'ner'`, or leave `legacy=None`
    to infer POS vs NER (and NER+CRF vs NER softmax) from tensor shapes and keys.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "config" in ckpt:
        cfg = ckpt["config"]
        w = None if cfg.get("random_embedding") else weights_np
        model = build_bilstm_tagger(
            cfg["num_labels"],
            w,
            freeze=cfg.get("frozen", True),
            bidirectional=cfg.get("bidirectional", True),
            use_dropout=cfg.get("use_dropout", True),
            random_embedding=cfg.get("random_embedding", False),
            use_crf=cfg.get("use_crf", False),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model, cfg

    sd = _extract_state_dict(ckpt)
    if sd is not None and legacy is None:
        return _infer_plain_checkpoint(sd, weights_np)

    if legacy == "pos":
        cfg = make_config("pos", len(POS_TAGS), True, True, True, False, False)
        model = build_bilstm_tagger(len(POS_TAGS), weights_np, freeze=True, use_crf=False)
    elif legacy == "ner":
        raw_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        assert isinstance(raw_sd, dict)
        use_crf = any(str(k).startswith("crf.") for k in raw_sd)
        cfg = make_config("ner", len(NER_TAGS), True, True, True, False, use_crf)
        model = build_bilstm_tagger(len(NER_TAGS), weights_np, freeze=True, use_crf=use_crf)
    else:
        raise ValueError(
            f"Unknown checkpoint format: {path} "
            "(use a {state_dict, config} save, a plain state_dict, or legacy='pos'|'ner')."
        )
    raw_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt and "config" not in ckpt else ckpt
    model.load_state_dict(raw_sd)
    return model, cfg

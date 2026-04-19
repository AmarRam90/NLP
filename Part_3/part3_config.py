"""
Part 3 — Transformer tagger hyperparameters (align with notebook master table).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PART3_ROOT = Path(__file__).resolve().parent
_P1_STR = str(_PART3_ROOT.parent / "Part_1")
if _P1_STR not in sys.path:
    sys.path.insert(0, _P1_STR)

from part1_vocab_tfidf import EMB_DIR  # noqa: E402

# Architecture (defaults match CS4063_NLP_Assignment2.ipynb)
TF_HEADS = 4
TF_DMODEL = 128
TF_DFF = 512
TF_LAYERS = 4
TF_DROPOUT = 0.1

TF_LR = 5e-4
TF_WD = 0.01
TF_WARMUP = 50
TF_EPOCHS = 20
TF_MAX_SEQ = 256
TF_BATCH = 16
TF_PATIENCE = 5

MODEL_DIR = str(_PART3_ROOT / "models")
EMB_PATH = str(Path(EMB_DIR) / "embeddings_w2v.npy")
WORD2IDX_PATH = str(Path(EMB_DIR) / "word2idx.json")

PAD_IDX = 0
UNK_IDX = 1

NER_AUX_CE_WEIGHT = 0.4

"""
Part 2.3 — Test metrics, confusion heatmaps; helpers for tables & analysis.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_P1 = str(Path(__file__).resolve().parent.parent / "Part_1")
_P2 = str(Path(__file__).resolve().parent)
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
if _P2 not in sys.path:
    sys.path.insert(0, _P2)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from part2_checkpoint import load_tagger_for_eval
from part2_dataset import DATA_DIR, IDX2NER, POS_TAGS
from part2_train import (
    EMB_PATH,
    MODEL_DIR,
    WORD2IDX_PATH,
    NerDataset,
    PosDataset,
    ner_collate,
    pos_collate,
    read_ner_conll,
    read_pos_conll,
)


def load_w2i() -> dict[str, int]:
    with open(WORD2IDX_PATH, encoding="utf-8") as f:
        return json.load(f)


def pos_test_metrics(model_path: str, legacy: str | None = "pos") -> tuple[float, float]:
    """Token accuracy and macro-F1 on pos_test.conll."""
    w2i = load_w2i()
    weights = np.load(EMB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_tagger_for_eval(model_path, weights, device, legacy=legacy)
    model = model.to(device)
    model.eval()
    test_s = read_pos_conll(Path(DATA_DIR) / "pos_test.conll")
    ds = PosDataset(test_s, w2i)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=pos_collate)
    ys: list[int] = []
    pr: list[int] = []
    with torch.no_grad():
        for input_ids, lengths, targets in loader:
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            logits = model(input_ids, lengths)
            pred = logits.argmax(-1).cpu().numpy()
            tg = targets.numpy()
            for b in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    if tg[b, t] == -100:
                        continue
                    ys.append(tg[b, t])
                    pr.append(pred[b, t])
    acc = accuracy_score(ys, pr)
    macro = f1_score(ys, pr, average="macro", labels=list(range(len(POS_TAGS))), zero_division=0)
    return float(acc), float(macro)


def ner_test_macro_f1(model_path: str, legacy: str | None = "ner") -> float:
    """Macro-averaged entity F1 (seqeval) on ner_test.conll."""
    w2i = load_w2i()
    weights = np.load(EMB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_tagger_for_eval(model_path, weights, device, legacy=legacy)
    model = model.to(device)
    model.eval()
    test_s = read_ner_conll(Path(DATA_DIR) / "ner_test.conll")
    ds = NerDataset(test_s, w2i)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ner_collate)
    y_true_seq: list[list[str]] = []
    y_pred_seq: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, lengths, targets, mask = batch
            input_ids, lengths, targets, mask = (
                input_ids.to(device),
                lengths.to(device),
                targets.to(device),
                mask.to(device),
            )
            if model.crf is not None:
                pred_ids = model(input_ids, lengths, tags=None, mask=mask)
            else:
                pred_ids = model(input_ids, lengths).argmax(-1)
            pred_ids = pred_ids.cpu().numpy()
            tg = targets.cpu().numpy()
            for b in range(pred_ids.shape[0]):
                tt, pp = [], []
                for t in range(pred_ids.shape[1]):
                    if tg[b, t] == -100:
                        continue
                    tt.append(IDX2NER[int(tg[b, t])])
                    pp.append(IDX2NER[int(pred_ids[b, t])])
                y_true_seq.append(tt)
                y_pred_seq.append(pp)
    rep = classification_report(y_true_seq, y_pred_seq, output_dict=True, zero_division=0)
    return float(rep.get("macro avg", {}).get("f1-score", 0.0))


def evaluate_pos(model_path: str, name: str, legacy: str | None = "pos") -> None:
    acc, macro = pos_test_metrics(model_path, legacy=legacy)
    print(f"\n=== POS {name} ===\nAccuracy: {acc:.4f}\nMacro-F1: {macro:.4f}")
    w2i = load_w2i()
    weights = np.load(EMB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_tagger_for_eval(model_path, weights, device, legacy=legacy)
    model = model.to(device)
    model.eval()
    test_s = read_pos_conll(Path(DATA_DIR) / "pos_test.conll")
    ds = PosDataset(test_s, w2i)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=pos_collate)
    ys: list[int] = []
    pr: list[int] = []
    with torch.no_grad():
        for input_ids, lengths, targets in loader:
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            logits = model(input_ids, lengths)
            pred = logits.argmax(-1).cpu().numpy()
            tg = targets.numpy()
            for b in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    if tg[b, t] == -100:
                        continue
                    ys.append(tg[b, t])
                    pr.append(pred[b, t])
    cm = confusion_matrix(ys, pr, labels=list(range(len(POS_TAGS))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=POS_TAGS, yticklabels=POS_TAGS)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(f"POS confusion ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"pos_confusion_{name}.png"), dpi=120)
    plt.close()


def evaluate_ner(model_path: str, name: str, legacy: str | None = "ner") -> None:
    w2i = load_w2i()
    weights = np.load(EMB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_tagger_for_eval(model_path, weights, device, legacy=legacy)
    model = model.to(device)
    model.eval()
    test_s = read_ner_conll(Path(DATA_DIR) / "ner_test.conll")
    ds = NerDataset(test_s, w2i)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ner_collate)
    y_true_seq: list[list[str]] = []
    y_pred_seq: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, lengths, targets, mask = batch
            input_ids, lengths, targets, mask = (
                input_ids.to(device),
                lengths.to(device),
                targets.to(device),
                mask.to(device),
            )
            if model.crf is not None:
                pred_ids = model(input_ids, lengths, tags=None, mask=mask)
            else:
                pred_ids = model(input_ids, lengths).argmax(-1)
            pred_ids = pred_ids.cpu().numpy()
            tg = targets.cpu().numpy()
            for b in range(pred_ids.shape[0]):
                tt, pp = [], []
                for t in range(pred_ids.shape[1]):
                    if tg[b, t] == -100:
                        continue
                    tt.append(IDX2NER[int(tg[b, t])])
                    pp.append(IDX2NER[int(pred_ids[b, t])])
                y_true_seq.append(tt)
                y_pred_seq.append(pp)
    print(f"\n=== NER ({name}) test ===")
    print(classification_report(y_true_seq, y_pred_seq, digits=4))


def main() -> None:
    pos_pt = os.path.join(MODEL_DIR, "bilstm_pos.pt")
    ner_pt = os.path.join(MODEL_DIR, "bilstm_ner.pt")
    pos_ft = os.path.join(MODEL_DIR, "bilstm_pos_ft.pt")
    ner_ft = os.path.join(MODEL_DIR, "bilstm_ner_ft.pt")

    if os.path.isfile(pos_pt):
        evaluate_pos(pos_pt, "frozen")
    if os.path.isfile(pos_ft):
        evaluate_pos(pos_ft, "finetuned", legacy=None)
    if os.path.isfile(ner_pt):
        evaluate_ner(ner_pt, "crf_frozen")
    if os.path.isfile(ner_ft):
        evaluate_ner(ner_ft, "crf_finetuned", legacy=None)


if __name__ == "__main__":
    main()

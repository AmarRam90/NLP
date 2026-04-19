"""
Part 2 — Error analysis: top confused POS pairs + examples; NER FP/FN spans.
Run from assignment root:  python Part_2/part2_analysis.py
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

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from part2_checkpoint import load_tagger_for_eval
from part2_dataset import DATA_DIR, IDX2NER, IDX2POS, POS_TAGS
from part2_train import EMB_PATH, MODEL_DIR, WORD2IDX_PATH, NerDataset, PosDataset, ner_collate, pos_collate, read_ner_conll, read_pos_conll


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_w2i() -> dict[str, int]:
    with open(WORD2IDX_PATH, encoding="utf-8") as f:
        return json.load(f)


def bio_entity_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """Entity spans as (type, start inclusive, end exclusive)."""
    spans: set[tuple[str, int, int]] = set()
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t.startswith("B-"):
            et = t[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{et}":
                j += 1
            spans.add((et, i, j))
            i = j
        else:
            i += 1
    return spans


def surface(tokens: list[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end])


def pos_top_confusions_with_examples(
    model_path: str,
    *,
    legacy: str | None = "pos",
    top_k: int = 3,
    examples_per_pair: int = 2,
) -> None:
    """Print top-k off-diagonal POS confusion counts + example sentences from pos_test."""
    w2i = _load_w2i()
    weights = np.load(EMB_PATH)
    device = _device()
    model, _cfg = load_tagger_for_eval(model_path, weights, device, legacy=legacy)
    model = model.to(device)
    model.eval()
    test_s = read_pos_conll(Path(DATA_DIR) / "pos_test.conll")
    ds = PosDataset(test_s, w2i)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=pos_collate)

    ys: list[int] = []
    pr: list[int] = []
    # Per-sentence aligned gold/pred for examples
    sent_gold: list[list[int]] = []
    sent_pred: list[list[int]] = []
    sent_toks: list[list[str]] = []

    with torch.no_grad():
        si = 0
        for input_ids, lengths, targets in loader:
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            logits = model(input_ids, lengths)
            pred = logits.argmax(-1).cpu().numpy()
            tg = targets.numpy()
            for b in range(pred.shape[0]):
                toks, _ = test_s[si]
                si += 1
                glist, plist = [], []
                for t in range(pred.shape[1]):
                    if tg[b, t] == -100:
                        continue
                    gi = int(tg[b, t])
                    pi = int(pred[b, t])
                    ys.append(gi)
                    pr.append(pi)
                    glist.append(gi)
                    plist.append(pi)
                sent_toks.append(toks)
                sent_gold.append(glist)
                sent_pred.append(plist)

    cm = confusion_matrix(ys, pr, labels=list(range(len(POS_TAGS))))
    off_diag: list[tuple[int, int, int]] = []
    for gi in range(len(POS_TAGS)):
        for pj in range(len(POS_TAGS)):
            if gi != pj:
                off_diag.append((int(cm[gi, pj]), gi, pj))
    off_diag.sort(key=lambda x: -x[0])
    top = off_diag[:top_k]

    print(f"\n=== Top {top_k} confused POS pairs (gold → pred), counts on pos_test ===\n")
    for count, gi, pj in top:
        gname, pname = IDX2POS[gi], IDX2POS[pj]
        print(f"{gname} → {pname}: {count} tokens")
        shown = 0
        for toks, gl, pl in zip(sent_toks, sent_gold, sent_pred):
            for ti, (g, p) in enumerate(zip(gl, pl)):
                if g == gi and p == pj:
                    # window around token
                    lo = max(0, ti - 6)
                    hi = min(len(toks), ti + 7)
                    ctx = " ".join(toks[lo:hi])
                    print(f"  ex{shown + 1}: … {ctx} …")
                    shown += 1
                    if shown >= examples_per_pair:
                        break
            if shown >= examples_per_pair:
                break
        if shown == 0:
            print("  (no sentence-level example collected — check alignment)")
        print()


def ner_fp_fn_examples(
    model_path: str,
    *,
    legacy: str | None = "ner",
    n_each: int = 5,
) -> None:
    """Hand-pick-style list: first n FP and n FN entity spans (pred vs gold) on ner_test."""
    w2i = _load_w2i()
    weights = np.load(EMB_PATH)
    device = _device()
    model, _cfg = load_tagger_for_eval(model_path, weights, device, legacy=legacy)
    model = model.to(device)
    model.eval()
    test_s = read_ner_conll(Path(DATA_DIR) / "ner_test.conll")
    ds = NerDataset(test_s, w2i)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ner_collate)

    fp_list: list[str] = []
    fn_list: list[str] = []

    with torch.no_grad():
        si = 0
        done = False
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
                toks, _p, _n = test_s[si]
                si += 1
                true_tags: list[str] = []
                pred_tags: list[str] = []
                for t in range(pred_ids.shape[1]):
                    if tg[b, t] == -100:
                        continue
                    true_tags.append(IDX2NER[int(tg[b, t])])
                    pred_tags.append(IDX2NER[int(pred_ids[b, t])])
                gold_spans = bio_entity_spans(true_tags)
                pred_spans = bio_entity_spans(pred_tags)
                for s in pred_spans - gold_spans:
                    et, a, c = s
                    if len(fp_list) < n_each:
                        fp_list.append(
                            f"FP: predicted {et} [{a}:{c}] «{surface(toks, a, c)}» "
                            f"(gold tags there: {' '.join(true_tags[a:c])})"
                        )
                for s in gold_spans - pred_spans:
                    et, a, c = s
                    if len(fn_list) < n_each:
                        fn_list.append(
                            f"FN: missed {et} [{a}:{c}] «{surface(toks, a, c)}» "
                            f"(pred tags there: {' '.join(pred_tags[a:c])})"
                        )
                if len(fp_list) >= n_each and len(fn_list) >= n_each:
                    done = True
                    break
            if done:
                break

    print("\n=== NER: false positives (predicted span not in gold) — test set (first matches) ===\n")
    for line in fp_list[:n_each]:
        print(line)
    print("\n=== NER: false negatives (gold span not predicted) ===\n")
    for line in fn_list[:n_each]:
        print(line)
    print()


def main() -> None:
    pos_pt = os.path.join(MODEL_DIR, "bilstm_pos.pt")
    ner_pt = os.path.join(MODEL_DIR, "bilstm_ner.pt")
    if os.path.isfile(pos_pt):
        pos_top_confusions_with_examples(pos_pt, legacy="pos")
    else:
        print(f"Skip POS confusion analysis: missing {pos_pt}")
    if os.path.isfile(ner_pt):
        ner_fp_fn_examples(ner_pt, legacy="ner")
    else:
        print(f"Skip NER error analysis: missing {ner_pt}")


if __name__ == "__main__":
    main()

"""
Part 2.2–2.3 — Train BiLSTM POS / NER, early stopping, save checkpoints (+ config).
Run from assignment root:  python Part_2/part2_train.py
Or:  cd Part_2  then  python part2_train.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PART2_ROOT = Path(__file__).resolve().parent
_P1_STR = str(_PART2_ROOT.parent / "Part_1")
if _P1_STR not in sys.path:
    sys.path.insert(0, _P1_STR)
if str(_PART2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from part1_vocab_tfidf import BILSTM_LR, BILSTM_PATIENCE, BILSTM_WD, EMB_DIR
from part2_checkpoint import make_config, save_checkpoint
from part2_dataset import DATA_DIR, IDX2NER, NER2IDX, NER_TAGS, POS2IDX, POS_TAGS
from part2_models import build_bilstm_tagger

EMB_PATH = os.path.join(EMB_DIR, "embeddings_w2v.npy")
MODEL_DIR = str(_PART2_ROOT / "models")
WORD2IDX_PATH = os.path.join(EMB_DIR, "word2idx.json")
UNK_IDX = 1
PAD_IDX = 0

# NER: heavy O-class imbalance — weighted token CE + CRF; longer horizon before early stop
NER_AUX_CE_WEIGHT = 0.4
NER_MIN_EPOCHS = 12
NER_TRAIN_PATIENCE = 12
NER_LR = 5e-4


def compute_ner_class_weights(train_ner_sents: list[tuple[list[str], list[str], list[str]]]) -> torch.Tensor:
    """Inverse-frequency weights (mean 1.0); cap extremes so rare BIO tags get gradient without exploding."""
    counts = np.zeros(len(NER_TAGS), dtype=np.float64)
    for _toks, _pos, ner in train_ner_sents:
        for tag in ner:
            counts[NER2IDX[tag]] += 1
    total = float(max(counts.sum(), 1.0))
    n_classes = len(NER_TAGS)
    w = total / (n_classes * np.maximum(counts, 1.0))
    med = float(np.median(w))
    w = np.clip(w, med * 0.2, med * 30.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def assert_embeddings() -> None:
    assert os.path.isfile(EMB_PATH), f"Missing {EMB_PATH} — run Part 1 Skip-gram first."


def load_word2idx() -> dict[str, int]:
    with open(WORD2IDX_PATH, encoding="utf-8") as f:
        return json.load(f)


def read_pos_conll(path: Path) -> list[tuple[list[str], list[str]]]:
    sents: list[tuple[list[str], list[str]]] = []
    toks: list[str] = []
    pos: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if toks:
                    sents.append((toks, pos))
                    toks, pos = [], []
            else:
                a, b = line.split("\t", 1)
                toks.append(a)
                pos.append(b)
    if toks:
        sents.append((toks, pos))
    return sents


def read_ner_conll(path: Path) -> list[tuple[list[str], list[str], list[str]]]:
    sents: list[tuple[list[str], list[str], list[str]]] = []
    toks: list[str] = []
    pos: list[str] = []
    ner: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if toks:
                    sents.append((toks, pos, ner))
                    toks, pos, ner = [], [], []
            else:
                a, b, c = line.split("\t", 2)
                toks.append(a)
                pos.append(b)
                ner.append(c.strip())
    if toks:
        sents.append((toks, pos, ner))
    return sents


class PosDataset(Dataset):
    def __init__(self, sentences: list[tuple[list[str], list[str]]], w2i: dict[str, int]) -> None:
        self.sentences = sentences
        self.w2i = w2i

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        toks, pos = self.sentences[idx]
        ids = [self.w2i.get(t, UNK_IDX) for t in toks]
        y = [POS2IDX[p] for p in pos]
        return ids, y


class NerDataset(Dataset):
    def __init__(self, sentences: list[tuple[list[str], list[str], list[str]]], w2i: dict[str, int]) -> None:
        self.sentences = sentences
        self.w2i = w2i

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        toks, _pos, ner = self.sentences[idx]
        ids = [self.w2i.get(t, UNK_IDX) for t in toks]
        y = [NER2IDX[n] for n in ner]
        return ids, y


def pos_collate(batch):
    max_len = max(len(x[0]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), PAD_IDX, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (ids, y) in enumerate(batch):
        L = len(ids)
        lengths[i] = L
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        targets[i, :L] = torch.tensor(y, dtype=torch.long)
    return input_ids, lengths, targets


def ner_collate(batch):
    max_len = max(len(x[0]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), PAD_IDX, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (ids, y) in enumerate(batch):
        L = len(ids)
        lengths[i] = L
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        targets[i, :L] = torch.tensor(y, dtype=torch.long)
    mask = (input_ids != PAD_IDX).float()
    return input_ids, lengths, targets, mask


def train_pos(
    frozen: bool,
    out_name: str = "bilstm_pos.pt",
    bidirectional: bool = True,
    use_dropout: bool = True,
    random_embedding: bool = False,
    plot_curves_path: str | None = None,
) -> dict[str, float]:
    assert_embeddings()
    w2i = load_word2idx()
    weights = np.load(EMB_PATH)
    train_s = read_pos_conll(Path(DATA_DIR) / "pos_train.conll")
    val_s = read_pos_conll(Path(DATA_DIR) / "pos_val.conll")

    train_ds = PosDataset(train_s, w2i)
    val_ds = PosDataset(val_s, w2i)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=pos_collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=pos_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_np = None if random_embedding else weights
    model = build_bilstm_tagger(
        len(POS_TAGS),
        w_np,
        freeze=frozen,
        bidirectional=bidirectional,
        use_dropout=use_dropout,
        random_embedding=random_embedding,
        use_crf=False,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=BILSTM_LR, weight_decay=BILSTM_WD)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = -1.0
    patience = 0
    history_loss: list[float] = []
    history_f1: list[float] = []

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for input_ids, lengths, targets in tqdm(train_loader, desc=f"POS ep{epoch+1}"):
            input_ids, lengths, targets = input_ids.to(device), lengths.to(device), targets.to(device)
            opt.zero_grad()
            logits = model(input_ids, lengths)
            loss = crit(logits.reshape(-1, len(POS_TAGS)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(len(train_loader), 1)
        history_loss.append(avg_loss)

        model.eval()
        ys: list[int] = []
        pr: list[int] = []
        with torch.no_grad():
            for input_ids, lengths, targets in val_loader:
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
        macro_f1 = f1_score(ys, pr, average="macro", labels=list(range(len(POS_TAGS))), zero_division=0)
        history_f1.append(float(macro_f1))

        print(f"epoch {epoch+1} loss={avg_loss:.4f} val_macro_f1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            out_path = os.path.join(MODEL_DIR, out_name)
            cfg = make_config(
                "pos",
                len(POS_TAGS),
                frozen,
                bidirectional,
                use_dropout,
                random_embedding,
                False,
            )
            save_checkpoint(out_path, model, cfg)
        else:
            patience += 1
            if patience >= BILSTM_PATIENCE:
                print("Early stopping POS")
                break

    pcurve = plot_curves_path or os.path.join(MODEL_DIR, "pos_training_curves.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history_loss)
    ax[0].set_title("POS train loss")
    ax[1].plot(history_f1)
    ax[1].set_title("POS val macro-F1")
    plt.tight_layout()
    plt.savefig(pcurve, dpi=120)
    plt.close()
    return {"best_val_macro_f1": float(best_f1)}


def train_ner(
    frozen: bool,
    use_crf: bool,
    out_name: str = "bilstm_ner.pt",
    bidirectional: bool = True,
    use_dropout: bool = True,
    random_embedding: bool = False,
    plot_curves_path: str | None = None,
) -> dict[str, float]:
    assert_embeddings()
    from seqeval.metrics import classification_report

    w2i = load_word2idx()
    weights = np.load(EMB_PATH)
    train_s = read_ner_conll(Path(DATA_DIR) / "ner_train.conll")
    val_s = read_ner_conll(Path(DATA_DIR) / "ner_val.conll")

    train_ds = NerDataset(train_s, w2i)
    val_ds = NerDataset(val_s, w2i)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=ner_collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=ner_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_np = None if random_embedding else weights
    class_w = compute_ner_class_weights(train_s).to(device)
    model = build_bilstm_tagger(
        len(NER_TAGS),
        w_np,
        freeze=frozen,
        bidirectional=bidirectional,
        use_dropout=use_dropout,
        random_embedding=random_embedding,
        use_crf=use_crf,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=NER_LR, weight_decay=BILSTM_WD)
    crit = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

    best_f1 = -1.0
    patience = 0
    history_loss: list[float] = []
    history_f1: list[float] = []

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"NER ep{epoch+1}"):
            if use_crf:
                input_ids, lengths, targets, mask = batch
                input_ids, lengths, targets, mask = (
                    input_ids.to(device),
                    lengths.to(device),
                    targets.to(device),
                    mask.to(device),
                )
                opt.zero_grad()
                emissions = model.encode(input_ids, lengths)
                loss_crf = model.crf(emissions, targets, mask)
                loss_ce = F.cross_entropy(
                    emissions.reshape(-1, len(NER_TAGS)),
                    targets.reshape(-1),
                    weight=class_w,
                    ignore_index=-100,
                )
                loss = loss_crf + NER_AUX_CE_WEIGHT * loss_ce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total_loss += float(loss.item())
            else:
                input_ids, lengths, targets, _mask = batch
                input_ids, lengths, targets = input_ids.to(device), lengths.to(device), targets.to(device)
                opt.zero_grad()
                logits = model(input_ids, lengths)
                loss = crit(logits.reshape(-1, len(NER_TAGS)), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total_loss += float(loss.item())
        avg_loss = total_loss / max(len(train_loader), 1)
        history_loss.append(avg_loss)

        model.eval()
        y_true_seq: list[list[str]] = []
        y_pred_seq: list[list[str]] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, lengths, targets, mask = batch
                input_ids, lengths, targets, mask = (
                    input_ids.to(device),
                    lengths.to(device),
                    targets.to(device),
                    mask.to(device),
                )
                if use_crf:
                    pred_ids = model(input_ids, lengths, tags=None, mask=mask)
                else:
                    pred_ids = model(input_ids, lengths).argmax(-1)
                pred_ids = pred_ids.cpu().numpy()
                tg = targets.cpu().numpy()
                for b in range(pred_ids.shape[0]):
                    true_tags = []
                    pred_tags = []
                    for t in range(pred_ids.shape[1]):
                        if tg[b, t] == -100:
                            continue
                        true_tags.append(IDX2NER[int(tg[b, t])])
                        pred_tags.append(IDX2NER[int(pred_ids[b, t])])
                    y_true_seq.append(true_tags)
                    y_pred_seq.append(pred_tags)
        rep = classification_report(y_true_seq, y_pred_seq, output_dict=True, zero_division=0)
        ent_f1 = float(rep.get("macro avg", {}).get("f1-score", 0.0))

        history_f1.append(ent_f1)
        print(f"epoch {epoch+1} loss={avg_loss:.4f} val_entity_macro_f1={ent_f1:.4f}")

        if ent_f1 > best_f1:
            best_f1 = ent_f1
            patience = 0
            out_path = os.path.join(MODEL_DIR, out_name)
            cfg = make_config(
                "ner",
                len(NER_TAGS),
                frozen,
                bidirectional,
                use_dropout,
                random_embedding,
                use_crf,
            )
            save_checkpoint(out_path, model, cfg)
        else:
            patience += 1
        if epoch + 1 >= NER_MIN_EPOCHS and patience >= NER_TRAIN_PATIENCE:
            print("Early stopping NER")
            break

    pcurve = plot_curves_path or os.path.join(MODEL_DIR, "ner_training_curves.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history_loss)
    ax[0].set_title("NER train loss")
    ax[1].plot(history_f1)
    ax[1].set_title("NER val macro-F1")
    plt.tight_layout()
    plt.savefig(pcurve, dpi=120)
    plt.close()
    return {"best_val_macro_f1": float(best_f1)}


def run_finetune() -> None:
    """Fine-tune embeddings: POS then NER+CRF."""
    assert os.path.isfile(EMB_PATH), f"Missing {EMB_PATH}"
    assert_embeddings()
    print("=== Fine-tune POS (trainable embeddings) ===")
    train_pos(frozen=False, out_name="bilstm_pos_ft.pt", plot_curves_path=os.path.join(MODEL_DIR, "pos_training_curves_ft.png"))
    print("=== Fine-tune NER + CRF ===")
    train_ner(frozen=False, use_crf=True, out_name="bilstm_ner_ft.pt", plot_curves_path=os.path.join(MODEL_DIR, "ner_training_curves_ft.png"))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description="Train BiLSTM POS / NER taggers")
    ap.add_argument(
        "--finetune",
        action="store_true",
        help="Only fine-tune embeddings: save bilstm_pos_ft.pt and bilstm_ner_ft.pt",
    )
    args = ap.parse_args()

    assert os.path.isfile(EMB_PATH), f"Missing {EMB_PATH}"
    assert_embeddings()
    if args.finetune:
        run_finetune()
        return
    print("=== Train POS (frozen embeddings) ===")
    train_pos(frozen=True, out_name="bilstm_pos.pt")
    print("=== Train NER + CRF (frozen) ===")
    train_ner(frozen=True, use_crf=True, out_name="bilstm_ner.pt")


if __name__ == "__main__":
    main()

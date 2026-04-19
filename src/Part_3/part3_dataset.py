import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

_PART3_ROOT = Path(__file__).resolve().parent
for _p in (_PART3_ROOT.parent / "Part_1", _PART3_ROOT.parent / "Part_2", _PART3_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from part1_vocab_tfidf import load_document_lines, tokenize_whitespace, CLEANED_TXT, WORD2IDX_PATH, METADATA_PATH, UNK_TOKEN

class DocClassificationDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int], max_seq: int = 256, pad_idx: int = 0):
        self.sequences = sequences
        self.labels = labels
        self.max_seq = max_seq
        self.pad_idx = pad_idx

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx][:self.max_seq]
        length = len(seq)
        pad_len = self.max_seq - length
        padded_seq = seq + [self.pad_idx] * pad_len
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(length, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def build_loaders(batch_size: int = 16, max_seq: int = 256):
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(WORD2IDX_PATH, encoding="utf-8") as f:
        word2idx = json.load(f)

    # Classes: 1=Politics, 2=Sports, 3=Economy, 4=International, 5=Health&Society
    # We'll map them 0-4 for PyTorch 
    cat2idx = {
        "Politics": 0,
        "Sports": 1,
        "Economy": 2,
        "International": 3,
        "Health&Society": 4
    }
    
    doc_lines = load_document_lines(Path(CLEANED_TXT))
    
    assert len(doc_lines) == len(meta)
    
    X = []
    y = []
    
    unk_idx = word2idx.get(UNK_TOKEN, 1)
    
    for i, line in enumerate(doc_lines):
        toks = tokenize_whitespace(line)
        ids = [word2idx.get(t, unk_idx) for t in toks]
        X.append(ids)
        y.append(cat2idx[meta[i]["category"]])
        
    X_arr = []
    y_arr = []
    for xx, yy in zip(X, y):
        X_arr.append(xx)
        y_arr.append(yy)

    # Split 70/15/15 stratified
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss1.split(X_arr, y_arr))
    
    X_train = [X_arr[i] for i in train_idx]
    y_train = [y_arr[i] for i in train_idx]
    
    X_temp = [X_arr[i] for i in temp_idx]
    y_temp = [y_arr[i] for i in temp_idx]
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    
    X_val = [X_temp[i] for i in val_idx]
    y_val = [y_temp[i] for i in val_idx]
    
    X_test = [X_temp[i] for i in test_idx]
    y_test = [y_temp[i] for i in test_idx]
    
    train_ds = DocClassificationDataset(X_train, y_train, max_seq=max_seq)
    val_ds = DocClassificationDataset(X_val, y_val, max_seq=max_seq)
    test_ds = DocClassificationDataset(X_test, y_test, max_seq=max_seq)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, list(cat2idx.keys())

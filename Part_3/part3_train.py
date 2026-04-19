import os
import sys
from pathlib import Path

_PART3_ROOT = Path(__file__).resolve().parent
for _p in (_PART3_ROOT.parent / "Part_1", _PART3_ROOT.parent / "Part_2", _PART3_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np

from part3_dataset import build_loaders
from part3_model import TransformerClassifier

def build_scheduler(optimizer, warmup_steps, total_steps, initial_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Hyperparams from prompt
    from part1_vocab_tfidf import VOCAB_SIZE
    TF_HEADS = 4
    TF_DMODEL = 128
    TF_DK = 32
    TF_DFF = 512
    TF_LAYERS = 4
    TF_LR = 5e-4
    TF_WD = 0.01
    TF_WARMUP = 50
    TF_EPOCHS = 20
    TF_MAX_SEQ = 256
    TF_BATCH = 32 # Can adjust

    train_loader, val_loader, test_loader, classes = build_loaders(batch_size=TF_BATCH, max_seq=TF_MAX_SEQ)
    
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=TF_DMODEL,
        heads=TF_HEADS,
        dk=TF_DK,
        d_ff=TF_DFF,
        layers=TF_LAYERS,
        num_classes=5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=TF_LR, weight_decay=TF_WD)
    total_steps = len(train_loader) * TF_EPOCHS
    scheduler = build_scheduler(optimizer, TF_WARMUP, total_steps, TF_LR)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0

    print("Starting training:")
    for epoch in range(TF_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for input_ids, lengths, targets in train_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            padding_mask = (input_ids == 0)
            
            optimizer.zero_grad()
            logits = model(input_ids, padding_mask)
            loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += input_ids.size(0)
            
        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for input_ids, lengths, targets in val_loader:
                input_ids, targets = input_ids.to(device), targets.to(device)
                padding_mask = (input_ids == 0)
                logits = model(input_ids, padding_mask)
                loss = criterion(logits, targets)
                v_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=-1)
                v_corr += (preds == targets).sum().item()
                v_total += input_ids.size(0)
                
        val_loss = v_loss / v_total
        val_acc = v_corr / v_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{TF_EPOCHS} - TrLoss: {train_loss:.4f} TrAcc: {train_acc:.4f} - ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(str(_PART3_ROOT / "models"), exist_ok=True)
            torch.save(model.state_dict(), str(_PART3_ROOT / "models" / "transformer_cls.pt"))
            
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(train_losses, label='Train')
    ax[0].plot(val_losses, label='Val')
    ax[0].set_title('Loss')
    ax[0].legend()
    
    ax[1].plot(train_accs, label='Train')
    ax[1].plot(val_accs, label='Val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    
    plt.savefig(str(_PART3_ROOT / "models" / "training_curves_cls.png"), dpi=120)
    print("Saved training curves.")

if __name__ == "__main__":
    train_classifier()

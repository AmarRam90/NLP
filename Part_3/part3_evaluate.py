import os
import sys
from pathlib import Path

_PART3_ROOT = Path(__file__).resolve().parent
for _p in (_PART3_ROOT.parent / "Part_1", _PART3_ROOT.parent / "Part_2", _PART3_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

from part3_dataset import build_loaders
from part3_model import TransformerClassifier

def load_word2idx():
    from part1_vocab_tfidf import WORD2IDX_PATH
    with open(WORD2IDX_PATH, encoding="utf-8") as f:
        return json.load(f)

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating on device:", device)

    from part1_vocab_tfidf import VOCAB_SIZE
    TF_HEADS = 4
    TF_DMODEL = 128
    TF_DK = 32
    TF_DFF = 512
    TF_LAYERS = 4
    TF_MAX_SEQ = 256
    
    _, _, test_loader, classes = build_loaders(batch_size=16, max_seq=TF_MAX_SEQ)
    
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=TF_DMODEL,
        heads=TF_HEADS,
        dk=TF_DK,
        d_ff=TF_DFF,
        layers=TF_LAYERS,
        num_classes=5
    ).to(device)
    
    model_path = str(_PART3_ROOT / "models" / "transformer_cls.pt")
    if not os.path.exists(model_path):
        print("Model checkpoint not found. Please train first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    y_true = []
    y_pred = []
    
    correct_samples = [] # tuples of (input_ids, lengths, weights)

    with torch.no_grad():
        for input_ids, lengths, targets in test_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            padding_mask = (input_ids == 0)
            
            logits, all_weights = model(input_ids, padding_mask, return_attn=True)
            preds = logits.argmax(dim=-1)
            
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            
            final_layer_weights = all_weights[-1] # (B, heads, T+1, T+1)
            
            for i in range(input_ids.size(0)):
                if preds[i] == targets[i] and len(correct_samples) < 5:
                    correct_samples.append({
                        'input_ids': input_ids[i].cpu().numpy(),
                        'weights': final_layer_weights[i].cpu().numpy(), # (heads, T+1, T+1)
                        'length': lengths[i].item()
                    })

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    
    rep = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:\n", rep)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Transformer (Document Classification)')
    plt.tight_layout()
    plt.savefig(str(_PART3_ROOT / "models" / "transformer_confusion_matrix.png"), dpi=120)
    plt.close()
    
    # ----------------------------------------------------
    # Attention Heatmaps
    idx2word = {int(v): k for k, v in load_word2idx().items()}
    # 0 is PAD_TOKEN, 1 is UNK_TOKEN which is default in word2idx
    
    for c_idx, sample in enumerate(correct_samples[:3]):
        # Extract first 30 tokens max
        seq_len = min(30, sample['length'])
        token_ids = sample['input_ids'][:seq_len]
        tokens = ["[CLS]"] + [idx2word.get(idx, "<UNK>") for idx in token_ids]
        
        # sample['weights'] is shape (heads, T+1, T+1)
        # T+1 because of CLS string
        
        attn_weights = sample['weights'][:, :seq_len+1, :seq_len+1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for h in range(2):
            w = attn_weights[h]
            sns.heatmap(w, cmap='Blues', ax=axes[h], xticklabels=tokens, yticklabels=tokens)
            axes[h].set_title(f"Article {c_idx+1} \u2014 Head {h+1} Attention Weights")
            
        plt.tight_layout()
        plt.savefig(str(_PART3_ROOT / "models" / f"attention_article_{c_idx+1}.png"), dpi=120)
        plt.close()
        
    print("Evaluation complete. Generated confusion matrix and attention heatmaps.")

if __name__ == "__main__":
    run_evaluation()

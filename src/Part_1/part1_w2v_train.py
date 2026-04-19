"""
Part 1.4 — Skip-gram Word2Vec with negative sampling (CS-4063).
No Gensim; PyTorch only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from part1_vocab_tfidf import (
    CLEANED_TXT,
    EMB_DIR,
    NOISE_TABLE_SIZE,
    VOCAB_SIZE,
    W2V_BATCH,
    W2V_DIM,
    W2V_EPOCHS,
    W2V_LR,
    W2V_NEG_SAMPLES,
    W2V_WINDOW,
    build_vocab_indices,
    docs_to_ids,
    ensure_utf8_stdout,
    load_document_lines,
    setup_working_directory,
    tokenize_whitespace,
)

LOG_EVERY = 500


class SkipGram(nn.Module):
    """Skip-gram with negative sampling (sigmoid / BCE on logits)."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, dim)
        self.context_embeddings = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center_embeddings.weight, -0.5 / dim, 0.5 / dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5 / dim, 0.5 / dim)

    def forward(self, center_ids: torch.Tensor, context_ids: torch.Tensor, neg_ids: torch.Tensor) -> torch.Tensor:
        # Input:  center_ids (B,), context_ids (B,), neg_ids (B, K)
        # Output: scalar loss
        v = self.center_embeddings(center_ids)
        u_pos = self.context_embeddings(context_ids)
        u_neg = self.context_embeddings(neg_ids)
        pos_logits = (v * u_pos).sum(dim=-1)
        neg_logits = (u_neg * v.unsqueeze(1)).sum(dim=-1)
        logits = torch.cat([pos_logits, neg_logits.reshape(-1)], dim=0)
        targets = torch.cat(
            [
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits.reshape(-1)),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(logits, targets)


def build_noise_table(unigram: np.ndarray, table_size: int) -> np.ndarray:
    """P_n(w) ∝ f(w)^(3/4); table holds vocab indices."""
    f = unigram.astype(np.float64).copy()
    f[0] = 0.0
    weights = np.power(f, 0.75)
    s = weights.sum()
    if s <= 0:
        raise ValueError("unigram sum is zero")
    probs = weights / s
    rng = np.random.default_rng(42)
    table = rng.choice(np.arange(len(f), dtype=np.int64), size=table_size, p=probs)
    return table


class SkipGramDataset(Dataset):
    def __init__(
        self,
        doc_ids: list[list[int]],
        window: int,
        noise_table: np.ndarray,
        neg_k: int,
    ) -> None:
        self.pairs: list[tuple[int, int]] = []
        for ids in doc_ids:
            L = len(ids)
            for i in range(L):
                lo = max(0, i - window)
                hi = min(L, i + window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    self.pairs.append((ids[i], ids[j]))
        self.noise_table = noise_table
        self.neg_k = neg_k
        self.rng = np.random.default_rng(123)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        c, o = self.pairs[idx]
        negs: list[int] = []
        while len(negs) < self.neg_k:
            n = int(self.noise_table[self.rng.integers(0, len(self.noise_table))])
            if n != c and n != o:
                negs.append(n)
        return c, o, np.array(negs, dtype=np.int64)


def collate_batch(batch: list[tuple[int, int, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negs = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.long)
    return centers, contexts, negs


def train_skipgram(
    corpus_name: str,
    out_npy_path: str,
    dim: int,
    loss_plot_path: str,
) -> None:
    ensure_utf8_stdout()
    root = setup_working_directory()
    corpus_path = root / corpus_name
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)

    print(f"=== Skip-gram training: corpus={corpus_name} dim={dim} ===")
    doc_lines = load_document_lines(corpus_path)
    tokenized = [tokenize_whitespace(L) for L in doc_lines]
    word2idx, _idx2word, unigram = build_vocab_indices(tokenized)
    doc_ids = docs_to_ids(tokenized, word2idx)

    noise_table = build_noise_table(unigram, NOISE_TABLE_SIZE)
    ds = SkipGramDataset(doc_ids, W2V_WINDOW, noise_table, W2V_NEG_SAMPLES)
    loader = DataLoader(
        ds,
        batch_size=W2V_BATCH,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkipGram(VOCAB_SIZE, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=W2V_LR)

    global_step = 0
    step_losses: list[tuple[int, float]] = []

    for epoch in range(W2V_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            centers, contexts, negs = (t.to(device) for t in batch)
            opt.zero_grad()
            loss = model(centers, contexts, negs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            global_step += 1
            lv = float(loss.detach().cpu())
            epoch_loss += lv
            n_batches += 1
            step_losses.append((global_step, lv))
            if global_step % LOG_EVERY == 0:
                print(f"step {global_step}  BCE loss {lv:.5f}  (epoch {epoch + 1}/{W2V_EPOCHS})")

        print(f"Epoch {epoch + 1}/{W2V_EPOCHS} mean batch loss: {epoch_loss / max(n_batches, 1):.5f}")

    V = model.center_embeddings.weight
    U = model.context_embeddings.weight
    final = 0.5 * (V.detach().cpu() + U.detach().cpu())
    arr = final.numpy().astype(np.float32)
    np.save(out_npy_path, arr)
    print(f"Saved embeddings -> {out_npy_path} shape={arr.shape}")

    # Loss curve
    if step_losses:
        xs = [s for s, _ in step_losses]
        ys = [y for _, y in step_losses]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, linewidth=0.8)
        plt.xlabel("Training Steps")
        plt.ylabel("BCE Loss")
        plt.title("Skip-gram Training Loss")
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=120)
        plt.close()
        print(f"Saved loss plot -> {loss_plot_path}")


def main() -> None:
    setup_working_directory()
    os.makedirs(EMB_DIR, exist_ok=True)
    # C3: cleaned, dim=100 (primary artifact for Part 2)
    train_skipgram(
        CLEANED_TXT,
        os.path.join(EMB_DIR, "embeddings_w2v.npy"),
        W2V_DIM,
        os.path.join(EMB_DIR, "skipgram_loss_cleaned_d100.png"),
    )


if __name__ == "__main__":
    main()

"""
Part 1.3 — PPMI co-occurrence matrix + t-SNE plot (CS-4063).
Uses scipy.sparse for co-occurrence construction; dense PPMI saved to disk.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.manifold import TSNE

from part1_vocab_tfidf import (
    CLEANED_TXT,
    EMB_DIR,
    VOCAB_SIZE,
    W2V_WINDOW,
    build_vocab_indices,
    docs_to_ids,
    ensure_utf8_stdout,
    load_document_lines,
    setup_working_directory,
    tokenize_whitespace,
)

PPMI_PATH = os.path.join(EMB_DIR, "ppmi_matrix.npy")

# t-SNE (assignment)
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_TOP_N = 200
PPMI_EPS = 1e-12

# Five legend categories for t-SNE (keyword buckets for Urdu tokens)
TSNE_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "politics": ("حکومت", "وزیر", "انتخاب", "پارلیمنٹ", "سیاست", "جماعت", "قانون"),
    "sports": ("کرکٹ", "میچ", "کھلاڑی", "ورلڈ", "کپ", "ٹیم", "فٹبال"),
    "geography": ("پاکستان", "انڈیا", "افغانستان", "چین", "شہر", "صوب", "دنیا"),
    "economy": ("معیشت", "بینک", "ڈالر", "تجارت", "بجٹ", "مارکیٹ"),
    "religion": ("مسلم", "اسلام", "عیسائی", "ہندو", "مذہب", "مسجد", "چرچ"),
}


def assign_tsne_category(token: str) -> str:
    scores = {k: 0 for k in TSNE_CATEGORY_KEYWORDS}
    for cat, keys in TSNE_CATEGORY_KEYWORDS.items():
        for kw in keys:
            if kw in token:
                scores[cat] += 1
    best = max(scores, key=lambda c: scores[c])
    if scores[best] == 0:
        return "other"
    return best


def build_cooccurrence_coo(doc_ids: list[list[int]], vocab_size: int, window: int) -> csr_matrix:
    """Ordered pairs (center, context) within symmetric window; excludes i==j."""
    rows: list[int] = []
    cols: list[int] = []
    for ids in doc_ids:
        L = len(ids)
        for i in range(L):
            wi = ids[i]
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                wj = ids[j]
                rows.append(wi)
                cols.append(wj)
    data = np.ones(len(rows), dtype=np.float64)
    C = coo_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
    C = C.tocsr()
    C.sum_duplicates()
    return C


def ppmi_from_cooccurrence(
    C_csr,
    unigram: np.ndarray,
    eps: float = PPMI_EPS,
) -> np.ndarray:
    """
    PPMI(w1,w2) = max(0, log2( P(w1,w2) / (P(w1)*P(w2)) ) ) with epsilon in ratio.
    P(w1,w2) from joint co-occurrence; P(w) from unigram over full corpus.
    """
    C = C_csr.toarray().astype(np.float64)
    total_cooc = float(C.sum())
    total_tokens = float(unigram.sum())
    if total_cooc == 0 or total_tokens == 0:
        raise ValueError("empty statistics")
    P_joint = C / total_cooc
    P_w = unigram.astype(np.float64) / total_tokens
    outer = np.outer(P_w, P_w)
    ratio = (P_joint + eps) / (outer + eps)
    ppmi = np.maximum(0.0, np.log2(ratio))
    return ppmi.astype(np.float32)


def run_ppmi_and_tsne() -> None:
    ensure_utf8_stdout()
    root = setup_working_directory()
    corpus = root / CLEANED_TXT
    doc_lines = load_document_lines(corpus)
    tokenized = [tokenize_whitespace(L) for L in doc_lines]
    word2idx, idx2word, unigram = build_vocab_indices(tokenized)
    doc_ids = docs_to_ids(tokenized, word2idx)

    print("=== Part 1.3 — Co-occurrence (sparse COO -> CSR) ===")
    print("Window k =", W2V_WINDOW)
    C_csr = build_cooccurrence_coo(doc_ids, VOCAB_SIZE, W2V_WINDOW)
    print("Co-occ nnz:", C_csr.nnz, "shape:", C_csr.shape)

    print("=== PPMI (dense) ===")
    ppmi = ppmi_from_cooccurrence(C_csr, unigram)
    np.save(PPMI_PATH, ppmi)
    print(f"Saved {PPMI_PATH} shape={ppmi.shape} dtype={ppmi.dtype}")

    print("=== t-SNE on top-200 frequent tokens (PPMI rows) ===")
    # Exclude PAD from frequency ranking
    freq_scores = unigram.copy()
    freq_scores[0] = 0
    top_idx = np.argsort(-freq_scores)[:TSNE_TOP_N]
    X = ppmi[top_idx]
    labels = [assign_tsne_category(idx2word[int(i)]) for i in top_idx]

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_N_ITER,
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    XY = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    categories = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(categories), 1)))
    for ci, cat in enumerate(categories):
        mask = [l == cat for l in labels]
        xs = XY[mask, 0]
        ys = XY[mask, 1]
        plt.scatter(xs, ys, s=12, label=cat, color=colors[ci % len(colors)])
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title("t-SNE of Top-200 Urdu Tokens (PPMI Vectors)")
    plt.legend(title="Category")
    plt.tight_layout()
    fig_path = os.path.join(EMB_DIR, "tsne_ppmi_top200.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure {fig_path}")
    plt.close()
    print("=== Part 1.3 done ===")


if __name__ == "__main__":
    run_ppmi_and_tsne()

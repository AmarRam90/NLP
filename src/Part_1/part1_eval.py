"""
Part 1.3 / 1.5 — PPMI neighbours, Word2Vec neighbours, analogies, four-condition comparison + MRR.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from part1_vocab_tfidf import EMB_DIR, ensure_utf8_stdout, setup_working_directory

PPMI_PATH = os.path.join(EMB_DIR, "ppmi_matrix.npy")
W2V_CLEAN = os.path.join(EMB_DIR, "embeddings_w2v.npy")
W2V_RAW = os.path.join(EMB_DIR, "embeddings_w2v_raw.npy")
W2V_D200 = os.path.join(EMB_DIR, "embeddings_w2v_dim200.npy")
WORD2IDX_PATH = os.path.join(EMB_DIR, "word2idx.json")
MRR_PAIRS_PATH = os.path.join(os.path.dirname(EMB_DIR), "word_pairs_mrr.json")

# Assignment 2.5 queries (Urdu script)
EVAL_QUERIES_25 = [
    "پاکستان",
    "حکومت",
    "عدالت",
    "معیشت",
    "فوج",
    "صحت",
    "تعلیم",
    "آبادی",
]

# ≥10 words for PPMI neighbour report (section 2.3)
PPMI_QUERY_WORDS = [
    "پاکستان",
    "حکومت",
    "کرکٹ",
    "میچ",
    "معیشت",
    "بینک",
    "امریکہ",
    "فلسطین",
    "صحت",
    "تعلیم",
    "ورلڈ",
    "عدالت",
]


def load_word2idx() -> dict[str, int]:
    with open(WORD2IDX_PATH, encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def top_neighbors(
    vectors: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    query: str,
    k: int = 10,
    exclude_self: bool = True,
) -> list[tuple[str, float]]:
    if query not in word2idx:
        return [(f"<OOV:{query}>", float("nan"))]
    qi = word2idx[query]
    Vn = normalize_rows(vectors)
    nv = Vn[qi]
    sims = Vn @ nv
    sims = sims.copy()
    if exclude_self:
        sims[qi] = -1e9
    order = np.argsort(-sims)
    out: list[tuple[str, float]] = []
    for j in order[:k]:
        out.append((idx2word.get(int(j), str(j)), float(sims[j])))
    return out


def report_neighbors_table(
    name: str,
    vectors: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    queries: list[str],
    k: int,
) -> pd.DataFrame:
    rows = []
    for q in queries:
        nbs = top_neighbors(vectors, word2idx, idx2word, q, k=k + 1)
        for rank, (tok, sc) in enumerate(nbs[:k], start=1):
            rows.append({"condition": name, "query": q, "rank": rank, "neighbor": tok, "cosine": sc})
    return pd.DataFrame(rows)


def analogy_top3(
    vectors: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    a: str,
    b: str,
    c: str,
) -> tuple[list[tuple[str, float]], str]:
    """candidate = argmax cos(b - a + c, v) excluding a,b,c."""
    for w in (a, b, c):
        if w not in word2idx:
            return [], f"missing: {w}"
    ia, ib, ic = word2idx[a], word2idx[b], word2idx[c]
    va, vb, vc = vectors[ia], vectors[ib], vectors[ic]
    target = vb - va + vc
    target = target / (np.linalg.norm(target) + 1e-12)
    Vn = normalize_rows(vectors)
    sims = Vn @ target
    sims[ia] = -1e9
    sims[ib] = -1e9
    sims[ic] = -1e9
    order = np.argsort(-sims)
    top3 = [(idx2word.get(int(j), str(j)), float(sims[j])) for j in order[:3]]
    return top3, "ok"


def mrr_score(
    vectors: np.ndarray,
    word2idx: dict[str, int],
    pairs: list[tuple[str, str]],
) -> float:
    """Mean reciprocal rank of correct neighbour in full-vocab cosine ranking."""
    ranks = []
    Vn = normalize_rows(vectors)
    for q, gold in pairs:
        if q not in word2idx or gold not in word2idx:
            ranks.append(0.0)
            continue
        qi, gi = word2idx[q], word2idx[gold]
        v = Vn[qi]
        sims = Vn @ v
        sims[qi] = -1e9
        order = np.argsort(-sims)
        pos = np.where(order == gi)[0]
        if len(pos) == 0:
            ranks.append(0.0)
        else:
            rank = int(pos[0]) + 1
            ranks.append(1.0 / rank)
    return float(np.mean(ranks)) if ranks else 0.0


def default_analogy_tests() -> list[tuple[str, str, str]]:
    """≥10 analogy tuples (Urdu)."""
    return [
        ("پاکستان", "کرکٹ", "انڈیا"),
        ("حکومت", "وزیر", "عدالت"),
        ("میچ", "کھلاڑی", "ورلڈ"),
        ("بینک", "روپیہ", "تجارت"),
        ("صحت", "ہسپتال", "تعلیم"),
        ("امریکہ", "چین", "روس"),
        ("فوج", "جنگ", "امن"),
        ("کرکٹ", "میچ", "فٹبال"),
        ("پارلیمنٹ", "قانون", "عدالت"),
        ("معیشت", "بجٹ", "ٹیکس"),
        ("پاکستان", "اسلام", "انڈیا"),
        ("کھلاڑی", "ٹیم", "کپ"),
    ]


def run_all_reports() -> None:
    ensure_utf8_stdout()
    setup_working_directory()
    word2idx = load_word2idx()
    idx2word = {int(i): w for w, i in word2idx.items()}

    print("=== PPMI: top-5 cosine neighbours (≥10 query words) ===")
    ppmi = np.load(PPMI_PATH)
    rows_ppmi = []
    seen_q = []
    for q in PPMI_QUERY_WORDS:
        if q in seen_q:
            continue
        seen_q.append(q)
        if q not in word2idx:
            print(f"  (skip OOV) {q}")
            continue
        nbs = top_neighbors(ppmi, word2idx, idx2word, q, k=6, exclude_self=True)[:5]
        for rank, (t, s) in enumerate(nbs, start=1):
            rows_ppmi.append({"query": q, "rank": rank, "neighbor": t, "cosine": s})
    df_ppmi = pd.DataFrame(rows_ppmi)
    print(df_ppmi.to_string(index=False))

    print("\n=== Word2Vec (cleaned, d=100): top-10 neighbours (assignment queries) ===")
    w2v = np.load(W2V_CLEAN)
    df_w2v = report_neighbors_table("C3_skipgram_clean_d100", w2v, word2idx, idx2word, EVAL_QUERIES_25, 10)
    print(df_w2v.to_string(index=False))

    print("\n=== Analogy tests (top-3 + score) ===")
    analogies = default_analogy_tests()
    for a, b, c in analogies:
        top3, status = analogy_top3(w2v, word2idx, idx2word, a, b, c)
        if status != "ok":
            print(f"{a} : {b} :: {c} : [{status}]")
            continue
        tstr = ", ".join([f"{t} ({sc:.3f})" for t, sc in top3])
        print(f"{a} : {b} :: {c} : [{tstr}]")

    print("\n=== Four-condition comparison (top-5 for 5 queries; MRR on 20 pairs) ===")
    if not os.path.isfile(MRR_PAIRS_PATH):
        raise FileNotFoundError(MRR_PAIRS_PATH)
    with open(MRR_PAIRS_PATH, encoding="utf-8") as f:
        mrr_pairs_raw = json.load(f)
    pairs: list[tuple[str, str]] = [(p["query"], p["gold"]) for p in mrr_pairs_raw]

    five_queries = ["پاکستان", "حکومت", "کرکٹ", "معیشت", "صحت"]
    conditions: list[tuple[str, str | None]] = [
        ("C1_PPMI", PPMI_PATH),
        ("C2_skipgram_raw_d100", W2V_RAW if os.path.isfile(W2V_RAW) else None),
        ("C3_skipgram_clean_d100", W2V_CLEAN if os.path.isfile(W2V_CLEAN) else None),
        ("C4_skipgram_clean_d200", W2V_D200 if os.path.isfile(W2V_D200) else None),
    ]

    summary_rows = []
    for name, path in conditions:
        if path is None or not os.path.isfile(path):
            summary_rows.append({"condition": name, "MRR": float("nan"), "notes": "embedding file missing"})
            print(f"\n--- {name} (missing file) ---")
            continue
        vecs = np.load(path)
        mrr = mrr_score(vecs, word2idx, pairs)
        summary_rows.append({"condition": name, "MRR": mrr, "notes": ""})
        print(f"\n--- {name} MRR={mrr:.4f} ---")
        df5 = report_neighbors_table(name, vecs, word2idx, idx2word, five_queries, 5)
        print(df5.to_string(index=False))

    print("\n=== Comparison table ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    run_all_reports()

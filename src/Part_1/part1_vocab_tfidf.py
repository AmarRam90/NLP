"""
Part 1 — vocabulary + TF-IDF (CS-4063).
Run from assignment root:  python Part_1/part1_vocab_tfidf.py
Or:  cd Part_1  then  python part1_vocab_tfidf.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# --- Master constants (source of truth) ---
VOCAB_SIZE = 10_000
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

# Word2Vec / PPMI window (shared k)
W2V_WINDOW = 5
W2V_DIM = 100
W2V_NEG_SAMPLES = 10
W2V_BATCH = 512
W2V_EPOCHS = 5
W2V_LR = 0.001
# Noise distribution table (negative sampling)
NOISE_TABLE_SIZE = 10**7

_PART1_ROOT = Path(__file__).resolve().parent
# Corpus + artifacts live under Part_1/ (absolute paths so Part_2 can import constants without chdir)
CLEANED_TXT = str(_PART1_ROOT / "cleaned.txt")
RAW_TXT = str(_PART1_ROOT / "raw.txt")
EMB_DIR = str(_PART1_ROOT / "embeddings")

# BiLSTM (Part 2)
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.5
BILSTM_LR = 1e-3
BILSTM_WD = 1e-4
BILSTM_PATIENCE = 5
WORD2IDX_PATH = os.path.join(EMB_DIR, "word2idx.json")
TFIDF_PATH = os.path.join(EMB_DIR, "tfidf_matrix.npy")
METADATA_PATH = str(_PART1_ROOT / "Metadata.json")

# Topic labels (align with assignment Part 3 five-way split)
TOPIC_POLITICS = "Politics"
TOPIC_SPORTS = "Sports"
TOPIC_ECONOMY = "Economy"
TOPIC_INTERNATIONAL = "International"
TOPIC_HEALTH_SOCIETY = "Health&Society"
TOPIC_NAMES = (
    TOPIC_POLITICS,
    TOPIC_SPORTS,
    TOPIC_ECONOMY,
    TOPIC_INTERNATIONAL,
    TOPIC_HEALTH_SOCIETY,
)

# Urdu / mixed keyword hints per category (scores are summed per document line)
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    TOPIC_POLITICS: (
        "حکومت",
        "وزیر",
        "وزارت",
        "انتخاب",
        "پارلیمنٹ",
        "سیاست",
        "عوامی",
        "قانون",
        "عدالت",
        "صدر",
        "وزیراعظم",
        "جماعت",
        "حزب",
        "اپوزیشن",
    ),
    TOPIC_SPORTS: (
        "کرکٹ",
        "میچ",
        "کھلاڑی",
        "ورلڈ",
        "کپ",
        "ٹیم",
        "فٹبال",
        "ہاکی",
        "اسپورٹس",
        "بیٹنگ",
        "بال",
        "اسٹیڈیم",
        "کوچ",
    ),
    TOPIC_ECONOMY: (
        "معیشت",
        "بینک",
        "روپیہ",
        "ڈالر",
        "تجارت",
        "برآمد",
        "درآمد",
        "بجٹ",
        "کاروبار",
        "مارکیٹ",
        "شرح",
        "سرمایہ",
    ),
    TOPIC_INTERNATIONAL: (
        "امریکہ",
        "چین",
        "روس",
        "یورپ",
        "اقوام",
        "متحدہ",
        "بین",
        "الاقوامی",
        "سفارت",
        "معاہدہ",
        "جنگ",
        "سرحد",
        "غزہ",
        "اسرائیل",
        "فلسطین",
        "افغانستان",
        "بحر",
        "خلیج",
    ),
    TOPIC_HEALTH_SOCIETY: (
        "صحت",
        "ہسپتال",
        "بیماری",
        "ویکسین",
        "ڈاکٹر",
        "علاج",
        "تعلیم",
        "سیلاب",
        "آبادی",
        "خاندان",
        "خواتین",
        "بچ",
        "اسکول",
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def load_document_lines(corpus_path: Path) -> list[str]:
    """Each non-empty line is one document; skip article header lines."""
    lines: list[str] = []
    header_re = re.compile(r"^\s*Article\s+\d+\s*:\s*$")
    with corpus_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or header_re.match(line):
                continue
            lines.append(line)
    return lines


def tokenize_whitespace(text: str) -> list[str]:
    return text.split()


def classify_topic_line(text: str) -> str:
    """Heuristic single-label assignment for Metadata (Urdu keyword overlap)."""
    scores = {name: 0 for name in TOPIC_NAMES}
    for topic, keys in TOPIC_KEYWORDS.items():
        for k in keys:
            if k in text:
                scores[topic] += 1
    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return TOPIC_INTERNATIONAL
    return best


def build_vocab_indices(tokenized_docs: list[list[str]]) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    """
    word2idx maps PAD->0, UNK->1, then top (VOCAB_SIZE-2) frequent tokens.
    Total embedding/TF-IDF width = VOCAB_SIZE.
    """
    counter: Counter[str] = Counter()
    for toks in tokenized_docs:
        counter.update(toks)

    top_n = VOCAB_SIZE - 2
    most = counter.most_common(top_n)
    word2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, (w, _) in enumerate(most, start=2):
        word2idx[w] = i

    idx2word = {i: w for w, i in word2idx.items()}
    unigram = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for toks in tokenized_docs:
        for t in toks:
            idx = word2idx.get(t, 1)
            unigram[idx] += 1
    return word2idx, idx2word, unigram


def docs_to_ids(tokenized_docs: list[list[str]], word2idx: dict[str, int]) -> list[list[int]]:
    unk = word2idx[UNK_TOKEN]
    return [[word2idx.get(t, unk) for t in toks] for toks in tokenized_docs]


def build_tfidf_sparse(
    doc_ids: list[list[int]],
    vocab_size: int,
) -> tuple[csr_matrix, np.ndarray]:
    """
    TF(w,d) = count/tokens_in_d ; IDF = log(N/(1+df)) ; TF-IDF = TF*IDF.
    Returns CSR float32 and the IDF vector (vocab_size,).
    """
    n_docs = len(doc_ids)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    df = np.zeros(vocab_size, dtype=np.int64)
    for d, ids in enumerate(doc_ids):
        seen = set(ids)
        for j in seen:
            df[j] += 1
        total = len(ids)
        if total == 0:
            continue
        counts: Counter[int] = Counter(ids)
        for j, c in counts.items():
            tf = c / total
            rows.append(d)
            cols.append(j)
            data.append(tf)

    idf = np.log(n_docs / (1.0 + df.astype(np.float64))).astype(np.float32)
    data_scaled: list[float] = []
    for r, c, tf in zip(rows, cols, data, strict=True):
        data_scaled.append(tf * float(idf[c]))

    mat = csr_matrix(
        (np.asarray(data_scaled, dtype=np.float32), (rows, cols)),
        shape=(n_docs, vocab_size),
        dtype=np.float32,
    )
    return mat, idf


def top_tfidf_per_topic(
    tfidf_dense: np.ndarray,
    topics: list[str],
    idx2word: dict[int, str],
    top_k: int,
) -> pd.DataFrame:
    """Mean TF-IDF per term within each topic; report top_k tokens per topic."""
    rows_out: list[dict[str, object]] = []
    topic_names = sorted(set(topics))
    for topic in topic_names:
        mask = np.array([t == topic for t in topics], dtype=bool)
        if not np.any(mask):
            continue
        mean_vec = tfidf_dense[mask].mean(axis=0)
        # skip PAD column (0) for readability
        order = np.argsort(-mean_vec)[1 : top_k + 1]
        for rank, j in enumerate(order, start=1):
            rows_out.append(
                {
                    "topic": topic,
                    "rank": rank,
                    "token": idx2word.get(int(j), str(j)),
                    "mean_tfidf": float(mean_vec[j]),
                }
            )
    return pd.DataFrame(rows_out)


def ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def setup_working_directory() -> Path:
    root = _project_root()
    os.chdir(root)
    os.makedirs(EMB_DIR, exist_ok=True)
    return root


def step1_load_corpus(root: Path) -> tuple[list[str], list[list[str]]]:
    """Load cleaned.txt: one document per non-empty line (excluding Article headers)."""
    corpus = root / CLEANED_TXT
    if not corpus.is_file():
        raise FileNotFoundError(f"Missing {corpus}")
    print("=== Part 1.1 — load & tokenize ===")
    doc_lines = load_document_lines(corpus)
    tokenized = [tokenize_whitespace(L) for L in doc_lines]
    print(f"Documents (lines): {len(doc_lines)}")
    print(f"Total whitespace tokens: {sum(len(t) for t in tokenized)}")
    print("Example (first doc, first 25 tokens):", tokenized[0][:25])
    return doc_lines, tokenized


def step2_vocab_and_word2idx(
    tokenized: list[list[str]],
) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    print("\n=== Vocabulary & word2idx ===")
    word2idx, idx2word, unigram = build_vocab_indices(tokenized)
    print(f"VOCAB_SIZE (indices 0..{VOCAB_SIZE - 1}): PAD=0, UNK=1, content={VOCAB_SIZE - 2}")
    print("Sample word2idx entries:", list(word2idx.items())[:12])
    with open(WORD2IDX_PATH, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    print(f"Saved {WORD2IDX_PATH}")
    print("Unigram total count sum:", int(unigram.sum()))
    return word2idx, idx2word, unigram


def step3_metadata(
    doc_lines: list[str],
) -> list[str]:
    print("\n=== Metadata.json (topic labels for TF-IDF report) ===")
    meta_records = []
    topics: list[str] = []
    for i, line in enumerate(doc_lines):
        cat = classify_topic_line(line)
        topics.append(cat)
        meta_records.append({"doc_id": i, "category": cat, "preview": line[:120]})
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, ensure_ascii=False, indent=2)
    print(f"Saved {METADATA_PATH} ({len(meta_records)} entries)")
    print("Topic distribution:", dict(Counter(topics)))
    return topics


def step4_tfidf_and_report(
    tokenized: list[list[str]],
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    topics: list[str],
) -> None:
    print("\n=== Part 1.2 — TF-IDF (sparse CSR, then dense save) ===")
    doc_ids = docs_to_ids(tokenized, word2idx)
    tfidf_csr, _idf = build_tfidf_sparse(doc_ids, VOCAB_SIZE)
    print("CSR TF-IDF shape:", tfidf_csr.shape, "nnz:", tfidf_csr.nnz)
    tfidf_dense = tfidf_csr.toarray()
    np.save(TFIDF_PATH, tfidf_dense.astype(np.float32))
    print(f"Saved {TFIDF_PATH} shape={tfidf_dense.shape} dtype=float32")

    print("\n=== Top-10 TF-IDF tokens per topic (pandas DataFrame) ===")
    df_top = top_tfidf_per_topic(tfidf_dense, topics, idx2word, top_k=10)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 120)
    print(df_top.to_string(index=False))
    print("\n=== Part 1.1–1.2 complete (vocab + TF-IDF) ===")


def main() -> None:
    ensure_utf8_stdout()
    root = setup_working_directory()
    doc_lines, tokenized = step1_load_corpus(root)
    word2idx, idx2word, _u = step2_vocab_and_word2idx(tokenized)
    topics = step3_metadata(doc_lines)
    step4_tfidf_and_report(tokenized, word2idx, idx2word, topics)


if __name__ == "__main__":
    main()

"""
Part 2.1 — Stratified 500-sentence sample, POS/NER annotation, CoNLL export, plots.
Run from assignment root:  python Part_2/part2_dataset.py
Or:  cd Part_2  then  python part2_dataset.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_PART2_ROOT = Path(__file__).resolve().parent
_P2_STR = str(_PART2_ROOT)
_P1_STR = str(_PART2_ROOT.parent / "Part_1")
if _P1_STR not in sys.path:
    sys.path.insert(0, _P1_STR)
if _P2_STR not in sys.path:
    sys.path.insert(0, _P2_STR)

from part1_vocab_tfidf import CLEANED_TXT, METADATA_PATH, load_document_lines, tokenize_whitespace
from part2_lexicons import (
    gazetteer_phrases,
    merge_frequency_buckets,
    rule_pos,
    tag_sentence_ner,
)

DATA_DIR = str(_PART2_ROOT / "data")

# 12 POS tags (assignment); PROPN = proper noun from gazetteer / capital-like
POS_TAGS = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "CONJ",
    "POST",
    "NUM",
    "PUNC",
    "UNK",
    "PROPN",
]
POS2IDX = {t: i for i, t in enumerate(POS_TAGS)}
IDX2POS = {i: t for t, i in POS2IDX.items()}

NER_TAGS = [
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-MISC",
    "I-MISC",
]
NER2IDX = {t: i for i, t in enumerate(NER_TAGS)}
IDX2NER = {i: t for t, i in NER2IDX.items()}

RNG = np.random.default_rng(42)
THREE_CATS = ("International", "Sports", "Politics")
N_PER_CAT = 100
N_TOTAL = 500


def load_topics_ordered() -> list[str]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    meta.sort(key=lambda x: int(x["doc_id"]))
    return [m["category"] for m in meta]


def sample_line_indices(topics: list[str]) -> list[int]:
    from collections import defaultdict

    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(topics):
        by_cat[t].append(i)
    for c in THREE_CATS:
        if len(by_cat[c]) < N_PER_CAT:
            raise RuntimeError(f"Category {c} has only {len(by_cat[c])} lines; need {N_PER_CAT}")
    picked: set[int] = set()
    for c in THREE_CATS:
        choice = RNG.choice(by_cat[c], size=N_PER_CAT, replace=False)
        picked.update(int(x) for x in choice.tolist())
    pool = [i for i in range(len(topics)) if i not in picked]
    extra = RNG.choice(pool, size=N_TOTAL - len(picked), replace=False)
    picked.update(int(x) for x in extra.tolist())
    return sorted(picked)


def token_is_in_gazetteer(token: str, phrase_toks: set[str]) -> bool:
    return token in phrase_toks


def build_gazetteer_token_set(phrases: list[tuple[tuple[str, ...], str]]) -> set[str]:
    s: set[str] = set()
    for ph, _ in phrases:
        for t in ph:
            s.add(t)
    return s


def annotate_sentence(
    tokens: list[str],
    pos_lex: dict[str, str],
    gaz_tokens: set[str],
    ner_phrases: list[tuple[tuple[str, ...], str]],
) -> tuple[list[str], list[str]]:
    pos_tags: list[str] = []
    for tok in tokens:
        if tok in gaz_tokens:
            pos_tags.append("PROPN")
        else:
            pos_tags.append(rule_pos(tok, pos_lex))
    ner_tags = tag_sentence_ner(tokens, ner_phrases)
    return pos_tags, ner_tags


def build_dataset() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    corpus = Path(CLEANED_TXT)
    lines = load_document_lines(corpus)
    topics = load_topics_ordered()
    assert len(lines) == len(topics)

    counter: Counter[str] = Counter()
    for L in lines:
        counter.update(tokenize_whitespace(L))
    pos_lex = merge_frequency_buckets(counter)

    ner_phrases = gazetteer_phrases()
    gaz_tokens = build_gazetteer_token_set(ner_phrases)

    indices = sample_line_indices(topics)
    sampled_topics = [topics[i] for i in indices]

    sentences: list[tuple[list[str], list[str], list[str], str]] = []
    for idx, top in zip(indices, sampled_topics, strict=True):
        toks = tokenize_whitespace(lines[idx])
        if not toks:
            continue
        pt, nt = annotate_sentence(toks, pos_lex, gaz_tokens, ner_phrases)
        sentences.append((toks, pt, nt, top))

    # 70 / 15 / 15 stratified by coarse topic (avoids rare-class errors in sklearn)
    def coarse_topic(t: str) -> str:
        return t if t in THREE_CATS else "Other"

    y_strat = [coarse_topic(s[3]) for s in sentences]
    idx_all = np.arange(len(sentences))
    train_i, temp_i = train_test_split(
        idx_all, test_size=0.3, stratify=y_strat, random_state=42
    )
    y_temp = [coarse_topic(sentences[int(i)][3]) for i in temp_i]
    val_i, test_i = train_test_split(
        temp_i, test_size=0.5, stratify=y_temp, random_state=42
    )

    splits = {
        "train": train_i.tolist(),
        "val": val_i.tolist(),
        "test": test_i.tolist(),
    }

    def subset(rows: list[int]) -> list[tuple[list[str], list[str], list[str]]]:
        return [(sentences[i][0], sentences[i][1], sentences[i][2]) for i in rows]

    for name, rows in splits.items():
        data = subset(rows)
        pos_path = os.path.join(DATA_DIR, f"pos_{name}.conll")
        ner_path = os.path.join(DATA_DIR, f"ner_{name}.conll")
        with open(pos_path, "w", encoding="utf-8") as fp:
            for toks, pt, nt in data:
                for t, p in zip(toks, pt, strict=True):
                    fp.write(f"{t}\t{p}\n")
                fp.write("\n")
        with open(ner_path, "w", encoding="utf-8") as fn:
            for toks, pt, nt in data:
                for t, p, n in zip(toks, pt, nt, strict=True):
                    fn.write(f"{t}\t{p}\t{n}\n")
                fn.write("\n")

    # Distribution plots
    pos_counts = Counter()
    ner_counts = Counter()
    for toks, pt, nt in subset(list(train_i)):
        pos_counts.update(pt)
        ner_counts.update(nt)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pos_counts.keys(), pos_counts.values(), color="steelblue")
    ax.set_title("POS tag counts (train)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "pos_dist_train.png"), dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    keys = list(ner_counts.keys())
    ax.bar(keys, [ner_counts[k] for k in keys], color="darkorange")
    ax.set_title("NER tag counts (train)")
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "ner_dist_train.png"), dpi=120)
    plt.close()

    df_pos = pd.DataFrame(sorted(pos_counts.items(), key=lambda x: -x[1]), columns=["POS", "count"])
    df_ner = pd.DataFrame(sorted(ner_counts.items(), key=lambda x: -x[1]), columns=["NER", "count"])
    print("POS train distribution:\n", df_pos.to_string(index=False))
    print("\nNER train distribution:\n", df_ner.to_string(index=False))
    print(
        f"\nSplit sizes: train={len(train_i)} val={len(val_i)} test={len(test_i)} | "
        f"three categories ≥100 each: {THREE_CATS}"
    )


if __name__ == "__main__":
    build_dataset()

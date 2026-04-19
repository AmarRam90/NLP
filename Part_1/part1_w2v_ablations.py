"""Train C2 (raw corpus) and C4 (cleaned, W2V_DIM=200). Requires raw.txt present."""
from __future__ import annotations

import os

from part1_vocab_tfidf import CLEANED_TXT, EMB_DIR, RAW_TXT, ensure_utf8_stdout, setup_working_directory
from part1_w2v_train import train_skipgram


def main() -> None:
    ensure_utf8_stdout()
    setup_working_directory()
    os.makedirs(EMB_DIR, exist_ok=True)
    if not os.path.isfile(RAW_TXT):
        raise FileNotFoundError(
            f"{RAW_TXT} missing — copy your raw BBC Urdu dump here (see README)."
        )
    train_skipgram(
        RAW_TXT,
        os.path.join(EMB_DIR, "embeddings_w2v_raw.npy"),
        100,
        os.path.join(EMB_DIR, "skipgram_loss_raw_d100.png"),
    )
    train_skipgram(
        CLEANED_TXT,
        os.path.join(EMB_DIR, "embeddings_w2v_dim200.npy"),
        200,
        os.path.join(EMB_DIR, "skipgram_loss_clean_d200.png"),
    )


if __name__ == "__main__":
    main()

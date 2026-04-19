"""Sequential Part 1 pipeline (after TF-IDF / word2idx already exist)."""
from __future__ import annotations

from part1_eval import run_all_reports
from part1_ppmi_tsne import run_ppmi_and_tsne
from part1_vocab_tfidf import ensure_utf8_stdout, setup_working_directory
from part1_w2v_ablations import main as ablations
from part1_w2v_train import main as w2v_clean


def main() -> None:
    ensure_utf8_stdout()
    setup_working_directory()
    run_ppmi_and_tsne()
    w2v_clean()
    ablations()
    run_all_reports()


if __name__ == "__main__":
    main()

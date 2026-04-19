"""
Part 2 — Ablations A1–A4 + frozen vs fine-tuned summary table.

  python part2_ablations.py              # print tables from checkpoints on disk
  python part2_ablations.py --train      # run all ablation trainings (long)
  python part2_ablations.py --finetune   # only fine-tune POS+NER (trainable embeddings)
  python part2_ablations.py --baseline   # frozen POS + NER+CRF only (same as part2_train.py default)

Use the project venv, e.g. NLP_Assignment_2\\nlpvenv\\Scripts\\activate (Windows).

Checkpoints (after --train):
  A1 uni-LSTM:  models/abl_a1_pos.pt, abl_a1_ner.pt
  A2 no dropout: abl_a2_pos.pt, abl_a2_ner.pt
  A3 random emb: abl_a3_pos.pt, abl_a3_ner.pt
  A4 NER softmax: abl_a4_ner.pt  (POS uses frozen baseline bilstm_pos.pt)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_P1 = str(Path(__file__).resolve().parent.parent / "Part_1")
_P2 = str(Path(__file__).resolve().parent)
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
if _P2 not in sys.path:
    sys.path.insert(0, _P2)

import pandas as pd

from part2_evaluate import ner_test_macro_f1, pos_test_metrics
from part2_train import MODEL_DIR, run_finetune, train_ner, train_pos


def _pos_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)


def _safe_pos_metrics(path: str, legacy: str | None, label: str, err_log: list[str]) -> tuple[float, float]:
    try:
        return pos_test_metrics(path, legacy=legacy)
    except Exception as e:
        err_log.append(f"{label} — POS: {e}")
        return float("nan"), float("nan")


def _safe_ner_f1(path: str, legacy: str | None, label: str, err_log: list[str]) -> float:
    try:
        return ner_test_macro_f1(path, legacy=legacy)
    except Exception as e:
        err_log.append(f"{label} — NER: {e}")
        return float("nan")


def _row(
    label: str,
    pos_ckpt: str | None,
    ner_ckpt: str | None,
    pos_legacy: str | None,
    ner_legacy: str | None,
    err_log: list[str],
    pos_reuse_baseline: bool = False,
) -> dict[str, Any]:
    """One table row; POS may be copied from frozen baseline when pos_reuse_baseline."""
    baseline_pos = _pos_path("bilstm_pos.pt")
    if pos_reuse_baseline:
        if os.path.isfile(baseline_pos):
            pa, pm = _safe_pos_metrics(baseline_pos, "pos", label, err_log)
        else:
            pa, pm = float("nan"), float("nan")
    elif pos_ckpt and os.path.isfile(pos_ckpt):
        pa, pm = _safe_pos_metrics(pos_ckpt, pos_legacy, label, err_log)
    else:
        pa, pm = float("nan"), float("nan")

    if ner_ckpt and os.path.isfile(ner_ckpt):
        nf = _safe_ner_f1(ner_ckpt, ner_legacy, label, err_log)
    else:
        nf = float("nan")

    return {
        "setting": label,
        "POS_acc": pa,
        "POS_macro_F1": pm,
        "NER_macro_F1": nf,
    }


def frozen_vs_finetune_df(err_log: list[str]) -> pd.DataFrame:
    rows = [
        _row(
            "frozen (baseline)",
            _pos_path("bilstm_pos.pt"),
            _pos_path("bilstm_ner.pt"),
            "pos",
            "ner",
            err_log,
        ),
        _row(
            "fine-tuned",
            _pos_path("bilstm_pos_ft.pt"),
            _pos_path("bilstm_ner_ft.pt"),
            None,
            None,
            err_log,
        ),
    ]
    return pd.DataFrame(rows)


def ablations_df(err_log: list[str]) -> pd.DataFrame:
    rows = [
        _row("A1_uni-LSTM", _pos_path("abl_a1_pos.pt"), _pos_path("abl_a1_ner.pt"), None, None, err_log),
        _row("A2_dropout_0", _pos_path("abl_a2_pos.pt"), _pos_path("abl_a2_ner.pt"), None, None, err_log),
        _row("A3_random_emb", _pos_path("abl_a3_pos.pt"), _pos_path("abl_a3_ner.pt"), None, None, err_log),
        _row(
            "A4_NER_softmax_no_CRF",
            None,
            _pos_path("abl_a4_ner.pt"),
            None,
            None,
            err_log,
            pos_reuse_baseline=True,
        ),
    ]
    return pd.DataFrame(rows)


def train_ablations() -> None:
    """Train A1–A4 checkpoints (does not run frozen baseline or fine-tune)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=== A1: uni-directional LSTM ===")
    train_pos(
        frozen=True,
        out_name="abl_a1_pos.pt",
        bidirectional=False,
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a1_pos_curves.png"),
    )
    train_ner(
        frozen=True,
        use_crf=True,
        out_name="abl_a1_ner.pt",
        bidirectional=False,
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a1_ner_curves.png"),
    )
    print("=== A2: no dropout ===")
    train_pos(
        frozen=True,
        out_name="abl_a2_pos.pt",
        use_dropout=False,
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a2_pos_curves.png"),
    )
    train_ner(
        frozen=True,
        use_crf=True,
        out_name="abl_a2_ner.pt",
        use_dropout=False,
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a2_ner_curves.png"),
    )
    print("=== A3: random embedding init ===")
    train_pos(
        frozen=True,
        out_name="abl_a3_pos.pt",
        random_embedding=True,
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a3_pos_curves.png"),
    )
    train_ner(
        frozen=True,
        use_crf=True,
        out_name="abl_a3_ner.pt",
        random_embedding=True,
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a3_ner_curves.png"),
    )
    print("=== A4: NER without CRF (softmax); POS unchanged vs baseline column ===")
    train_ner(
        frozen=True,
        use_crf=False,
        out_name="abl_a4_ner.pt",
        plot_curves_path=os.path.join(MODEL_DIR, "abl_a4_ner_curves.png"),
    )


def print_missing_checkpoint_hints() -> None:
    """Explain NaN cells: list missing files and the command that produces them."""
    ft_pos = _pos_path("bilstm_pos_ft.pt")
    ft_ner = _pos_path("bilstm_ner_ft.pt")
    if not os.path.isfile(ft_pos) or not os.path.isfile(ft_ner):
        print("\n--- Missing fine-tuned checkpoints (fine-tuned row will show NaN) ---")
        if not os.path.isfile(ft_pos):
            print(f"  Expected: {ft_pos}")
        if not os.path.isfile(ft_ner):
            print(f"  Expected: {ft_ner}")
        print("  Generate:  python part2_train.py --finetune\n")

    abl = [
        ("A1", ["abl_a1_pos.pt", "abl_a1_ner.pt"]),
        ("A2", ["abl_a2_pos.pt", "abl_a2_ner.pt"]),
        ("A3", ["abl_a3_pos.pt", "abl_a3_ner.pt"]),
        ("A4", ["abl_a4_ner.pt"]),
    ]
    missing_lines: list[str] = []
    for name, names in abl:
        for n in names:
            p = _pos_path(n)
            if not os.path.isfile(p):
                missing_lines.append(f"  [{name}] {p}")
    if missing_lines:
        print("--- Missing ablation checkpoints (A1–A4 rows may show NaN) ---")
        print("\n".join(missing_lines))
        print("  A4 POS columns reuse frozen `bilstm_pos.pt` (no extra POS checkpoint).\n")
        print("  Generate ablations:  python part2_ablations.py --train\n")

    base_pos = _pos_path("bilstm_pos.pt")
    base_ner = _pos_path("bilstm_ner.pt")
    if not os.path.isfile(base_pos) or not os.path.isfile(base_ner):
        print("\n--- Missing frozen baseline (train Part 2 models first) ---")
        if not os.path.isfile(base_pos):
            print(f"  Expected: {base_pos}")
        if not os.path.isfile(base_ner):
            print(f"  Expected: {base_ner}")
        print("  Generate:  python part2_train.py\n")


def save_tables() -> None:
    err_log: list[str] = []
    ft = frozen_vs_finetune_df(err_log)
    ab = ablations_df(err_log)
    ft_path = os.path.join(MODEL_DIR, "part2_frozen_vs_finetune.csv")
    ab_path = os.path.join(MODEL_DIR, "part2_ablations.csv")
    os.makedirs(MODEL_DIR, exist_ok=True)
    ft.to_csv(ft_path, index=False)
    ab.to_csv(ab_path, index=False)
    print("\n=== Frozen vs fine-tuned (test) ===\n")
    print(ft.to_string(index=False))
    print(f"\nSaved: {ft_path}")
    print("\n=== Ablations A1–A4 (test) ===\n")
    print(ab.to_string(index=False))
    print(f"\nSaved: {ab_path}")
    if err_log:
        print("\n--- Load / metric errors (see messages below) ---\n")
        for line in err_log:
            print(line)
    print_missing_checkpoint_hints()


def train_baseline() -> None:
    """Frozen embedding baseline (same as `python part2_train.py`)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=== Train POS (frozen embeddings) ===")
    train_pos(frozen=True, out_name="bilstm_pos.pt")
    print("=== Train NER + CRF (frozen) ===")
    train_ner(frozen=True, use_crf=True, out_name="bilstm_ner.pt")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description="Part 2 ablation training + metric tables")
    ap.add_argument("--train", action="store_true", help="Run A1–A4 training jobs")
    ap.add_argument("--finetune", action="store_true", help="Fine-tune POS+NER only (bilstm_*_ft.pt)")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Train frozen baseline only (bilstm_pos.pt, bilstm_ner.pt); same as part2_train.py",
    )
    args = ap.parse_args()

    if args.baseline:
        train_baseline()
    if args.finetune:
        run_finetune()
    if args.train:
        train_ablations()

    save_tables()


if __name__ == "__main__":
    main()

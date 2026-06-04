#!/usr/bin/env python3
"""
F001: Data preparation and matrix standardisation.

Extract and standardise Y matrix (2749x100), E_old SBERT embedding (100xd),
item_text, trait_id, and reverse_id from the BIG5 dataset CSV files.
Output .npy and .parquet files to data/processed/.

Usage:
    python scripts/prepare_data.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root is parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SRC = PROJECT_ROOT / "embeddings" / "BIG5"
DATA_OUT = PROJECT_ROOT / "data" / "processed"

# Expected dimensions
EXPECTED_N_ITEMS = 100
EXPECTED_N_SUBJECTS = 2749


def ensure_output_dir():
    """Create output directory if it does not exist."""
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {DATA_OUT}")


def load_questions_text():
    """
    Load item metadata from big5_questions_text.csv.

    Returns
    -------
    qt : pd.DataFrame  (100 x 5)
        Columns: question-id(index), item, grammartical_item, construct, encoding
    """
    fp = DATA_SRC / "big5_questions_text.csv"
    qt = pd.read_csv(fp, index_col=0)
    print(f"[OK] Loaded questions_text: {qt.shape[0]} items, columns={list(qt.columns)}")
    return qt


def load_responses():
    """
    Load raw participant responses from big5_responses.csv.

    The CSV has 100 rows (items) x 2751 columns (question-id, item, 2749 subjects).
    We transpose to get a (2749 x 100) subject-by-item matrix Y.

    Returns
    -------
    Y : np.ndarray  (2749, 100)  dtype=float32
        Raw 1-5 responses, NO reverse scoring applied.
    subject_ids : list[str]
        Hashed subject identifiers (column headers from CSV).
    item_ids : list[str]
        Item identifiers q1..q100 (row labels from CSV).
    """
    fp = DATA_SRC / "big5_responses.csv"
    df = pd.read_csv(fp, index_col=0)

    # Drop the 'item' text column — keep only numeric response columns
    if "item" in df.columns:
        df = df.drop(columns=["item"])

    item_ids = list(df.index)  # q1..q100
    subject_ids = list(df.columns)  # hashed subject IDs

    # Transpose: items (rows) -> subjects (rows)
    Y = df.values.T.astype(np.float32)

    print(f"[OK] Loaded responses: Y shape={Y.shape}, "
          f"value range=[{Y.min():.0f}, {Y.max():.0f}]")
    return Y, subject_ids, item_ids


def load_sbert_embeddings(item_ids):
    """
    Load original SBERT embeddings and L2-normalise each row.

    Parameters
    ----------
    item_ids : list[str]
        Expected item order q1..q100.

    Returns
    -------
    E_old : np.ndarray  (100, d)  dtype=float32, L2-normalised.
    """
    fp = DATA_SRC / "big5_questions_embeddings_SENTENCEBERT.csv"
    df = pd.read_csv(fp, index_col=0)

    # Ensure item order matches
    df = df.loc[item_ids]
    E = df.values.astype(np.float32)

    # L2 normalise each row
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding row detected — cannot L2-normalise.")
    E_norm = E / norms

    print(f"[OK] Loaded SBERT embeddings: shape={E_norm.shape}, "
          f"L2 norms min={norms.min():.6f} max={norms.max():.6f}")
    return E_norm


def build_metadata(qt):
    """
    Build metadata arrays from questions_text.

    Parameters
    ----------
    qt : pd.DataFrame
        Questions text with 'construct' and 'encoding' columns.

    Returns
    -------
    metadata : pd.DataFrame  (100 x 4)
        Columns: item_text, trait_id, reverse_id, question_id
    """
    metadata = pd.DataFrame(index=qt.index)
    metadata["question_id"] = qt.index.tolist()
    metadata["item_text"] = qt["item"].values
    metadata["trait_id"] = qt["construct"].values  # O, C, E, A, N
    metadata["reverse_id"] = (qt["encoding"] == -1).astype(int).values  # 1 if reverse, 0 if forward

    print(f"[OK] Built metadata: {len(metadata)} items, "
          f"traits={metadata['trait_id'].value_counts().to_dict()}, "
          f"reverse ratio={metadata['reverse_id'].mean():.2f}")
    return metadata


def _check(condition, pass_msg, fail_msg):
    """Print [PASS]/[FAIL] and return True iff the check passed."""
    if condition:
        print(f"[PASS] {pass_msg} ✓")
        return True
    print(f"[FAIL] {fail_msg}")
    return False


def validate(Y, E_old, metadata):
    """
    Run integrity checks and print a data report.

    Returns
    -------
    passed : bool
    """
    all_ok = True

    # --- Y matrix ---
    print("\n--- Y matrix validation ---")
    all_ok &= _check(
        Y.shape == (EXPECTED_N_SUBJECTS, EXPECTED_N_ITEMS),
        f"Y shape = {Y.shape}",
        f"Y shape is {Y.shape}, expected ({EXPECTED_N_SUBJECTS}, {EXPECTED_N_ITEMS})")
    all_ok &= _check(not np.any(np.isnan(Y)),
                     "No missing values",
                     "Y contains NaN values")
    y_min, y_max = Y.min(), Y.max()
    all_ok &= _check(1 <= y_min and y_max <= 5,
                     f"Y value range = [{y_min:.0f}, {y_max:.0f}]",
                     f"Y value range [{y_min}, {y_max}] outside [1, 5]")

    # --- E_old matrix ---
    print("\n--- E_old matrix validation ---")
    all_ok &= _check(E_old.shape[0] == EXPECTED_N_ITEMS,
                     f"E_old rows = {E_old.shape[0]}",
                     f"E_old has {E_old.shape[0]} rows, expected {EXPECTED_N_ITEMS}")
    d = E_old.shape[1]
    all_ok &= _check(d >= 384,
                     f"E_old dimension d={d}",
                     f"E_old dimension d={d} < 384")
    norms = np.linalg.norm(E_old, axis=1)
    all_ok &= _check(np.allclose(norms, 1.0, atol=1e-5),
                     f"E_old L2 norms ≈ 1.0 (tolerance 1e-5)",
                     f"E_old not L2-normalised: norm range [{norms.min():.6f}, {norms.max():.6f}]")

    # --- Metadata ---
    print("\n--- Metadata validation ---")
    trait_counts = metadata["trait_id"].value_counts()
    expected_traits = {"O", "C", "E", "A", "N"}
    all_ok &= _check(
        set(trait_counts.index) == expected_traits and all(v == 20 for v in trait_counts.values),
        f"5 traits × 20 items each",
        f"trait distribution: {trait_counts.to_dict()} (expected 5×20)")

    n_reverse = metadata["reverse_id"].sum()
    n_forward = len(metadata) - n_reverse
    print(f"[INFO] Forward: {n_forward}, Reverse: {n_reverse} (ratio={n_reverse / len(metadata):.2f})")

    # --- Cross-consistency ---
    print("\n--- Cross-consistency ---")
    all_ok &= _check(len(metadata) == Y.shape[1] == E_old.shape[0],
                     "All sources agree on n_items=100",
                     f"Item count mismatch: metadata={len(metadata)}, Y_cols={Y.shape[1]}, E_rows={E_old.shape[0]}")

    return all_ok


def save_outputs(Y, E_old, metadata, subject_ids):
    """
    Save Y, E_old as .npy, metadata as .parquet, and subject_ids as .txt.
    """
    np.save(DATA_OUT / "Y.npy", Y)
    print(f"[OK] Saved Y.npy  ({Y.shape}, {Y.dtype})")

    np.save(DATA_OUT / "E_old.npy", E_old)
    print(f"[OK] Saved E_old.npy  ({E_old.shape}, {E_old.dtype})")

    metadata.to_parquet(DATA_OUT / "metadata.parquet", index=True)
    print(f"[OK] Saved metadata.parquet  ({len(metadata)} rows × {len(metadata.columns)} cols)")

    with open(DATA_OUT / "subject_ids.txt", "w") as f:
        f.write("\n".join(subject_ids))
    print(f"[OK] Saved subject_ids.txt  ({len(subject_ids)} subjects)")


def main():
    print("=" * 60)
    print("F001: Data Preparation & Matrix Standardisation")
    print("=" * 60)

    ensure_output_dir()

    # 1. Load item metadata
    qt = load_questions_text()

    # 2. Build Y matrix [2749, 100]
    Y, subject_ids, item_ids = load_responses()

    # 3. Build E_old matrix [100, d], L2-normalised
    E_old = load_sbert_embeddings(item_ids)

    # 4. Build metadata table
    metadata = build_metadata(qt)

    # 5. Validate
    passed = validate(Y, E_old, metadata)

    # 6. Save
    if passed:
        save_outputs(Y, E_old, metadata, subject_ids)
        print("\n" + "=" * 60)
        print("F001 COMPLETE — all checks passed.")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("F001 FAILED validation. See [FAIL] messages above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

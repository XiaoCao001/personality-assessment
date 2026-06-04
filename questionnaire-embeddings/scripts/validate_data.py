#!/usr/bin/env python3
"""
Validation script for F001 outputs.

Checks that Y.npy, E_old.npy, metadata.parquet, and subject_ids.txt
exist and are valid.

Usage:
    python scripts/validate_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_FILES = ["Y.npy", "E_old.npy", "metadata.parquet", "subject_ids.txt"]


def _check(condition, pass_msg, fail_msg):
    """Print [PASS]/[FAIL] and return True iff the check passed."""
    if condition:
        print(f"[PASS] {pass_msg} ✓")
        return True
    print(f"[FAIL] {fail_msg}")
    return False


def main():
    all_ok = True

    # --- File existence ---
    for fname in REQUIRED_FILES:
        fp = DATA_DIR / fname
        all_ok &= _check(fp.exists(),
                         f"File exists: {fname}",
                         f"Missing file: {fp}")

    if not all_ok:
        print("\nSome required files are missing. Aborting.")
        return 1

    # --- Y matrix ---
    print("\n--- Y.npy ---")
    Y = np.load(DATA_DIR / "Y.npy")
    print(f"  shape={Y.shape}, dtype={Y.dtype}, range=[{Y.min():.0f}, {Y.max():.0f}]")
    all_ok &= _check(Y.shape == (2749, 100),
                     f"Y shape = {Y.shape}",
                     f"Y shape {Y.shape} != (2749, 100)")
    all_ok &= _check(Y.min() >= 1 and Y.max() <= 5,
                     "Y values in [1, 5]",
                     f"Y values [{Y.min()}, {Y.max()}] out of [1, 5]")
    all_ok &= _check(not np.any(np.isnan(Y)),
                     "No NaN values",
                     "Y contains NaN")

    # --- E_old matrix ---
    print("\n--- E_old.npy ---")
    E = np.load(DATA_DIR / "E_old.npy")
    print(f"  shape={E.shape}, dtype={E.dtype}")
    all_ok &= _check(E.shape[0] == 100,
                     f"E_old rows = {E.shape[0]}",
                     f"E_old has {E.shape[0]} rows, expected 100")
    all_ok &= _check(E.shape[1] >= 384,
                     f"E_old dimension d={E.shape[1]}",
                     f"E_old dimension {E.shape[1]} < 384")
    norms = np.linalg.norm(E, axis=1)
    all_ok &= _check(np.allclose(norms, 1.0, atol=1e-5),
                     f"L2 norms ≈ 1.0 (range [{norms.min():.6f}, {norms.max():.6f}])",
                     f"L2 normalisation failed: norms range [{norms.min():.6f}, {norms.max():.6f}]")

    # --- Metadata ---
    print("\n--- metadata.parquet ---")
    meta = pd.read_parquet(DATA_DIR / "metadata.parquet")
    print(f"  shape={meta.shape}, columns={list(meta.columns)}")
    all_ok &= _check(len(meta) == 100,
                     f"metadata rows = {len(meta)}",
                     f"metadata has {len(meta)} rows, expected 100")
    all_ok &= _check(set(meta["trait_id"].unique()) == {"O", "C", "E", "A", "N"},
                     "All 5 traits present",
                     f"Unexpected traits: {set(meta['trait_id'].unique())}")
    trait_counts = meta["trait_id"].value_counts()
    all_ok &= _check(all(v == 20 for v in trait_counts.values),
                     "5 traits × 20 items each",
                     f"Uneven trait distribution: {trait_counts.to_dict()}")

    # --- Subject IDs ---
    print("\n--- subject_ids.txt ---")
    with open(DATA_DIR / "subject_ids.txt") as f:
        subject_ids = [line.strip() for line in f if line.strip()]
    all_ok &= _check(len(subject_ids) == 2749,
                     f"subject count = {len(subject_ids)}",
                     f"subject count = {len(subject_ids)}, expected 2749")
    all_ok &= _check(len(set(subject_ids)) == len(subject_ids),
                     "All subject IDs unique",
                     f"Duplicate subject IDs: {len(subject_ids) - len(set(subject_ids))}")

    # --- Cross-consistency ---
    print("\n--- Cross-consistency ---")
    all_ok &= _check(len(subject_ids) == Y.shape[0],
                     f"subject_ids ({len(subject_ids)}) == Y rows ({Y.shape[0]})",
                     f"subject_ids ({len(subject_ids)}) != Y rows ({Y.shape[0]})")
    all_ok &= _check(len(meta) == Y.shape[1] == E.shape[0],
                     f"{len(meta)} items = {Y.shape[1]} Y cols = {E.shape[0]} E rows",
                     f"Item count mismatch: meta={len(meta)}, Y_cols={Y.shape[1]}, E_rows={E.shape[0]}")

    # --- Summary ---
    print("\n" + "=" * 40)
    if all_ok:
        print("All validations PASSED.")
        return 0
    else:
        print("Some validations FAILED. See [FAIL] above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

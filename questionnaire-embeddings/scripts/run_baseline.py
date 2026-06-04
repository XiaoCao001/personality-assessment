#!/usr/bin/env python3
"""
F002: Original paper baseline reproduction — 10-fold item-level cross-validation.

Reproduces the per-participant 10-fold item CV from the original paper using
SBERT embeddings + KNN K=5 regressor. The pipeline is an exact reimplementation
of `modelPerformance(m=4, par=5, d="BIG5", e="sentencebert")` from
`scripts/functions.py`.

Uses **non-reversed** response data (big5_responses_nonReversed.csv) — this matches
the original paper's methodology (modelPerformance uses R=2 → non-reversed data).
F001's Y.npy is the reversed version for downstream trait-score computation;
the non-reversed version is loaded here directly for baseline fidelity.

Usage:
    python scripts/run_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_SRC = PROJECT_ROOT / "embeddings" / "BIG5"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline"

RANDOM_STATE = 0
K_NEIGHBORS = 5
N_FOLDS = 10


def mean_confidence_interval(data, confidence=0.95):
    """
    Compute mean and 95 % confidence interval.

    Replicates `mean_confidence_interval` from scripts/functions.py exactly.
    """
    a = np.asarray(data, dtype=np.float64)
    a = a[~np.isnan(a)]
    n = len(a)
    mean = np.mean(a)
    se = sp_stats.sem(a)
    h = se * sp_stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h


def load_data():
    """
    Load input data.

    Uses F001's E_old.npy for embeddings, but loads non-reversed responses
    directly from the original CSV — this matches the paper's methodology
    (modelPerformance uses R=2 → non-reversed data).

    Returns
    -------
    Y : np.ndarray  (2749, 100)  float64
        Non-reversed 1-5 responses.
    E_old : np.ndarray  (100, 1024)  float32
        L2-normalised SBERT embeddings from F001.
    subject_ids : list[str]
        Subject identifiers in row order.
    """
    # --- Load E_old from F001 (embedding standardisation is fine) ---
    E_old = np.load(DATA_DIR / "E_old.npy")

    # --- Load non-reversed responses (matches paper baseline) ---
    fp = RAW_DATA_SRC / "big5_responses_nonReversed.csv"
    df = pd.read_csv(fp, index_col=0)
    if "item" in df.columns:
        df = df.drop(columns=["item"])

    item_ids = list(df.index)
    subject_ids = list(df.columns)

    # Transpose to (n_subjects, n_items)
    Y = df.values.T.astype(np.float64)

    n_subjects, n_items = Y.shape
    print(f"[OK] Loaded Y (non-rev): {Y.shape}  range=[{Y.min():.0f}, {Y.max():.0f}]")
    print(f"[OK] Loaded E_old:      {E_old.shape}  L2-norm")
    print(f"[OK] Subjects:          {n_subjects}, Items: {n_items}")

    return Y, E_old, subject_ids


def preprocess_embeddings(E):
    """
    Reproduce the original embedding pipeline: StandardScaler → PCA(0.9).

    This is exactly what `getEmbeddings()` in scripts/functions.py does.

    Returns
    -------
    E_pca : np.ndarray  (100, n_components)
    pca : PCA
        Fitted PCA object (for reporting).
    """
    scaler = StandardScaler()
    E_stand = scaler.fit_transform(E)

    pca = PCA(0.9, random_state=RANDOM_STATE)
    E_pca = pca.fit_transform(E_stand)

    print(f"[OK] StandardScaler + PCA(0.9): "
          f"{E.shape[1]} dims → {E_pca.shape[1]} PCs "
          f"({pca.explained_variance_ratio_.sum():.3f} variance retained)")
    return E_pca, pca


def run_baseline_cv(Y, E_pca):
    """
    Run the original paper's per-participant 10-fold item CV.

    For each participant:
        - 10-fold split of 100 items into 90 train / 10 test
        - Train KNN K=5 on the 90 training-item embeddings + responses
        - Predict the 10 held-out items
        - Round predictions to int, clamp to [1, 5]

    Parameters
    ----------
    Y : np.ndarray  (n_subjects, 100)
    E_pca : np.ndarray  (100, n_pca_dims)

    Returns
    -------
    total_preds : np.ndarray  (n_subjects, 100)
        Predicted responses (may contain NaN if unreachable).
    """
    n_subjects, n_items = Y.shape
    n_folds = N_FOLDS

    # Initialise prediction matrix with NaN
    total_preds = np.full((n_subjects, n_items), np.nan, dtype=np.float64)

    kf = KFold(n_splits=n_folds, random_state=RANDOM_STATE, shuffle=True)
    folds = list(kf.split(E_pca))

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{n_folds}  "
              f"(train={len(train_idx)}, test={len(test_idx)}) ...", end=" ")

        X_train = E_pca[train_idx]   # (90, pca_dims)
        X_test = E_pca[test_idx]     # (10, pca_dims)

        for subj in range(n_subjects):
            y_train = Y[subj, train_idx].astype(np.float64)

            # Drop items the participant did not answer (shouldn't happen, but safety)
            valid = ~np.isnan(y_train)
            if valid.sum() < 1:
                # All training responses missing — can't predict
                continue

            knn = KNeighborsRegressor(n_neighbors=min(K_NEIGHBORS, valid.sum()))
            knn.fit(X_train[valid], y_train[valid])

            y_pred = knn.predict(X_test)

            # Round and clamp to valid Likert scale
            y_pred = np.round(y_pred)
            y_pred[y_pred < 1] = 1
            y_pred[y_pred > 5] = 5

            total_preds[subj, test_idx] = y_pred

        print("done")

    return total_preds


def evaluate_per_participant(Y, total_preds, subject_ids):
    """
    Compute per-participant Pearson r and MAE between true and predicted responses.

    Returns
    -------
    results : pd.DataFrame
        Columns: subject_id, pearson_r, p_value, mae
    """
    n_subjects = Y.shape[0]
    rows = []

    for i in range(n_subjects):
        y_true = Y[i]
        y_pred = total_preds[i]

        # Only evaluate items that were predicted (non-NaN)
        valid = ~np.isnan(y_pred)
        n_valid = valid.sum()

        if n_valid < 2:
            rows.append({
                "subject_id": subject_ids[i],
                "pearson_r": np.nan,
                "p_value": np.nan,
                "mae": np.nan,
            })
            continue

        r, p = sp_stats.pearsonr(y_true[valid], y_pred[valid])
        mae = np.mean(np.abs(y_true[valid] - y_pred[valid]))

        rows.append({
            "subject_id": subject_ids[i],
            "pearson_r": r,
            "p_value": p,
            "mae": mae,
        })

    results = pd.DataFrame(rows)
    return results


def print_summary(results):
    """Print summary statistics and diagnostic information."""
    r_values = results["pearson_r"].dropna().values

    mean, ci_low, ci_high = mean_confidence_interval(r_values)
    _, pval_total = sp_stats.ttest_1samp(r_values, popmean=0)

    n_valid = len(r_values)
    n_total = len(results)
    n_nan = n_total - n_valid

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Subjects evaluated:      {n_valid}/{n_total}")
    if n_nan > 0:
        print(f"  Subjects with NaN r:     {n_nan}")
    print(f"  Mean Pearson r:          {mean:.4f}")
    print(f"  95% CI:                  [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  t-statistic vs 0:        {sp_stats.ttest_1samp(r_values, popmean=0).statistic:.2f}")
    print(f"  p-value (H0: r=0):       {pval_total:.6f}")
    print(f"  Mean MAE:                {results['mae'].mean():.4f}")

    # Diagnostic: check if mean r is in expected range
    if 0.40 <= mean <= 0.50:
        print(f"\n  ✓ Mean r {mean:.4f} within expected range [0.40, 0.50]")
    else:
        delta = abs(mean - 0.45)
        print(f"\n  ⚠ Mean r {mean:.4f} outside expected [0.40, 0.50] — deviation {delta:.4f}")
        print(f"    Possible causes: distance metric, item order, reverse-scoring timing, embedding version")

    return mean, ci_low, ci_high


def main():
    print("=" * 60)
    print("F002: Original Baseline — 10-Fold Item CV (KNN K=5)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data ...")
    Y, E_old, subject_ids = load_data()

    # 2. Preprocess embeddings
    print("\n[2/5] Preprocessing embeddings (StandardScaler + PCA 0.9) ...")
    E_pca, pca = preprocess_embeddings(E_old)

    # 3. Run baseline CV
    print(f"\n[3/5] Running {N_FOLDS}-fold item CV for {Y.shape[0]} participants ...")
    total_preds = run_baseline_cv(Y, E_pca)

    # 4. Evaluate
    print("\n[4/5] Evaluating per-participant Pearson r ...")
    results = evaluate_per_participant(Y, total_preds, subject_ids)

    # 5. Summary
    print_summary(results)

    # 6. Save
    print("\n[5/5] Saving results ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "original_10fold_itemcv_results.csv"
    results.to_csv(output_path, index=False)
    print(f"[OK] Saved {len(results)} rows to {output_path}")

    # Also save predictions matrix for debugging
    preds_path = OUTPUT_DIR / "predictions.npy"
    np.save(preds_path, total_preds)
    print(f"[OK] Saved predictions matrix to {preds_path}")

    print("\n" + "=" * 60)
    print("F002 COMPLETE.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

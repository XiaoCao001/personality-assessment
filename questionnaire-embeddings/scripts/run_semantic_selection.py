#!/usr/bin/env python3
"""
F005: Semantic Item Selection Strategy Evaluation.

Runs 5-fold participant-level CV with Coverage and Coverage+Diversity item
selection across four item-count ratios (10, 30, 50, 90 % of 100 items).

Strategies evaluated:
- **Coverage**: greedy facility-location maximising semantic coverage
- **Coverage+Div(λ=0.25)**: coverage minus light redundancy penalty
- **Coverage+Div(λ=0.5)**: coverage minus moderate redundancy penalty
- **Coverage+Div(λ=1.0)**: coverage minus strong redundancy penalty

For Coverage+Diversity, the best λ is selected **per fold/ratio** via inner
validation on train participants (AC003).  The inner-validation winner is
recorded alongside test-set results.

Prediction: per-subject item-level KNN (K=5, cosine distance on E_old
embeddings).  Evaluation: item-level Pearson r and trait-level per-trait r
via ``cv_framework.evaluate_predictions``.

Usage::

    python scripts/run_semantic_selection.py              # full run
    python scripts/run_semantic_selection.py --quick       # 1 fold, reduced ratios
    python scripts/run_semantic_selection.py --smoke       # 1 fold, 1 ratio
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress scipy ConstantInputWarning during per-subject correlation computation
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1"

RANDOM_STATE = 0
K_NEIGHBORS = 5
N_FOLDS = 5
RATIOS = (10, 30, 50, 90)
LAMBDA_CANDIDATES = (0.25, 0.5, 1.0)
TRAIT_ORDER = ("O", "C", "E", "A", "N")

# Strategy labels (λ‑selected variant uses inner validation to pick λ)
COVERAGE_STRATEGIES = ("Coverage",)  # baseline — no λ
COVDIV_STRATEGIES = tuple(f"Coverage+Div(λ={lam:.2f})" for lam in LAMBDA_CANDIDATES)
ALL_STRATEGIES = COVERAGE_STRATEGIES + COVDIV_STRATEGIES


def _resolve_imports():
    """Make local scripts importable."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


_resolve_imports()
from cv_framework import (  # noqa: E402
    participant_cv_split,
    inner_validation_split,
    evaluate_predictions,
)
from selection import CoverageSelector, CoverageDiversitySelector  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load F001 pre-processed data.

    Returns
    -------
    Y : np.ndarray  (2749, 100) float32 — reverse-scored responses
    E_old : np.ndarray  (100, 1024) float32 — L2-normalised SBERT embeddings
    metadata : pd.DataFrame  (100, 4)
    """
    Y = np.load(DATA_DIR / "Y.npy").astype(np.float64)
    E_old = np.load(DATA_DIR / "E_old.npy").astype(np.float64)
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    print(f"[OK] Loaded Y:      {Y.shape}  range=[{Y.min():.0f}, {Y.max():.0f}]")
    print(f"[OK] Loaded E_old:  {E_old.shape}")
    print(f"[OK] Loaded metadata: {len(metadata)} items")
    return Y, E_old, metadata


# ---------------------------------------------------------------------------
# Precompute distance matrix & vectorised KNN prediction
# ---------------------------------------------------------------------------


def _precompute_cosine_dist(E: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix from L2-normalised embeddings.

    For L2-normalised vectors, ``cosine_dist(i,j) = 1 - dot(e_i, e_j)``.
    Returns shape ``(n_items, n_items)``.
    """
    sim = E @ E.T  # (100, 100)
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)  # prevent self-match
    return dist


def _predict_held_out_batch(
    y_test: np.ndarray,
    dist: np.ndarray,
    S: np.ndarray,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """Predict held-out items for ALL test subjects in one vectorised pass.

    Parameters
    ----------
    y_test : np.ndarray  shape (n_test, n_items)
        Test-subject response matrix (reverse-scored).
    dist : np.ndarray  shape (n_items, n_items)
        Precomputed cosine distance matrix.
    S : np.ndarray  shape (|S|,)
        Indices of selected (observed) items.
    k : int
        Number of neighbours (capped at ``|S|``).

    Returns
    -------
    y_pred : np.ndarray  shape (n_test, n_items)
        Predictions.  Items in *S* are ``NaN``.
    """
    n_items = dist.shape[0]
    T_mask = np.ones(n_items, dtype=bool)
    T_mask[S] = False
    T = np.where(T_mask)[0]

    y_pred = np.full_like(y_test, np.nan)

    if len(T) == 0:
        return y_pred

    k_eff = min(k, len(S))

    # For each held-out item j, find its k nearest neighbours in S
    d_st = dist[np.ix_(S, T)]  # (|S|, |T|)

    if k_eff < len(S):
        nn_idx_in_S = np.argpartition(d_st, k_eff - 1, axis=0)[:k_eff]
    else:
        nn_idx_in_S = np.arange(len(S))[:, None].repeat(len(T), axis=1)

    nn_items = S[nn_idx_in_S]  # (k_eff, |T|)

    for si in range(y_test.shape[0]):
        y_subj = y_test[si]
        preds = y_subj[nn_items].mean(axis=0)
        preds = np.round(preds)
        preds = np.clip(preds, 1, 5)
        y_pred[si, T] = preds

    return y_pred


# ---------------------------------------------------------------------------
# Single fold evaluation
# ---------------------------------------------------------------------------


def _eval_fold(
    y_train: np.ndarray,
    y_test: np.ndarray,
    D: np.ndarray,
    cs: CoverageSelector,
    E_old: np.ndarray,
    trait_ids: np.ndarray,
    reverse_ids: np.ndarray,
    fold_idx: int,
    quick: bool = False,
    smoke: bool = False,
) -> list[dict]:
    """Evaluate all semantic strategy × ratio combinations for one outer fold.

    For Coverage+Diversity strategies, inner validation on *y_train* selects
    the best λ per ratio (AC003).  The inner-validation score is recorded.

    Parameters
    ----------
    y_train : np.ndarray  shape (n_train, 100)
        Train-subject response matrix — used for λ selection inner validation.
    y_test : np.ndarray  shape (n_test, 100)
        Test-subject response matrix.
    D : np.ndarray  shape (100, 100)
        Precomputed cosine distance matrix.
    cs : CoverageSelector
        Pre-built CoverageSelector (used for coverage/redundancy queries).
    E_old : np.ndarray  shape (100, d)
        L2-normalised SBERT embeddings.
    trait_ids, reverse_ids : np.ndarray
        Item metadata for ``evaluate_predictions``.
    fold_idx : int
        Fold number (for logging).
    quick : bool
        If True, evaluate only ratios 10 and 50.
    smoke : bool
        If True, evaluate only ratio 10.

    Returns
    -------
    rows : list of dict
        One dict per strategy×ratio with metric values and diagnostics.
    """
    ratios = (10,) if smoke else ((10, 50) if quick else RATIOS)
    rows: list[dict] = []

    # ---- inner validation split (for λ tuning) ---------------------------
    n_train = y_train.shape[0]
    train_inner_idx, valid_inner_idx = inner_validation_split(
        np.arange(n_train), val_ratio=0.2, seed=RANDOM_STATE + fold_idx
    )
    y_train_inner = y_train[train_inner_idx]   # (n_train_inner, 100)
    y_valid_inner = y_train[valid_inner_idx]   # (n_valid_inner, 100)

    for m in ratios:
        t0 = time.time()

        # ---- Coverage -----------------------------------------------------
        S_cov = cs.select(m)
        y_pred = _predict_held_out_batch(y_test, D, S_cov, k=K_NEIGHBORS)
        metrics = evaluate_predictions(y_test, y_pred, trait_ids, reverse_ids)
        cov_val = cs.compute_coverage(S_cov)
        red_val = cs.compute_redundancy(S_cov) if m >= 2 else 0.0

        rows.append(_make_row(
            strategy="Coverage", ratio=m, fold=fold_idx,
            metrics=metrics, selected_S=S_cov,
            coverage_val=cov_val, redundancy_val=red_val,
            selected_lam=None, inner_score=None,
        ))

        # ---- Coverage+Diversity — per λ candidates ------------------------
        best_lam = None
        best_inner_score = -np.inf
        lam_rows: list[dict] = []

        for lam in LAMBDA_CANDIDATES:
            cds = CoverageDiversitySelector(E_old, lam=lam)
            S_div = cds.select(m)

            # Inner validation: predict valid-inner using this λ's S
            y_pred_inner = _predict_held_out_batch(
                y_valid_inner, D, S_div, k=K_NEIGHBORS
            )
            metrics_inner = evaluate_predictions(
                y_valid_inner, y_pred_inner, trait_ids, reverse_ids
            )
            inner_score = metrics_inner["item_level"]["pearson_r"][0]

            # Track best λ
            if inner_score > best_inner_score:
                best_inner_score = inner_score
                best_lam = lam

            # Evaluate on test
            y_pred_test = _predict_held_out_batch(y_test, D, S_div, k=K_NEIGHBORS)
            metrics_test = evaluate_predictions(
                y_test, y_pred_test, trait_ids, reverse_ids
            )
            cov_val_div = cs.compute_coverage(S_div)
            red_val_div = cs.compute_redundancy(S_div) if m >= 2 else 0.0

            label = f"Coverage+Div(λ={lam:.2f})"
            lam_rows.append(_make_row(
                strategy=label, ratio=m, fold=fold_idx,
                metrics=metrics_test, selected_S=S_div,
                coverage_val=cov_val_div, redundancy_val=red_val_div,
                selected_lam=lam, inner_score=inner_score,
                is_best_lam=False,  # assigned after loop
            ))

        # Mark the single best λ row
        for r in lam_rows:
            if r["selected_lam"] == best_lam:
                r["is_best_lam"] = True
                break
        rows.extend(lam_rows)

        elapsed = time.time() - t0
        # Log the fold result in compact form
        cov_row = rows[-4]  # Coverage row (first in this m block)
        best_row = [r for r in rows[-3:] if r.get("is_best_lam")][0]
        print(f"  [Fold {fold_idx + 1}] m={m:>2}  "
              f"Coverage: item_r={cov_row['item_r']:.4f}  "
              f"best λ={best_row['selected_lam']:.2f}  "
              f"item_r={best_row['item_r']:.4f}  "
              f"({elapsed:.1f}s)")

    return rows


def _make_row(
    strategy: str,
    ratio: int,
    fold: int,
    metrics: dict,
    selected_S: np.ndarray,
    coverage_val: float,
    redundancy_val: float,
    selected_lam: float | None,
    inner_score: float | None,
    is_best_lam: bool = False,
) -> dict:
    """Pack evaluation results into a flat dictionary row."""
    il = metrics["item_level"]
    tl = metrics["trait_level"]
    pc = metrics["profile_correlation"]

    row = {
        "strategy": strategy,
        "ratio": ratio,
        "fold": fold,
        "item_r": il["pearson_r"][0],
        "item_r_ci_lower": il["pearson_r"][1],
        "item_r_ci_upper": il["pearson_r"][2],
        "item_mae": il["mae"][0],
        "item_rmse": il["rmse"][0],
        "trait_r_O": tl["per_trait_r"].get("O", np.nan),
        "trait_r_C": tl["per_trait_r"].get("C", np.nan),
        "trait_r_E": tl["per_trait_r"].get("E", np.nan),
        "trait_r_A": tl["per_trait_r"].get("A", np.nan),
        "trait_r_N": tl["per_trait_r"].get("N", np.nan),
        "trait_r_mean": tl["mean_big5_r"],
        "profile_r": pc["mean"],
        "coverage": coverage_val,
        "redundancy": redundancy_val,
        "selected_lam": selected_lam,
        "inner_val_score": inner_score,
        "is_best_lam": is_best_lam,
        "selected_S": ",".join(str(i) for i in selected_S),
    }
    return row


# ---------------------------------------------------------------------------
# Aggregate across folds
# ---------------------------------------------------------------------------


def _aggregate_results(detail_rows: list[dict]) -> pd.DataFrame:
    """Compute per-strategy×ratio mean metrics by averaging across folds."""
    df_detail = pd.DataFrame(detail_rows)
    metric_cols = [
        "item_r", "item_mae", "item_rmse",
    ] + [f"trait_r_{t}" for t in TRAIT_ORDER] + [
        "trait_r_mean", "profile_r", "coverage", "redundancy",
    ]
    group_cols = ["strategy", "ratio"]
    agg = df_detail.groupby(group_cols)[metric_cols].mean().reset_index()
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    quick = "--quick" in sys.argv
    smoke = "--smoke" in sys.argv

    mode_str = "SMOKE (1 fold, 1 ratio)" if smoke else (
        "QUICK (1 fold, 2 ratios)" if quick else "FULL"
    )
    print("=" * 60)
    print(f"F005: Semantic Selection Evaluation — {mode_str}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data ...")
    Y, E_old, metadata = load_data()
    trait_ids = metadata["trait_id"].values
    reverse_ids = metadata["reverse_id"].values.astype(np.float64)
    n_subjects = Y.shape[0]

    # Precompute cosine distance matrix (once for entire run)
    D = _precompute_cosine_dist(E_old)
    print(f"[OK] Cosine distance matrix: {D.shape}")

    # Coverage selector for diagnostics (coverage/redundancy queries)
    cs = CoverageSelector(E_old)

    # 2. Create Lambda summary tracker
    print("\n[2/5] Strategy overview ...")
    print(f"    Baseline: Coverage (greedy facility-location)")
    for lam in LAMBDA_CANDIDATES:
        print(f"    Variant:  Coverage+Div(λ={lam:.2f})")
    print(f"    Inner validation: 80/20 split of train participants")
    print(f"    λ grid: {LAMBDA_CANDIDATES}")

    # 3. Participant CV split
    print(f"\n[3/5] Running {N_FOLDS}-fold participant CV ...")
    folds = participant_cv_split(n_subjects, n_folds=N_FOLDS, seed=RANDOM_STATE)

    actual_folds = 1 if smoke else N_FOLDS
    all_rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds[:actual_folds]):
        y_train = Y[train_idx]  # (n_train, 100) — for inner validation
        y_test = Y[test_idx]    # (n_test, 100)
        print(f"\n--- Fold {fold_idx + 1}/{actual_folds} "
              f"(train={len(train_idx)}, test={len(test_idx)}) ---")

        fold_rows = _eval_fold(
            y_train=y_train,
            y_test=y_test,
            D=D,
            cs=cs,
            E_old=E_old,
            trait_ids=trait_ids,
            reverse_ids=reverse_ids,
            fold_idx=fold_idx,
            quick=quick,
            smoke=smoke,
        )
        all_rows.extend(fold_rows)

    # 4. Aggregate
    print("\n[4/5] Aggregating results ...")
    df_detail = pd.DataFrame(all_rows)
    df_agg = _aggregate_results(all_rows)

    # Print summary
    print("\n--- Per-strategy × ratio (averaged across folds) ---")
    summary = df_agg.copy()
    print(f"{'Strategy':<24} {'m':>3}  {'item_r':>8}  {'big5_r':>8}  "
          f"{'coverage':>8}  {'redund':>8}")
    print("-" * 75)
    for _, row in summary.iterrows():
        print(f"{row['strategy']:<24} {int(row['ratio']):>3}  "
              f"{row['item_r']:>8.4f}  {row['trait_r_mean']:>8.4f}  "
              f"{row['coverage']:>8.4f}  {row['redundancy']:>8.4f}")

    # λ selection summary (best λ per fold/ratio for Coverage+Div)
    best_rows = df_detail[df_detail["is_best_lam"] == True]  # noqa: E712
    if len(best_rows) > 0:
        print("\n--- Best λ per fold/ratio (inner validation) ---")
        for _, row in best_rows.iterrows():
            print(f"  Fold {int(row['fold']) + 1}  m={int(row['ratio']):>2}  "
                  f"best λ={row['selected_lam']:.2f}  "
                  f"inner score={row['inner_val_score']:.4f}  "
                  f"test item_r={row['item_r']:.4f}")

    # 5. Save
    print(f"\n[5/5] Saving results ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTPUT_DIR / "semantic_selection_detail.csv"
    df_detail.to_csv(detail_path, index=False)
    print(f"[OK] Detail results ({len(df_detail)} rows) → {detail_path}")

    agg_path = OUTPUT_DIR / "semantic_selection_aggregated.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"[OK] Aggregated results ({len(df_agg)} rows) → {agg_path}")

    summary_path = OUTPUT_DIR / "semantic_selection_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Summary ({len(summary)} rows) → {summary_path}")

    print("\n" + "=" * 60)
    print("F005 COMPLETE.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

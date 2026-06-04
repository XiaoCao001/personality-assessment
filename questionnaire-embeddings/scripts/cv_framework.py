#!/usr/bin/env python3
"""
F003: Participant-level cross-validation framework.

Provides the shared CV infrastructure for the full experiment pipeline
(F004–F010).  Implements 5-fold participant-level outer CV with inner
validation for hyperparameter selection, reverse scoring, trait score
computation, profile correlation, and a unified evaluation function.

Usage (import):
    from cv_framework import (
        participant_cv_split,
        inner_validation_split,
        reverse_score,
        compute_trait_scores,
        compute_profile_correlation,
        evaluate_predictions,
        simulate_real_testing,
    )

Usage (stand-alone demo / smoke test):
    python scripts/cv_framework.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 0
N_FOLDS = 5
TRAIT_ORDER = ("O", "C", "E", "A", "N")  # standard Big-Five OCEAN order
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# 1. Participant-level CV split
# ---------------------------------------------------------------------------


def participant_cv_split(
    n_subjects: int,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split *n_subjects* into *n_folds* folds of train (~80 %) / test (~20 %).

    Uses ``sklearn.model_selection.KFold`` with shuffling and a fixed random
    seed so that every call with the same arguments returns identical splits.
    Each subject appears in exactly one test fold across the returned list.

    Parameters
    ----------
    n_subjects : int
        Total number of participants.
    n_folds : int
        Number of outer CV folds (default 5).
    seed : int
        Random seed for the shuffle.

    Returns
    -------
    folds : list of (train_idx, test_idx)
        Each element is a tuple of two 1-D numpy arrays.  The training
        arrays collectively cover ~80 % of subjects per fold; test arrays
        cover the complementary ~20 %.
    """
    indices = np.arange(n_subjects)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(indices)]
    return folds


# ---------------------------------------------------------------------------
# 2. Inner validation split
# ---------------------------------------------------------------------------


def inner_validation_split(
    train_indices: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split *train_indices* into train-inner and validation-inner subsets.

    This is used for hyperparameter selection *within* a single outer
    training fold — the validation subset is never seen during final
    evaluation.

    Parameters
    ----------
    train_indices : np.ndarray
        1-D array of subject indices that belong to the outer training fold.
    val_ratio : float
        Fraction of *train_indices* to reserve for validation (default 0.2).
    seed : int
        Random seed.

    Returns
    -------
    train_inner : np.ndarray
    valid_inner : np.ndarray
    """
    rng = np.random.RandomState(seed)
    n = len(train_indices)
    n_val = max(1, int(np.round(n * val_ratio)))
    perm = rng.permutation(n)
    valid_inner = train_indices[perm[:n_val]]
    train_inner = train_indices[perm[n_val:]]
    return train_inner, valid_inner


# ---------------------------------------------------------------------------
# 3. Reverse scoring
# ---------------------------------------------------------------------------


def reverse_score(
    responses: np.ndarray,
    reverse_ids: np.ndarray,
) -> np.ndarray:
    """Apply reverse scoring to a response matrix.

    Forward-coded items are left unchanged; reverse-coded items are
    transformed as ``score = 6 - raw`` so that higher scores consistently
    indicate higher levels of the trait.

    Parameters
    ----------
    responses : np.ndarray  shape (n_subjects, n_items)
        Raw 1–5 Likert responses.
    reverse_ids : np.ndarray  shape (n_items,)
        Binary array where 1 indicates a reverse-coded item, 0 a forward
        item.

    Returns
    -------
    scored : np.ndarray  shape (n_subjects, n_items)
        Responses with reverse scoring applied.  A copy is always returned;
        the input is never mutated.
    """
    responses = np.asarray(responses, dtype=np.float64)
    reverse_ids = np.asarray(reverse_ids, dtype=np.float64)
    scored = responses.copy()
    rev_mask = reverse_ids == 1
    scored[:, rev_mask] = 6.0 - scored[:, rev_mask]
    return scored


# ---------------------------------------------------------------------------
# 4. Trait scores
# ---------------------------------------------------------------------------


def compute_trait_scores(
    responses: np.ndarray,
    trait_ids: np.ndarray,
    trait_order: tuple[str, ...] = TRAIT_ORDER,
) -> np.ndarray:
    """Compute per-subject mean trait scores.

    For each of the five Big-Five dimensions, the score is the simple
    (unweighted) average of the responses to all items belonging to that
    trait.

    Parameters
    ----------
    responses : np.ndarray  shape (n_subjects, n_items)
        Response matrix (reverse-scored if desired).
    trait_ids : np.ndarray  shape (n_items,)
        Trait labels (e.g. ``"O"``, ``"C"``, …) for each item.
    trait_order : tuple of str
        Order in which traits appear in the output columns (default OCEAN).

    Returns
    -------
    trait_scores : np.ndarray  shape (n_subjects, len(trait_order))
        Mean score per trait per subject.  Columns follow *trait_order*.
    """
    responses = np.asarray(responses, dtype=np.float64)
    trait_ids = np.asarray(trait_ids)

    n_subjects = responses.shape[0]
    n_traits = len(trait_order)
    scores = np.full((n_subjects, n_traits), np.nan, dtype=np.float64)

    for col, trait in enumerate(trait_order):
        mask = trait_ids == trait
        if mask.sum() == 0:
            continue
        scores[:, col] = np.nanmean(responses[:, mask], axis=1)

    return scores


# ---------------------------------------------------------------------------
# 5. Profile correlation
# ---------------------------------------------------------------------------


def compute_profile_correlation(
    trait_scores_true: np.ndarray,
    trait_scores_pred: np.ndarray,
) -> dict:
    """Compute per-subject Pearson *r* between true and predicted trait
    profiles, then return the mean and confidence interval across subjects.

    A *profile* is the vector of five Big-Five trait scores for a single
    participant.  Profile correlation measures how well the predicted
    *shape* of personality matches the true shape, regardless of absolute
    level.

    Parameters
    ----------
    trait_scores_true : np.ndarray  shape (n_subjects, 5)
        True trait scores.
    trait_scores_pred : np.ndarray  shape (n_subjects, 5)
        Predicted trait scores.

    Returns
    -------
    result : dict
        Keys: ``mean``, ``ci_lower``, ``ci_upper``, ``per_subject``.
    """
    n_subjects = trait_scores_true.shape[0]
    per_subject = np.full(n_subjects, np.nan, dtype=np.float64)

    for i in range(n_subjects):
        tv = trait_scores_true[i]
        pv = trait_scores_pred[i]
        valid = ~np.isnan(tv) & ~np.isnan(pv)
        if valid.sum() < 3:
            continue
        r, _ = sp_stats.pearsonr(tv[valid], pv[valid])
        per_subject[i] = r

    valid_rs = per_subject[~np.isnan(per_subject)]
    if len(valid_rs) == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "per_subject": per_subject}

    mean, ci_low, ci_high = _mean_ci(valid_rs)
    return {"mean": mean, "ci_lower": ci_low, "ci_upper": ci_high,
            "per_subject": per_subject}


# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_ids: np.ndarray,
    reverse_ids: np.ndarray,
    trait_order: tuple[str, ...] = TRAIT_ORDER,
    confidence: float = 0.95,
) -> dict:
    """Compute a comprehensive suite of prediction-performance metrics.

    Metrics are computed **per subject** and then averaged (with confidence
    intervals where applicable).  The function computes three sets of trait
    scores following Decision 5:

    * **short_form** — trait scores using only the items the participant
      actually answered (i.e., only non-NaN entries in *y_pred*).
    * **imputed_full** — trait scores where observed answers are used for
      available items and predictions fill in the rest.
    * **held_out** — trait scores using only the predicted items.

    The ``imputed_full`` version is the primary scenario; the other two are
    diagnostic.

    Parameters
    ----------
    y_true : np.ndarray  shape (n_subjects, n_items)
        Ground-truth responses (reverse-scored).
    y_pred : np.ndarray  shape (n_subjects, n_items)
        Predicted responses.  Items that were not predicted should be
        ``NaN``; items that were observed (not predicted) should also be
        ``NaN``.
    trait_ids : np.ndarray  shape (n_items,)
        Trait label per item.
    reverse_ids : np.ndarray  shape (n_items,)
        Binary reverse indicator per item.
    trait_order : tuple of str
        Trait column order (default OCEAN).
    confidence : float
        Confidence level for intervals (default 0.95).

    Returns
    -------
    metrics : dict
        A nested dictionary with the following top-level keys:

        * ``item_level`` — per-subject Pearson *r*, MAE, RMSE, rounded-accuracy
          between true and predicted item responses.
        * ``trait_level`` — per-trait Pearson *r*, MAE, and RMSE for the
          imputed-full trait scores, plus ``mean_big5_r``.
        * ``profile_correlation`` — result of :func:`compute_profile_correlation`
          on the imputed-full trait scores.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    trait_ids = np.asarray(trait_ids)
    reverse_ids = np.asarray(reverse_ids, dtype=np.float64)

    n_subjects, n_items = y_true.shape

    # ---- item-level metrics (only on predicted items) --------------------
    item_r = np.full(n_subjects, np.nan, dtype=np.float64)
    item_mae = np.full(n_subjects, np.nan, dtype=np.float64)
    item_rmse = np.full(n_subjects, np.nan, dtype=np.float64)
    item_rounded_acc = np.full(n_subjects, np.nan, dtype=np.float64)

    for i in range(n_subjects):
        mask = ~np.isnan(y_pred[i])
        if mask.sum() < 2:
            continue
        tv = y_true[i, mask]
        pv = y_pred[i, mask]
        r, _ = sp_stats.pearsonr(tv, pv)
        item_r[i] = r
        item_mae[i] = np.mean(np.abs(tv - pv))
        item_rmse[i] = np.sqrt(np.mean((tv - pv) ** 2))
        item_rounded_acc[i] = np.mean(np.round(pv) == np.round(tv))

    item_level = {
        "pearson_r": _mean_ci(item_r, confidence),
        "mae": _mean_ci(item_mae, confidence),
        "rmse": _mean_ci(item_rmse, confidence),
        "rounded_accuracy": _mean_ci(item_rounded_acc, confidence),
        "per_subject_r": item_r,
    }

    # ---- trait-level metrics ----------------------------------------------
    # True trait scores (reverse-scored)
    trait_true = compute_trait_scores(y_true, trait_ids, trait_order)

    # Short-form: only observed items (where y_pred is NOT NaN → answered)
    y_short = y_true.copy()
    y_short[~np.isnan(y_pred)] = np.nan
    trait_short = compute_trait_scores(y_short, trait_ids, trait_order)

    # Imputed full: true where available, predicted where not
    y_imputed = y_true.copy()
    pred_mask = ~np.isnan(y_pred)
    y_imputed[pred_mask] = y_pred[pred_mask]
    trait_imputed = compute_trait_scores(y_imputed, trait_ids, trait_order)

    # Held-out: only predicted items
    y_held = np.full_like(y_true, np.nan)
    y_held[pred_mask] = y_pred[pred_mask]
    trait_held = compute_trait_scores(y_held, trait_ids, trait_order)

    # Per-trait Pearson r for imputed full
    per_trait_r = {}
    per_trait_mae = {}
    per_trait_rmse = {}
    for col, trait in enumerate(trait_order):
        tv = trait_true[:, col]
        pv = trait_imputed[:, col]
        valid = ~np.isnan(tv) & ~np.isnan(pv)
        if valid.sum() < 2:
            per_trait_r[trait] = np.nan
            per_trait_mae[trait] = np.nan
            per_trait_rmse[trait] = np.nan
            continue
        r, _ = sp_stats.pearsonr(tv[valid], pv[valid])
        per_trait_r[trait] = r
        per_trait_mae[trait] = np.mean(np.abs(tv[valid] - pv[valid]))
        per_trait_rmse[trait] = np.sqrt(np.mean((tv[valid] - pv[valid]) ** 2))

    # Mean Big Five r
    valid_rs = [v for v in per_trait_r.values() if not np.isnan(v)]
    mean_big5_r = np.mean(valid_rs) if valid_rs else np.nan

    trait_level = {
        "per_trait_r": per_trait_r,
        "per_trait_mae": per_trait_mae,
        "per_trait_rmse": per_trait_rmse,
        "mean_big5_r": mean_big5_r,
        "trait_scores_true": trait_true,
        "trait_scores_short_form": trait_short,
        "trait_scores_imputed_full": trait_imputed,
        "trait_scores_held_out": trait_held,
    }

    # ---- profile correlation ---------------------------------------------
    profile = compute_profile_correlation(trait_true, trait_imputed)

    return {
        "item_level": item_level,
        "trait_level": trait_level,
        "profile_correlation": profile,
    }


# ---------------------------------------------------------------------------
# 7. Simulate real testing
# ---------------------------------------------------------------------------


def simulate_real_testing(
    y_test: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a realistic testing scenario where only a subset of items is
    administered.

    Given the ground-truth response matrix for test participants and a set
    *S* of selected item indices, returns:

    * ``y_observed`` — the responses that are actually available (items in
      *S* are kept; all other items are set to ``NaN``).
    * ``y_held_out`` — the complementary set (items **not** in *S* are kept;
      items in *S* are set to ``NaN``).

    This allows downstream code to train on ``y_observed`` and evaluate
    predictions against ``y_held_out``.

    Parameters
    ----------
    y_test : np.ndarray  shape (n_subjects, n_items)
        Full response matrix for test participants.
    S : np.ndarray  shape (|S|,)
        Indices of the selected / administered items.

    Returns
    -------
    y_observed : np.ndarray  shape (n_subjects, n_items)
    y_held_out : np.ndarray  shape (n_subjects, n_items)
    """
    y_test = np.asarray(y_test, dtype=np.float64)
    S = np.asarray(S, dtype=np.intp)

    n_items = y_test.shape[1]
    mask = np.zeros(n_items, dtype=bool)
    mask[S] = True

    y_observed = y_test.copy()
    y_observed[:, ~mask] = np.nan

    y_held_out = y_test.copy()
    y_held_out[:, mask] = np.nan

    return y_observed, y_held_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_ci(
    data: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and confidence interval, ignoring NaNs.

    Mirrors ``mean_confidence_interval`` from ``scripts/functions.py``.
    """
    a = np.asarray(data, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    mean = np.mean(a)
    se = sp_stats.sem(a)
    h = se * sp_stats.t.ppf((1 + confidence) / 2.0, len(a) - 1)
    return (mean, mean - h, mean + h)


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke test
# ---------------------------------------------------------------------------


def _demo():
    """Run a quick smoke test using the real F001 data."""
    print("=" * 60)
    print("F003 cv_framework — smoke test")
    print("=" * 60)

    # --- Load data ---------------------------------------------------------
    Y = np.load(DATA_DIR / "Y.npy")  # (2749, 100) — reversed
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    reverse_ids = metadata["reverse_id"].values.astype(np.float64)
    trait_ids = metadata["trait_id"].values

    n_subjects, n_items = Y.shape
    print(f"[OK] Loaded Y:      {Y.shape}")
    print(f"[OK] Loaded metadata: {len(metadata)} items")
    print(f"    Traits: {metadata['trait_id'].value_counts().to_dict()}")
    print(f"    Reverse ratio: {reverse_ids.mean():.2f}")

    # --- 1. Participant CV split -------------------------------------------
    folds = participant_cv_split(n_subjects, n_folds=5, seed=0)
    print(f"\n[1] Participant CV split: {len(folds)} folds")
    all_test = set()
    for f_idx, (train, test) in enumerate(folds):
        ratio = len(train) / n_subjects
        overlap = set(train) & set(test)
        all_test.update(test)
        assert len(overlap) == 0, f"Fold {f_idx}: train/test overlap!"
        assert abs(ratio - 0.8) < 0.01, f"Fold {f_idx}: ratio={ratio:.4f}"
        print(f"    Fold {f_idx + 1}: train={len(train)}, test={len(test)}, "
              f"ratio={ratio:.3f}  [OK]")
    assert len(all_test) == n_subjects, (
        f"Coverage: {len(all_test)}/{n_subjects}"
    )
    print(f"  [OK] All {n_subjects} subjects covered exactly once across folds")

    # --- 2. Inner validation split -----------------------------------------
    train_idx, _ = folds[0]
    ti, vi = inner_validation_split(train_idx, val_ratio=0.2, seed=0)
    print(f"\n[2] Inner validation: train_inner={len(ti)}, "
          f"valid_inner={len(vi)}  [OK]")

    # --- 3. Reverse scoring -------------------------------------------------
    # Y.npy already IS reverse-scored, so applying reverse_score again
    # should "un-reverse" it.  We test the transform is correct.
    # Use the raw (non-reversed) data from source to verify.
    raw_csv = PROJECT_ROOT / "embeddings" / "BIG5" / "big5_responses_nonReversed.csv"
    df_raw = pd.read_csv(raw_csv, index_col=0)
    if "item" in df_raw.columns:
        df_raw = df_raw.drop(columns=["item"])
    Y_raw = df_raw.values.T.astype(np.float64)  # non-reversed

    Y_rev = reverse_score(Y_raw, reverse_ids)
    # After reverse scoring, should match Y.npy (the reversed version)
    diff = np.abs(Y_rev - Y)
    print(f"\n[3] Reverse scoring: max|Y_rev - Y_npy| = {diff.max():.6f}")
    assert diff.max() < 0.01, "Reverse scoring mismatch with F001 Y.npy!"
    print(f"  [OK] Reverse scoring reproduces F001 Y.npy")

    # --- 4. Trait scores ---------------------------------------------------
    trait_scores = compute_trait_scores(Y, trait_ids)
    print(f"\n[4] Trait scores: shape={trait_scores.shape}")
    for col, trait in enumerate(TRAIT_ORDER):
        ts = trait_scores[:, col]
        print(f"    {trait}: mean={np.nanmean(ts):.3f}, "
              f"std={np.nanstd(ts):.3f}, "
              f"range=[{np.nanmin(ts):.1f}, {np.nanmax(ts):.1f}]")
    print(f"  [OK] 5 traits computed")

    # --- 5. Profile correlation --------------------------------------------
    # Compare trait scores from first half vs second half of items
    half = n_items // 2
    ts_first = compute_trait_scores(Y[:, :half], trait_ids[:half])
    ts_second = compute_trait_scores(Y[:, half:], trait_ids[half:])
    profile = compute_profile_correlation(ts_first, ts_second)
    print(f"\n[5] Profile correlation (split-half): "
          f"mean={profile['mean']:.4f}, "
          f"95% CI=[{profile['ci_lower']:.4f}, {profile['ci_upper']:.4f}]")
    assert 0.3 < profile["mean"] < 0.9, (
        f"Unexpected profile correlation {profile['mean']:.4f}"
    )
    print(f"  [OK] Profile correlation in expected range")

    # --- 6. Simulate real testing ------------------------------------------
    S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 10% = 10 items
    y_obs, y_held = simulate_real_testing(Y, S)
    assert np.all(~np.isnan(y_obs[:, S])), "Observed items should be non-NaN"
    assert np.all(np.isnan(y_held[:, S])), "Held-out items should be NaN for S"
    assert np.all(~np.isnan(y_held[:, 10:])), "Non-S items should be non-NaN in held"
    print(f"\n[6] Simulate testing: S={S}, y_obs NaN={np.isnan(y_obs).sum()}, "
          f"y_held NaN={np.isnan(y_held).sum()}  [OK]")

    # --- 7. Full evaluate_predictions --------------------------------------
    # Quick check: use a trivial predictor (mean of observed items)
    print(f"\n[7] evaluate_predictions demo ...")
    train_idx, test_idx = folds[0]
    y_test_true = Y[test_idx]

    # Dummy predictor: predict each held-out item as mean of observed items
    y_test_pred = np.full_like(y_test_true, np.nan)
    for si in range(len(test_idx)):
        obs = y_obs[test_idx[si]]
        obs_mean = np.nanmean(obs)
        if np.isnan(obs_mean):
            obs_mean = 3.0
        y_test_pred[si] = obs_mean
    y_test_pred[:, S] = np.nan  # observed items not predicted

    metrics = evaluate_predictions(y_test_true, y_test_pred, trait_ids, reverse_ids)
    il = metrics["item_level"]
    tl = metrics["trait_level"]
    pc = metrics["profile_correlation"]
    print(f"    Item r:        {il['pearson_r'][0]:.4f} "
          f"[{il['pearson_r'][1]:.4f}, {il['pearson_r'][2]:.4f}]")
    print(f"    Item MAE:      {il['mae'][0]:.4f}")
    print(f"    Mean Big5 r:   {tl['mean_big5_r']:.4f}")
    print(f"    Profile corr:  {pc['mean']:.4f}")
    print(f"  [OK] evaluate_predictions runs without error")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("F003 cv_framework — ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(_demo())

#!/usr/bin/env python3
"""
Unit tests for cv_framework.py (F003).

Covers all acceptance criteria:
  AC001 — 5-fold split ratios and no leakage
  AC002 — reverse scoring correctness
  AC003 — trait score computation
  AC004 — evaluation output format

Usage:
    python scripts/test_cv_framework.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Ensure we can import from the scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_framework import (  # noqa: E402
    participant_cv_split,
    inner_validation_split,
    reverse_score,
    compute_trait_scores,
    compute_profile_correlation,
    evaluate_predictions,
    simulate_real_testing,
    TRAIT_ORDER,
    DATA_DIR,
    _mean_ci,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
PASSED = 0
FAILED = 0


def check(condition: bool, label: str) -> bool:
    """Report a single check."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}")
    return condition


# ===================================================================
# AC001: 5-fold split ratios and no leakage
# ===================================================================


def test_cv_split_ratios():
    print("\n--- AC001: participant_cv_split ---")

    n = 2749
    folds = participant_cv_split(n, n_folds=5, seed=0)

    check(len(folds) == 5, f"Returns 5 folds (got {len(folds)})")

    all_test = []
    for fi, (train, test) in enumerate(folds):
        ratio = len(train) / n
        check(
            abs(ratio - 0.8) < 0.01,
            f"Fold {fi}: train ratio = {ratio:.4f} (expected ~0.8)",
        )
        overlap = set(train) & set(test)
        check(len(overlap) == 0, f"Fold {fi}: no train/test overlap")
        all_test.extend(test)

    check(
        len(set(all_test)) == n,
        f"All {n} subjects covered exactly once (got {len(set(all_test))})",
    )

    # Determinism
    folds2 = participant_cv_split(n, n_folds=5, seed=0)
    for fi in range(5):
        check(
            np.array_equal(folds[fi][0], folds2[fi][0]),
            f"Fold {fi}: train indices deterministic",
        )
        check(
            np.array_equal(folds[fi][1], folds2[fi][1]),
            f"Fold {fi}: test indices deterministic",
        )

    # Different seeds give different splits
    folds3 = participant_cv_split(n, n_folds=5, seed=42)
    all_different = not np.array_equal(folds[0][0], folds3[0][0])
    check(all_different, "Different seeds produce different splits")

    # Edge: small n
    folds4 = participant_cv_split(10, n_folds=5, seed=0)
    check(len(folds4) == 5, "Works with n_subjects=10")

    # Edge: single fold
    folds5 = participant_cv_split(100, n_folds=2, seed=0)
    check(len(folds5) == 2, "Works with n_folds=2")
    check(
        abs(len(folds5[0][0]) / 100 - 0.5) < 0.05,
        "2-fold gives ~50/50 split",
    )


def test_inner_validation_split():
    print("\n--- inner_validation_split ---")

    rng = np.random.RandomState(0)
    train_idx = rng.permutation(1000)

    ti, vi = inner_validation_split(train_idx, val_ratio=0.2, seed=0)

    check(len(ti) + len(vi) == 1000, "All indices preserved")
    check(abs(len(vi) / 1000 - 0.2) < 0.02, f"Val ratio ~0.2 (got {len(vi)/1000:.3f})")
    check(len(set(ti) & set(vi)) == 0, "No overlap between train_inner and valid_inner")

    # Reproducibility
    ti2, vi2 = inner_validation_split(train_idx, val_ratio=0.2, seed=0)
    check(np.array_equal(ti, ti2), "Deterministic output")

    # Edge: val_ratio=0
    ti3, vi3 = inner_validation_split(train_idx, val_ratio=0.0, seed=0)
    check(len(vi3) == 1 and len(ti3) == 999, "val_ratio=0 gives 1 validation subject")


# ===================================================================
# AC002: Reverse scoring correctness
# ===================================================================


def test_reverse_score():
    print("\n--- AC002: reverse_score ---")

    # 3 subjects, 4 items; items 0 and 2 are reverse-coded
    responses = np.array([
        [1, 3, 5, 2],
        [2, 2, 2, 2],
        [5, 1, 1, 4],
    ], dtype=np.float64)
    reverse_ids = np.array([0, 1, 1, 0], dtype=np.float64)

    scored = reverse_score(responses, reverse_ids)

    # Item 0 forward: stays same
    check(np.array_equal(scored[:, 0], responses[:, 0]), "Forward items unchanged")
    # Item 1 reverse: 6 - y
    expected_item1 = np.array([3, 4, 5], dtype=np.float64)  # 6-3, 6-2, 6-1
    check(np.array_equal(scored[:, 1], expected_item1), "Reverse items: score = 6 - y")
    # Item 2 reverse
    expected_item2 = np.array([1, 4, 5], dtype=np.float64)  # 6-5, 6-2, 6-1
    check(np.array_equal(scored[:, 2], expected_item2), "Reverse item 2 correct")
    # Item 3 forward
    check(np.array_equal(scored[:, 3], responses[:, 3]), "Forward item 3 unchanged")

    # Original not mutated
    check(np.array_equal(responses[0], [1, 3, 5, 2]), "Input not mutated")

    # Idempotency: reversing twice gives original
    scored_twice = reverse_score(scored, reverse_ids)
    check(
        np.allclose(scored_twice, responses),
        "Double reverse = identity",
    )

    # Integration: real data from F001
    Y = np.load(DATA_DIR / "Y.npy")
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    real_rev = metadata["reverse_id"].values.astype(np.float64)

    # Y.npy is already reversed; applying reverse_score should "un-reverse"
    Y_unrev = reverse_score(Y, real_rev)
    # Verify that un-reversed Y has reasonable relationship with reversed Y
    # Forward items should be identical
    fwd_mask = real_rev == 0
    check(
        np.allclose(Y[:, fwd_mask], Y_unrev[:, fwd_mask]),
        "Real data: forward items unchanged by reverse_score",
    )
    # Reverse items: Y + Y_unrev ≈ 6
    rev_mask = real_rev == 1
    sums = Y[:, rev_mask] + Y_unrev[:, rev_mask]
    check(
        np.allclose(sums, 6.0, atol=0.01),
        f"Real data: Y + unrev(Y) ≈ 6 on reverse items (max dev: {np.abs(sums - 6).max():.4f})",
    )


# ===================================================================
# AC003: Trait score computation
# ===================================================================


def test_compute_trait_scores():
    print("\n--- AC003: compute_trait_scores ---")

    # 2 subjects, 5 items (1 per trait)
    responses = np.array([
        [4, 3, 5, 2, 1],   # O=4, C=3, E=5, A=2, N=1
        [2, 4, 2, 5, 3],   # O=2, C=4, E=2, A=5, N=3
    ], dtype=np.float64)
    trait_ids = np.array(["O", "C", "E", "A", "N"])

    scores = compute_trait_scores(responses, trait_ids)

    check(scores.shape == (2, 5), f"Shape (2, 5) — got {scores.shape}")
    # Subject 0
    check(np.allclose(scores[0], [4, 3, 5, 2, 1]), "Subject 0 trait scores correct")
    # Subject 1
    check(np.allclose(scores[1], [2, 4, 2, 5, 3]), "Subject 1 trait scores correct")

    # Multiple items per trait
    responses2 = np.array([
        [4, 2, 3, 3, 5, 3, 2, 4, 1, 2],  # 2 per trait
    ], dtype=np.float64)
    trait_ids2 = np.array(["O", "O", "C", "C", "E", "E", "A", "A", "N", "N"])
    scores2 = compute_trait_scores(responses2, trait_ids2)
    expected2 = np.array([[3.0, 3.0, 4.0, 3.0, 1.5]])  # means
    check(np.allclose(scores2, expected2), "Multi-item trait means correct")
    check(scores2.shape == (1, 5), "Single subject shape (1, 5)")

    # Missing trait
    trait_ids3 = np.array(["O", "C", "E", "A", "X"])  # X not in TRAIT_ORDER
    scores3 = compute_trait_scores(responses, trait_ids3)
    check(np.isnan(scores3[0, 4]), "Missing trait (N) is NaN")
    check(not np.isnan(scores3[0, 0]), "Present trait (O) is not NaN")

    # NaN handling in responses
    responses4 = responses.astype(float).copy()
    responses4[0, 0] = np.nan  # O item for subject 0 → O should be NaN for subj 0
    scores4 = compute_trait_scores(responses4, trait_ids)
    check(np.isnan(scores4[0, 0]), "NaN item makes that trait NaN for that subject")
    check(scores4[0, 1] == 3.0, "Other traits (C) unaffected by NaN in O")

    # Integration: real data
    Y = np.load(DATA_DIR / "Y.npy")
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    real_traits = metadata["trait_id"].values
    real_scores = compute_trait_scores(Y, real_traits)
    check(real_scores.shape == (2749, 5), f"Real data shape (2749, 5): got {real_scores.shape}")
    check(not np.any(np.isnan(real_scores)), "Real data: no NaN trait scores")
    # All scores should be in [1, 5]
    check(
        real_scores.min() >= 1.0 and real_scores.max() <= 5.0,
        f"Real data: trait scores in [1, 5] (got [{real_scores.min():.2f}, {real_scores.max():.2f}])",
    )


# ===================================================================
# Profile correlation
# ===================================================================


def test_profile_correlation():
    print("\n--- compute_profile_correlation ---")

    rng = np.random.RandomState(42)
    n = 100
    true = rng.randn(n, 5)
    # Perfect correlation: pred = true (just shifted)
    pred = true + 2.0
    result = compute_profile_correlation(true, pred)
    check(
        abs(result["mean"] - 1.0) < 0.01,
        f"Perfect correlation (shifted): r={result['mean']:.4f}",
    )

    # Zero correlation
    pred2 = rng.randn(n, 5)
    result2 = compute_profile_correlation(true, pred2)
    check(
        abs(result2["mean"]) < 0.3,
        f"Near-zero correlation: r={result2['mean']:.4f}",
    )

    # NaN handling — insufficient valid traits
    true3 = np.array([[1, 2, np.nan, np.nan, np.nan]])
    pred3 = np.array([[2, 3, 4, 5, 1]])
    result3 = compute_profile_correlation(true3, pred3)
    check(
        np.isnan(result3["per_subject"][0]),
        "Insufficient valid traits → NaN per_subject",
    )
    check(np.isnan(result3["mean"]), "Mean NaN when no valid subjects")


# ===================================================================
# simulate_real_testing
# ===================================================================


def test_simulate_real_testing():
    print("\n--- simulate_real_testing ---")

    y = np.arange(1, 21, dtype=np.float64).reshape(2, 10)  # 2 subj × 10 items
    S = np.array([0, 3, 7])

    y_obs, y_held = simulate_real_testing(y, S)

    # Observed: S items kept, rest NaN
    check(np.all(~np.isnan(y_obs[:, S])), "y_observed: S items are non-NaN")
    non_S = np.array([i for i in range(10) if i not in S])
    check(np.all(np.isnan(y_obs[:, non_S])), "y_observed: non-S items are NaN")

    # Held-out: S items NaN, rest kept
    check(np.all(np.isnan(y_held[:, S])), "y_held_out: S items are NaN")
    check(np.all(~np.isnan(y_held[:, non_S])), "y_held_out: non-S items are non-NaN")

    # Original not mutated
    check(y[0, 0] == 1.0, "Input y_test not mutated")


# ===================================================================
# AC004: Evaluation output format
# ===================================================================


def test_evaluate_predictions_format():
    print("\n--- AC004: evaluate_predictions output format ---")

    rng = np.random.RandomState(0)
    n_subj, n_items = 50, 20
    y_true = rng.randint(1, 6, (n_subj, n_items)).astype(np.float64)
    # Predict every other item
    y_pred = np.full_like(y_true, np.nan)
    pred_items = np.arange(0, n_items, 2)
    y_pred[:, pred_items] = np.clip(y_true[:, pred_items] + rng.randn(n_subj, len(pred_items)), 1, 5)

    trait_ids = np.array(["O", "C", "E", "A", "N"] * 4)  # 20 items
    reverse_ids = np.tile([0, 1, 0, 1, 0], 4).astype(np.float64)

    metrics = evaluate_predictions(y_true, y_pred, trait_ids, reverse_ids)

    # Top-level keys
    for key in ["item_level", "trait_level", "profile_correlation"]:
        check(key in metrics, f"Top-level key '{key}' present")

    # item_level keys
    il = metrics["item_level"]
    for key in ["pearson_r", "mae", "rmse", "rounded_accuracy", "per_subject_r"]:
        check(key in il, f"item_level key '{key}' present")
    # Each metric tuple is (mean, ci_low, ci_high)
    for key in ["pearson_r", "mae", "rmse", "rounded_accuracy"]:
        tup = il[key]
        check(len(tup) == 3, f"{key} is 3-tuple (mean, ci_low, ci_high)")
    check(len(il["per_subject_r"]) == n_subj, f"per_subject_r has {n_subj} entries")

    # trait_level keys
    tl = metrics["trait_level"]
    for key in ["per_trait_r", "per_trait_mae", "per_trait_rmse",
                "mean_big5_r", "trait_scores_true", "trait_scores_short_form",
                "trait_scores_imputed_full", "trait_scores_held_out"]:
        check(key in tl, f"trait_level key '{key}' present")
    for trait in TRAIT_ORDER:
        check(trait in tl["per_trait_r"], f"per_trait_r has '{trait}'")
    check(
        tl["trait_scores_true"].shape == (n_subj, 5),
        f"trait_scores_true shape ({n_subj}, 5)",
    )

    # profile_correlation keys
    pc = metrics["profile_correlation"]
    for key in ["mean", "ci_lower", "ci_upper", "per_subject"]:
        check(key in pc, f"profile_correlation key '{key}' present")
    check(len(pc["per_subject"]) == n_subj, f"per_subject has {n_subj} entries")


def test_evaluate_predictions_perfect():
    """Perfect predictions should give r~1."""
    print("\n--- evaluate_predictions: perfect prediction ---")

    rng = np.random.RandomState(0)
    y_true = rng.randint(1, 6, (50, 10)).astype(np.float64)
    y_pred = y_true.copy()  # perfect
    trait_ids = np.array(["O", "C", "E", "A", "N"] * 2)
    reverse_ids = np.zeros(10, dtype=np.float64)

    metrics = evaluate_predictions(y_true, y_pred, trait_ids, reverse_ids)
    il = metrics["item_level"]
    check(
        abs(il["pearson_r"][0] - 1.0) < 0.01,
        f"Perfect prediction: r={il['pearson_r'][0]:.4f} ≈ 1.0",
    )
    check(il["mae"][0] < 0.01, "Perfect prediction: MAE ≈ 0")


# ===================================================================
# _mean_ci helper
# ===================================================================


def test_mean_ci():
    print("\n--- _mean_ci ---")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = _mean_ci(data)
    check(abs(mean - 3.0) < 0.01, f"Mean = {mean:.4f}")
    check(lo < mean < hi, "CI bounds are ordered")

    # All NaN
    nan_data = np.array([np.nan, np.nan])
    m2, lo2, hi2 = _mean_ci(nan_data)
    check(np.isnan(m2) and np.isnan(lo2) and np.isnan(hi2), "All NaN → all NaN")

    # Single value
    single = np.array([42.0])
    m3, lo3, hi3 = _mean_ci(single)
    check(abs(m3 - 42.0) < 0.01 and np.isnan(lo3), "Single value: mean OK, CI NaN")


# ===================================================================
# Integration: full pipeline with real data
# ===================================================================


def test_integration_real_data():
    print("\n--- Integration: real data pipeline ---")

    Y = np.load(DATA_DIR / "Y.npy")
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    trait_ids = metadata["trait_id"].values
    reverse_ids = metadata["reverse_id"].values.astype(np.float64)

    n_subj, n_items = Y.shape
    check(n_subj == 2749, f"Y has 2749 subjects (got {n_subj})")
    check(n_items == 100, f"Y has 100 items (got {n_items})")

    # Full pipeline: split → simulate → evaluate
    folds = participant_cv_split(n_subj, n_folds=5, seed=0)
    train_idx, test_idx = folds[0]

    # Simulate: select first 10 items as S
    S = np.arange(10)
    y_test_true = Y[test_idx]
    y_obs, y_held = simulate_real_testing(y_test_true, S)

    check(y_obs.shape == y_test_true.shape, "y_obs shape matches")
    check(y_held.shape == y_test_true.shape, "y_held shape matches")

    # Dummy prediction: mean of observed items + tiny noise per subject
    # (pure constant predictions make Pearson r undefined)
    rng = np.random.RandomState(0)
    y_pred = np.full_like(y_test_true, np.nan)
    for si in range(len(test_idx)):
        obs_vals = y_obs[si, S]
        pred_val = np.mean(obs_vals)
        # Add tiny jitter so Pearson r is computable
        noise = rng.randn(n_items) * 0.01
        y_pred[si] = pred_val + noise
    y_pred[:, S] = np.nan  # don't "predict" observed items

    # Evaluate
    metrics = evaluate_predictions(y_test_true, y_pred, trait_ids, reverse_ids)

    # Check all keys present
    check("item_level" in metrics, "Integration: item_level present")
    check("trait_level" in metrics, "Integration: trait_level present")
    check("profile_correlation" in metrics, "Integration: profile_correlation present")

    il = metrics["item_level"]
    check(
        not np.isnan(il["pearson_r"][0]),
        f"Integration: item r = {il['pearson_r'][0]:.4f} (not NaN)",
    )

    tl = metrics["trait_level"]
    check(
        not np.isnan(tl["mean_big5_r"]),
        f"Integration: mean Big5 r = {tl['mean_big5_r']:.4f} (not NaN)",
    )

    pc = metrics["profile_correlation"]
    check(
        not np.isnan(pc["mean"]),
        f"Integration: profile r = {pc['mean']:.4f} (not NaN)",
    )

    # trait scores should have valid shapes
    check(tl["trait_scores_true"].shape == (len(test_idx), 5), "True trait scores shape")
    check(tl["trait_scores_imputed_full"].shape == (len(test_idx), 5), "Imputed trait scores shape")
    check(tl["trait_scores_short_form"].shape == (len(test_idx), 5), "Short form trait scores shape")
    check(tl["trait_scores_held_out"].shape == (len(test_idx), 5), "Held-out trait scores shape")

    print("  [OK] Full integration pipeline completed")


# ===================================================================
# Main
# ===================================================================


def main():
    global PASSED, FAILED
    print("=" * 60)
    print("F003: cv_framework — Unit Tests")
    print("=" * 60)

    # AC001
    test_cv_split_ratios()
    test_inner_validation_split()

    # AC002
    test_reverse_score()

    # AC003
    test_compute_trait_scores()
    test_profile_correlation()

    # simulate
    test_simulate_real_testing()

    # AC004
    test_evaluate_predictions_format()
    test_evaluate_predictions_perfect()
    test_mean_ci()

    # Integration
    test_integration_real_data()

    # Summary
    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"Results: {PASSED}/{total} passed, {FAILED}/{total} failed")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
F004: Random and Balanced-Random Selection Strategy Evaluation.

Runs 5-fold participant-level CV with Random and Balanced-Random item selection
across four item-count ratios (10, 30, 50, 90 % of 100 items), repeating each
random selection 50 times per outer fold for stable baselines.

Prediction: per-subject item-level KNN (K=5, cosine distance on E_old embeddings).
Evaluation: item-level Pearson r and trait-level per-trait r via
``cv_framework.evaluate_predictions``.

Usage::

    python scripts/run_selection_baselines.py              # full run
    python scripts/run_selection_baselines.py --quick       # 1 fold, 2 repeats
    python scripts/run_selection_baselines.py --smoke       # fastest — 1 fold, 1 repeat, 1 ratio
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
N_REPEATS = 50
RATIOS = (10, 30, 50, 90)
STRATEGIES = ("Random", "BalancedRandom")
TRAIT_ORDER = ("O", "C", "E", "A", "N")


def _resolve_imports():
    """Make local scripts importable."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


_resolve_imports()
from cv_framework import participant_cv_split, evaluate_predictions  # noqa: E402
from selection import RandomSelector, BalancedRandomSelector          # noqa: E402


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
    # E is already L2-normalised, so E @ E.T gives cosine similarities
    sim = E @ E.T  # (100, 100)
    # Clamp to [-1, 1] to avoid fp rounding issues
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
    # dist[S, :][:, T] → shape (|S|, |T|)
    d_st = dist[np.ix_(S, T)]  # (|S|, |T|)

    # Sort distances for each column (held-out item); take top k
    # For small |S|, argsort is fast
    if k_eff < len(S):
        # partial sort — only need k smallest
        nn_idx_in_S = np.argpartition(d_st, k_eff - 1, axis=0)[:k_eff]
    else:
        nn_idx_in_S = np.arange(len(S))[:, None].repeat(len(T), axis=1)

    # Map back to item indices
    nn_items = S[nn_idx_in_S]  # (k_eff, |T|)

    # For each test subject, average the responses of the k nearest neighbours
    for si in range(y_test.shape[0]):
        y_subj = y_test[si]
        # y_subj[nn_items] → (k_eff, |T|)
        preds = y_subj[nn_items].mean(axis=0)

        # Round and clamp
        preds = np.round(preds)
        preds = np.clip(preds, 1, 5)

        y_pred[si, T] = preds

    return y_pred


# ---------------------------------------------------------------------------
# Single fold evaluation
# ---------------------------------------------------------------------------


def _eval_fold(
    y_test: np.ndarray,
    D: np.ndarray,
    selectors: dict,
    trait_ids: np.ndarray,
    reverse_ids: np.ndarray,
    fold_idx: int,
    quick: bool = False,
    smoke: bool = False,
) -> list[dict]:
    """Evaluate all strategy × ratio × repeat combinations for one outer fold.

    Parameters
    ----------
    y_test : np.ndarray  shape (n_test, 100)
        Test-subject response matrix (reverse-scored).
    D : np.ndarray  shape (100, 100)
        Precomputed cosine distance matrix.
    selectors : dict
        ``{"Random": RandomSelector, "BalancedRandom": BalancedRandomSelector}``.
    trait_ids, reverse_ids : np.ndarray
        Item metadata for ``evaluate_predictions``.
    fold_idx : int
        Fold number (for logging).
    quick : bool
        If True, use reduced repeats (2 instead of 50).
    smoke : bool
        If True, use minimal repeats (1) and only one ratio (10).

    Returns
    -------
    rows : list of dict
        One dict per strategy×ratio×repeat with metric values.
    """
    n_test = y_test.shape[0]
    ratios = (10,) if smoke else RATIOS
    n_repeats = 1 if smoke else (2 if quick else N_REPEATS)
    rows = []

    for strategy_name in STRATEGIES:
        sel = selectors[strategy_name]
        for m in ratios:
            t0 = time.time()
            acc_item_r = np.full(n_repeats, np.nan)
            acc_trait_r: dict[str, list[float]] = {t: [] for t in TRAIT_ORDER}
            acc_mean_big5_r = np.full(n_repeats, np.nan)

            for rep in range(n_repeats):
                S = sel.select(m)

                # Predict all test subjects at once (vectorised)
                y_pred = _predict_held_out_batch(y_test, D, S, k=K_NEIGHBORS)

                # Evaluate
                metrics = evaluate_predictions(
                    y_test, y_pred, trait_ids, reverse_ids
                )

                # Collect metrics
                item_r = metrics["item_level"]["pearson_r"][0]
                acc_item_r[rep] = item_r
                acc_mean_big5_r[rep] = metrics["trait_level"]["mean_big5_r"]

                for trait in TRAIT_ORDER:
                    acc_trait_r[trait].append(
                        metrics["trait_level"]["per_trait_r"].get(trait, np.nan)
                    )

                rows.append({
                    "strategy": strategy_name,
                    "ratio": m,
                    "fold": fold_idx,
                    "repeat": rep,
                    "item_r": item_r,
                    "trait_r_O": metrics["trait_level"]["per_trait_r"].get("O", np.nan),
                    "trait_r_C": metrics["trait_level"]["per_trait_r"].get("C", np.nan),
                    "trait_r_E": metrics["trait_level"]["per_trait_r"].get("E", np.nan),
                    "trait_r_A": metrics["trait_level"]["per_trait_r"].get("A", np.nan),
                    "trait_r_N": metrics["trait_level"]["per_trait_r"].get("N", np.nan),
                    "trait_r_mean": metrics["trait_level"]["mean_big5_r"],
                })

            elapsed = time.time() - t0
            mean_r = np.nanmean(acc_item_r)
            mean_big5 = np.nanmean(acc_mean_big5_r)
            print(f"  [{strategy_name:>16}] m={m:>2}  "
                  f"item_r={mean_r:.4f}  big5_r={mean_big5:.4f}  "
                  f"({n_repeats} reps, {elapsed:.1f}s)")

    return rows


# ---------------------------------------------------------------------------
# Aggregate across repeats
# ---------------------------------------------------------------------------


def _aggregate_results(detail_rows: list[dict]) -> pd.DataFrame:
    """Compute per-fold mean metrics by averaging across repeats.

    Returns a DataFrame with columns: strategy, ratio, fold, item_r,
    trait_r_O/C/E/A/N, trait_r_mean (means across repeats).
    """
    df_detail = pd.DataFrame(detail_rows)
    group_cols = ["strategy", "ratio", "fold"]
    metric_cols = ["item_r"] + [f"trait_r_{t}" for t in TRAIT_ORDER] + ["trait_r_mean"]

    agg = df_detail.groupby(group_cols)[metric_cols].mean().reset_index()
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    quick = "--quick" in sys.argv
    smoke = "--smoke" in sys.argv

    mode_str = "SMOKE (1 fold, 1 rep, 1 ratio)" if smoke else (
        "QUICK (reduced repeats)" if quick else "FULL"
    )
    print("=" * 60)
    print(f"F004: Selection Strategy Evaluation — {mode_str}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data ...")
    Y, E_old, metadata = load_data()
    trait_ids = metadata["trait_id"].values
    reverse_ids = metadata["reverse_id"].values.astype(np.float64)
    n_subjects = Y.shape[0]

    # 1b. Precompute cosine distance matrix (once for entire run)
    D = _precompute_cosine_dist(E_old)
    print(f"[OK] Cosine distance matrix: {D.shape}")

    # 2. Create selectors
    print("\n[2/5] Creating selectors ...")
    selectors = {
        "Random": RandomSelector(n_items=100, seed=RANDOM_STATE),
        "BalancedRandom": BalancedRandomSelector(
            trait_ids=trait_ids, seed=RANDOM_STATE
        ),
    }
    for name, sel in selectors.items():
        print(f"    {name}: {type(sel).__name__} [OK]")

    # 3. Participant CV split
    print(f"\n[3/5] Running {N_FOLDS}-fold participant CV ...")
    folds = participant_cv_split(n_subjects, n_folds=N_FOLDS, seed=RANDOM_STATE)

    actual_folds = 1 if smoke else N_FOLDS
    all_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds[:actual_folds]):
        y_test = Y[test_idx]  # (n_test, 100), reverse-scored
        print(f"\n--- Fold {fold_idx + 1}/{actual_folds} "
              f"(train={len(train_idx)}, test={len(test_idx)}) ---")

        fold_rows = _eval_fold(
            y_test=y_test,
            D=D,
            selectors=selectors,
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

    print("\n--- Per-strategy × ratio (averaged across folds & repeats) ---")
    summary_group = ["strategy", "ratio"]
    summary = df_agg.groupby(summary_group)[
        ["item_r", "trait_r_mean"] + [f"trait_r_{t}" for t in TRAIT_ORDER]
    ].mean().reset_index()
    for _, row in summary.iterrows():
        parts = [f"{row['item_r']:.4f}"]
        parts += [f"{row[f'trait_r_{t}']:.4f}" for t in TRAIT_ORDER]
        print(f"  {row['strategy']:>16}  m={int(row['ratio']):>2}  "
              f"item_r={'  '.join(parts[:1])}  "
              f"big5={row['trait_r_mean']:.4f}")

    # 5. Save
    print(f"\n[5/5] Saving results ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTPUT_DIR / "random_baseline_detail.csv"
    df_detail.to_csv(detail_path, index=False)
    print(f"[OK] Detail results ({len(df_detail)} rows) → {detail_path}")

    agg_path = OUTPUT_DIR / "random_baseline_aggregated.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"[OK] Aggregated results ({len(df_agg)} rows) → {agg_path}")

    summary_path = OUTPUT_DIR / "random_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Summary ({len(summary)} rows) → {summary_path}")

    print("\n" + "=" * 60)
    print("F004 COMPLETE.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

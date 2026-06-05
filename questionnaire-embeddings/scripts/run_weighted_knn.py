#!/usr/bin/env python3
"""
F008: Cosine Weighted KNN Predictor Evaluation.

Runs 5-fold participant-level CV comparing Cosine-Weighted KNN against
Uniform-Weight KNN (baseline) across four item-count ratios (10, 30, 50, 90
items selected via Coverage strategy from Phase 1).

For each fold and ratio, the best K ∈ {3, 5, 7, 10, 15} is selected
**per predictor** via inner validation on train participants (AC002).
Final evaluation uses the tuned K on the held-out test participants.

Prediction: ``predictors.CosineWeightedKNN`` and ``predictors.UniformKNN``.
Evaluation: item-level Pearson r and trait-level per-trait r via
``cv_framework.evaluate_predictions``.

Usage::

    python scripts/run_weighted_knn.py              # full run
    python scripts/run_weighted_knn.py --quick       # 1 fold, 2 ratios
    python scripts/run_weighted_knn.py --smoke       # 1 fold, 1 ratio
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2"

RANDOM_STATE = 0
N_FOLDS = 5
RATIOS = (10, 30, 50, 90)
K_CANDIDATES = (3, 5, 7, 10, 15)
PREDICTOR_NAMES = ("UniformKNN", "CosineWeightedKNN")
TRAIT_ORDER = ("O", "C", "E", "A", "N")


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
from selection import CoverageSelector  # noqa: E402
from predictors import UniformKNN, CosineWeightedKNN  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load F001 pre-processed data.

    Returns
    -------
    Y : np.ndarray  (2749, 100) float64 — reverse-scored responses
    E_old : np.ndarray  (100, 1024) float64 — L2-normalised SBERT embeddings
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
# Precompute cosine similarity matrix
# ---------------------------------------------------------------------------


def _precompute_cosine_sim(E: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix from L2-normalised embeddings.

    For L2-normalised vectors, ``cosine_sim(i,j) = dot(e_i, e_j)``.
    Returns shape ``(n_items, n_items)`` with values in [-1, 1].
    """
    sim = E @ E.T  # (100, 100)
    sim = np.clip(sim, -1.0, 1.0)  # guard against fp rounding
    return sim


# ---------------------------------------------------------------------------
# Fold evaluation (with inner validation for K tuning)
# ---------------------------------------------------------------------------


def _make_row(
    strategy: str,
    ratio: int,
    fold: int,
    best_K: int,
    item_metrics: dict,
    trait_metrics: dict,
    profile_r: dict,
    coverage_val: float,
    redundancy_val: float,
    selected_S: np.ndarray,
    inner_val_scores: dict[str, float],
) -> dict:
    """Build a single results row."""
    per_trait = trait_metrics.get("per_trait_r", {})
    return {
        "predictor": strategy,
        "ratio": ratio,
        "fold": fold,
        "best_K": best_K,
        "item_r": item_metrics["pearson_r"][0],
        "item_r_ci_lower": item_metrics["pearson_r"][1],
        "item_r_ci_upper": item_metrics["pearson_r"][2],
        "item_mae": item_metrics["mae"][0],
        "item_rmse": item_metrics["rmse"][0],
        "item_rounded_accuracy": item_metrics["rounded_accuracy"][0],
        "trait_r_O": per_trait.get("O", np.nan),
        "trait_r_C": per_trait.get("C", np.nan),
        "trait_r_E": per_trait.get("E", np.nan),
        "trait_r_A": per_trait.get("A", np.nan),
        "trait_r_N": per_trait.get("N", np.nan),
        "trait_r_mean": trait_metrics.get("mean_big5_r", np.nan),
        "profile_r": profile_r.get("mean", np.nan),
        "coverage": coverage_val,
        "redundancy": redundancy_val,
        "selected_S": ",".join(map(str, sorted(selected_S))),
        "inner_val_K_scores": str(inner_val_scores),
    }


def _eval_fold(
    y_train: np.ndarray,
    y_test: np.ndarray,
    sim: np.ndarray,
    E: np.ndarray,
    trait_ids: np.ndarray,
    reverse_ids: np.ndarray,
    fold_idx: int,
    quick: bool = False,
    smoke: bool = False,
) -> list[dict]:
    """Evaluate all predictor × ratio × K combinations for one outer fold.

    For each ratio *m*, selects items via Coverage on the full *E* matrix
    (semantic coverage is unsupervised — it does not use *y_train*).
    Then performs inner validation on train participants to choose the
    best K per predictor, and evaluates on test participants with the
    tuned K.

    Parameters
    ----------
    y_train : np.ndarray  shape (n_train, 100)
    y_test : np.ndarray  shape (n_test, 100)
    sim : np.ndarray  shape (100, 100) — cosine similarity
    E : np.ndarray  shape (100, 1024) — L2-normalised embeddings
    trait_ids, reverse_ids : np.ndarray — item metadata
    fold_idx : int
    quick, smoke : bool — mode flags

    Returns
    -------
    rows : list[dict]
    """
    ratios = (10,) if smoke else ((10, 50) if quick else RATIOS)
    n_train = len(y_train)
    rows: list[dict] = []

    # Coverage selector (unsupervised — only needs embeddings)
    cov_sel = CoverageSelector(E)

    # --- Inner validation split (AC002) ---
    train_inner_idx, valid_inner_idx = inner_validation_split(
        np.arange(n_train), val_ratio=0.2, seed=RANDOM_STATE + fold_idx
    )
    y_train_inner = y_train[train_inner_idx]
    y_valid_inner = y_train[valid_inner_idx]
    n_train_inner = len(y_train_inner)
    n_valid_inner = len(y_valid_inner)

    for m in ratios:
        print(f"  [{fold_idx+1}] ratio m={m}: selecting items via Coverage ...")

        # Select items via Coverage (unsupervised)
        S = cov_sel.select(m)
        cov_val = cov_sel.compute_coverage(S)
        red_val = cov_sel.compute_redundancy(S)

        # --- Inner validation: tune K for each predictor ---
        best_K: dict[str, int] = {}
        inner_val_scores: dict[str, dict[str, float]] = {}

        for pred_name in PREDICTOR_NAMES:
            best_inner_score = -np.inf
            best_k = K_CANDIDATES[0]

            for K in K_CANDIDATES:
                # Build predictor
                if pred_name == "UniformKNN":
                    pred = UniformKNN(K=K)
                else:
                    pred = CosineWeightedKNN(K=K)

                # Predict on valid-inner subjects
                y_pred_inner = pred.predict(y_valid_inner, sim, S)

                # Evaluate
                metrics = evaluate_predictions(
                    y_valid_inner, y_pred_inner, trait_ids, reverse_ids
                )
                inner_score = metrics["item_level"]["pearson_r"][0]

                if inner_score > best_inner_score:
                    best_inner_score = inner_score
                    best_k = K

            best_K[pred_name] = best_k
            inner_val_scores[pred_name] = {
                str(k): float(
                    evaluate_predictions(
                        y_valid_inner,
                        (
                            UniformKNN(K=k).predict(y_valid_inner, sim, S)
                            if pred_name == "UniformKNN"
                            else CosineWeightedKNN(K=k).predict(y_valid_inner, sim, S)
                        ),
                        trait_ids,
                        reverse_ids,
                    )["item_level"]["pearson_r"][0]
                )
                for k in K_CANDIDATES
            }
            print(
                f"    [{pred_name}] inner val: best K={best_k} "
                f"(item_r={best_inner_score:.4f})"
            )

        # --- Test evaluation with tuned K ---
        for pred_name in PREDICTOR_NAMES:
            K_best = best_K[pred_name]

            if pred_name == "UniformKNN":
                pred = UniformKNN(K=K_best)
            else:
                pred = CosineWeightedKNN(K=K_best)

            y_pred_test = pred.predict(y_test, sim, S)
            metrics = evaluate_predictions(
                y_test, y_pred_test, trait_ids, reverse_ids
            )

            row = _make_row(
                strategy=pred_name,
                ratio=m,
                fold=fold_idx,
                best_K=K_best,
                item_metrics=metrics["item_level"],
                trait_metrics=metrics["trait_level"],
                profile_r=metrics["profile_correlation"],
                coverage_val=cov_val,
                redundancy_val=red_val,
                selected_S=S,
                inner_val_scores=inner_val_scores[pred_name],
            )
            rows.append(row)

            item_r = metrics["item_level"]["pearson_r"]
            trait_r = metrics["trait_level"].get("mean_big5_r", np.nan)
            print(
                f"    [{pred_name}] test: K={K_best}  "
                f"item_r={item_r[0]:.4f} [{item_r[1]:.4f}, {item_r[2]:.4f}]  "
                f"big5_r={trait_r:.4f}"
            )

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_results(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-fold results into per-predictor×ratio summaries.

    Parameters
    ----------
    rows : list[dict]
        Per-fold detail rows.

    Returns
    -------
    df_detail : pd.DataFrame
    df_agg : pd.DataFrame — mean ± CI across folds per predictor × ratio
    """
    df = pd.DataFrame(rows)

    metric_cols = [
        "item_r", "item_mae", "item_rmse", "item_rounded_accuracy",
        "trait_r_O", "trait_r_C", "trait_r_E", "trait_r_A", "trait_r_N",
        "trait_r_mean", "profile_r", "coverage", "redundancy",
    ]

    agg_parts = []
    for (pred_name, ratio), grp in df.groupby(["predictor", "ratio"]):
        agg_row = {"predictor": pred_name, "ratio": ratio, "n_folds": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            if len(vals) > 0:
                mean_val = vals.mean()
                # CI via t-distribution over folds
                from scipy import stats as sp_stats
                se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                t_crit = sp_stats.t.ppf(0.975, df=max(1, len(vals) - 1))
                h = t_crit * se
                agg_row[col] = mean_val
                agg_row[f"{col}_ci_lower"] = mean_val - h
                agg_row[f"{col}_ci_upper"] = mean_val + h
            else:
                agg_row[col] = np.nan
                agg_row[f"{col}_ci_lower"] = np.nan
                agg_row[f"{col}_ci_upper"] = np.nan
        # Best K: mode across folds
        if "best_K" in grp.columns:
            agg_row["best_K_mode"] = int(grp["best_K"].mode().iloc[0]) if len(grp["best_K"].mode()) > 0 else np.nan
        agg_parts.append(agg_row)

    df_agg = pd.DataFrame(agg_parts)
    return df, df_agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point."""
    quick = "--quick" in sys.argv
    smoke = "--smoke" in sys.argv

    mode_str = "SMOKE (1 fold, 1 ratio)" if smoke else (
        "QUICK (1 fold, 2 ratios)" if quick else "FULL"
    )

    print("=" * 60)
    print(f"F008: Weighted KNN Evaluation — {mode_str}")
    print("=" * 60)
    print()

    # ---- 1. Load data ----
    print("[1/5] Loading data ...")
    Y, E_old, metadata = load_data()
    trait_ids = metadata["trait_id"].values
    reverse_ids = metadata["reverse_id"].values.astype(np.float64)
    n_subjects = Y.shape[0]
    print()

    # ---- 2. Precompute similarity matrix ----
    print("[2/5] Computing cosine similarity matrix ...")
    sim = _precompute_cosine_sim(E_old)
    print(f"[OK] sim shape: {sim.shape}  range=[{sim.min():.4f}, {sim.max():.4f}]")
    print()

    # ---- 3. Create predictors reference ----
    print("[3/5] Predictors: UniformKNN, CosineWeightedKNN")
    print(f"      K candidates: {list(K_CANDIDATES)}")
    print(f"      Ratios: {list(RATIOS)}")
    print()

    # ---- 4. Run 5-fold CV ----
    print("[4/5] Running participant CV ...")
    folds = participant_cv_split(n_subjects, n_folds=N_FOLDS, seed=RANDOM_STATE)
    actual_folds = 1 if smoke else N_FOLDS
    all_rows: list[dict] = []

    t0 = time.perf_counter()

    for fold_idx, (train_idx, test_idx) in enumerate(folds[:actual_folds]):
        y_train = Y[train_idx]
        y_test = Y[test_idx]

        print(f"--- Fold {fold_idx + 1}/{actual_folds} "
              f"(train={len(train_idx)}, test={len(test_idx)}) ---")

        fold_rows = _eval_fold(
            y_train=y_train,
            y_test=y_test,
            sim=sim,
            E=E_old,
            trait_ids=trait_ids,
            reverse_ids=reverse_ids,
            fold_idx=fold_idx,
            quick=quick,
            smoke=smoke,
        )
        all_rows.extend(fold_rows)
        print()

    elapsed = time.perf_counter() - t0
    print(f"[OK] CV completed in {elapsed:.1f}s ({len(all_rows)} rows)")
    print()

    # ---- 5. Aggregate & save ----
    print("[5/5] Aggregating and saving results ...")
    df_detail, df_agg = _aggregate_results(all_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTPUT_DIR / "weighted_knn_detail.csv"
    df_detail.to_csv(detail_path, index=False)
    print(f"[OK] Detail results ({len(df_detail)} rows) → {detail_path}")

    agg_path = OUTPUT_DIR / "weighted_knn_aggregated.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"[OK] Aggregated results ({len(df_agg)} rows) → {agg_path}")

    # Summary: re-save aggregated as summary (same format as Phase 1 convention)
    summary_path = OUTPUT_DIR / "weighted_knn_summary.csv"
    df_agg.to_csv(summary_path, index=False)
    print(f"[OK] Summary results → {summary_path}")

    # --- Print summary table ---
    print()
    print("--- Per-predictor × ratio (averaged across folds) ---")
    header = f"{'Predictor':<22s} {'m':>3s}  {'item_r':>8s}  {'big5_r':>8s}  {'best_K':>6s}"
    print(header)
    print("-" * len(header))
    for _, row in df_agg.iterrows():
        best_k_str = str(int(row.get("best_K_mode", np.nan))) if not pd.isna(row.get("best_K_mode", np.nan)) else "—"
        print(
            f"{row['predictor']:<22s} "
            f"{int(row['ratio']):3d}  "
            f"{row['item_r']:8.4f}  "
            f"{row['trait_r_mean']:8.4f}  "
            f"{best_k_str:>6s}"
        )

    # Compare: compute Δr per ratio
    print()
    print("--- Weighted KNN vs Uniform KNN (Δ item_r) ---")
    for ratio in sorted(df_detail["ratio"].unique()):
        subset = df_detail[df_detail["ratio"] == ratio]
        uni = subset[subset["predictor"] == "UniformKNN"]["item_r"]
        cos = subset[subset["predictor"] == "CosineWeightedKNN"]["item_r"]
        if len(uni) > 0 and len(cos) > 0:
            delta = (cos.values - uni.values).mean()
            print(f"  m={int(ratio):3d}: Δr = {delta:+.4f}")

    print()
    print("=" * 60)
    print("F008 COMPLETE.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

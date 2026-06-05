#!/usr/bin/env python3
"""
F008: KNN predictor module — Cosine Weighted KNN and Uniform KNN baseline.

Provides two predictor classes with a shared vectorised prediction interface:

* ``UniformKNN`` — standard KNN with uniform neighbour weights (baseline).
* ``CosineWeightedKNN`` — KNN where neighbours are weighted by semantic
  similarity ``sim+(i,j) = (cos(e_i,e_j)+1)/2``.

Both accept a precomputed cosine-similarity matrix (from L2-normalised
embeddings) and a selected-item set *S*, and return predictions for the
held-out items *T*.

Usage (import)::

    from predictors import UniformKNN, CosineWeightedKNN

Usage (stand-alone demo / smoke test)::

    python scripts/predictors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 0
N_ITEMS = 100
K_CANDIDATES = (3, 5, 7, 10, 15)


def _resolve_project_root() -> Path:
    """Return the project root (questionnaire-embeddings/)."""
    return Path(__file__).resolve().parent.parent


def _resolve_imports():
    """Make local scripts importable."""
    scripts_dir = str(_resolve_project_root() / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


# ---------------------------------------------------------------------------
# Base predictor with shared vectorised prediction logic
# ---------------------------------------------------------------------------


class _BaseKNN:
    """Shared base: neighbour lookup + subject-level prediction loop.

    Subclasses override ``_compute_weights`` to implement different
    weighting schemes (uniform, cosine-similarity, etc.).
    """

    def __init__(self, K: int = 5):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        y_test: np.ndarray,
        sim: np.ndarray,
        S: np.ndarray,
    ) -> np.ndarray:
        """Predict held-out items for all test subjects.

        Parameters
        ----------
        y_test : np.ndarray  shape ``(n_test, n_items)``
            Reverse-scored test-subject response matrix.
        sim : np.ndarray  shape ``(n_items, n_items)``
            Precomputed cosine-similarity matrix (values in [-1, 1]).
            For L2-normalised embeddings this is ``E @ E.T``.
        S : np.ndarray  shape ``(|S|,)``
            Indices of selected (observed / administered) items.

        Returns
        -------
        y_pred : np.ndarray  shape ``(n_test, n_items)``
            Predictions.  Items in *S* are ``NaN`` (observed, not predicted).
        """
        n_items = sim.shape[0]

        # Identify held-out items T
        T_mask = np.ones(n_items, dtype=bool)
        T_mask[S] = False
        T = np.where(T_mask)[0]

        y_pred = np.full_like(y_test, np.nan, dtype=np.float64)

        if len(T) == 0:
            return y_pred

        k_eff = min(self.K, len(S))  # AC003: cap K to |S|

        # --- Find k nearest neighbours in S for each held-out item ---
        # Extract similarity submatrix: rows=S (candidates), cols=T (targets)
        sim_st = sim[np.ix_(S, T)]  # (|S|, |T|)

        # Mask self-similarity: an item j in T might also be in S
        # (though typically S ∩ T = ∅).  Prevent self-match by setting
        # self-sim to -inf so it sorts last.
        for ti, tj in enumerate(T):
            if tj in S:
                si = np.where(S == tj)[0][0]
                sim_st[si, ti] = -np.inf

        # Partial sort — get indices of k_eff largest similarities per column
        if k_eff < len(S):
            nn_idx_in_S = np.argpartition(-sim_st, k_eff - 1, axis=0)[:k_eff]
        else:
            nn_idx_in_S = np.arange(len(S))[:, None].repeat(len(T), axis=1)

        # Map row indices → actual item indices
        nn_items = S[nn_idx_in_S]  # (k_eff, |T|)  item indices of neighbours

        # --- Precompute weights for each (target, neighbour) pair ---
        # sim_st[nn_idx_in_S, range(|T|)] → (k_eff, |T|) raw similarities
        nn_sims = np.take_along_axis(sim_st, nn_idx_in_S, axis=0)
        weights = self._compute_weights(nn_sims)  # (k_eff, |T|)

        # --- Per-subject weighted average ---
        for si in range(y_test.shape[0]):
            y_subj = y_test[si]  # (n_items,)
            neighbour_responses = y_subj[nn_items]  # (k_eff, |T|)

            # Weighted mean: ŷ_j = Σ(w_ij × y_i) / Σ(w_ij)
            numerator = np.sum(weights * neighbour_responses, axis=0)
            denominator = np.sum(weights, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                preds = np.where(denominator > 0, numerator / denominator, np.nan)

            # Round and clamp (AC004)
            preds = np.round(preds)
            preds = np.clip(preds, 1.0, 5.0)

            y_pred[si, T] = preds

        return y_pred

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _compute_weights(self, similarities: np.ndarray) -> np.ndarray:
        """Convert raw cosine similarities to prediction weights.

        Parameters
        ----------
        similarities : np.ndarray  shape ``(k_eff, |T|)``
            Cosine similarity between each neighbour and each target item.
            Values in [-1, 1].

        Returns
        -------
        weights : np.ndarray  shape ``(k_eff, |T|)``
            Non-negative weights used in the weighted average.
        """
        raise NotImplementedError("subclass must implement _compute_weights")


# ---------------------------------------------------------------------------
# UniformKNN — baseline KNN with equal weights
# ---------------------------------------------------------------------------


class UniformKNN(_BaseKNN):
    """Standard KNN regression with uniform neighbour weights.

    This reproduces the behaviour of ``_predict_held_out_batch`` in the
    Phase 1 runner scripts.  Every neighbour contributes equally to the
    prediction, regardless of semantic distance.
    """

    def _compute_weights(self, similarities: np.ndarray) -> np.ndarray:
        """Return uniform weights — all ones."""
        return np.ones_like(similarities)


# ---------------------------------------------------------------------------
# CosineWeightedKNN — KNN weighted by shifted cosine similarity
# ---------------------------------------------------------------------------


class CosineWeightedKNN(_BaseKNN):
    """KNN regression with cosine-similarity-based weights.

    Weight for neighbour *i* predicting target *j*::

        w_ij = sim+(i,j) = (cos(e_i, e_j) + 1) / 2

    This maps cosine similarity from [-1, 1] to [0, 1], giving higher
    weight to semantically closer neighbours.
    """

    def _compute_weights(self, similarities: np.ndarray) -> np.ndarray:
        """Convert cosine similarities to sim+ weights in [0, 1]."""
        return (similarities + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Stand-alone smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> int:
    """Quick sanity check with synthetic data."""
    print("=" * 60)
    print("predictors.py — Smoke Test")
    print("=" * 60)

    # Synthetic data: 5 subjects, 10 items, dim=8
    n_subj, n_items, d = 5, 10, 8
    rng = np.random.default_rng(0)

    y = rng.integers(1, 6, size=(n_subj, n_items)).astype(np.float64)
    E = rng.normal(size=(n_items, d)).astype(np.float64)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)  # L2-normalise

    sim = np.clip(E @ E.T, -1.0, 1.0)
    S = np.array([0, 2, 5, 7])  # 4 selected items

    for name, cls in [("UniformKNN", UniformKNN), ("CosineWeightedKNN", CosineWeightedKNN)]:
        pred = cls(K=3)
        y_pred = pred.predict(y, sim, S)
        assert y_pred.shape == (n_subj, n_items), f"{name}: bad shape {y_pred.shape}"
        assert np.all(np.isnan(y_pred[:, S])), f"{name}: S items should be NaN"
        # Held-out items should have valid predictions (not NaN)
        held_out = np.setdiff1d(np.arange(n_items), S)
        assert np.all(~np.isnan(y_pred[:, held_out])), f"{name}: held-out items should not be NaN"
        assert np.all((y_pred[:, held_out] >= 1) & (y_pred[:, held_out] <= 5)), \
            f"{name}: predictions out of [1,5] range"
        print(f"  [OK] {name}(K=3): {n_subj}×{n_items}, |S|={len(S)}, |T|={len(held_out)}")

    # Test AC003: K > |S| for CosineWeightedKNN
    pred_AC003 = CosineWeightedKNN(K=10)
    y_pred_ac3 = pred_AC003.predict(y, sim, S)
    assert y_pred_ac3.shape == (n_subj, n_items)
    assert np.all(~np.isnan(y_pred_ac3[:, held_out])), "AC003: K>|S| should still produce predictions"
    print("  [OK] AC003: K=10 > |S|=4 → K_eff=4, predictions valid")

    # Test AC004: predictions in [1,5]
    assert np.all((y_pred_ac3[:, held_out] >= 1) & (y_pred_ac3[:, held_out] <= 5)), \
        "AC004: predictions outside [1,5]"
    print("  [OK] AC004: All predictions in [1,5]")

    # Verify weighted differs from uniform (non-trivial with non-uniform sim)
    y_uniform = UniformKNN(K=3).predict(y, sim, S)
    y_weighted = CosineWeightedKNN(K=3).predict(y, sim, S)
    if not np.allclose(y_uniform, y_weighted, equal_nan=True):
        print("  [OK] Weighted ≠ Uniform (different predictions)")
    else:
        print("  [INFO] Weighted == Uniform (synthetic data may produce ties)")

    # Test with real data if available
    data_dir = _resolve_project_root() / "data" / "processed"
    if data_dir.exists():
        try:
            Y_real = np.load(data_dir / "Y.npy").astype(np.float64)
            E_real = np.load(data_dir / "E_old.npy").astype(np.float64)
            sim_real = np.clip(E_real @ E_real.T, -1.0, 1.0)
            S_real = np.array([0, 2, 5, 7, 10, 15, 20, 25, 30, 35])

            for name, cls in [("UniformKNN", UniformKNN), ("CosineWeightedKNN", CosineWeightedKNN)]:
                p = cls(K=5)
                yp = p.predict(Y_real[:100], sim_real, S_real)
                held = np.setdiff1d(np.arange(100), S_real)
                valid = yp[:100, held]
                assert valid.shape[1] == 90, f"{name}: bad T shape"
                assert np.all((valid >= 1) & (valid <= 5)), \
                    f"{name}: predictions out of [1,5] with real data"
                print(f"  [OK] {name} on real data: preds range [{valid.min():.0f}, {valid.max():.0f}]")
        except FileNotFoundError:
            print("  [SKIP] Real data not found, skipping integration test")

    print()
    print("=" * 60)
    print("Smoke test PASSED.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())

#!/usr/bin/env python3
"""Phase 4 prediction helpers.

This module intentionally does not modify ``scripts/predictors.py``.  The
historical Phase 1/2 predictors round then clip predictions; F013 Version A
uses continuous predictions clipped to [1, 5] as the primary analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftmaxPredictionPlan:
    """Precomputed neighbour plan for one S/K/tau/similarity matrix."""

    selected_items: np.ndarray
    heldout_items: np.ndarray
    nn_items: np.ndarray
    weights: np.ndarray


class ContinuousSoftmaxKNN:
    """Softmax-weighted KNN with continuous clip-only post-processing.

    This matches the Phase 2 SoftmaxKNN neighbour geometry and softmax weighting
    but deliberately skips rounding.  Predictions are weighted averages of
    observed responses and are clipped defensively to the valid Likert range.
    """

    def __init__(self, K: int = 5, tau: float = 0.1, clip_min: float = 1.0, clip_max: float = 5.0):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.K = int(K)
        self.tau = float(tau)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def compile(self, sim: np.ndarray, S: np.ndarray) -> SoftmaxPredictionPlan:
        """Precompute neighbours and softmax weights for a selected set."""
        sim = np.asarray(sim, dtype=np.float64)
        S = np.asarray(S, dtype=np.intp)
        n_items = sim.shape[0]

        t_mask = np.ones(n_items, dtype=bool)
        t_mask[S] = False
        T = np.where(t_mask)[0]

        if len(T) == 0:
            return SoftmaxPredictionPlan(S.copy(), T, np.empty((0, 0), dtype=np.intp), np.empty((0, 0), dtype=np.float64))

        k_eff = min(self.K, len(S))
        sim_st = sim[np.ix_(S, T)].copy()

        # Defensive self-match mask.  In F013 S and T are disjoint by construction.
        for ti, tj in enumerate(T):
            hit = np.where(S == tj)[0]
            if len(hit):
                sim_st[hit[0], ti] = -np.inf

        if k_eff < len(S):
            nn_idx_in_S = np.argpartition(-sim_st, k_eff - 1, axis=0)[:k_eff]
        else:
            nn_idx_in_S = np.arange(len(S))[:, None].repeat(len(T), axis=1)

        nn_items = S[nn_idx_in_S]
        nn_sims = np.take_along_axis(sim_st, nn_idx_in_S, axis=0)
        shifted = nn_sims - np.max(nn_sims, axis=0, keepdims=True)
        exp_sim = np.exp(shifted / self.tau)
        weights = exp_sim / np.sum(exp_sim, axis=0, keepdims=True)

        return SoftmaxPredictionPlan(S.copy(), T, nn_items, weights)

    def predict(self, y: np.ndarray, sim: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Predict held-out items; selected items remain NaN."""
        return self.predict_with_plan(y, self.compile(sim, S))

    def predict_with_plan(self, y: np.ndarray, plan: SoftmaxPredictionPlan) -> np.ndarray:
        """Predict using a precomputed plan."""
        y = np.asarray(y, dtype=np.float64)
        y_pred = np.full_like(y, np.nan, dtype=np.float64)
        T = plan.heldout_items
        if len(T) == 0:
            return y_pred

        # Vectorized over subjects.  ``plan.nn_items`` has shape (k, |T|), so
        # advanced indexing returns (n_subjects, k, |T|).  This preserves the
        # Phase 2 weighted-average math while avoiding a Python subject loop.
        neighbour_responses = y[:, plan.nn_items]
        numerator = np.sum(neighbour_responses * plan.weights[None, :, :], axis=1)
        denominator = np.sum(plan.weights, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            preds = np.where(denominator[None, :] > 0, numerator / denominator[None, :], np.nan)
        y_pred[:, T] = np.clip(preds, self.clip_min, self.clip_max)
        return y_pred


def round_clip_predictions(y_pred: np.ndarray, clip_min: float = 1.0, clip_max: float = 5.0) -> np.ndarray:
    """Supplemental rounded prediction copy; not used for primary F013 scoring."""
    rounded = np.asarray(y_pred, dtype=np.float64).copy()
    mask = ~np.isnan(rounded)
    rounded[mask] = np.clip(np.round(rounded[mask]), clip_min, clip_max)
    return rounded

#!/usr/bin/env python3
"""
F004: Random and Balanced-Random item-selection strategies.

Provides two baseline selectors for the Phase 1 experiment:

* ``RandomSelector`` — uniformly random selection of *m* items from the full pool.
* ``BalancedRandomSelector`` — balanced random selection ensuring each Big-Five
  trait contributes *m* // 5 items (remainder allocated randomly).

Usage (import)::

    from selection import RandomSelector, BalancedRandomSelector

Usage (stand-alone demo / smoke test)::

    python scripts/selection.py
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
TRAIT_ORDER = ("O", "C", "E", "A", "N")  # standard Big-Five OCEAN order
RATIOS = (10, 30, 50, 90)                # item counts (m)
N_REPEATS = 50                            # repeats per outer fold


def _resolve_project_root() -> Path:
    """Return the project root (questionnaire-embeddings/)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# RandomSelector
# ---------------------------------------------------------------------------


class RandomSelector:
    """Select *m* items uniformly at random from the pool of *n_items*.

    Parameters
    ----------
    n_items : int
        Total number of items in the pool (default 100).
    seed : int or None
        Random seed for reproducibility.  Pass an integer for deterministic
        selection; pass ``None`` for non-deterministic selection.
    """

    def __init__(self, n_items: int = N_ITEMS, seed: int | None = RANDOM_STATE):
        self.n_items = n_items
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def select(self, m: int) -> np.ndarray:
        """Select *m* item indices uniformly at random (without replacement).

        Parameters
        ----------
        m : int
            Number of items to select.

        Returns
        -------
        selected : np.ndarray  shape (m,)
            Sorted indices of the selected items.
        """
        if m < 1:
            return np.array([], dtype=np.intp)
        if m > self.n_items:
            raise ValueError(
                f"Cannot select {m} items from a pool of {self.n_items}"
            )
        indices = self.rng.choice(self.n_items, size=m, replace=False)
        indices.sort()
        return indices

    def select_multi(self, m: int, n_repeats: int = N_REPEATS) -> list[np.ndarray]:
        """Return *n_repeats* independent selections of size *m*.

        Parameters
        ----------
        m : int
            Number of items to select per repeat.
        n_repeats : int
            Number of independent draws (default 50).

        Returns
        -------
        selections : list of np.ndarray
            Each element is a sorted array of *m* item indices.
        """
        return [self.select(m) for _ in range(n_repeats)]


# ---------------------------------------------------------------------------
# BalancedRandomSelector
# ---------------------------------------------------------------------------


class BalancedRandomSelector:
    """Select *m* items with balanced representation across Big-Five traits.

    Each trait contributes ``m // 5`` items.  Any remainder (*r* = *m* % 5) is
    distributed by randomly picking *r* traits and giving each one extra item.
    Within each trait, items are selected uniformly at random.

    Parameters
    ----------
    trait_ids : np.ndarray  shape (n_items,)
        Trait label per item (e.g. ``"O"``, ``"C"``, …).
    trait_order : tuple of str
        Ordered trait labels (default OCEAN).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        trait_ids: np.ndarray,
        trait_order: tuple[str, ...] = TRAIT_ORDER,
        seed: int | None = RANDOM_STATE,
    ):
        self.trait_ids = np.asarray(trait_ids)
        self.trait_order = trait_order
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        # Build per-trait item pools
        self._pools: dict[str, np.ndarray] = {}
        for trait in trait_order:
            idx = np.where(self.trait_ids == trait)[0]
            if len(idx) == 0:
                raise ValueError(f"Trait '{trait}' not found in trait_ids")
            self._pools[trait] = idx

    def select(self, m: int) -> np.ndarray:
        """Select *m* items with balanced per-trait allocation.

        Parameters
        ----------
        m : int
            Number of items to select.

        Returns
        -------
        selected : np.ndarray  shape (m,)
            Sorted indices of the selected items.
        """
        if m < 1:
            return np.array([], dtype=np.intp)

        n_traits = len(self.trait_order)
        per_trait = m // n_traits
        remainder = m % n_traits

        selected = []

        # Allocate the extra items to *remainder* randomly chosen traits
        extra_traits = set(
            self.rng.choice(n_traits, size=remainder, replace=False)
        )

        for t_idx, trait in enumerate(self.trait_order):
            pool = self._pools[trait]
            n_pick = per_trait + (1 if t_idx in extra_traits else 0)
            if n_pick > len(pool):
                raise ValueError(
                    f"Cannot select {n_pick} items from trait '{trait}' "
                    f"(pool size {len(pool)})"
                )
            if n_pick > 0:
                picked = self.rng.choice(pool, size=n_pick, replace=False)
                selected.append(picked)

        if not selected:
            return np.array([], dtype=np.intp)
        result = np.concatenate(selected)
        result.sort()
        return result

    def select_multi(self, m: int, n_repeats: int = N_REPEATS) -> list[np.ndarray]:
        """Return *n_repeats* independent balanced selections of size *m*."""
        return [self.select(m) for _ in range(n_repeats)]


# ---------------------------------------------------------------------------
# CoverageSelector — greedy facility-location
# ---------------------------------------------------------------------------


class CoverageSelector:
    """Select *m* items by greedy maximisation of semantic coverage.

    Uses the **facility location** objective:

        Coverage(S) = mean_j max_{i∈S} sim⁺(i, j)

    where ``sim⁺(i, j) = (cos(e_i, e_j) + 1) / 2`` is the shifted cosine
    similarity in [0, 1].  The selector starts from an empty set and at each
    step adds the item that maximises Coverage(S ∪ {i}).

    Coverage is monotone and submodular, so greedy achieves a (1 − 1/e)
    approximation guarantee.

    Parameters
    ----------
    embeddings : np.ndarray  shape (n_items, d)
        Item embedding matrix.  Should already be L2-normalised for cosine
        similarity via dot product.
    """

    def __init__(self, embeddings: np.ndarray):
        self._embs = np.asarray(embeddings, dtype=np.float64)
        self.n_items = self._embs.shape[0]

        # Precompute sim⁺ matrix — shifted cosine similarity
        sim = self._embs @ self._embs.T
        sim = np.clip(sim, -1.0, 1.0)
        self._sim_plus = (sim + 1.0) / 2.0  # (n_items, n_items)

    # -- public API ----------------------------------------------------------

    def select(self, m: int) -> np.ndarray:
        """Greedy facility-location selection of *m* items.

        Parameters
        ----------
        m : int
            Number of items to select.

        Returns
        -------
        selected : np.ndarray  shape (m,)
            Sorted indices of the selected items.
        """
        if m < 1:
            return np.array([], dtype=np.intp)
        if m >= self.n_items:
            return np.arange(self.n_items, dtype=np.intp)

        S: list[int] = []
        remaining = list(range(self.n_items))  # list for deterministic iteration

        for _ in range(m):
            best_i = None
            best_cov = -np.inf
            for i in remaining:
                cand = tuple(S) + (i,)
                cov = self._coverage(cand)
                if cov > best_cov:
                    best_cov = cov
                    best_i = i
            S.append(best_i)
            remaining.remove(best_i)

        result = np.array(S, dtype=np.intp)
        result.sort()
        return result

    def select_multi(self, m: int, n_repeats: int = 1) -> list[np.ndarray]:
        """Return *n_repeats* selections (deterministic — all identical)."""
        s = self.select(m)
        return [s.copy() for _ in range(max(n_repeats, 1))]

    # -- metric accessors (for smoke tests / acceptance criteria) -------------

    def compute_coverage(self, S: np.ndarray) -> float:
        """Compute Coverage(S) for a given set of item indices."""
        if len(S) == 0:
            return 0.0
        return self._coverage(tuple(S))

    def compute_redundancy(self, S: np.ndarray) -> float:
        """Compute Redundancy(S) — mean pairwise sim⁺ within S."""
        if len(S) < 2:
            return 0.0
        return self._redundancy(tuple(S))

    @property
    def sim_plus_matrix(self) -> np.ndarray:
        """The precomputed sim⁺ matrix (n_items, n_items)."""
        return self._sim_plus

    # -- internal helpers ----------------------------------------------------

    def _coverage(self, S_indices: tuple[int, ...]) -> float:
        """Coverage = mean over all items j of max_{i∈S} sim⁺(i, j)."""
        if len(S_indices) == 0:
            return 0.0
        idx = list(S_indices)
        max_sim = self._sim_plus[idx, :].max(axis=0)  # (n_items,)
        return float(np.mean(max_sim))

    def _redundancy(self, S_indices: tuple[int, ...]) -> float:
        """Redundancy = mean of all unique pairwise sim⁺ within S."""
        n = len(S_indices)
        if n < 2:
            return 0.0
        idx = list(S_indices)
        sub = self._sim_plus[np.ix_(idx, idx)]
        # Upper triangle (k=1) excludes diagonal self-similarities
        triu_idx = np.triu_indices_from(sub, k=1)
        return float(np.mean(sub[triu_idx]))


# ---------------------------------------------------------------------------
# CoverageDiversitySelector — semantic coverage with redundancy penalty
# ---------------------------------------------------------------------------


class CoverageDiversitySelector:
    """Select *m* items balancing semantic coverage against within-S redundancy.

    At each greedy step, candidates are scored by:

        Score(i) = Coverage_z(S ∪ {i}) − λ × Redundancy_z(S ∪ {i})

    where Coverage_z and Redundancy_z are **z-score normalised** across all
    candidates at that step (so the two objectives are commensurable before
    the λ trade-off).

    Parameters
    ----------
    embeddings : np.ndarray  shape (n_items, d)
        Item embedding matrix (L2-normalised).
    lam : float
        Redundancy penalty weight (λ ∈ {0.25, 0.5, 1.0} recommended).
    """

    def __init__(self, embeddings: np.ndarray, lam: float = 0.5):
        self._embs = np.asarray(embeddings, dtype=np.float64)
        self.n_items = self._embs.shape[0]
        self.lam = lam

        # Precompute sim⁺ matrix
        sim = self._embs @ self._embs.T
        sim = np.clip(sim, -1.0, 1.0)
        self._sim_plus = (sim + 1.0) / 2.0  # (n_items, n_items)

    # -- public API ----------------------------------------------------------

    def select(self, m: int) -> np.ndarray:
        """Greedy Coverage−λ·Redundancy selection of *m* items.

        Parameters
        ----------
        m : int
            Number of items to select.

        Returns
        -------
        selected : np.ndarray  shape (m,)
            Sorted indices of the selected items.
        """
        if m < 1:
            return np.array([], dtype=np.intp)
        if m >= self.n_items:
            return np.arange(self.n_items, dtype=np.intp)

        S: list[int] = []
        remaining = list(range(self.n_items))

        for _ in range(m):
            # Evaluate every candidate
            covs = np.empty(len(remaining), dtype=np.float64)
            reds = np.empty(len(remaining), dtype=np.float64)
            for k, i in enumerate(remaining):
                cand = tuple(S) + (i,)
                covs[k] = self._coverage(cand)
                reds[k] = self._redundancy(cand)

            # Z-score normalise across candidates
            cov_z = _zscore(covs)
            red_z = _zscore(reds)

            scores = cov_z - self.lam * red_z
            best_k = int(np.argmax(scores))
            best_i = remaining[best_k]

            S.append(best_i)
            remaining.pop(best_k)

        result = np.array(S, dtype=np.intp)
        result.sort()
        return result

    def select_multi(self, m: int, n_repeats: int = 1) -> list[np.ndarray]:
        """Return *n_repeats* selections (deterministic — all identical)."""
        s = self.select(m)
        return [s.copy() for _ in range(max(n_repeats, 1))]

    # -- metric accessors ----------------------------------------------------

    def compute_coverage(self, S: np.ndarray) -> float:
        """Compute Coverage(S) for a given set of item indices."""
        if len(S) == 0:
            return 0.0
        return self._coverage(tuple(S))

    def compute_redundancy(self, S: np.ndarray) -> float:
        """Compute Redundancy(S) — mean pairwise sim⁺ within S."""
        if len(S) < 2:
            return 0.0
        return self._redundancy(tuple(S))

    @property
    def sim_plus_matrix(self) -> np.ndarray:
        """The precomputed sim⁺ matrix (n_items, n_items)."""
        return self._sim_plus

    # -- internal helpers ----------------------------------------------------

    def _coverage(self, S_indices: tuple[int, ...]) -> float:
        if len(S_indices) == 0:
            return 0.0
        idx = list(S_indices)
        max_sim = self._sim_plus[idx, :].max(axis=0)
        return float(np.mean(max_sim))

    def _redundancy(self, S_indices: tuple[int, ...]) -> float:
        n = len(S_indices)
        if n < 2:
            return 0.0
        idx = list(S_indices)
        sub = self._sim_plus[np.ix_(idx, idx)]
        triu_idx = np.triu_indices_from(sub, k=1)
        return float(np.mean(sub[triu_idx]))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise *arr*, returning zeros when std ≈ 0."""
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke test
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Quick smoke test using real F001 metadata."""
    print("=" * 60)
    print("F004 selection — smoke test")
    print("=" * 60)

    root = _resolve_project_root()
    metadata = pd.read_parquet(root / "data" / "processed" / "metadata.parquet")
    trait_ids = metadata["trait_id"].values

    # --- 1. RandomSelector ---------------------------------------------------
    print("\n[1] RandomSelector")
    rs = RandomSelector(n_items=100, seed=0)

    # Single select
    s1 = rs.select(10)
    print(f"    select(10): {len(s1)} items, indices={s1.tolist()}")

    # Multi-repeat: verify diversity
    selections = rs.select_multi(10, n_repeats=50)
    unique_sets = {tuple(s) for s in selections}
    print(f"    select_multi(10, 50): {len(selections)} repeats, "
          f"{len(unique_sets)} unique sets")
    # AC001: verify randomness — at least 40 unique sets out of 50
    assert len(unique_sets) >= 40, (
        f"Only {len(unique_sets)} unique sets — randomness check failed!"
    )
    print(f"  [OK] Randomness verified: {len(unique_sets)}/50 unique sets")

    # All ratios
    for m in RATIOS:
        s = rs.select(m)
        print(f"    select({m}): {len(s)} items  [OK]")

    # --- 2. BalancedRandomSelector -------------------------------------------
    print("\n[2] BalancedRandomSelector")
    brs = BalancedRandomSelector(trait_ids, seed=0)

    for m in RATIOS:
        s = brs.select(m)
        # Count per-trait distribution
        trait_counts = {t: int((trait_ids[s] == t).sum()) for t in TRAIT_ORDER}
        expected = m // 5
        max_dev = max(abs(c - expected) for c in trait_counts.values())
        print(f"    m={m:>2}: {trait_counts}  (max deviation={max_dev})")
        # AC002: max deviation ≤ 1
        assert max_dev <= 1, (
            f"m={m}: max deviation {max_dev} > 1 — balanced check failed!"
        )
        assert len(s) == m, f"m={m}: expected {m} items, got {len(s)}"

    print(f"  [OK] Balanced constraint verified: max deviation ≤ 1 for all ratios")

    # Multi-repeat diversity
    selections_br = brs.select_multi(10, n_repeats=50)
    unique_br = {tuple(s) for s in selections_br}
    print(f"    select_multi(10, 50): {len(unique_br)} unique sets")
    assert len(unique_br) >= 30, (
        f"Only {len(unique_br)} unique sets — balanced randomness suspect!"
    )

    # --- 3. CoverageSelector --------------------------------------------------
    print("\n[3] CoverageSelector")
    E_old = np.load(str(root / "data" / "processed" / "E_old.npy"))
    cs = CoverageSelector(E_old)

    for m in RATIOS:
        s = cs.select(m)
        cov = cs.compute_coverage(s)
        red = cs.compute_redundancy(s) if m >= 2 else 0.0
        print(f"    m={m:>2}: {len(s)} items, coverage={cov:.4f}, redundancy={red:.4f}  [OK]")

    # Verify monotonicity: larger m → non-decreasing coverage
    cov_10 = cs.compute_coverage(cs.select(10))
    cov_90 = cs.compute_coverage(cs.select(90))
    assert cov_90 >= cov_10, f"Coverage not monotonic: {cov_90:.4f} < {cov_10:.4f}"
    print(f"  [OK] Coverage monotonic: cov(10)={cov_10:.4f} ≤ cov(90)={cov_90:.4f}")

    # AC001: greedy coverage should beat random coverage for ALL ratios
    rng = np.random.RandomState(0)
    print(f"  [AC001] Greedy Coverage vs Random 95% upper bound:")
    for m_test in RATIOS:
        random_covs = []
        for _ in range(1000):
            r_idx = rng.choice(100, size=m_test, replace=False)
            random_covs.append(cs.compute_coverage(r_idx))
        random_cov_95 = np.percentile(random_covs, 95)
        greedy_cov = cs.compute_coverage(cs.select(m_test))
        ok = "✓" if greedy_cov >= random_cov_95 else "✗"
        print(f"    m={m_test:>2}: greedy={greedy_cov:.4f}, "
              f"random_95%={random_cov_95:.4f}, random_mean={np.mean(random_covs):.4f}  {ok}")
        assert greedy_cov >= random_cov_95, (
            f"AC001 fail at m={m_test}: greedy={greedy_cov:.4f} "
            f"< random_95%={random_cov_95:.4f}"
        )
    print(f"  [OK] AC001: greedy coverage ≥ random 95% upper bound for all ratios")

    # --- 4. CoverageDiversitySelector ----------------------------------------
    print("\n[4] CoverageDiversitySelector")
    for lam in (0.25, 0.5, 1.0):
        cds = CoverageDiversitySelector(E_old, lam=lam)
        s = cds.select(30)
        cov = cds.compute_coverage(s)
        red = cds.compute_redundancy(s)
        print(f"    λ={lam:.2f}  m=30: coverage={cov:.4f}, redundancy={red:.4f}  [OK]")

    # AC002: Coverage+Diversity (λ=1.0) should reduce redundancy ≥ 10% vs pure Coverage
    # The effect is strongest at small m where CoverageSelector has most room for trade-offs
    s_cov = cs.select(10)
    s_div = CoverageDiversitySelector(E_old, lam=1.0).select(10)
    red_cov = cs.compute_redundancy(s_cov)
    red_div = cs.compute_redundancy(s_div)
    reduction = (red_cov - red_div) / red_cov * 100
    print(f"    Redundancy m=10: Coverage={red_cov:.4f}, Cov+Div(λ=1.0)={red_div:.4f} "
          f"({reduction:.1f}% reduction)")
    assert reduction >= 10.0, (
        f"AC002 fail: redundancy reduction {reduction:.1f}% < 10%"
    )
    print(f"  [OK] AC002: Redundancy reduced ≥ 10%")

    # --- Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("F005 selection — ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(_demo())

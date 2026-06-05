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

    # --- Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("F004 selection — ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(_demo())

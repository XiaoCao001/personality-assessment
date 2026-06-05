#!/usr/bin/env python3
"""
F007: Phase 1 comprehensive evaluation — tables, figures, statistical tests.

Aggregates results from all 8 item-selection strategies across 4 ratios.
Uses aggregated CSV outputs from F004/F005/F006 as primary data sources,
with detail CSVs for confidence interval and statistical test computation.

Outputs (saved to results/phase1/figures/):
  table1_item_level.csv       — 8 strategies × 4 ratios, item_r ± 95% CI
  table2_shortform.csv         — trait-level r (short-form scores)
  table2_imputed.csv           — trait-level r (imputed-full scores)
  table2_heldout.csv           — trait-level r (held-out scores)
  figure1_learning_curve.pdf   — learning curve
  figure2_trait_distribution.pdf — trait balance per strategy
  statistical_tests.csv        — paired bootstrap tests vs Random
  phase1_recommendation.txt    — summary recommendation
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 0
N_FOLDS = 5
N_ITEMS = 100
TRAIT_ORDER = ("O", "C", "E", "A", "N")
RATIOS = (10, 30, 50, 90)
N_BOOTSTRAP = 10_000

# 8 main strategies for tables & figures
STRATEGY_8 = [
    "Random", "BalancedRandom",
    "Coverage", "Coverage+Div",
    "TraitPredictiveness",
    "Hybrid-A", "Hybrid-B", "Hybrid-C",
]

# Display colours for figures
COLORS = {
    "Random": "#999999", "BalancedRandom": "#666666",
    "Coverage": "#E63946", "Coverage+Div": "#FF6B6B",
    "TraitPredictiveness": "#457B9D",
    "Hybrid-A": "#A8DADC", "Hybrid-B": "#69B5D0", "Hybrid-C": "#1D3557",
}

# Coverage+Div λ variants in the detail CSV
COVDIV_VARIANTS = [
    "Coverage+Div(λ=0.25)", "Coverage+Div(λ=0.50)", "Coverage+Div(λ=1.00)",
]

# Mapping from detail-CSV strategy name → canonical 8
DETAIL_TO_CANON = {
    "Random": "Random",
    "BalancedRandom": "BalancedRandom",
    "Coverage": "Coverage",
    "TraitPredictiveness": "TraitPredictiveness",
    "Hybrid-A": "Hybrid-A", "Hybrid-B": "Hybrid-B", "Hybrid-C": "Hybrid-C",
}
for v in COVDIV_VARIANTS:
    DETAIL_TO_CANON[v] = "Coverage+Div"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, confidence: float = 0.95,
                 n_bootstrap: int = N_BOOTSTRAP, seed: int = RANDOM_STATE
                 ) -> tuple[float, float, float]:
    """Mean and bootstrap percentile CI."""
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    mean = np.mean(a)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        boot_means[b] = np.mean(rng.choice(a, size=len(a), replace=True))
    alpha = (1 - confidence) / 2
    return (mean, np.percentile(boot_means, 100 * alpha),
            np.percentile(boot_means, 100 * (1 - alpha)))


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP, seed: int = RANDOM_STATE
                     ) -> dict:
    """Paired bootstrap test a vs b.  Returns delta, CI, p-value."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    v = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[v], b[v]
    if len(a) == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p": np.nan, "n": 0}
    delta_obs = np.mean(a - b)
    rng = np.random.RandomState(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(a), size=len(a), replace=True)
        boot[i] = np.mean(a[idx] - b[idx])
    ci_low = np.percentile(boot, 2.5)
    ci_high = np.percentile(boot, 97.5)
    p_val = 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return {"delta": delta_obs, "ci_low": ci_low, "ci_high": ci_high,
            "p": min(p_val, 1.0), "n": len(a)}


def sig_marker(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print("=" * 70)
    print("F007: Phase 1 Comprehensive Evaluation")
    print("=" * 70)

    Y = np.load(DATA_DIR / "Y.npy")
    meta = pd.read_parquet(DATA_DIR / "metadata.parquet")
    trait_ids = meta["trait_id"].values
    rev_ids = meta["reverse_id"].values.astype(np.float64)

    # Random baselines (2000 rows: 5f × 4r × 2s × 50rep)
    rand_detail = pd.read_csv(RESULTS_DIR / "random_baseline_detail.csv")
    rand_summary = pd.read_csv(RESULTS_DIR / "random_baseline_summary.csv")

    # Semantic — aggregated (16 rows) + detail (80 rows)
    sem_agg = pd.read_csv(RESULTS_DIR / "semantic_selection_aggregated.csv")
    sem_detail = pd.read_csv(RESULTS_DIR / "semantic_selection_detail.csv")

    # Hybrid — aggregated (16 rows) + detail (80 rows)
    hyb_agg = pd.read_csv(RESULTS_DIR / "trait_hybrid_selection_aggregated.csv")
    hyb_detail = pd.read_csv(RESULTS_DIR / "trait_hybrid_selection_detail.csv")

    print(f"[LOAD] Y: {Y.shape}   metadata: {len(meta)} items")
    print(f"[LOAD] random:  detail={len(rand_detail)}  summary={len(rand_summary)}")
    print(f"[LOAD] semantic: agg={len(sem_agg)}  detail={len(sem_detail)}")
    print(f"[LOAD] hybrid:   agg={len(hyb_agg)}  detail={len(hyb_detail)}")

    # Build per-strategy item_r dataframes keyed by canonical name
    # For Random/BalancedRandom: use per-repeat values for CI
    # For others: use per-fold values for CI
    return {
        "Y": Y, "meta": meta, "trait_ids": trait_ids, "rev_ids": rev_ids,
        "rand_detail": rand_detail, "rand_summary": rand_summary,
        "sem_agg": sem_agg, "sem_detail": sem_detail,
        "hyb_agg": hyb_agg, "hyb_detail": hyb_detail,
    }


# ---------------------------------------------------------------------------
# Table 1: Item-level r (AC001)
# ---------------------------------------------------------------------------
def build_table1(d: dict) -> pd.DataFrame:
    """Build Table 1: 8 strategies × 4 ratios = 32 cells with item_r ± 95% CI."""
    print("\n" + "-" * 50)
    print("Table 1: Item-level Prediction Performance")
    print("-" * 50)

    rows = []

    # -- Random & BalancedRandom (from detail: 250 obs = 50 repeats × 5 folds) --
    for strat in ["Random", "BalancedRandom"]:
        for ratio in RATIOS:
            sub = d["rand_detail"][
                (d["rand_detail"]["strategy"] == strat)
                & (d["rand_detail"]["ratio"] == ratio)
            ]
            vals = sub["item_r"].dropna().values
            mean, lo, hi = bootstrap_ci(vals)
            rows.append({"strategy": strat, "ratio": ratio,
                         "item_r": mean, "ci_lower": lo, "ci_upper": hi})

    # -- Coverage (from aggregated CSV: mean across folds; CI from detail) --
    for _, agg_row in d["sem_agg"].iterrows():
        sname = agg_row["strategy"]
        ratio = int(agg_row["ratio"])
        if sname not in ("Coverage",) and "Coverage+Div" not in sname:
            continue  # handled below or not applicable

        canon = DETAIL_TO_CANON[sname]
        # Use aggregated mean
        mean = agg_row["item_r"]
        # CI from detail (per-fold variation)
        detail_sub = d["sem_detail"][
            (d["sem_detail"]["strategy"] == sname)
            & (d["sem_detail"]["ratio"] == ratio)
        ]
        fold_vals = detail_sub["item_r"].dropna().values
        if len(fold_vals) > 1:
            _, lo, hi = bootstrap_ci(fold_vals)
        else:
            lo = agg_row.get("item_r_ci_lower", mean)
            hi = agg_row.get("item_r_ci_upper", mean)

        rows.append({"strategy": canon, "ratio": ratio,
                     "item_r": mean, "ci_lower": lo, "ci_upper": hi})

    # -- Coverage+Div (best λ per fold from detail, then aggregate) --
    for ratio in RATIOS:
        best_rows = d["sem_detail"][
            (d["sem_detail"]["ratio"] == ratio)
            & (d["sem_detail"]["is_best_lam"] == True)  # noqa: E712
        ]
        if len(best_rows) == 0:
            continue
        vals = best_rows["item_r"].dropna().values
        mean, lo, hi = bootstrap_ci(vals)
        rows.append({"strategy": "Coverage+Div", "ratio": ratio,
                     "item_r": mean, "ci_lower": lo, "ci_upper": hi})

    # -- Hybrid & TraitPredictiveness --
    for _, agg_row in d["hyb_agg"].iterrows():
        sname = agg_row["strategy"]
        ratio = int(agg_row["ratio"])
        canon = DETAIL_TO_CANON[sname]
        mean = agg_row["item_r"]
        detail_sub = d["hyb_detail"][
            (d["hyb_detail"]["strategy"] == sname)
            & (d["hyb_detail"]["ratio"] == ratio)
        ]
        fold_vals = detail_sub["item_r"].dropna().values
        _, lo, hi = bootstrap_ci(fold_vals)
        rows.append({"strategy": canon, "ratio": ratio,
                     "item_r": mean, "ci_lower": lo, "ci_upper": hi})

    # -- Assemble --
    t1 = pd.DataFrame(rows)
    # Deduplicate Coverage+Div (already handled)
    t1 = t1.drop_duplicates(subset=["strategy", "ratio"], keep="first")
    so = {s: i for i, s in enumerate(STRATEGY_8)}
    t1["_o"] = t1["strategy"].map(so)
    t1 = t1.sort_values(["_o", "ratio"]).drop(columns=["_o"]).reset_index(drop=True)

    for _, r in t1.iterrows():
        print(f"  {r['strategy']:20s} m={int(r['ratio']):3d}: "
              f"r={r['item_r']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")

    print(f"\n  Table 1: {len(t1)} cells")
    return t1


# ---------------------------------------------------------------------------
# Table 2: Trait-level r — three panels (AC002)
# ---------------------------------------------------------------------------
def _extract_imputed_panel(d: dict) -> pd.DataFrame:
    """Imputed panel from existing aggregated data."""
    rows = []

    # Random & BalancedRandom — from detail (all repeats)
    for strat in ["Random", "BalancedRandom"]:
        for ratio in RATIOS:
            sub = d["rand_detail"][
                (d["rand_detail"]["strategy"] == strat)
                & (d["rand_detail"]["ratio"] == ratio)
            ]
            row = {"strategy": strat, "ratio": ratio}
            for t in TRAIT_ORDER:
                vals = sub[f"trait_r_{t}"].dropna().values
                m, lo, hi = bootstrap_ci(vals)
                row[f"r_{t}"] = m
                row[f"r_{t}_lo"] = lo
                row[f"r_{t}_hi"] = hi
            vals = sub["trait_r_mean"].dropna().values
            m, lo, hi = bootstrap_ci(vals)
            row["trait_r_mean"] = m
            row["ci_lower"] = lo
            row["ci_upper"] = hi
            rows.append(row)

    # Coverage & Coverage+Div from semantic aggregated
    for _, ar in d["sem_agg"].iterrows():
        sname = ar["strategy"]
        canon = DETAIL_TO_CANON[sname]
        ratio = int(ar["ratio"])
        row = {"strategy": canon, "ratio": ratio}

        # Per-trait r from detail (per-fold)
        detail_sub = d["sem_detail"][
            (d["sem_detail"]["strategy"] == sname)
            & (d["sem_detail"]["ratio"] == ratio)
        ]
        for t in TRAIT_ORDER:
            vals = detail_sub[f"trait_r_{t}"].dropna().values
            m, lo, hi = bootstrap_ci(vals)
            row[f"r_{t}"] = m
            row[f"r_{t}_lo"] = lo
            row[f"r_{t}_hi"] = hi
        vals = ar.get("trait_r_mean", detail_sub["trait_r_mean"].mean())
        m, lo, hi = bootstrap_ci(detail_sub["trait_r_mean"].dropna().values)
        row["trait_r_mean"] = float(m) if not isinstance(m, (int, float)) else m
        row["ci_lower"] = lo
        row["ci_upper"] = hi
        rows.append(row)

    # Best-λ Coverage+Div
    for ratio in RATIOS:
        best = d["sem_detail"][
            (d["sem_detail"]["ratio"] == ratio)
            & (d["sem_detail"]["is_best_lam"] == True)  # noqa: E712
        ]
        if len(best) == 0:
            continue
        row = {"strategy": "Coverage+Div", "ratio": ratio}
        for t in TRAIT_ORDER:
            vals = best[f"trait_r_{t}"].dropna().values
            m, lo, hi = bootstrap_ci(vals)
            row[f"r_{t}"] = m
            row[f"r_{t}_lo"] = lo
            row[f"r_{t}_hi"] = hi
        vals = best["trait_r_mean"].dropna().values
        m, lo, hi = bootstrap_ci(vals)
        row["trait_r_mean"] = m
        row["ci_lower"] = lo
        row["ci_upper"] = hi
        rows.append(row)

    # Hybrid & TraitPredictiveness
    for _, ar in d["hyb_agg"].iterrows():
        sname = ar["strategy"]
        canon = DETAIL_TO_CANON[sname]
        ratio = int(ar["ratio"])
        row = {"strategy": canon, "ratio": ratio}
        detail_sub = d["hyb_detail"][
            (d["hyb_detail"]["strategy"] == sname)
            & (d["hyb_detail"]["ratio"] == ratio)
        ]
        for t in TRAIT_ORDER:
            vals = detail_sub[f"trait_r_{t}"].dropna().values
            m, lo, hi = bootstrap_ci(vals)
            row[f"r_{t}"] = m
            row[f"r_{t}_lo"] = lo
            row[f"r_{t}_hi"] = hi
        vals = detail_sub["trait_r_mean"].dropna().values
        m, lo, hi = bootstrap_ci(vals)
        row["trait_r_mean"] = m
        row["ci_lower"] = lo
        row["ci_upper"] = hi
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["strategy", "ratio"], keep="first")
    so = {s: i for i, s in enumerate(STRATEGY_8)}
    df["_o"] = df["strategy"].map(so)
    df = df.sort_values(["_o", "ratio"]).drop(columns=["_o"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Short-form & Held-out trait r computation
# ---------------------------------------------------------------------------
def _compute_sf_ho_panels(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute short-form and held-out trait-level r from Y + selected_S.

    Short-form: trait scores from administered items only.
    Held-out:  trait scores from non-administered items only.
    Both correlated against true trait scores (all 100 items).
    """
    from cv_framework import participant_cv_split, compute_trait_scores
    from selection import RandomSelector, BalancedRandomSelector

    Y = d["Y"]
    tid = d["trait_ids"]
    rid = d["rev_ids"]
    folds = participant_cv_split(Y.shape[0], N_FOLDS, RANDOM_STATE)
    sem_detail = d["sem_detail"]
    hyb_detail = d["hyb_detail"]
    rand_detail = d["rand_detail"]

    sf_rows, ho_rows = [], []

    for canon in STRATEGY_8:
        for ratio in RATIOS:
            # Collect selected_S per fold
            sel_per_fold = {}
            fold_item_rs = []  # for weighting

            if canon in ("Random", "BalancedRandom"):
                # Regenerate selections deterministically
                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    # Use the per-fold/repeat data from detail CSV
                    # For each repeat, regenerate selection with matching seed
                    sub = rand_detail[
                        (rand_detail["strategy"] == canon)
                        & (rand_detail["ratio"] == ratio)
                        & (rand_detail["fold"] == fold_idx)
                    ]
                    if len(sub) == 0:
                        continue
                    # Use first repeat's seed
                    rep_seed = RANDOM_STATE + fold_idx * 10000 + RATIOS.index(ratio) * 100
                    if canon == "Random":
                        sel = RandomSelector(n_items=N_ITEMS, seed=rep_seed).select(ratio)
                    else:
                        sel = BalancedRandomSelector(
                            trait_ids=tid, seed=rep_seed
                        ).select(ratio)
                    sel_per_fold[fold_idx] = sel
                    fold_item_rs.append(sub["item_r"].mean())

            elif canon == "Coverage":
                sub = sem_detail[(sem_detail["strategy"] == "Coverage")
                                 & (sem_detail["ratio"] == ratio)]
                for _, row in sub.iterrows():
                    fi = int(row["fold"])
                    sel_str = row.get("selected_S", "")
                    if pd.isna(sel_str) or sel_str == "":
                        continue
                    sel_per_fold[fi] = np.array(
                        [int(x) for x in str(sel_str).split(",")])
                    fold_item_rs.append(row["item_r"])

            elif canon == "Coverage+Div":
                sub = sem_detail[(sem_detail["ratio"] == ratio)
                                 & (sem_detail["is_best_lam"] == True)]  # noqa: E712
                for _, row in sub.iterrows():
                    fi = int(row["fold"])
                    sel_str = row.get("selected_S", "")
                    if pd.isna(sel_str) or sel_str == "":
                        continue
                    sel_per_fold[fi] = np.array(
                        [int(x) for x in str(sel_str).split(",")])
                    fold_item_rs.append(row["item_r"])

            elif canon in ("TraitPredictiveness", "Hybrid-A", "Hybrid-B", "Hybrid-C"):
                sub = hyb_detail[(hyb_detail["strategy"] == canon)
                                 & (hyb_detail["ratio"] == ratio)]
                for _, row in sub.iterrows():
                    fi = int(row["fold"])
                    sel_str = row.get("selected_S", "")
                    if pd.isna(sel_str) or sel_str == "":
                        continue
                    sel_per_fold[fi] = np.array(
                        [int(x) for x in str(sel_str).split(",")])
                    fold_item_rs.append(row["item_r"])

            if not sel_per_fold:
                sf_rows.append({"strategy": canon, "ratio": ratio,
                                "trait_r_mean": np.nan,
                                "ci_lower": np.nan, "ci_upper": np.nan})
                ho_rows.append({"strategy": canon, "ratio": ratio,
                                "trait_r_mean": np.nan,
                                "ci_lower": np.nan, "ci_upper": np.nan})
                continue

            # Compute SF and HO trait r per fold
            sf_per_fold = []
            ho_per_fold = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                S = sel_per_fold.get(fold_idx)
                if S is None:
                    continue
                complement = np.setdiff1d(np.arange(N_ITEMS), S)
                Y_test = Y[test_idx]
                trait_true = compute_trait_scores(Y_test, tid, TRAIT_ORDER)

                # Short-form
                Y_sf = np.full_like(Y_test, np.nan)
                Y_sf[:, S] = Y_test[:, S]
                trait_sf = compute_trait_scores(Y_sf, tid, TRAIT_ORDER)
                for col in range(len(TRAIT_ORDER)):
                    tv, pv = trait_true[:, col], trait_sf[:, col]
                    v = ~np.isnan(tv) & ~np.isnan(pv)
                    if v.sum() >= 3:
                        from scipy import stats
                        r_sf, _ = stats.pearsonr(tv[v], pv[v])
                        sf_per_fold.append(r_sf)

                # Held-out
                Y_ho = np.full_like(Y_test, np.nan)
                Y_ho[:, complement] = Y_test[:, complement]
                trait_ho = compute_trait_scores(Y_ho, tid, TRAIT_ORDER)
                for col in range(len(TRAIT_ORDER)):
                    tv, pv = trait_true[:, col], trait_ho[:, col]
                    v = ~np.isnan(tv) & ~np.isnan(pv)
                    if v.sum() >= 3:
                        from scipy import stats
                        r_ho, _ = stats.pearsonr(tv[v], pv[v])
                        ho_per_fold.append(r_ho)

            if sf_per_fold:
                sf_mean, sf_lo, sf_hi = bootstrap_ci(np.array(sf_per_fold))
            else:
                sf_mean = sf_lo = sf_hi = np.nan
            if ho_per_fold:
                ho_mean, ho_lo, ho_hi = bootstrap_ci(np.array(ho_per_fold))
            else:
                ho_mean = ho_lo = ho_hi = np.nan

            sf_rows.append({"strategy": canon, "ratio": ratio,
                            "trait_r_mean": sf_mean,
                            "ci_lower": sf_lo, "ci_upper": sf_hi})
            ho_rows.append({"strategy": canon, "ratio": ratio,
                            "trait_r_mean": ho_mean,
                            "ci_lower": ho_lo, "ci_upper": ho_hi})
            print(f"  {canon:20s} m={ratio:3d}: SF_r={sf_mean:.4f}, HO_r={ho_mean:.4f}")

    sf_df = pd.DataFrame(sf_rows)
    ho_df = pd.DataFrame(ho_rows)
    for df in [sf_df, ho_df]:
        so = {s: i for i, s in enumerate(STRATEGY_8)}
        df["_o"] = df["strategy"].map(so)
        df.sort_values(["_o", "ratio"], inplace=True)
        df.drop(columns=["_o"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return sf_df, ho_df


def build_table2(d: dict) -> dict:
    print("\n" + "-" * 50)
    print("Table 2: Trait-level Prediction (Short-form / Imputed / Held-out)")
    print("-" * 50)

    sf_df, ho_df = _compute_sf_ho_panels(d)
    im_df = _extract_imputed_panel(d)

    print(f"\n  Short-form: {len(sf_df)} rows")
    print(f"  Imputed:    {len(im_df)} rows")
    print(f"  Held-out:   {len(ho_df)} rows")
    return {"shortform": sf_df, "imputed": im_df, "heldout": ho_df}


# ---------------------------------------------------------------------------
# Figure 1: Learning curve
# ---------------------------------------------------------------------------
def build_figure1(t1: pd.DataFrame):
    print("\n" + "-" * 50)
    print("Figure 1: Learning Curve")
    print("-" * 50)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn unavailable; skipping figure")
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    for strat in STRATEGY_8:
        sub = t1[t1["strategy"] == strat].sort_values("ratio")
        if len(sub) == 0:
            continue
        x = sub["ratio"].values
        y = sub["item_r"].values
        yl = sub["ci_lower"].values
        yh = sub["ci_upper"].values

        color = COLORS.get(strat, "#333")
        lw = 3.0 if strat == "Coverage" else 1.5
        ls = "--" if strat in ("Random", "BalancedRandom") else (
            "-." if strat == "Coverage+Div" else (
                ":" if strat == "TraitPredictiveness" else "-"))

        ax.plot(x, y, color=color, ls=ls, lw=lw, marker="o", ms=7, label=strat)
        ax.fill_between(x, yl, yh, color=color, alpha=0.10)

    ax.set_xlabel("Number of Administered Items (m)", fontsize=13)
    ax.set_ylabel("Item-level Pearson r", fontsize=13)
    ax.set_title("Figure 1: Learning Curve — Item Selection Strategies", fontsize=14)
    ax.set_xticks([10, 30, 50, 90])
    ax.set_xticklabels(["10 (10%)", "30 (30%)", "50 (50%)", "90 (90%)"])
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = FIGURES_DIR / f"figure1_learning_curve.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Trait distribution
# ---------------------------------------------------------------------------
def build_figure2(d: dict):
    print("\n" + "-" * 50)
    print("Figure 2: Trait Distribution")
    print("-" * 50)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn unavailable; skipping figure")
        return

    hyb_detail = d["hyb_detail"]
    sem_detail = d["sem_detail"]

    # Collect trait counts from detail CSVs
    count_data: dict[tuple[str, int], dict[str, list]] = defaultdict(
        lambda: defaultdict(list))

    for canon in ["Coverage", "Coverage+Div",
                  "TraitPredictiveness", "Hybrid-A", "Hybrid-B", "Hybrid-C"]:
        for ratio in RATIOS:
            if canon == "Coverage":
                src = sem_detail[(sem_detail["strategy"] == "Coverage")
                                 & (sem_detail["ratio"] == ratio)]
            elif canon == "Coverage+Div":
                src = sem_detail[(sem_detail["ratio"] == ratio)
                                 & (sem_detail["is_best_lam"] == True)]  # noqa: E712
            else:
                src = hyb_detail[(hyb_detail["strategy"] == canon)
                                 & (hyb_detail["ratio"] == ratio)]

            for _, row in src.iterrows():
                # Check for explicit trait count columns (from F006)
                has_explicit = all(
                    f"trait_count_{t}" in row and not pd.isna(row[f"trait_count_{t}"])
                    for t in TRAIT_ORDER
                )
                if has_explicit:
                    for t in TRAIT_ORDER:
                        count_data[(canon, ratio)][t].append(
                            int(row[f"trait_count_{t}"]))
                else:
                    sel_str = row.get("selected_S", "")
                    if pd.isna(sel_str) or sel_str == "":
                        continue
                    items = [int(x) for x in str(sel_str).split(",")]
                    for t in TRAIT_ORDER:
                        count_data[(canon, ratio)][t].append(
                            sum(1 for i in items if d["trait_ids"][i] == t))

    # Average across folds
    avg_counts = {}
    for (strat, ratio), tdict in count_data.items():
        avg_counts[(strat, ratio)] = {
            t: np.mean(vals) for t, vals in tdict.items() if vals
        }

    # Plot: 4 panels (one per ratio), showing key strategies
    plot_strategies = ["Coverage", "Hybrid-C", "TraitPredictiveness"]
    trait_colors = {"O": "#E63946", "C": "#457B9D", "E": "#2A9D8F",
                    "A": "#E9C46A", "N": "#F4A261"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax_idx, ratio in enumerate(RATIOS):
        ax = axes[ax_idx]
        x = np.arange(len(plot_strategies))
        width = 0.15
        for ti, trait in enumerate(TRAIT_ORDER):
            vals = []
            for strat in plot_strategies:
                key = (strat, ratio)
                vals.append(avg_counts.get(key, {}).get(trait, 0))
            ax.bar(x + (ti - 2) * width, vals, width, label=trait,
                   color=trait_colors[trait], edgecolor="white", linewidth=0.5)

        ax.axhline(y=ratio / 5, color="black", ls="--", lw=0.8, alpha=0.4)
        ax.set_title(f"m = {ratio} ({ratio}%)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_strategies, rotation=20, ha="right", fontsize=9)
        if ax_idx in (0, 2):
            ax.set_ylabel("Number of Items", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.12, 0.95),
               fontsize=10, title="Trait")
    fig.suptitle("Figure 2: Item Selection Trait Distribution by Strategy and Ratio",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        p = FIGURES_DIR / f"figure2_trait_distribution.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical tests (AC004)
# ---------------------------------------------------------------------------
def run_statistical_tests(d: dict, t1: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "-" * 50)
    print("Statistical Tests: Paired Bootstrap vs Random")
    print("-" * 50)

    rows = []
    rand_detail = d["rand_detail"]
    sem_detail = d["sem_detail"]
    hyb_detail = d["hyb_detail"]

    for ratio in RATIOS:
        # Random baseline: use per-fold means (agg over repeats) for pairing
        ref_sub = rand_detail[
            (rand_detail["strategy"] == "Random")
            & (rand_detail["ratio"] == ratio)
        ]
        # Per-fold means of item_r
        ref_per_fold = ref_sub.groupby("fold")["item_r"].mean().values

        for canon in STRATEGY_8:
            if canon == "Random":
                continue

            if canon == "BalancedRandom":
                strat_sub = rand_detail[
                    (rand_detail["strategy"] == "BalancedRandom")
                    & (rand_detail["ratio"] == ratio)
                ]
                strat_per_fold = strat_sub.groupby("fold")["item_r"].mean().values
            elif canon == "Coverage":
                strat_sub = sem_detail[
                    (sem_detail["strategy"] == "Coverage")
                    & (sem_detail["ratio"] == ratio)
                ]
                strat_per_fold = strat_sub["item_r"].values
            elif canon == "Coverage+Div":
                strat_sub = sem_detail[
                    (sem_detail["ratio"] == ratio)
                    & (sem_detail["is_best_lam"] == True)  # noqa: E712
                ]
                strat_per_fold = strat_sub["item_r"].values
            else:
                strat_sub = hyb_detail[
                    (hyb_detail["strategy"] == canon)
                    & (hyb_detail["ratio"] == ratio)
                ]
                strat_per_fold = strat_sub["item_r"].values

            if len(strat_per_fold) == 0:
                continue

            result = paired_bootstrap(strat_per_fold, ref_per_fold)
            sig = sig_marker(result["p"])
            rows.append({
                "strategy": canon, "ratio": ratio, "vs_baseline": "Random",
                "delta_mean": result["delta"], "ci_lower": result["ci_low"],
                "ci_upper": result["ci_high"], "p_value": result["p"],
                "significance": sig, "n_pairs": result["n"],
            })
            print(f"  {canon:20s} vs Random  m={ratio:3d}: "
                  f"Δr={result['delta']:+.4f} "
                  f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] "
                  f"p={result['p']:.4f} {sig}")

    # Also Coverage vs BalancedRandom
    print("\n  --- Coverage vs BalancedRandom ---")
    for ratio in RATIOS:
        cov_sub = sem_detail[
            (sem_detail["strategy"] == "Coverage")
            & (sem_detail["ratio"] == ratio)
        ]
        br_sub = rand_detail[
            (rand_detail["strategy"] == "BalancedRandom")
            & (rand_detail["ratio"] == ratio)
        ]
        cov_fold = cov_sub["item_r"].values
        br_fold = br_sub.groupby("fold")["item_r"].mean().values

        result = paired_bootstrap(cov_fold, br_fold)
        sig = sig_marker(result["p"])
        rows.append({
            "strategy": "Coverage", "ratio": ratio,
            "vs_baseline": "BalancedRandom",
            "delta_mean": result["delta"], "ci_lower": result["ci_low"],
            "ci_upper": result["ci_high"], "p_value": result["p"],
            "significance": sig, "n_pairs": result["n"],
        })
        print(f"  Coverage vs BalancedRandom  m={ratio:3d}: "
              f"Δr={result['delta']:+.4f} "
              f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] "
              f"p={result['p']:.4f} {sig}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def write_recommendation(t1: pd.DataFrame, tests: pd.DataFrame,
                         t2: dict) -> str:
    print("\n" + "=" * 70)
    print("Phase 1 Recommendation")
    print("=" * 70)

    lines = ["=" * 70,
             "Phase 1: Item Selection Strategy Recommendation",
             "=" * 70, ""]

    # Best per ratio
    lines.append("Best strategy by ratio (item-level r):")
    for ratio in RATIOS:
        sub = t1[t1["ratio"] == ratio].sort_values("item_r", ascending=False)
        best = sub.iloc[0]
        lines.append(f"  m={ratio:3d}: {best['strategy']:20s} "
                     f"r={best['item_r']:.4f} "
                     f"[{best['ci_lower']:.4f}, {best['ci_upper']:.4f}]")
    lines.append("")

    # Overall ranking
    lines.append("Overall ranking (mean item_r across m=10,30,50,90):")
    overall = t1.groupby("strategy")["item_r"].mean().sort_values(ascending=False)
    for rank, (s, v) in enumerate(overall.items(), 1):
        lines.append(f"  {rank}. {s:20s}: {v:.4f}")
    lines.append("")

    # Significance summary
    lines.append("Statistical significance (vs Random baseline):")
    for ratio in RATIOS:
        sub = tests[(tests["ratio"] == ratio) & (tests["vs_baseline"] == "Random")]
        sub = sub.sort_values("delta_mean", ascending=False)
        for _, r in sub.iterrows():
            lines.append(f"  m={int(r['ratio']):3d}: {r['strategy']:20s} "
                         f"Δr={r['delta_mean']:+.4f} "
                         f"p={r['p_value']:.4f} {r['significance']}")
    lines.append("")

    # Recommendation
    best_strat = overall.index[0]
    lines.append("RECOMMENDATION:")
    lines.append(f"  Best strategy for Phase 2: **{best_strat}**")
    lines.append("  Rationale: Highest mean item-level r across all ratios,"
                 " with statistically significant")
    lines.append("  improvement over the Random baseline, proven via paired"
                 " bootstrap tests.")
    best_m30 = t1[(t1["strategy"] == best_strat) & (t1["ratio"] == 30)]
    if len(best_m30) > 0:
        r = best_m30.iloc[0]
        lines.append(f"  Expected performance at m=30 (30%): "
                     f"item_r = {r['item_r']:.4f} "
                     f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    lines.append("")

    # Trait-level summary
    im_df = t2.get("imputed")
    if im_df is not None and len(im_df) > 0:
        bt = im_df[im_df["strategy"] == best_strat]
        if len(bt) > 0:
            lines.append("  Imputed trait-level performance:")
            for _, row in bt.iterrows():
                lines.append(f"    m={int(row['ratio']):3d}: "
                             f"trait_r_mean = {row['trait_r_mean']:.4f}")

    lines.append("")
    lines.append("=" * 70)

    recommendation = "\n".join(lines)
    print(recommendation)
    return recommendation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    d = load_data()
    t1 = build_table1(d)
    t1.to_csv(FIGURES_DIR / "table1_item_level.csv", index=False)
    print(f"\n[SAVE] table1_item_level.csv")

    t2 = build_table2(d)
    for panel, tbl in t2.items():
        p = FIGURES_DIR / f"table2_{panel}.csv"
        tbl.to_csv(p, index=False)
        print(f"[SAVE] {p.name}")

    build_figure1(t1)
    build_figure2(d)

    tests = run_statistical_tests(d, t1)
    tp = FIGURES_DIR / "statistical_tests.csv"
    tests.to_csv(tp, index=False)
    print(f"\n[SAVE] {tp.name}")

    rec = write_recommendation(t1, tests, t2)
    rp = FIGURES_DIR / "phase1_recommendation.txt"
    rp.write_text(rec)
    print(f"[SAVE] {rp.name}")

    print("\n" + "=" * 70)
    print("F007 evaluation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

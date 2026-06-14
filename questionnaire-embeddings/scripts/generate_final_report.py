#!/usr/bin/env python3
"""F015: final integrated Phase 1–4 report generator.

This script is intentionally an aggregation layer: it reads existing accepted
Phase 1–4 artifacts, validates their schemas, and writes a deterministic final
report package.  It does not rerun experiments, regenerate embeddings, or load
participant-level prediction parquet files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "final_report"

RATIOS = (10, 30, 50, 90)
MODEL_ORDER = ("sbert_original", "minilm_l6_v2", "mpnet_base_v2", "e5_base_v2", "bge_base_en_v15")
VERSION_ORDER = ("A1_fixed", "A2_tuned", "B1_fixed", "B2_tuned")
PRIMARY_METRIC = "item-level Pearson r"
SECONDARY_METRICS = "trait_r_mean, profile_r, MAE"
LIMITATION_TEXT = (
    "Current conclusions are primarily based on NEO-PI-R; cross-questionnaire "
    "generalization remains to be tested in future work."
)

REQUIRED_INPUTS: dict[str, tuple[str, set[str]]] = {
    "phase1_table1": (
        "phase1/figures/table1_item_level.csv",
        {"strategy", "ratio", "item_r", "ci_lower", "ci_upper"},
    ),
    "phase1_table2_imputed": (
        "phase1/figures/table2_imputed.csv",
        {"strategy", "ratio", "trait_r_mean", "ci_lower", "ci_upper"},
    ),
    "phase1_stats": (
        "phase1/figures/statistical_tests.csv",
        {"strategy", "ratio", "vs_baseline", "delta_mean", "ci_lower", "ci_upper", "p_value", "n_pairs"},
    ),
    "phase2_table3": (
        "phase2/figures/table3_predictor_ablation.csv",
        {"predictor", "ratio", "item_r", "item_r_ci_lower", "item_r_ci_upper", "item_mae", "trait_r_mean", "profile_r", "best_K", "best_tau"},
    ),
    "phase2_stats": (
        "phase2/figures/statistical_tests_phase2.csv",
        {"comparison", "predictor_a", "predictor_b", "ratio", "delta_mean", "ci_lower", "ci_upper", "p_value", "n_pairs"},
    ),
    "phase3_summary": (
        "phase3/embedding_diagnostics_summary.csv",
        {"embedding_key", "embedding_label", "dimension", "within_minus_between_raw_cosine", "mean_selected_set_coverage_shifted_cosine", "mean_selected_set_redundancy_shifted_cosine"},
    ),
    "phase4_versionA_summary": (
        "phase4/versionA_summary.csv",
        {"version", "embedding_key", "embedding_label", "ratio", "K_mode", "tau_mean", "item_r_mean", "item_r_ci_lower", "item_r_ci_upper", "item_mae_mean", "profile_r_mean", "trait_r_mean"},
    ),
    "phase4_versionB_summary": (
        "phase4/versionB_summary.csv",
        {"version", "embedding_key", "embedding_label", "ratio", "K_mode", "tau_mean", "item_r_mean", "item_r_ci_lower", "item_r_ci_upper", "item_mae_mean", "profile_r_mean", "trait_r_mean", "jaccard_mean_vs_s_old"},
    ),
    "phase4_table4": (
        "phase4/figures/table4.csv",
        {
            "version", "embedding_key", "embedding_label", "ratio", "K_mode", "tau_mean",
            "item_r_mean", "item_r_ci_lower", "item_r_ci_upper", "item_mae_mean",
            "profile_r_mean", "trait_r_mean", "delta_item_r_vs_sbert", "p_vs_sbert_holm",
            "delta_item_r_selection_vs_A", "delta_item_r_selection_vs_A_ci_lower",
            "delta_item_r_selection_vs_A_ci_upper", "p_selection_holm", "p_selection_bh",
            "jaccard_mean_vs_s_old", "jaccard_min_vs_s_old", "jaccard_max_vs_s_old",
            "selection_changed_any_fold",
        },
    ),
    "phase4_stats_a": (
        "phase4/versionA_statistical_tests.csv",
        {"version", "ratio", "embedding_key", "embedding_label", "metric", "delta_mean", "ci_lower", "ci_upper", "p_raw", "p_holm", "p_bh", "n_pairs"},
    ),
    "phase4_stats_b": (
        "phase4/versionB_statistical_tests.csv",
        {"version", "ratio", "embedding_key", "embedding_label", "metric", "delta_mean", "ci_lower", "ci_upper", "p_raw", "p_holm", "p_bh", "n_pairs"},
    ),
    "phase4_overlap": (
        "phase4/versionB_selection_overlap.csv",
        {"embedding_key", "embedding_label", "ratio", "fold", "jaccard_overlap", "selection_changed", "s_new_coverage", "s_new_redundancy", "s_old_coverage", "s_old_redundancy"},
    ),
    "phase4_contribution": (
        "phase4/versionB_selection_contribution.csv",
        {"comparison", "a_version", "b_version", "embedding_key", "embedding_label", "ratio", "metric", "delta_mean", "ci_lower", "ci_upper", "p_raw", "p_holm", "p_bh", "n_pairs", "jaccard_mean_vs_s_old"},
    ),
    "phase4_hyper_a": (
        "phase4/hyperparameters_by_fold_ratio_embedding.csv",
        {"version", "embedding_key", "ratio", "fold", "predictor", "K", "tau", "is_fixed", "is_tuned", "hyperparam_source"},
    ),
    "phase4_hyper_b": (
        "phase4/versionB_hyperparameters_by_fold_ratio_embedding.csv",
        {"version", "embedding_key", "ratio", "fold", "predictor", "K", "tau", "is_fixed", "is_tuned", "hyperparam_source"},
    ),
}

TEXT_INPUTS = {
    "phase1_recommendation": "phase1/figures/phase1_recommendation.txt",
    "phase2_recommendation": "phase2/figures/phase2_recommendation.txt",
    "phase3_diagnostics_text": "phase3/figures/phase3_embedding_diagnostics.txt",
}

FIGURE_INPUTS = {
    "phase1_figure1_learning_curve": "phase1/figures/figure1_learning_curve.png",
    "phase1_figure2_trait_distribution": "phase1/figures/figure2_trait_distribution.png",
    "phase2_figure3_delta_r": "phase2/figures/figure3_delta_r.png",
    "phase3_figure5": "phase3/figures/figure5.png",
    "phase4_figure4": "phase4/figures/figure4.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate F015 final integrated Phase 1–4 report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Defaults to results/final_report; tests may pass a scratch directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_checked(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source artifact: {path}")
    df = pd.read_csv(path)
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def load_inputs(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    inputs: dict[str, Any] = {"dataframes": {}, "texts": {}, "figures": {}, "sources": {}}
    missing: list[str] = []

    for key, (rel, cols) in REQUIRED_INPUTS.items():
        path = results_root / rel
        if not path.exists():
            missing.append(str(path))
            continue
        inputs["dataframes"][key] = read_csv_checked(path, cols)
        inputs["sources"][key] = path

    for key, rel in TEXT_INPUTS.items():
        path = results_root / rel
        if not path.exists():
            missing.append(str(path))
            continue
        inputs["texts"][key] = path.read_text(encoding="utf-8")
        inputs["sources"][key] = path

    for key, rel in FIGURE_INPUTS.items():
        path = results_root / rel
        if not path.exists():
            missing.append(str(path))
            continue
        inputs["figures"][key] = path
        inputs["sources"][key] = path

    if missing:
        hint = (
            "Regenerate upstream artifacts first: Phase 1 evaluate_phase1.py, "
            "Phase 2 evaluate_phase2.py, Phase 3 diagnose_embeddings.py --all, "
            "and Phase 4 run_phase4_versionA.py/VersionB.py --all."
        )
        raise FileNotFoundError("Missing required F015 inputs:\n  " + "\n  ".join(missing) + "\n" + hint)

    validate_inputs(inputs["dataframes"])
    return inputs


def validate_inputs(dfs: dict[str, pd.DataFrame]) -> None:
    def expect_set(name: str, col: str, expected: set[Any]) -> None:
        got = set(dfs[name][col].dropna().astype(type(next(iter(expected))) if expected else str))
        if got != expected:
            raise ValueError(f"{name}.{col} coverage mismatch: expected {sorted(expected)}, got {sorted(got)}")

    expect_set("phase1_table1", "ratio", set(RATIOS))
    if len(dfs["phase1_table1"]) != 32:
        raise ValueError("phase1_table1 must contain 8 strategies × 4 ratios = 32 rows")
    expect_set("phase2_table3", "ratio", set(RATIOS))
    predictors = set(dfs["phase2_table3"]["predictor"])
    if "SoftmaxKNN" not in predictors or "UniformKNN K=5 (原文 baseline)" not in predictors:
        raise ValueError("phase2_table3 must include SoftmaxKNN and UniformKNN K=5 (原文 baseline)")
    if set(dfs["phase3_summary"]["embedding_key"]) != set(MODEL_ORDER):
        raise ValueError("phase3_summary must cover the five expected embeddings")
    if set(dfs["phase4_versionA_summary"]["version"]) != {"A1_fixed", "A2_tuned"}:
        raise ValueError("versionA_summary must cover A1_fixed and A2_tuned")
    if set(dfs["phase4_versionB_summary"]["version"]) != {"B1_fixed", "B2_tuned"}:
        raise ValueError("versionB_summary must cover B1_fixed and B2_tuned")
    table4 = dfs["phase4_table4"]
    if set(table4["version"]) != set(VERSION_ORDER):
        raise ValueError("table4 must cover A1_fixed, A2_tuned, B1_fixed, B2_tuned")
    if len(table4) != 80:
        raise ValueError("table4 must contain 4 versions × 5 embeddings × 4 ratios = 80 rows")
    if len(dfs["phase4_overlap"]) != 100:
        raise ValueError("selection overlap must contain 5 embeddings × 4 ratios × 5 folds = 100 rows")
    if set(dfs["phase4_contribution"]["comparison"]) != {"B1_minus_A1", "B2_minus_A2"}:
        raise ValueError("selection contribution must contain B1_minus_A1 and B2_minus_A2")
    if len(dfs["phase4_contribution"]) != 40:
        raise ValueError("selection contribution must contain 2 comparisons × 5 embeddings × 4 ratios = 40 rows")


def sort_phase4(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_version_order"] = out["version"].map({v: i for i, v in enumerate(VERSION_ORDER)})
    out["_model_order"] = out["embedding_key"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["_version_order", "ratio", "_model_order"]).drop(columns=["_version_order", "_model_order"])


def build_phase4_integrated_synthesis(
    table4: pd.DataFrame,
    contribution: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Build Table 5 by joining performance, B−A contribution, and overlap artifacts."""
    rows: list[dict[str, Any]] = []
    pairs = [("fixed_params", "A1_fixed", "B1_fixed", "B1_minus_A1"), ("tuned_params", "A2_tuned", "B2_tuned", "B2_minus_A2")]
    overlap_summary = overlap.groupby(["embedding_key", "ratio"]).agg(
        jaccard_mean_vs_s_old=("jaccard_overlap", "mean"),
        jaccard_min_vs_s_old=("jaccard_overlap", "min"),
        jaccard_max_vs_s_old=("jaccard_overlap", "max"),
        selection_changed_any_fold=("selection_changed", "any"),
    ).reset_index()

    for family, a_version, b_version, comparison in pairs:
        for embedding_key in MODEL_ORDER:
            for ratio in RATIOS:
                a = table4[(table4["version"] == a_version) & (table4["embedding_key"] == embedding_key) & (table4["ratio"] == ratio)]
                b = table4[(table4["version"] == b_version) & (table4["embedding_key"] == embedding_key) & (table4["ratio"] == ratio)]
                c = contribution[(contribution["comparison"] == comparison) & (contribution["embedding_key"] == embedding_key) & (contribution["ratio"] == ratio)]
                o = overlap_summary[(overlap_summary["embedding_key"] == embedding_key) & (overlap_summary["ratio"] == ratio)]
                if a.empty or b.empty or c.empty or o.empty:
                    raise ValueError(f"Missing Phase 4 synthesis source rows for {family} {embedding_key} ratio={ratio}")
                ar = a.iloc[0]
                br = b.iloc[0]
                cr = c.iloc[0]
                orow = o.iloc[0]

                # Fail fast if table4's embedded contribution/overlap fields drift from
                # their canonical source artifacts.  This preserves table4 as a convenient
                # performance source without allowing stale derived columns to slip through.
                checks = [
                    (br["delta_item_r_selection_vs_A"], cr["delta_mean"], "delta_item_r_selection_vs_A"),
                    (br["p_selection_holm"], cr["p_holm"], "p_selection_holm"),
                    (br["p_selection_bh"], cr["p_bh"], "p_selection_bh"),
                    (br["jaccard_mean_vs_s_old"], orow["jaccard_mean_vs_s_old"], "jaccard_mean_vs_s_old"),
                    (br["jaccard_min_vs_s_old"], orow["jaccard_min_vs_s_old"], "jaccard_min_vs_s_old"),
                    (br["jaccard_max_vs_s_old"], orow["jaccard_max_vs_s_old"], "jaccard_max_vs_s_old"),
                ]
                for from_table4, from_source, field in checks:
                    if not np.isclose(float(from_table4), float(from_source), atol=1e-10, equal_nan=True):
                        raise ValueError(f"table4 {field} drift for {comparison} {embedding_key} m={ratio}: {from_table4} vs {from_source}")

                rows.append({
                    "family": family,
                    "a_version": a_version,
                    "b_version": b_version,
                    "embedding_key": embedding_key,
                    "embedding_label": br["embedding_label"],
                    "ratio": int(ratio),
                    "a_item_r_mean": ar["item_r_mean"],
                    "a_item_r_ci_lower": ar["item_r_ci_lower"],
                    "a_item_r_ci_upper": ar["item_r_ci_upper"],
                    "b_item_r_mean": br["item_r_mean"],
                    "b_item_r_ci_lower": br["item_r_ci_lower"],
                    "b_item_r_ci_upper": br["item_r_ci_upper"],
                    "delta_selection_vs_a": cr["delta_mean"],
                    "delta_selection_vs_a_ci_lower": cr["ci_lower"],
                    "delta_selection_vs_a_ci_upper": cr["ci_upper"],
                    "p_selection_holm": cr["p_holm"],
                    "p_selection_bh": cr["p_bh"],
                    "jaccard_mean_vs_s_old": orow["jaccard_mean_vs_s_old"],
                    "jaccard_min_vs_s_old": orow["jaccard_min_vs_s_old"],
                    "jaccard_max_vs_s_old": orow["jaccard_max_vs_s_old"],
                    "selection_changed_any_fold": bool(orow["selection_changed_any_fold"]),
                    "delta_a_vs_sbert": ar["delta_item_r_vs_sbert"],
                    "delta_b_vs_sbert": br["delta_item_r_vs_sbert"],
                    "best_of_pair": b_version if br["item_r_mean"] >= ar["item_r_mean"] else a_version,
                    "evidence_source_table4": "results/phase4/figures/table4.csv",
                    "evidence_source_contribution": "results/phase4/versionB_selection_contribution.csv",
                    "evidence_source_overlap": "results/phase4/versionB_selection_overlap.csv",
                })
    return pd.DataFrame(rows)


def build_best_pipeline_table(table4: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        sub = table4[table4["ratio"] == ratio].copy()
        sub["_fixed_pref"] = sub["version"].map(lambda v: 0 if v.endswith("fixed") else 1)
        sub = sub.sort_values(
            ["item_r_mean", "item_mae_mean", "trait_r_mean", "profile_r_mean", "_fixed_pref", "version", "embedding_key"],
            ascending=[False, True, False, False, True, True, True],
        )
        best = sub.iloc[0]
        version = str(best["version"])
        rows.append({
            "ratio": int(ratio),
            "recommended_version": version,
            "embedding_key": best["embedding_key"],
            "embedding_label": best["embedding_label"],
            "selection_strategy": "Coverage",
            "selection_scope": "S_old_fixed" if version.startswith("A") else "S_new_embedding_specific_reselected",
            "predictor": "SoftmaxKNN",
            "hyperparam_policy": "fixed_phase2_preregistered" if version.endswith("fixed") else "embedding_specific_train_inner_tuned",
            "K_mode": int(best["K_mode"]),
            "tau_mean": float(best["tau_mean"]),
            "item_r_mean": best["item_r_mean"],
            "item_r_ci_lower": best["item_r_ci_lower"],
            "item_r_ci_upper": best["item_r_ci_upper"],
            "trait_r_mean": best["trait_r_mean"],
            "profile_r_mean": best["profile_r_mean"],
            "item_mae_mean": best["item_mae_mean"],
            "delta_item_r_vs_sbert": best["delta_item_r_vs_sbert"],
            "p_vs_sbert_holm": best["p_vs_sbert_holm"],
            "delta_item_r_selection_vs_A": best["delta_item_r_selection_vs_A"],
            "p_selection_holm": best["p_selection_holm"],
            "source_summary_artifact": "results/phase4/figures/table4.csv",
        })
    return pd.DataFrame(rows)


def build_final_summary(dfs: dict[str, pd.DataFrame], best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    phase1_trait = dfs["phase1_table2_imputed"][["strategy", "ratio", "trait_r_mean"]]
    phase1 = dfs["phase1_table1"].merge(phase1_trait, on=["strategy", "ratio"], how="left")
    for row in phase1.itertuples(index=False):
        rows.append({
            "section": "phase1_selection_strategy",
            "phase": "phase1",
            "ratio": row.ratio,
            "version": "",
            "embedding_key": "sbert_original",
            "embedding_label": "SBERT (original)",
            "selection_strategy": row.strategy,
            "selection_scope": "phase1_selected_items",
            "predictor": "UniformKNN K=5",
            "hyperparam_policy": "original_fixed_K5",
            "K": 5,
            "tau": np.nan,
            "item_r_mean": row.item_r,
            "item_r_ci_lower": row.ci_lower,
            "item_r_ci_upper": row.ci_upper,
            "trait_r_mean": row.trait_r_mean,
            "profile_r_mean": np.nan,
            "item_mae_mean": np.nan,
            "reference": "Phase 1 item-selection comparison",
            "source_artifact": "results/phase1/figures/table1_item_level.csv",
        })

    for row in dfs["phase2_table3"].itertuples(index=False):
        rows.append({
            "section": "phase2_predictor_ablation",
            "phase": "phase2",
            "ratio": row.ratio,
            "version": "",
            "embedding_key": "sbert_original",
            "embedding_label": "SBERT (original)",
            "selection_strategy": "Coverage",
            "selection_scope": "S_old_phase1_coverage",
            "predictor": row.predictor,
            "hyperparam_policy": "train_inner_tuned" if "Tuned" in row.predictor or row.predictor in {"SoftmaxKNN", "KernelSmoothing", "CosineWeightedKNN"} else "original_fixed_K5",
            "K": row.best_K,
            "tau": row.best_tau,
            "item_r_mean": row.item_r,
            "item_r_ci_lower": row.item_r_ci_lower,
            "item_r_ci_upper": row.item_r_ci_upper,
            "trait_r_mean": row.trait_r_mean,
            "profile_r_mean": row.profile_r,
            "item_mae_mean": row.item_mae,
            "reference": "Phase 2 predictor ablation",
            "source_artifact": "results/phase2/figures/table3_predictor_ablation.csv",
        })

    for row in dfs["phase3_summary"].itertuples(index=False):
        rows.append({
            "section": "phase3_embedding_diagnostics",
            "phase": "phase3",
            "ratio": np.nan,
            "version": "",
            "embedding_key": row.embedding_key,
            "embedding_label": row.embedding_label,
            "selection_strategy": "Coverage diagnostics",
            "selection_scope": "embedding_space_full100_and_selected_sets",
            "predictor": "",
            "hyperparam_policy": "",
            "K": np.nan,
            "tau": np.nan,
            "item_r_mean": np.nan,
            "item_r_ci_lower": np.nan,
            "item_r_ci_upper": np.nan,
            "trait_r_mean": np.nan,
            "profile_r_mean": np.nan,
            "item_mae_mean": np.nan,
            "within_minus_between_raw_cosine": row.within_minus_between_raw_cosine,
            "mean_selected_set_coverage_shifted_cosine": row.mean_selected_set_coverage_shifted_cosine,
            "mean_selected_set_redundancy_shifted_cosine": row.mean_selected_set_redundancy_shifted_cosine,
            "reference": "Phase 3 embedding-space diagnostics",
            "source_artifact": "results/phase3/embedding_diagnostics_summary.csv",
        })

    for row in sort_phase4(dfs["phase4_table4"]).itertuples(index=False):
        rows.append({
            "section": "phase4_version_result",
            "phase": "phase4",
            "ratio": row.ratio,
            "version": row.version,
            "embedding_key": row.embedding_key,
            "embedding_label": row.embedding_label,
            "selection_strategy": "Coverage",
            "selection_scope": "S_old_fixed" if row.version.startswith("A") else "S_new_embedding_specific_reselected",
            "predictor": "SoftmaxKNN",
            "hyperparam_policy": "fixed_phase2_preregistered" if row.version.endswith("fixed") else "embedding_specific_train_inner_tuned",
            "K": row.K_mode,
            "tau": row.tau_mean,
            "item_r_mean": row.item_r_mean,
            "item_r_ci_lower": row.item_r_ci_lower,
            "item_r_ci_upper": row.item_r_ci_upper,
            "trait_r_mean": row.trait_r_mean,
            "profile_r_mean": row.profile_r_mean,
            "item_mae_mean": row.item_mae_mean,
            "delta_item_r_vs_sbert": row.delta_item_r_vs_sbert,
            "p_vs_sbert_holm": row.p_vs_sbert_holm,
            "delta_item_r_selection_vs_A": row.delta_item_r_selection_vs_A,
            "p_selection_holm": row.p_selection_holm,
            "jaccard_mean_vs_s_old": row.jaccard_mean_vs_s_old,
            "reference": "Phase 4 A/B embedding and re-selection comparison",
            "source_artifact": "results/phase4/figures/table4.csv",
        })

    for row in best.itertuples(index=False):
        rows.append({
            "section": "recommended_pipeline",
            "phase": "final",
            "ratio": row.ratio,
            "version": row.recommended_version,
            "embedding_key": row.embedding_key,
            "embedding_label": row.embedding_label,
            "selection_strategy": row.selection_strategy,
            "selection_scope": row.selection_scope,
            "predictor": row.predictor,
            "hyperparam_policy": row.hyperparam_policy,
            "K": row.K_mode,
            "tau": row.tau_mean,
            "item_r_mean": row.item_r_mean,
            "item_r_ci_lower": row.item_r_ci_lower,
            "item_r_ci_upper": row.item_r_ci_upper,
            "trait_r_mean": row.trait_r_mean,
            "profile_r_mean": row.profile_r_mean,
            "item_mae_mean": row.item_mae_mean,
            "delta_item_r_vs_sbert": row.delta_item_r_vs_sbert,
            "p_vs_sbert_holm": row.p_vs_sbert_holm,
            "delta_item_r_selection_vs_A": row.delta_item_r_selection_vs_A,
            "p_selection_holm": row.p_selection_holm,
            "reference": "Best observed pipeline by primary metric",
            "source_artifact": row.source_summary_artifact,
        })

    return pd.DataFrame(rows)


def build_statistical_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in dfs["phase1_stats"].itertuples(index=False):
        rows.append({
            "phase": "phase1",
            "comparison_family": "phase1_vs_random",
            "comparison": f"{row.strategy} vs {row.vs_baseline}",
            "ratio": row.ratio,
            "version": "",
            "embedding_key": "sbert_original",
            "embedding_label": "SBERT (original)",
            "metric": "item_r",
            "delta_mean": row.delta_mean,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "p_raw": row.p_value,
            "p_holm": np.nan,
            "p_bh": np.nan,
            "n_pairs": row.n_pairs,
            "bootstrap_unit": "folds_or_repeats_as_phase1_reported",
            "source_artifact": "results/phase1/figures/statistical_tests.csv",
        })
    for row in dfs["phase2_stats"].itertuples(index=False):
        rows.append({
            "phase": "phase2",
            "comparison_family": "phase2_predictor_ablation",
            "comparison": row.comparison,
            "ratio": row.ratio,
            "version": "",
            "embedding_key": "sbert_original",
            "embedding_label": "SBERT (original)",
            "metric": "item_r",
            "delta_mean": row.delta_mean,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "p_raw": row.p_value,
            "p_holm": np.nan,
            "p_bh": np.nan,
            "n_pairs": row.n_pairs,
            "bootstrap_unit": "paired_by_fold_as_phase2_reported",
            "source_artifact": "results/phase2/figures/statistical_tests_phase2.csv",
        })
    for source_key, rel, family in [
        ("phase4_stats_a", "results/phase4/versionA_statistical_tests.csv", "phase4_vs_sbert"),
        ("phase4_stats_b", "results/phase4/versionB_statistical_tests.csv", "phase4_vs_sbert"),
        ("phase4_contribution", "results/phase4/versionB_selection_contribution.csv", "phase4_selection_contribution"),
    ]:
        for row in dfs[source_key].itertuples(index=False):
            if source_key == "phase4_contribution":
                comparison = row.comparison
                version = f"{row.a_version}->{row.b_version}"
            else:
                comparison = f"{row.embedding_key} vs sbert_original"
                version = row.version
            rows.append({
                "phase": "phase4",
                "comparison_family": family,
                "comparison": comparison,
                "ratio": row.ratio,
                "version": version,
                "embedding_key": row.embedding_key,
                "embedding_label": row.embedding_label,
                "metric": row.metric,
                "delta_mean": row.delta_mean,
                "ci_lower": row.ci_lower,
                "ci_upper": row.ci_upper,
                "p_raw": row.p_raw,
                "p_holm": row.p_holm,
                "p_bh": row.p_bh,
                "n_pairs": row.n_pairs,
                "bootstrap_unit": getattr(row, "bootstrap_unit", "subjects_resampled_within_outer_fold"),
                "source_artifact": rel,
            })
    return pd.DataFrame(rows)


def copy_assets(inputs: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for key, src in inputs["figures"].items():
        dst = figures_dir / f"{key}{src.suffix}"
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


def plot_figure6(synthesis: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = {k: v for k, v in zip(synthesis["embedding_key"], synthesis["embedding_label"])}
    heat = synthesis.pivot_table(index=["family", "embedding_key"], columns="ratio", values="delta_selection_vs_a")
    heat = heat.reindex(pd.MultiIndex.from_product([["fixed_params", "tuned_params"], MODEL_ORDER], names=["family", "embedding_key"]))
    im = axes[0].imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    axes[0].set_xticks(range(len(RATIOS)), [str(r) for r in RATIOS])
    axes[0].set_yticks(range(len(heat.index)), [f"{fam.replace('_', ' ')}\n{labels.get(key, key)}" for fam, key in heat.index], fontsize=7)
    axes[0].set_title("B−A selection contribution (item_r)")
    axes[0].set_xlabel("Administered items (m)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    cmap = plt.get_cmap("tab10")
    color_map = {key: cmap(i) for i, key in enumerate(MODEL_ORDER)}
    marker_map = {"fixed_params": "o", "tuned_params": "s"}
    for row in synthesis.itertuples(index=False):
        axes[1].scatter(
            row.jaccard_mean_vs_s_old,
            row.delta_selection_vs_a,
            color=color_map[row.embedding_key],
            marker=marker_map[row.family],
            s=45,
            alpha=0.85,
        )
    for key in MODEL_ORDER:
        axes[1].scatter([], [], color=color_map[key], label=labels.get(key, key))
    axes[1].set_title("Selection change vs performance contribution")
    axes[1].set_xlabel("Mean Jaccard(S_new, S_old)")
    axes[1].set_ylabel("B−A Δ item_r")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Figure 6 — Integrated Phase 4 selection/performance synthesis")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def f4(x: Any) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def build_report_markdown(dfs: dict[str, pd.DataFrame], best: pd.DataFrame, synthesis: pd.DataFrame) -> str:
    phase1_best = dfs["phase1_table1"].sort_values("item_r", ascending=False).iloc[0]
    phase2_best_overall = dfs["phase2_table3"].groupby("predictor")["item_r"].mean().sort_values(ascending=False).index[0]
    phase3_cov = dfs["phase3_summary"].sort_values("mean_selected_set_coverage_shifted_cosine", ascending=False).iloc[0]
    phase3_gap = dfs["phase3_summary"].sort_values("within_minus_between_raw_cosine", ascending=False).iloc[0]
    phase4_best = best.sort_values("item_r_mean", ascending=False).iloc[0]
    max_sel = synthesis.sort_values("delta_selection_vs_a", ascending=False).iloc[0]

    lines = [
        "# Final Report: Questionnaire Embeddings Phase 1–4 Integrated Analysis",
        "",
        "## Scope and deliverables",
        "This F015 report aggregates existing accepted artifacts only. It does not rerun Phase 1–4 experiments, regenerate embeddings, or re-estimate bootstrap tests.",
        f"Primary metric: **{PRIMARY_METRIC}**. Key secondary metrics: **{SECONDARY_METRICS}**.",
        "",
        "## Executive summary",
        f"- Phase 1 selected **{phase1_best['strategy']}** as the strongest item-selection strategy in the observed Table 1 cells (best cell m={int(phase1_best['ratio'])}, item_r={f4(phase1_best['item_r'])}).",
        f"- Phase 2 selected **{phase2_best_overall}** as the recommended predictor family over Coverage-selected items.",
        f"- Phase 3 diagnostics suggested **{phase3_cov['embedding_label']}** for highest selected-set coverage and **{phase3_gap['embedding_label']}** for strongest within-minus-between trait separation.",
        f"- Phase 4 best observed pipeline cell was **{phase4_best['recommended_version']} / {phase4_best['embedding_label']}** at m={int(phase4_best['ratio'])}, item_r={f4(phase4_best['item_r_mean'])}.",
        f"- Largest positive B−A re-selection contribution was **{max_sel['b_version']} minus {max_sel['a_version']}** for {max_sel['embedding_label']} at m={int(max_sel['ratio'])}, Δitem_r={f4(max_sel['delta_selection_vs_a'])}.",
        "",
        "## Phase 1: item-selection contribution",
        "Phase 1 compared random, balanced-random, semantic Coverage, Coverage+Diversity, TraitPredictiveness, and Hybrid strategies. The final report carries forward Coverage as the preferred semantic selection strategy and treats Phase 1 Table 1/Table 2 as the source of item-level and trait-level evidence.",
        "",
        "## Phase 2: prediction-algorithm contribution",
        "Phase 2 compared Tuned UniformKNN, CosineWeightedKNN, SoftmaxKNN, KernelSmoothing, and the cross-phase original-paper UniformKNN K=5 baseline. The final Phase 4 pipeline uses SoftmaxKNN because it gave the best overall item-level Pearson r in Phase 2 and supports both fixed Phase 2 hyperparameters and train-inner tuning.",
        "",
        "## Phase 3: embedding-space diagnostics",
        "Phase 3 compared SBERT original, MiniLM, MPNet, E5, and BGE using Coverage/Redundancy and raw-cosine trait-structure diagnostics. These diagnostics are explanatory hypotheses, not predictive-performance claims; Phase 4 is the predictive test.",
        "",
        "## Phase 4: A1/A2/B1/B2 attribution",
        "- **A1_fixed** = fixed historical SBERT Coverage `S_old` plus fixed Phase 2 SoftmaxKNN hyperparameters. This isolates embedding-neighbor geometry under a fixed administered item set.",
        "- **A2_tuned** = fixed `S_old` plus embedding-specific train-inner K/τ tuning. This estimates geometry contribution after calibration without changing item selection.",
        "- **B1_fixed** = embedding-specific Coverage re-selection `S_new` plus fixed Phase 2 hyperparameters. This adds re-selection contribution while keeping hyperparameters fixed.",
        "- **B2_tuned** = embedding-specific `S_new` plus embedding-specific train-inner K/τ tuning. This is the full re-selected+tuned Phase 4 pipeline.",
        "The B−A comparisons are paired within the same embedding and tuning regime, so B1−A1 and B2−A2 quantify selection contribution beyond the corresponding fixed-`S_old` baseline.",
        "",
        "## Best pipeline by administered-item ratio",
        "| m | version | embedding | selection | predictor | K | tau | item_r | trait_r_mean | profile_r | MAE |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {int(row.ratio)} | {row.recommended_version} | {row.embedding_label} | {row.selection_scope} | {row.predictor} | {int(row.K_mode)} | {f4(row.tau_mean)} | {f4(row.item_r_mean)} | {f4(row.trait_r_mean)} | {f4(row.profile_r_mean)} | {f4(row.item_mae_mean)} |"
        )
    lines.extend([
        "",
        "## Statistical inference",
        "The primary Phase 4 statistical comparisons use paired bootstrap over participants while preserving outer-fold pairing. New embeddings are compared against `sbert_original`; B−A selection-contribution tests compare B1 vs A1 and B2 vs A2 within the same embedding and ratio. Raw p-values plus Holm and Benjamini-Hochberg corrected p-values are preserved in `statistical_summary.csv`.",
        "",
        "## Reproducibility notes",
        "- Random state is fixed at 0 in upstream CV and selection code.",
        "- Outer folds are participant-level folds and are paired across Phase 4 A/B comparisons.",
        "- A2/B2 K and τ values are selected only on train-inner validation splits; test participants are held out for final evaluation.",
        "- Phase 4 primary predictions are continuous and clipped to [1,5] without rounding; rounded accuracy and rounded MAE are supplemental outputs only.",
        "- This report reads existing CSV/TXT/PNG artifacts and records source hashes in `report_manifest.json`.",
        "",
        "## Output inventory",
        "- `final_summary.csv`: normalized Phase 1–4 and recommended-pipeline rows.",
        "- `statistical_summary.csv`: normalized inferential tests from Phase 1, Phase 2, and Phase 4.",
        "- `best_pipeline_table.csv`: best observed Phase 4 pipeline by 10/30/50/90 administered items.",
        "- `figures/table5_phase4_integrated_synthesis.csv` (**Table 5**): A-vs-B synthesis table for Phase 4.",
        "- `figures/figure6_phase4_selection_vs_performance.pdf/png` (**Figure 6**): new integrated selection/performance synthesis figure.",
        "",
        "## Limitations",
        LIMITATION_TEXT,
        "No cross-questionnaire generalization experiment is added in F015; the report is an aggregation and synthesis of existing NEO-PI-R artifacts.",
        "",
    ])
    return "\n".join(lines)


def render_text_report(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            line = stripped.lstrip("#").lstrip()
        lines.append(line.replace("**", "").replace("`", ""))
    return "\n".join(lines)


def add_text_page(pdf: Any, title: str, paragraphs: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 0.95
    ax.text(0.07, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.06
    for para in paragraphs:
        for line in textwrap.wrap(para, width=95):
            ax.text(0.07, y, line, fontsize=9.5, va="top")
            y -= 0.022
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.95
        y -= 0.018
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: Any, title: str, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(f4)
    table = ax.table(cellText=display.astype(str).values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: Any, title: str, image_path: Path, caption: str) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    img = mpimg.imread(image_path)
    ax.imshow(img)
    fig.text(0.5, 0.03, caption, ha="center", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_pdf_report(pdf_path: Path, markdown: str, best: pd.DataFrame, copied_assets: dict[str, Path], figure6: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    paragraphs = [p.strip("- ") for p in markdown.split("\n") if p.strip() and not p.startswith("|")]
    metadata = {"Title": "F015 Final Report", "Author": "questionnaire-embeddings", "Subject": "Deterministic aggregation report", "Keywords": "F015 Phase 1 Phase 2 Phase 3 Phase 4", "CreationDate": None, "ModDate": None}
    with PdfPages(pdf_path, metadata=metadata) as pdf:
        add_text_page(pdf, "F015 Final Integrated Report", paragraphs[:18])
        add_text_page(pdf, "Phase 4 attribution, reproducibility, and limitations", paragraphs[18:])
        add_table_page(pdf, "Best pipeline by administered-item ratio", best[["ratio", "recommended_version", "embedding_label", "selection_scope", "K_mode", "tau_mean", "item_r_mean", "trait_r_mean", "profile_r_mean", "item_mae_mean"]])
        for key, title in [
            ("phase1_figure1_learning_curve", "Phase 1 learning curve"),
            ("phase2_figure3_delta_r", "Phase 2 predictor deltas"),
            ("phase3_figure5", "Phase 3 embedding diagnostics"),
            ("phase4_figure4", "Phase 4 A/B learning curves"),
        ]:
            add_image_page(pdf, title, copied_assets[key], f"Copied from accepted upstream artifact: {copied_assets[key].name}")
        add_image_page(pdf, "Figure 6 — new integrated Phase 4 synthesis", figure6, "Generated by F015 from table4, overlap, and B−A contribution artifacts.")


def write_manifest(inputs: dict[str, Any], output_dir: Path, output_files: dict[str, Path], row_counts: dict[str, int]) -> None:
    sources = []
    for key, path in sorted(inputs["sources"].items()):
        sources.append({
            "key": key,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    manifest = {
        "feature_id": "F015",
        "generated_at": "deterministic_not_recorded",
        "script_path": "scripts/generate_final_report.py",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "no_experiments_rerun": True,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": SECONDARY_METRICS,
        "limitation_cross_questionnaire_not_run": True,
        "source_artifacts": sources,
        "output_files": {k: str(v.relative_to(output_dir)) for k, v in sorted(output_files.items())},
        "row_counts": row_counts,
        "required_phase4_versions": list(VERSION_ORDER),
        "required_ratios": list(RATIOS),
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    dfs = inputs["dataframes"]
    table4 = dfs["phase4_table4"]

    synthesis = build_phase4_integrated_synthesis(table4, dfs["phase4_contribution"], dfs["phase4_overlap"])
    best = build_best_pipeline_table(table4)
    final_summary = build_final_summary(dfs, best)
    statistical_summary = build_statistical_summary(dfs)

    final_summary_path = output_dir / "final_summary.csv"
    stats_path = output_dir / "statistical_summary.csv"
    best_path = output_dir / "best_pipeline_table.csv"
    synth_path = figures_dir / "table5_phase4_integrated_synthesis.csv"
    figure6_pdf = figures_dir / "figure6_phase4_selection_vs_performance.pdf"
    figure6_png = figures_dir / "figure6_phase4_selection_vs_performance.png"
    md_path = output_dir / "final_report.md"
    txt_path = output_dir / "final_report.txt"
    pdf_path = output_dir / "final_report.pdf"

    final_summary.to_csv(final_summary_path, index=False)
    statistical_summary.to_csv(stats_path, index=False)
    best.to_csv(best_path, index=False)
    synthesis.to_csv(synth_path, index=False)
    plot_figure6(synthesis, figure6_pdf, figure6_png)

    copied_assets = copy_assets(inputs, output_dir)
    markdown = build_report_markdown(dfs, best, synthesis)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    txt_path.write_text(render_text_report(markdown) + "\n", encoding="utf-8")
    render_pdf_report(pdf_path, markdown, best, copied_assets, figure6_png)

    output_files = {
        "final_summary": final_summary_path,
        "statistical_summary": stats_path,
        "best_pipeline_table": best_path,
        "phase4_integrated_synthesis": synth_path,
        "figure6_pdf": figure6_pdf,
        "figure6_png": figure6_png,
        "final_report_md": md_path,
        "final_report_txt": txt_path,
        "final_report_pdf": pdf_path,
        "report_manifest": output_dir / "report_manifest.json",
    }
    row_counts = {
        "final_summary": len(final_summary),
        "statistical_summary": len(statistical_summary),
        "best_pipeline_table": len(best),
        "phase4_integrated_synthesis": len(synthesis),
    }
    write_manifest(inputs, output_dir, output_files, row_counts)

    print("=" * 70)
    print("F015 final report generated")
    print("=" * 70)
    for key, path in output_files.items():
        print(f"[SAVE] {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

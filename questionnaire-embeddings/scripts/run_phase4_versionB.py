#!/usr/bin/env python3
"""F014: Phase 4 Version B1/B2 embedding-specific re-selection experiment.

Version B re-runs semantic Coverage selection for each embedding and compares
that re-selected short form against F013 Version A, which held the original
SBERT Coverage short form fixed.  Prediction/scoring conventions intentionally
mirror F013: continuous clip-only SoftmaxKNN predictions, evaluated only on
unselected/held-out items.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cv_framework import evaluate_predictions, participant_cv_split  # noqa: E402
from phase4_common import (  # noqa: E402
    A1_FIXED_PARAMS,
    K_CANDIDATES,
    MODEL_ORDER,
    N_FOLDS,
    PHASE4_DIR,
    RANDOM_STATE,
    RATIOS,
    TAU_CANDIDATES,
    adjust_pvalues_bh,
    adjust_pvalues_holm,
    build_embedding_registry,
    load_core_data,
    load_embedding_matrix,
    load_fixed_s_old_by_fold_ratio,
    mean_ci,
    paired_bootstrap_by_fold,
    precompute_similarity,
    stable_hash,
    write_json,
    write_predictions_parquet,
)
from phase4_predictors import ContinuousSoftmaxKNN  # noqa: E402
from run_phase4_versionA import (  # noqa: E402
    SCORING_CONVENTION,
    aggregate_summary,
    build_bootstrap_tests,
    participant_metric_rows,
    rounded_mae_from_prediction,
    tune_a2,
)
from selection import CoverageSelector  # noqa: E402


VERSION_B1 = "B1_fixed"
VERSION_B2 = "B2_tuned"
PREDICTOR_NAME = "SoftmaxKNN"
A_TO_B = {
    VERSION_B1: "A1_fixed",
    VERSION_B2: "A2_tuned",
}
VERSION_LABELS = {
    "A1_fixed": "A1 fixed S_old + fixed params",
    "A2_tuned": "A2 fixed S_old + tuned params",
    VERSION_B1: "B1 S_new + fixed params",
    VERSION_B2: "B2 S_new + tuned params",
}


@dataclass(frozen=True)
class SelectedSetB:
    embedding_key: str
    embedding_label: str
    ratio: int
    selected_indices: np.ndarray
    selected_question_ids: list[str]
    coverage: float
    redundancy: float

    @property
    def selected_csv(self) -> str:
        return ",".join(str(int(i)) for i in self.selected_indices)

    @property
    def selected_hash(self) -> str:
        return stable_hash(self.selected_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F014 Phase 4 Version B re-selection experiment.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Run all 5 folds × 4 ratios × 5 embeddings × B1/B2.")
    mode.add_argument("--quick", action="store_true", help="Run 1 fold × ratios 10/30 × all embeddings × B1/B2.")
    mode.add_argument("--smoke", action="store_true", help="Run 1 fold × ratio 10 × sbert_original × B1/B2.")
    parser.add_argument("--n-bootstrap", type=int, default=None, help="Bootstrap iterations for statistical tests.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/phase4 for --all and scratch dirs for --quick/--smoke.",
    )
    parser.add_argument(
        "--version-a-dir",
        type=Path,
        default=PHASE4_DIR,
        help="Directory containing F013 Version A artifacts used for B−A contribution tests.",
    )
    return parser.parse_args()


def mode_config(args: argparse.Namespace) -> tuple[str, list[int], tuple[int, ...], tuple[str, ...], tuple[str, ...], int, Path]:
    if args.smoke:
        output_dir = args.output_dir or (PHASE4_DIR.parent / "phase4_versionB_smoke")
        return "SMOKE", [0], (10,), ("sbert_original",), (VERSION_B1, VERSION_B2), args.n_bootstrap or 200, output_dir
    if args.quick:
        output_dir = args.output_dir or (PHASE4_DIR.parent / "phase4_versionB_quick")
        return "QUICK", [0], (10, 30), MODEL_ORDER, (VERSION_B1, VERSION_B2), args.n_bootstrap or 500, output_dir
    output_dir = args.output_dir or PHASE4_DIR
    return "FULL", list(range(N_FOLDS)), RATIOS, MODEL_ORDER, (VERSION_B1, VERSION_B2), args.n_bootstrap or 10_000, output_dir


def version_b_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "predictions": output_dir / "versionB_predictions.parquet",
        "participant_metrics": output_dir / "versionB_participant_metrics.csv",
        "results": output_dir / "versionB_results.csv",
        "aggregate_metrics": output_dir / "versionB_aggregate_metrics.csv",
        "summary": output_dir / "versionB_summary.csv",
        "hyperparameters": output_dir / "versionB_hyperparameters_by_fold_ratio_embedding.csv",
        "selected_items": output_dir / "versionB_selected_items_by_fold_ratio_embedding.json",
        "stats": output_dir / "versionB_statistical_tests.csv",
        "selection_contribution": output_dir / "versionB_selection_contribution.csv",
        "selection_overlap": output_dir / "versionB_selection_overlap.csv",
        "folds": output_dir / "versionB_outer_folds_subject_ids.json",
        "table4": output_dir / "figures" / "table4.csv",
        "figure4_pdf": output_dir / "figures" / "figure4.pdf",
        "figure4_png": output_dir / "figures" / "figure4.png",
    }


def jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    set_a = {int(x) for x in np.asarray(a, dtype=np.intp)}
    set_b = {int(x) for x in np.asarray(b, dtype=np.intp)}
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return (float(inter / union) if union else np.nan, inter, union)


def build_selected_sets(
    matrices: dict[str, np.ndarray],
    specs: dict[str, Any],
    question_ids: list[str],
    ratios: tuple[int, ...],
) -> dict[tuple[str, int], SelectedSetB]:
    selected: dict[tuple[str, int], SelectedSetB] = {}
    for embedding_key, E in matrices.items():
        selector = CoverageSelector(E)
        for ratio in ratios:
            S = selector.select(ratio)
            selected[(embedding_key, ratio)] = SelectedSetB(
                embedding_key=embedding_key,
                embedding_label=specs[embedding_key].label,
                ratio=int(ratio),
                selected_indices=S,
                selected_question_ids=[question_ids[int(i)] for i in S],
                coverage=selector.compute_coverage(S),
                redundancy=selector.compute_redundancy(S),
            )
    return selected


def make_fold_result_row_b(
    *,
    version: str,
    embedding_key: str,
    embedding_label: str,
    fold_idx: int,
    ratio: int,
    K: int,
    tau: float,
    hyperparam_source: str,
    selected: SelectedSetB,
    s_old_hash: str,
    s_old_coverage: float,
    s_old_redundancy: float,
    jaccard_vs_s_old: float,
    intersection_n: int,
    union_n: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    trait_ids: np.ndarray,
    reverse_ids: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_predictions(y_test, y_pred, trait_ids, reverse_ids)
    item = metrics["item_level"]
    trait = metrics["trait_level"]
    profile = metrics["profile_correlation"]
    per_trait = trait["per_trait_r"]
    return {
        "version": version,
        "embedding_key": embedding_key,
        "embedding_label": embedding_label,
        "ratio": int(ratio),
        "fold": int(fold_idx),
        "predictor": PREDICTOR_NAME,
        "rounded": False,
        "scoring_convention": SCORING_CONVENTION,
        "hyperparam_source": hyperparam_source,
        "best_K": int(K),
        "best_tau": float(tau),
        "item_r": item["pearson_r"][0],
        "item_r_ci_lower": item["pearson_r"][1],
        "item_r_ci_upper": item["pearson_r"][2],
        "item_mae": item["mae"][0],
        "item_rmse": item["rmse"][0],
        "item_rounded_accuracy": item["rounded_accuracy"][0],
        "item_rounded_mae": rounded_mae_from_prediction(y_test, y_pred),
        "trait_r_O": per_trait.get("O", np.nan),
        "trait_r_C": per_trait.get("C", np.nan),
        "trait_r_E": per_trait.get("E", np.nan),
        "trait_r_A": per_trait.get("A", np.nan),
        "trait_r_N": per_trait.get("N", np.nan),
        "trait_r_mean": trait.get("mean_big5_r", np.nan),
        "profile_r": profile.get("mean", np.nan),
        "selected_S": selected.selected_csv,
        "selected_S_hash": selected.selected_hash,
        "selected_S_source": "coverage_selector_embedding_specific",
        "s_new_coverage": selected.coverage,
        "s_new_redundancy": selected.redundancy,
        "s_old_fold_hash": s_old_hash,
        "s_old_coverage": s_old_coverage,
        "s_old_redundancy": s_old_redundancy,
        "jaccard_vs_s_old": jaccard_vs_s_old,
        "intersection_n": int(intersection_n),
        "union_n": int(union_n),
        "n_test_subjects": int(y_test.shape[0]),
    }


def enrich_participant_outputs(
    records: list[dict[str, Any]],
    metric_df: pd.DataFrame,
    *,
    selected: SelectedSetB,
    s_old_hash: str,
    jaccard_vs_s_old: float,
) -> pd.DataFrame:
    extra = {
        "selected_S_hash": selected.selected_hash,
        "selected_S_source": "coverage_selector_embedding_specific",
        "s_new_coverage": selected.coverage,
        "s_new_redundancy": selected.redundancy,
        "s_old_fold_hash": s_old_hash,
        "jaccard_vs_s_old": jaccard_vs_s_old,
        "selection_scope": "embedding_ratio_materialized_by_fold",
    }
    for row in records:
        row.update(extra)
    for key, value in extra.items():
        metric_df[key] = value
    return metric_df


def build_overlap_rows(
    *,
    selected_sets: dict[tuple[str, int], SelectedSetB],
    s_old_sets: dict[tuple[int, int], Any],
    active_folds: list[int],
    ratios: tuple[int, ...],
    embeddings: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for embedding_key in embeddings:
        for ratio in ratios:
            selected = selected_sets[(embedding_key, ratio)]
            for fold_idx in active_folds:
                old = s_old_sets[(fold_idx, ratio)]
                jac, inter, union = jaccard(selected.selected_indices, old.selected_indices)
                only_new = sorted(set(map(int, selected.selected_indices)) - set(map(int, old.selected_indices)))
                only_old = sorted(set(map(int, old.selected_indices)) - set(map(int, selected.selected_indices)))
                rows.append({
                    "embedding_key": embedding_key,
                    "embedding_label": selected.embedding_label,
                    "ratio": int(ratio),
                    "fold": int(fold_idx),
                    "version_scope": "B1_and_B2_shared_selection",
                    "s_new_hash": selected.selected_hash,
                    "s_old_fold_hash": old.selected_hash,
                    "s_new_size": int(len(selected.selected_indices)),
                    "s_old_size": int(len(old.selected_indices)),
                    "intersection_n": int(inter),
                    "union_n": int(union),
                    "jaccard_overlap": jac,
                    "selection_changed": bool(jac < 1.0),
                    "items_only_in_s_new": ",".join(str(i) for i in only_new),
                    "items_only_in_s_old": ",".join(str(i) for i in only_old),
                    "s_new_coverage": selected.coverage,
                    "s_new_redundancy": selected.redundancy,
                    "s_old_coverage": old.coverage,
                    "s_old_redundancy": old.redundancy,
                })
    return rows


def validate_outputs_b(
    *,
    mode: str,
    participant_records: list[dict[str, Any]],
    participant_df: pd.DataFrame,
    fold_rows: list[dict[str, Any]],
    hyper_rows: list[dict[str, Any]],
    overlap_df: pd.DataFrame,
) -> None:
    if not participant_records:
        raise RuntimeError("No Version B prediction records produced")
    for row in participant_records[: min(25, len(participant_records))]:
        n = len(row["heldout_item_ids"])
        if len(row["y_true"]) != n or len(row["y_pred_continuous"]) != n:
            raise RuntimeError("Prediction vector lengths do not match heldout_item_ids")
        if any(x is None or np.isnan(float(x)) for x in row["y_pred_continuous"]):
            raise RuntimeError("Prediction record contains NaN y_pred_continuous")
        y_pred = np.asarray(row["y_pred_continuous"], dtype=np.float64)
        if np.nanmin(y_pred) < 1.0 or np.nanmax(y_pred) > 5.0:
            raise RuntimeError("Continuous predictions outside [1,5]")
    if participant_df["per_subject_mae"].isna().any():
        raise RuntimeError("Participant MAE contains NaN")
    if not fold_rows or not hyper_rows:
        raise RuntimeError("Missing fold-level or hyperparameter rows")
    hyper = pd.DataFrame(hyper_rows)
    if VERSION_B1 in set(hyper["version"]):
        if hyper[hyper["version"] == VERSION_B1]["is_tuned"].any():
            raise RuntimeError("B1 hyperparameter rows indicate tuning")
    if VERSION_B2 in set(hyper["version"]):
        if not hyper[hyper["version"] == VERSION_B2]["is_tuned"].all():
            raise RuntimeError("B2 hyperparameter rows are not marked tuned")
    if overlap_df.empty:
        raise RuntimeError("Missing selection overlap rows")
    if not overlap_df["jaccard_overlap"].between(0.0, 1.0).all():
        raise RuntimeError("Jaccard overlap outside [0,1]")
    if mode == "SMOKE" and set(participant_df["version"].unique()) != {VERSION_B1, VERSION_B2}:
        raise RuntimeError("Smoke mode should run B1 and B2")


def build_selection_contribution(
    b_participants: pd.DataFrame,
    a_dir: Path,
    n_bootstrap: int,
    overlap_df: pd.DataFrame,
) -> pd.DataFrame:
    a_path = a_dir / "versionA_participant_metrics.csv"
    if not a_path.exists():
        raise RuntimeError(f"Missing Version A participant metrics for B−A comparison: {a_path}")
    a = pd.read_csv(a_path)
    rows: list[dict[str, Any]] = []
    overlap_summary = overlap_df.groupby(["embedding_key", "ratio"]).agg(
        jaccard_mean_vs_s_old=("jaccard_overlap", "mean"),
        selection_changed_any_fold=("selection_changed", "any"),
    ).reset_index()

    for b_version, a_version in A_TO_B.items():
        b_sub = b_participants[b_participants["version"] == b_version]
        a_sub = a[a["version"] == a_version]
        for (embedding_key, embedding_label, ratio), b_grp in b_sub.groupby(["embedding_key", "embedding_label", "ratio"]):
            a_grp = a_sub[(a_sub["embedding_key"] == embedding_key) & (a_sub["ratio"] == ratio)]
            if a_grp.empty:
                raise RuntimeError(f"No Version A rows for {a_version}, {embedding_key}, ratio={ratio}")
            merged = b_grp[["subject_id", "outer_fold", "per_subject_item_r"]].merge(
                a_grp[["subject_id", "outer_fold", "per_subject_item_r"]],
                on=["subject_id", "outer_fold"],
                how="inner",
                suffixes=("_B", "_A"),
            )
            expected = len(b_grp)
            if len(merged) != expected:
                raise RuntimeError(
                    f"B−A pairing failed for {b_version} vs {a_version}, {embedding_key}, ratio={ratio}: "
                    f"merged={len(merged)} expected={expected}"
                )
            merged["diff"] = merged["per_subject_item_r_B"] - merged["per_subject_item_r_A"]
            result = paired_bootstrap_by_fold(
                merged,
                n_bootstrap=n_bootstrap,
                seed=RANDOM_STATE + int(ratio) + (2000 if b_version == VERSION_B2 else 1000),
            )
            ov = overlap_summary[(overlap_summary["embedding_key"] == embedding_key) & (overlap_summary["ratio"] == ratio)]
            rows.append({
                "comparison": "B1_minus_A1" if b_version == VERSION_B1 else "B2_minus_A2",
                "a_version": a_version,
                "b_version": b_version,
                "embedding_key": embedding_key,
                "embedding_label": embedding_label,
                "ratio": int(ratio),
                "metric": "per_subject_item_r",
                "delta_mean": result["delta"],
                "ci_lower": result["ci_low"],
                "ci_upper": result["ci_high"],
                "p_raw": result["p"],
                "n_pairs": result["n"],
                "bootstrap_unit": "subjects_resampled_within_outer_fold",
                "bootstrap_iterations": int(n_bootstrap),
                "selection_changed_any_fold": bool(ov["selection_changed_any_fold"].iloc[0]) if not ov.empty else np.nan,
                "jaccard_mean_vs_s_old": float(ov["jaccard_mean_vs_s_old"].iloc[0]) if not ov.empty else np.nan,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    df["p_bh"] = np.nan
    for comparison, idx in df.groupby("comparison").groups.items():
        pvals = df.loc[idx, "p_raw"].tolist()
        df.loc[idx, "p_holm"] = adjust_pvalues_holm(pvals)
        df.loc[idx, "p_bh"] = adjust_pvalues_bh(pvals)
    return df.sort_values(["comparison", "ratio", "embedding_key"]).reset_index(drop=True)


def build_table4_and_figure(
    *,
    output_dir: Path,
    a_dir: Path,
    b_summary: pd.DataFrame,
    b_stats: pd.DataFrame,
    contribution: pd.DataFrame,
    overlap_df: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    a_summary_path = a_dir / "versionA_summary.csv"
    if not a_summary_path.exists():
        raise RuntimeError(f"Missing Version A summary for Table 4/Figure 4: {a_summary_path}")
    a_summary = pd.read_csv(a_summary_path)
    a_stats_path = a_dir / "versionA_statistical_tests.csv"
    a_stats = pd.read_csv(a_stats_path) if a_stats_path.exists() else pd.DataFrame()

    perf = pd.concat([a_summary, b_summary], ignore_index=True, sort=False)
    rows = []
    overlap_summary = overlap_df.groupby(["embedding_key", "ratio"]).agg(
        jaccard_mean_vs_s_old=("jaccard_overlap", "mean"),
        jaccard_min_vs_s_old=("jaccard_overlap", "min"),
        jaccard_max_vs_s_old=("jaccard_overlap", "max"),
    ).reset_index()

    for row in perf.itertuples(index=False):
        out = row._asdict()
        version = out["version"]
        embedding_key = out["embedding_key"]
        ratio = int(out["ratio"])
        stats_source = b_stats if version in {VERSION_B1, VERSION_B2} else a_stats
        vs = pd.DataFrame()
        if not stats_source.empty and embedding_key != "sbert_original":
            vs = stats_source[
                (stats_source["version"] == version)
                & (stats_source["embedding_key"] == embedding_key)
                & (stats_source["ratio"] == ratio)
            ]
        out["delta_item_r_vs_sbert"] = float(vs["delta_mean"].iloc[0]) if not vs.empty else np.nan
        out["delta_item_r_vs_sbert_ci_lower"] = float(vs["ci_lower"].iloc[0]) if not vs.empty else np.nan
        out["delta_item_r_vs_sbert_ci_upper"] = float(vs["ci_upper"].iloc[0]) if not vs.empty else np.nan
        out["p_vs_sbert_raw"] = float(vs["p_raw"].iloc[0]) if not vs.empty else np.nan
        out["p_vs_sbert_holm"] = float(vs["p_holm"].iloc[0]) if not vs.empty else np.nan
        out["p_vs_sbert_bh"] = float(vs["p_bh"].iloc[0]) if not vs.empty else np.nan

        comp_name = "B1_minus_A1" if version == VERSION_B1 else "B2_minus_A2" if version == VERSION_B2 else None
        sel = pd.DataFrame()
        if comp_name and not contribution.empty:
            sel = contribution[
                (contribution["comparison"] == comp_name)
                & (contribution["embedding_key"] == embedding_key)
                & (contribution["ratio"] == ratio)
            ]
        out["delta_item_r_selection_vs_A"] = float(sel["delta_mean"].iloc[0]) if not sel.empty else np.nan
        out["delta_item_r_selection_vs_A_ci_lower"] = float(sel["ci_lower"].iloc[0]) if not sel.empty else np.nan
        out["delta_item_r_selection_vs_A_ci_upper"] = float(sel["ci_upper"].iloc[0]) if not sel.empty else np.nan
        out["p_selection_raw"] = float(sel["p_raw"].iloc[0]) if not sel.empty else np.nan
        out["p_selection_holm"] = float(sel["p_holm"].iloc[0]) if not sel.empty else np.nan
        out["p_selection_bh"] = float(sel["p_bh"].iloc[0]) if not sel.empty else np.nan

        ov = overlap_summary[(overlap_summary["embedding_key"] == embedding_key) & (overlap_summary["ratio"] == ratio)]
        out["jaccard_mean_vs_s_old"] = float(ov["jaccard_mean_vs_s_old"].iloc[0]) if not ov.empty and version in {VERSION_B1, VERSION_B2} else np.nan
        out["jaccard_min_vs_s_old"] = float(ov["jaccard_min_vs_s_old"].iloc[0]) if not ov.empty and version in {VERSION_B1, VERSION_B2} else np.nan
        out["jaccard_max_vs_s_old"] = float(ov["jaccard_max_vs_s_old"].iloc[0]) if not ov.empty and version in {VERSION_B1, VERSION_B2} else np.nan
        rows.append(out)

    table4 = pd.DataFrame(rows)
    paths["table4"].parent.mkdir(parents=True, exist_ok=True)
    table4.to_csv(paths["table4"], index=False)
    _plot_figure4(table4, paths["figure4_pdf"], paths["figure4_png"])


def _plot_figure4(table4: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    versions = [("fixed params", "A1_fixed", VERSION_B1), ("tuned params", "A2_tuned", VERSION_B2)]
    color_keys = [key for key in MODEL_ORDER if key in set(table4["embedding_key"])]
    cmap = plt.get_cmap("tab10")
    colors = {key: cmap(i % 10) for i, key in enumerate(color_keys)}
    labels = dict(zip(table4["embedding_key"], table4["embedding_label"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (title, a_version, b_version) in zip(axes, versions):
        for embedding_key in color_keys:
            for version, linestyle, marker in [(a_version, "--", "o"), (b_version, "-", "s")]:
                sub = table4[(table4["version"] == version) & (table4["embedding_key"] == embedding_key)].sort_values("ratio")
                if sub.empty:
                    continue
                yerr = np.vstack([
                    sub["item_r_mean"] - sub["item_r_ci_lower"],
                    sub["item_r_ci_upper"] - sub["item_r_mean"],
                ])
                ax.errorbar(
                    sub["ratio"],
                    sub["item_r_mean"],
                    yerr=yerr,
                    color=colors[embedding_key],
                    linestyle=linestyle,
                    marker=marker,
                    capsize=3,
                    linewidth=1.5,
                    label=f"{labels.get(embedding_key, embedding_key)} {'A' if version.startswith('A') else 'B'}",
                )
        ax.set_title(title)
        ax.set_xlabel("Administered items (m)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Primary item-level Pearson r")
    handles, labels_seen = axes[1].get_legend_handles_labels()
    dedup = dict(zip(labels_seen, handles))
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=7,
        frameon=True,
    )
    fig.text(
        0.5,
        0.15,
        "Dashed/circle = Version A fixed S_old; solid/square = Version B re-selected S_new",
        ha="center",
        fontsize=9,
    )
    fig.suptitle("Figure 4 — Phase 4 A/B embedding learning curves")
    fig.tight_layout(rect=[0, 0.22, 1, 0.95])
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_selected_items_audit_b(
    paths: dict[str, Path],
    *,
    selected_sets: dict[tuple[str, int], SelectedSetB],
    s_old_sets: dict[tuple[int, int], Any],
    overlap_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    subject_ids: list[str],
    active_folds: list[int],
    ratios: tuple[int, ...],
    embeddings: tuple[str, ...],
    versions: tuple[str, ...],
) -> None:
    records = []
    for version in versions:
        for embedding_key in embeddings:
            for fold_idx in active_folds:
                for ratio in ratios:
                    selected = selected_sets[(embedding_key, ratio)]
                    old = s_old_sets[(fold_idx, ratio)]
                    ov = overlap_df[
                        (overlap_df["embedding_key"] == embedding_key)
                        & (overlap_df["fold"] == fold_idx)
                        & (overlap_df["ratio"] == ratio)
                    ].iloc[0]
                    records.append({
                        "version": version,
                        "embedding_key": embedding_key,
                        "embedding_label": selected.embedding_label,
                        "fold": int(fold_idx),
                        "ratio": int(ratio),
                        "selection_source": "coverage_selector_embedding_specific",
                        "selection_scope": "embedding_ratio_materialized_by_fold",
                        "selected_item_indices": [int(i) for i in selected.selected_indices],
                        "selected_item_ids": selected.selected_question_ids,
                        "selected_S_legacy": selected.selected_csv,
                        "selected_S_hash": selected.selected_hash,
                        "coverage": selected.coverage,
                        "redundancy": selected.redundancy,
                        "s_old_item_indices": [int(i) for i in old.selected_indices],
                        "s_old_item_ids": old.selected_question_ids,
                        "s_old_hash": old.selected_hash,
                        "jaccard_vs_s_old": float(ov["jaccard_overlap"]),
                        "intersection_n": int(ov["intersection_n"]),
                        "union_n": int(ov["union_n"]),
                    })
    write_json(paths["selected_items"], {
        "feature_id": "F014",
        "selector": "CoverageSelector",
        "selection_scope": "embedding_ratio_materialized_by_fold",
        "reused_across_folds": True,
        "reused_across_versions": True,
        "reused_across_embeddings": False,
        "baseline_selection": "F013/Phase1 S_old by fold×ratio",
        "item_id_order": "same order as selected_item_indices and prediction vectors",
        "invariant": (
            "S_new is computed from each embedding and ratio with unsupervised CoverageSelector; "
            "it is materialized per fold for joins against fold-scoped S_old."
        ),
        "records": records,
    })

    fold_payload = {
        "feature_id": "F014",
        "fold_source": "participant_cv_split(n_subjects, n_folds=5, seed=0)",
        "folds": [],
    }
    for fold_idx in active_folds:
        train_idx, test_idx = folds[fold_idx]
        fold_payload["folds"].append({
            "fold": int(fold_idx),
            "train_subject_indices": [int(i) for i in train_idx],
            "test_subject_indices": [int(i) for i in test_idx],
            "train_subject_ids": [subject_ids[int(i)] for i in train_idx],
            "test_subject_ids": [subject_ids[int(i)] for i in test_idx],
        })
    write_json(paths["folds"], fold_payload)


def main() -> int:
    args = parse_args()
    mode, active_folds, ratios, embeddings, versions, n_bootstrap, output_dir = mode_config(args)
    paths = version_b_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"F014 Phase 4 Version B — {mode}")
    print("=" * 78)
    print(f"Folds: {active_folds}; ratios: {ratios}; embeddings: {embeddings}; versions: {versions}")
    print("Primary scoring: continuous clip-only predictions on held-out/unselected items only")
    print(f"Version A comparison artifacts: {args.version_a_dir}")

    t0 = time.perf_counter()
    core = load_core_data()
    s_old_sets = load_fixed_s_old_by_fold_ratio(core.question_ids)
    registry = build_embedding_registry(core)
    specs = {spec.key: spec for spec in registry}
    matrices = {key: load_embedding_matrix(specs[key]) for key in embeddings}
    sims = {key: precompute_similarity(E) for key, E in matrices.items()}
    folds = participant_cv_split(core.Y.shape[0], n_folds=N_FOLDS, seed=RANDOM_STATE)
    selected_sets = build_selected_sets(matrices, specs, core.question_ids, ratios)
    overlap_df = pd.DataFrame(build_overlap_rows(
        selected_sets=selected_sets,
        s_old_sets=s_old_sets,
        active_folds=active_folds,
        ratios=ratios,
        embeddings=embeddings,
    ))

    sbert_mismatches = overlap_df[(overlap_df["embedding_key"] == "sbert_original") & (overlap_df["jaccard_overlap"] < 1.0)]
    if not sbert_mismatches.empty:
        print(
            "[WARN] Recomputed SBERT Coverage S_new differs from historical S_old for "
            f"{len(sbert_mismatches)} fold×ratio rows. This is documented as possible due to "
            "tie-breaking/numerical drift; see versionB_selection_overlap.csv."
        )

    participant_records: list[dict[str, Any]] = []
    participant_metric_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []

    for fold_idx in active_folds:
        train_idx, test_idx = folds[fold_idx]
        y_train = core.Y[train_idx]
        y_test = core.Y[test_idx]
        print(f"\n--- Fold {fold_idx} train={len(train_idx)} test={len(test_idx)} ---")
        for ratio in ratios:
            for embedding_key in embeddings:
                spec = specs[embedding_key]
                sim = sims[embedding_key]
                selected = selected_sets[(embedding_key, ratio)]
                old = s_old_sets[(fold_idx, ratio)]
                jac, inter, union = jaccard(selected.selected_indices, old.selected_indices)
                S = selected.selected_indices
                print(
                    f"  m={ratio:2d} {embedding_key:16s}: S_new={selected.selected_hash} "
                    f"Jaccard(S_new,S_old)={jac:.3f}"
                )
                for version in versions:
                    if version == VERSION_B1:
                        K, tau = A1_FIXED_PARAMS[ratio]
                        best_inner = np.nan
                        inner_scores: dict[str, float] = {}
                        train_inner_n = 0
                        valid_inner_n = 0
                        hyper_source = "phase2_fixed_preregistered_softmax"
                        is_tuned = False
                    else:
                        K, tau, best_inner, inner_scores, train_inner_n, valid_inner_n = tune_a2(
                            y_train, sim, S, core.trait_ids, core.reverse_ids, fold_idx
                        )
                        hyper_source = "train_inner_tuned_nested"
                        is_tuned = True

                    pred = ContinuousSoftmaxKNN(K=K, tau=tau)
                    y_pred = pred.predict(y_test, sim, S)
                    records, metric_df = participant_metric_rows(
                        y_true=y_test,
                        y_pred=y_pred,
                        selected_items=S,
                        subject_indices=test_idx,
                        subject_ids=core.subject_ids,
                        question_ids=core.question_ids,
                        trait_ids=core.trait_ids,
                        version=version,
                        embedding_key=embedding_key,
                        embedding_label=spec.label,
                        fold_idx=fold_idx,
                        ratio=ratio,
                        K=K,
                        tau=tau,
                    )
                    metric_df = enrich_participant_outputs(
                        records,
                        metric_df,
                        selected=selected,
                        s_old_hash=old.selected_hash,
                        jaccard_vs_s_old=jac,
                    )
                    participant_records.extend(records)
                    participant_metric_parts.append(metric_df)
                    fold_rows.append(make_fold_result_row_b(
                        version=version,
                        embedding_key=embedding_key,
                        embedding_label=spec.label,
                        fold_idx=fold_idx,
                        ratio=ratio,
                        K=K,
                        tau=tau,
                        hyperparam_source=hyper_source,
                        selected=selected,
                        s_old_hash=old.selected_hash,
                        s_old_coverage=old.coverage,
                        s_old_redundancy=old.redundancy,
                        jaccard_vs_s_old=jac,
                        intersection_n=inter,
                        union_n=union,
                        y_test=y_test,
                        y_pred=y_pred,
                        trait_ids=core.trait_ids,
                        reverse_ids=core.reverse_ids,
                    ))
                    hyper_rows.append({
                        "version": version,
                        "embedding_key": embedding_key,
                        "embedding_label": spec.label,
                        "ratio": int(ratio),
                        "fold": int(fold_idx),
                        "predictor": PREDICTOR_NAME,
                        "K": int(K),
                        "tau": float(tau),
                        "is_fixed": not is_tuned,
                        "is_tuned": is_tuned,
                        "hyperparam_source": hyper_source,
                        "train_n": int(len(train_idx)),
                        "train_inner_n": int(train_inner_n),
                        "valid_inner_n": int(valid_inner_n),
                        "test_n": int(len(test_idx)),
                        "best_inner_item_r": best_inner,
                        "search_space_K": ",".join(str(k) for k in K_CANDIDATES) if is_tuned else "",
                        "search_space_tau": ",".join(str(t) for t in TAU_CANDIDATES) if is_tuned else "",
                        "inner_val_scores_json": json.dumps(inner_scores, sort_keys=True),
                        "selected_S_hash": selected.selected_hash,
                        "jaccard_vs_s_old": jac,
                    })
                    print(f"    {version:8s} K={K:2d} tau={tau:.3f} subjects={len(test_idx)}")

    participant_df = pd.concat(participant_metric_parts, ignore_index=True)
    validate_outputs_b(
        mode=mode,
        participant_records=participant_records,
        participant_df=participant_df,
        fold_rows=fold_rows,
        hyper_rows=hyper_rows,
        overlap_df=overlap_df,
    )

    results_df = pd.DataFrame(fold_rows).sort_values(["version", "ratio", "embedding_key", "fold"])
    hyper_df = pd.DataFrame(hyper_rows).sort_values(["version", "ratio", "embedding_key", "fold"])
    summary_df = aggregate_summary(participant_df, results_df)
    overlap_summary = overlap_df.groupby(["embedding_key", "ratio"]).agg(
        jaccard_mean_vs_s_old=("jaccard_overlap", "mean"),
        jaccard_min_vs_s_old=("jaccard_overlap", "min"),
        jaccard_max_vs_s_old=("jaccard_overlap", "max"),
        selection_changed_any_fold=("selection_changed", "any"),
    ).reset_index()
    summary_df = summary_df.merge(overlap_summary, on=["embedding_key", "ratio"], how="left")
    stats_df = build_bootstrap_tests(participant_df, n_bootstrap=n_bootstrap)
    contribution_df = build_selection_contribution(participant_df, args.version_a_dir, n_bootstrap, overlap_df)

    write_predictions_parquet(participant_records, paths["predictions"])
    participant_df.to_csv(paths["participant_metrics"], index=False)
    results_df.to_csv(paths["results"], index=False)
    summary_df.to_csv(paths["aggregate_metrics"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    hyper_df.to_csv(paths["hyperparameters"], index=False)
    stats_df.to_csv(paths["stats"], index=False)
    contribution_df.to_csv(paths["selection_contribution"], index=False)
    overlap_df.to_csv(paths["selection_overlap"], index=False)
    write_selected_items_audit_b(
        paths,
        selected_sets=selected_sets,
        s_old_sets=s_old_sets,
        overlap_df=overlap_df,
        folds=folds,
        subject_ids=core.subject_ids,
        active_folds=active_folds,
        ratios=ratios,
        embeddings=embeddings,
        versions=versions,
    )
    build_table4_and_figure(
        output_dir=output_dir,
        a_dir=args.version_a_dir,
        b_summary=summary_df,
        b_stats=stats_df,
        contribution=contribution_df,
        overlap_df=overlap_df,
        paths=paths,
    )

    # Compatibility aliases requested by the F014 handoff/test plan.  In the
    # canonical Phase 4 directory, generic selected-items/hyperparameter/fold
    # names may already be F013 audit files, so never overwrite existing generic
    # artifacts.  Fresh scratch output directories still receive the aliases.
    alias_paths = {
        "selection_contribution_alias": output_dir / "selection_contribution.csv",
        "selection_overlap_alias": output_dir / "selection_overlap.csv",
        "selected_items_alias": output_dir / "selected_items_by_fold_ratio_embedding.json",
        "hyperparameters_alias": output_dir / "hyperparameters_by_fold_ratio_embedding.csv",
        "folds_alias": output_dir / "outer_folds_subject_ids.json",
    }
    alias_sources = {
        "selection_contribution_alias": paths["selection_contribution"],
        "selection_overlap_alias": paths["selection_overlap"],
        "selected_items_alias": paths["selected_items"],
        "hyperparameters_alias": paths["hyperparameters"],
        "folds_alias": paths["folds"],
    }
    written_aliases: dict[str, Path] = {}
    skipped_aliases: dict[str, Path] = {}
    for alias_name, alias_path in alias_paths.items():
        if alias_path.exists():
            skipped_aliases[alias_name] = alias_path
            continue
        shutil.copyfile(alias_sources[alias_name], alias_path)
        written_aliases[alias_name] = alias_path

    elapsed = time.perf_counter() - t0
    print("\n[SAVE] Phase 4 Version B outputs")
    for name, path in paths.items():
        if path.exists():
            print(f"  {name:24s} {path}")
    for alias_name, alias_path in written_aliases.items():
        print(f"  {alias_name:24s} {alias_path}")
    for alias_name, alias_path in skipped_aliases.items():
        print(f"  {alias_name + '_skipped':24s} {alias_path} (already exists)")
    print(f"\n[OK] F014 Version B {mode} completed in {elapsed:.1f}s")
    print(f"Participant rows: {len(participant_records)}; fold rows: {len(results_df)}; hyper rows: {len(hyper_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

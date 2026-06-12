#!/usr/bin/env python3
"""F013: Phase 4 Version A1/A2 fixed-S_old embedding comparison.

Version A holds the original SBERT Coverage selected items fixed and swaps only
prediction embedding geometry.  Main scoring follows the Phase 2 convention:
predictions are made and evaluated only for unselected/held-out items; selected
items are observed inputs and remain NaN in the prediction matrix.

Outputs are written to ``results/phase4/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cv_framework import (  # noqa: E402
    TRAIT_ORDER,
    compute_trait_scores,
    evaluate_predictions,
    inner_validation_split,
    participant_cv_split,
)
from phase4_common import (  # noqa: E402
    A1_FIXED_PARAMS,
    K_CANDIDATES,
    MODEL_ORDER,
    N_FOLDS,
    PHASE1_DETAIL,
    PHASE4_DIR,
    RANDOM_STATE,
    RATIOS,
    TAU_CANDIDATES,
    build_embedding_registry,
    load_core_data,
    load_embedding_matrix,
    load_fixed_s_old_by_fold_ratio,
    mean_ci,
    output_paths,
    paired_bootstrap_by_fold,
    pearson_or_nan,
    precompute_similarity,
    write_json,
    write_predictions_parquet,
    adjust_pvalues_bh,
    adjust_pvalues_holm,
)
from phase4_predictors import ContinuousSoftmaxKNN, round_clip_predictions  # noqa: E402


VERSION_A1 = "A1_fixed"
VERSION_A2 = "A2_tuned"
PREDICTOR_NAME = "SoftmaxKNN"
SCORING_CONVENTION = "heldout_unselected_items_only_phase2_convention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F013 Phase 4 Version A fixed-S_old experiment.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Run all 5 folds × 4 ratios × 5 embeddings × A1/A2.")
    mode.add_argument("--quick", action="store_true", help="Run 1 fold × ratios 10/30 × all embeddings × A1/A2.")
    mode.add_argument("--smoke", action="store_true", help="Run 1 fold × ratio 10 × sbert_original × A1 only.")
    parser.add_argument("--n-bootstrap", type=int, default=None, help="Bootstrap iterations for statistical tests.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to results/phase4 for --all and to "
            "mode-specific scratch directories for --quick/--smoke so validation "
            "runs cannot overwrite canonical full-run artifacts."
        ),
    )
    return parser.parse_args()


def mode_config(args: argparse.Namespace) -> tuple[str, list[int], tuple[int, ...], tuple[str, ...], tuple[str, ...], int, Path]:
    if args.smoke:
        output_dir = args.output_dir or (PHASE4_DIR.parent / "phase4_smoke")
        return "SMOKE", [0], (10,), ("sbert_original",), (VERSION_A1,), args.n_bootstrap or 200, output_dir
    if args.quick:
        output_dir = args.output_dir or (PHASE4_DIR.parent / "phase4_quick")
        return "QUICK", [0], (10, 30), MODEL_ORDER, (VERSION_A1, VERSION_A2), args.n_bootstrap or 500, output_dir
    output_dir = args.output_dir or PHASE4_DIR
    return "FULL", list(range(N_FOLDS)), RATIOS, MODEL_ORDER, (VERSION_A1, VERSION_A2), args.n_bootstrap or 10_000, output_dir


def participant_metric_rows(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected_items: np.ndarray,
    subject_indices: np.ndarray,
    subject_ids: list[str],
    question_ids: list[str],
    trait_ids: np.ndarray,
    version: str,
    embedding_key: str,
    embedding_label: str,
    fold_idx: int,
    ratio: int,
    K: int,
    tau: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    heldout_items = np.where(np.isnan(np.full(y_true.shape[1], np.nan)))[0]  # placeholder overwritten below
    mask = np.ones(y_true.shape[1], dtype=bool)
    mask[selected_items] = False
    heldout_items = np.where(mask)[0]
    y_pred_rounded = round_clip_predictions(y_pred)

    y_imputed = y_true.copy()
    pred_mask = ~np.isnan(y_pred)
    y_imputed[pred_mask] = y_pred[pred_mask]
    trait_true = compute_trait_scores(y_true, trait_ids, TRAIT_ORDER)
    trait_imputed = compute_trait_scores(y_imputed, trait_ids, TRAIT_ORDER)

    records: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    selected_ids = [question_ids[int(i)] for i in selected_items]
    heldout_ids = [question_ids[int(i)] for i in heldout_items]

    for local_idx, subject_index in enumerate(subject_indices):
        tv = y_true[local_idx, heldout_items]
        pv = y_pred[local_idx, heldout_items]
        rv = y_pred_rounded[local_idx, heldout_items]
        if np.isnan(pv).any():
            raise RuntimeError(
                f"NaN prediction found for version={version}, embedding={embedding_key}, "
                f"fold={fold_idx}, ratio={ratio}, subject_index={int(subject_index)}"
            )
        item_r = pearson_or_nan(tv, pv)
        item_mae = float(np.mean(np.abs(tv - pv)))
        item_rmse = float(np.sqrt(np.mean((tv - pv) ** 2)))
        rounded_accuracy = float(np.mean(np.round(tv) == np.round(pv)))
        rounded_mae = float(np.mean(np.abs(tv - rv)))
        profile_r = pearson_or_nan(trait_true[local_idx], trait_imputed[local_idx])

        common = {
            "subject_id": subject_ids[int(subject_index)],
            "subject_index": int(subject_index),
            "outer_fold": int(fold_idx),
            "embedding": embedding_key,
            "embedding_key": embedding_key,
            "embedding_label": embedding_label,
            "version": version,
            "ratio": int(ratio),
            "m": int(ratio),
            "predictor": PREDICTOR_NAME,
            "K": int(K),
            "tau": float(tau),
            "scoring_convention": SCORING_CONVENTION,
            "prediction_mode": "continuous_clip_only_primary",
            "per_subject_item_r": item_r,
            "per_subject_mae": item_mae,
            "per_subject_rmse": item_rmse,
            "per_subject_rounded_accuracy": rounded_accuracy,
            "per_subject_rounded_mae": rounded_mae,
            "per_subject_profile_r": profile_r,
        }
        record = {
            **common,
            "selected_item_indices": [int(i) for i in selected_items],
            "selected_item_ids": selected_ids,
            "heldout_item_indices": [int(i) for i in heldout_items],
            "heldout_item_ids": heldout_ids,
            "predicted_item_ids": heldout_ids,
            "y_true": [float(x) for x in tv],
            "y_pred_continuous": [float(x) for x in pv],
            "y_pred_rounded_supplemental": [float(x) for x in rv],
            "trait_scores_true": [float(x) if not np.isnan(x) else None for x in trait_true[local_idx]],
            "trait_scores_imputed_continuous": [float(x) if not np.isnan(x) else None for x in trait_imputed[local_idx]],
        }
        records.append(record)
        scalar_rows.append(common)

    return records, pd.DataFrame(scalar_rows)


def rounded_mae_from_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_pred)
    rounded = round_clip_predictions(y_pred)
    return float(np.mean(np.abs(y_true[mask] - rounded[mask])))


def make_fold_result_row(
    *,
    version: str,
    embedding_key: str,
    embedding_label: str,
    fold_idx: int,
    ratio: int,
    K: int,
    tau: float,
    hyperparam_source: str,
    selected_csv: str,
    selected_hash: str,
    selected_coverage: float,
    selected_redundancy: float,
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
        "selected_S": selected_csv,
        "selected_S_hash": selected_hash,
        "selected_S_source": str(PHASE1_DETAIL.relative_to(PROJECT_ROOT)),
        "s_old_coverage": selected_coverage,
        "s_old_redundancy": selected_redundancy,
        "n_test_subjects": int(y_test.shape[0]),
    }


def tune_a2(
    y_train: np.ndarray,
    sim: np.ndarray,
    S: np.ndarray,
    trait_ids: np.ndarray,
    reverse_ids: np.ndarray,
    fold_idx: int,
) -> tuple[int, float, float, dict[str, float], int, int]:
    train_inner_idx, valid_inner_idx = inner_validation_split(
        np.arange(len(y_train)), val_ratio=0.2, seed=RANDOM_STATE + fold_idx
    )
    y_valid_inner = y_train[valid_inner_idx]
    best_score = -np.inf
    best_k = K_CANDIDATES[0]
    best_tau = TAU_CANDIDATES[0]
    scores: dict[str, float] = {}
    for K in K_CANDIDATES:
        for tau in TAU_CANDIDATES:
            pred = ContinuousSoftmaxKNN(K=K, tau=tau)
            y_pred_inner = pred.predict(y_valid_inner, sim, S)
            metrics = evaluate_predictions(y_valid_inner, y_pred_inner, trait_ids, reverse_ids)
            score = float(metrics["item_level"]["pearson_r"][0])
            key = f"K={K},tau={tau}"
            scores[key] = score
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_k = K
                best_tau = tau
    if not np.isfinite(best_score):
        raise RuntimeError("A2 tuning failed: all inner validation scores were NaN")
    return int(best_k), float(best_tau), float(best_score), scores, int(len(train_inner_idx)), int(len(valid_inner_idx))


def aggregate_summary(participants: pd.DataFrame, fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["version", "embedding_key", "embedding_label", "ratio"]
    fold_groups = {
        keys: grp for keys, grp in fold_results.groupby(group_cols)
    } if not fold_results.empty else {}
    for keys, grp in participants.groupby(group_cols):
        version, embedding_key, embedding_label, ratio = keys
        row: dict[str, Any] = {
            "version": version,
            "embedding_key": embedding_key,
            "embedding_label": embedding_label,
            "ratio": int(ratio),
            "n_participants": int(len(grp)),
            "n_folds": int(grp["outer_fold"].nunique()),
            "K_mode": int(grp["K"].mode().iloc[0]),
            "tau_mean": float(grp["tau"].mean()),
            "scoring_convention": SCORING_CONVENTION,
            "rounded_primary": False,
        }
        participant_metric_map = {
            "item_r": "per_subject_item_r",
            "item_mae": "per_subject_mae",
            "item_rmse": "per_subject_rmse",
            "rounded_accuracy": "per_subject_rounded_accuracy",
            "rounded_mae": "per_subject_rounded_mae",
            "profile_r": "per_subject_profile_r",
        }
        for out_name, col in participant_metric_map.items():
            mean, lo, hi = mean_ci(grp[col].to_numpy())
            row[f"{out_name}_mean"] = mean
            row[f"{out_name}_ci_lower"] = lo
            row[f"{out_name}_ci_upper"] = hi

        # Trait-level r is a cross-participant Pearson metric, so aggregate the
        # fold-level values rather than treating it as a per-subject scalar.
        fold_grp = fold_groups.get(keys)
        if fold_grp is not None:
            mean, lo, hi = mean_ci(fold_grp["trait_r_mean"].to_numpy())
            row["trait_r_mean"] = mean
            row["trait_r_mean_ci_lower"] = lo
            row["trait_r_mean_ci_upper"] = hi
        else:
            row["trait_r_mean"] = np.nan
            row["trait_r_mean_ci_lower"] = np.nan
            row["trait_r_mean_ci_upper"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["version", "ratio", "embedding_key"]).reset_index(drop=True)


def build_bootstrap_tests(participants: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for version in sorted(participants["version"].unique()):
        for ratio in sorted(participants["ratio"].unique()):
            base = participants[
                (participants["version"] == version)
                & (participants["ratio"] == ratio)
                & (participants["embedding_key"] == "sbert_original")
            ][["subject_id", "outer_fold", "per_subject_item_r"]].rename(
                columns={"per_subject_item_r": "base_item_r"}
            )
            if base.empty:
                continue
            for embedding_key in MODEL_ORDER[1:]:
                cur = participants[
                    (participants["version"] == version)
                    & (participants["ratio"] == ratio)
                    & (participants["embedding_key"] == embedding_key)
                ][["subject_id", "outer_fold", "embedding_label", "per_subject_item_r"]].rename(
                    columns={"per_subject_item_r": "new_item_r"}
                )
                if cur.empty:
                    continue
                merged = cur.merge(base, on=["subject_id", "outer_fold"], how="inner")
                if len(merged) != len(base):
                    raise RuntimeError(
                        f"Paired bootstrap alignment failed for {version} ratio={ratio} {embedding_key}: "
                        f"merged={len(merged)} baseline={len(base)}"
                    )
                merged["diff"] = merged["new_item_r"] - merged["base_item_r"]
                result = paired_bootstrap_by_fold(
                    merged,
                    n_bootstrap=n_bootstrap,
                    seed=RANDOM_STATE + int(ratio) + 1000 * (1 if version == VERSION_A2 else 0),
                )
                rows.append({
                    "version": version,
                    "ratio": int(ratio),
                    "embedding_key": embedding_key,
                    "embedding_label": cur["embedding_label"].iloc[0],
                    "baseline_embedding_key": "sbert_original",
                    "metric": "per_subject_item_r",
                    "delta_mean": result["delta"],
                    "ci_lower": result["ci_low"],
                    "ci_upper": result["ci_high"],
                    "p_raw": result["p"],
                    "n_pairs": result["n"],
                    "bootstrap_unit": "subjects_resampled_within_outer_fold",
                    "bootstrap_iterations": int(n_bootstrap),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    df["p_bh"] = np.nan
    for version, idx in df.groupby("version").groups.items():
        pvals = df.loc[idx, "p_raw"].tolist()
        df.loc[idx, "p_holm"] = adjust_pvalues_holm(pvals)
        df.loc[idx, "p_bh"] = adjust_pvalues_bh(pvals)
    return df.sort_values(["version", "ratio", "embedding_key"]).reset_index(drop=True)


def validate_outputs(
    *,
    mode: str,
    participant_records: list[dict[str, Any]],
    participant_df: pd.DataFrame,
    fold_rows: list[dict[str, Any]],
    hyper_rows: list[dict[str, Any]],
) -> None:
    if not participant_records:
        raise RuntimeError("No prediction records produced")
    for row in participant_records[: min(25, len(participant_records))]:
        n = len(row["heldout_item_ids"])
        if n == 0:
            raise RuntimeError("Prediction record has no heldout items")
        if len(row["y_true"]) != n or len(row["y_pred_continuous"]) != n:
            raise RuntimeError("Prediction vector lengths do not match heldout_item_ids")
        if any(x is None or np.isnan(float(x)) for x in row["y_pred_continuous"]):
            raise RuntimeError("Prediction record contains NaN y_pred_continuous")
    if participant_df["per_subject_mae"].isna().any():
        raise RuntimeError("Participant MAE contains NaN")
    if not fold_rows or not hyper_rows:
        raise RuntimeError("Missing fold-level or hyperparameter rows")
    if VERSION_A1 in participant_df["version"].unique():
        a1_h = [r for r in hyper_rows if r["version"] == VERSION_A1]
        if any(r["is_tuned"] for r in a1_h):
            raise RuntimeError("A1 hyperparameter rows indicate tuning")
    if mode == "SMOKE":
        expected_versions = {VERSION_A1}
        if set(participant_df["version"].unique()) != expected_versions:
            raise RuntimeError("Smoke mode should run A1 only")


def write_selected_items_audit(
    paths: dict[str, Path],
    selected_sets: dict[tuple[int, int], Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
    subject_ids: list[str],
    specs: dict[str, Any],
    active_folds: list[int],
    ratios: tuple[int, ...],
    embeddings: tuple[str, ...],
    versions: tuple[str, ...],
) -> None:
    selected_records = []
    for version in versions:
        for embedding_key in embeddings:
            for fold_idx in active_folds:
                for ratio in ratios:
                    fixed = selected_sets[(fold_idx, ratio)]
                    selected_records.append({
                        "version": version,
                        "embedding_key": embedding_key,
                        "embedding_label": specs[embedding_key].label,
                        "fold": int(fold_idx),
                        "ratio": int(ratio),
                        "selection_source": "phase1_coverage_history_by_fold_ratio",
                        "source_artifact": str(PHASE1_DETAIL.relative_to(PROJECT_ROOT)),
                        "selected_item_indices": [int(i) for i in fixed.selected_indices],
                        "selected_item_ids": fixed.selected_question_ids,
                        "selected_S_legacy": fixed.selected_csv,
                        "selected_S_hash": fixed.selected_hash,
                        "coverage": fixed.coverage,
                        "redundancy": fixed.redundancy,
                    })
    write_json(paths["selected_items"], {
        "feature_id": "F013",
        "source_artifact": str(PHASE1_DETAIL.relative_to(PROJECT_ROOT)),
        "source_strategy": "Coverage",
        "selection_scope": "fold_ratio",
        "reused_across_embeddings": True,
        "reused_across_versions": True,
        "item_id_order": "same order as selected_item_indices and prediction vectors",
        "invariant": "S_old is loaded by fold×ratio and reused unchanged across embeddings within each fold×ratio.",
        "records": selected_records,
    })

    fold_payload = {
        "feature_id": "F013",
        "fold_source": "participant_cv_split(n_subjects, n_folds=5, seed=0)",
        "phase1_participant_fold_artifact": None,
        "phase1_fold_verification": (
            "Existing Phase 1 artifacts contain fold numbers but do not store participant subject IDs; "
            "this file records the regenerated seed=0 subject IDs for audit and reuse."
        ),
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
    paths = output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"F013 Phase 4 Version A — {mode}")
    print("=" * 78)
    print(f"Folds: {active_folds}; ratios: {ratios}; embeddings: {embeddings}; versions: {versions}")
    print(f"Primary scoring: continuous clip-only predictions on held-out/unselected items only")

    t0 = time.perf_counter()
    core = load_core_data()
    selected_sets = load_fixed_s_old_by_fold_ratio(core.question_ids)
    registry = build_embedding_registry(core)
    specs = {spec.key: spec for spec in registry}
    matrices = {key: load_embedding_matrix(specs[key]) for key in embeddings}
    sims = {key: precompute_similarity(E) for key, E in matrices.items()}
    folds = participant_cv_split(core.Y.shape[0], n_folds=N_FOLDS, seed=RANDOM_STATE)

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
            fixed = selected_sets[(fold_idx, ratio)]
            S = fixed.selected_indices
            print(f"  m={ratio}: S_old hash={fixed.selected_hash} ({len(S)} items)")
            for embedding_key in embeddings:
                spec = specs[embedding_key]
                sim = sims[embedding_key]
                for version in versions:
                    if version == VERSION_A1:
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
                    participant_records.extend(records)
                    participant_metric_parts.append(metric_df)
                    fold_rows.append(make_fold_result_row(
                        version=version,
                        embedding_key=embedding_key,
                        embedding_label=spec.label,
                        fold_idx=fold_idx,
                        ratio=ratio,
                        K=K,
                        tau=tau,
                        hyperparam_source=hyper_source,
                        selected_csv=fixed.selected_csv,
                        selected_hash=fixed.selected_hash,
                        selected_coverage=fixed.coverage,
                        selected_redundancy=fixed.redundancy,
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
                    })
                    print(
                        f"    {version:8s} {embedding_key:16s} K={K:2d} tau={tau:.3f} "
                        f"subjects={len(test_idx)}"
                    )

    participant_df = pd.concat(participant_metric_parts, ignore_index=True)
    validate_outputs(
        mode=mode,
        participant_records=participant_records,
        participant_df=participant_df,
        fold_rows=fold_rows,
        hyper_rows=hyper_rows,
    )

    results_df = pd.DataFrame(fold_rows).sort_values(["version", "ratio", "embedding_key", "fold"])
    hyper_df = pd.DataFrame(hyper_rows).sort_values(["version", "ratio", "embedding_key", "fold"])
    summary_df = aggregate_summary(participant_df, results_df)
    stats_df = build_bootstrap_tests(participant_df, n_bootstrap=n_bootstrap)

    write_predictions_parquet(participant_records, paths["predictions"])
    participant_df.to_csv(paths["participant_metrics"], index=False)
    results_df.to_csv(paths["results"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    hyper_df.to_csv(paths["hyperparameters"], index=False)
    stats_df.to_csv(paths["stats"], index=False)
    write_selected_items_audit(paths, selected_sets, folds, core.subject_ids, specs, active_folds, ratios, embeddings, versions)

    elapsed = time.perf_counter() - t0
    print("\n[SAVE] Phase 4 Version A outputs")
    for name, path in paths.items():
        if path.exists():
            print(f"  {name:20s} {path}")
    print(f"\n[OK] F013 Version A {mode} completed in {elapsed:.1f}s")
    print(f"Participant rows: {len(participant_records)}; fold rows: {len(results_df)}; hyper rows: {len(hyper_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Lightweight smoke checks for F014 Phase 4 Version B outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_versionB_smoke"
VERSION_A_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_versionB_smoke_A"


def fresh_output_dir(base: Path) -> Path:
    """Return a non-existing smoke output directory without deleting old artifacts."""
    if not base.exists():
        return base
    for i in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find fresh smoke output directory near {base}")


def run_smoke() -> Path:
    version_a_dir = fresh_output_dir(VERSION_A_OUTPUT_DIR)
    subprocess.run(
        [sys.executable, "scripts/run_phase4_versionA.py", "--quick", "--output-dir", str(version_a_dir)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    output_dir = fresh_output_dir(OUTPUT_DIR)
    cmd = [
        sys.executable,
        "scripts/run_phase4_versionB.py",
        "--smoke",
        "--output-dir",
        str(output_dir),
        "--version-a-dir",
        str(version_a_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return output_dir


def check_outputs(output_dir: Path) -> None:
    predictions_path = output_dir / "versionB_predictions.parquet"
    metrics_path = output_dir / "versionB_participant_metrics.csv"
    results_path = output_dir / "versionB_results.csv"
    aggregate_path = output_dir / "versionB_aggregate_metrics.csv"
    hyper_path = output_dir / "versionB_hyperparameters_by_fold_ratio_embedding.csv"
    selected_path = output_dir / "versionB_selected_items_by_fold_ratio_embedding.json"
    overlap_path = output_dir / "versionB_selection_overlap.csv"
    contribution_path = output_dir / "versionB_selection_contribution.csv"
    table4_path = output_dir / "figures" / "table4.csv"
    figure4_path = output_dir / "figures" / "figure4.pdf"

    required = [
        predictions_path,
        metrics_path,
        results_path,
        aggregate_path,
        hyper_path,
        selected_path,
        overlap_path,
        contribution_path,
        table4_path,
        figure4_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"Missing smoke outputs: {missing}")

    pred = pd.read_parquet(predictions_path)
    assert len(pred) > 0, "predictions parquet is empty"
    required_pred_cols = {
        "subject_id",
        "outer_fold",
        "embedding_key",
        "version",
        "ratio",
        "predictor",
        "K",
        "tau",
        "selected_item_ids",
        "heldout_item_ids",
        "predicted_item_ids",
        "y_true",
        "y_pred_continuous",
        "per_subject_item_r",
        "per_subject_mae",
        "selected_S_hash",
        "jaccard_vs_s_old",
    }
    assert required_pred_cols <= set(pred.columns), sorted(required_pred_cols - set(pred.columns))
    assert set(pred["version"]) == {"B1_fixed", "B2_tuned"}, "smoke must run B1/B2"
    assert set(pred["embedding_key"]) == {"sbert_original"}, "smoke must run one embedding"
    assert set(pred["ratio"]) == {10}, "smoke must run one ratio"
    assert set(pred["outer_fold"]) == {0}, "smoke must run fold 0"

    for row in pred.head(20).itertuples(index=False):
        heldout = list(row.heldout_item_ids)
        y_true = list(row.y_true)
        y_pred = list(row.y_pred_continuous)
        assert heldout == list(row.predicted_item_ids), "predicted_item_ids must match heldout_item_ids"
        assert len(heldout) == len(y_true) == len(y_pred), "vector length mismatch"
        assert len(heldout) == 90, "m=10 should predict 90 held-out items"
        assert np.isfinite(y_pred).all(), "continuous predictions contain non-finite values"
        assert min(y_pred) >= 1.0 and max(y_pred) <= 5.0, "predictions outside [1,5]"

    metrics = pd.read_csv(metrics_path)
    assert not metrics["per_subject_mae"].isna().any(), "participant MAE contains NaN"

    results = pd.read_csv(results_path)
    assert results["rounded"].eq(False).all(), "primary result rows must be non-rounded"
    assert results["scoring_convention"].eq("heldout_unselected_items_only_phase2_convention").all()
    assert results["selected_S"].str.split(",").map(len).eq(10).all(), "S_new length mismatch"

    hyper = pd.read_csv(hyper_path)
    b1 = hyper[hyper["version"] == "B1_fixed"]
    b2 = hyper[hyper["version"] == "B2_tuned"]
    assert not b1.empty and not b2.empty, "missing B1/B2 hyperparameter rows"
    assert b1["is_tuned"].eq(False).all(), "B1 must not tune"
    assert b2["is_tuned"].eq(True).all(), "B2 must tune"
    assert b1["K"].eq(7).all(), "B1 m=10 fixed K should be 7"
    assert np.allclose(b1["tau"], 0.1), "B1 m=10 fixed tau should be 0.1"

    overlap = pd.read_csv(overlap_path)
    assert not overlap.empty, "selection overlap is empty"
    assert overlap["jaccard_overlap"].between(0.0, 1.0).all(), "Jaccard outside [0,1]"
    assert overlap["s_new_size"].eq(10).all(), "S_new size mismatch"

    contrib = pd.read_csv(contribution_path)
    assert {"B1_minus_A1", "B2_minus_A2"} <= set(contrib["comparison"]), "missing B−A contribution rows"
    assert contrib["n_pairs"].gt(0).all(), "B−A join produced no pairs"

    selected = json.loads(selected_path.read_text())
    assert selected["feature_id"] == "F014"
    assert selected["selection_scope"] == "embedding_ratio_materialized_by_fold"
    assert selected["reused_across_versions"] is True
    assert selected["records"], "selected-items audit records missing"

    table4 = pd.read_csv(table4_path)
    assert {"A1_fixed", "A2_tuned", "B1_fixed", "B2_tuned"} <= set(table4["version"]), "table4 missing A/B versions"
    assert figure4_path.stat().st_size > 0, "Figure 4 PDF is empty"


def main() -> int:
    output_dir = run_smoke()
    check_outputs(output_dir)
    print(f"F014 smoke checks passed: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

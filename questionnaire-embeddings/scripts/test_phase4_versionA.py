#!/usr/bin/env python3
"""Lightweight smoke checks for F013 Phase 4 Version A outputs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_smoke"


def run_smoke() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    cmd = [sys.executable, "scripts/run_phase4_versionA.py", "--smoke", "--output-dir", str(OUTPUT_DIR)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def check_outputs() -> None:
    predictions_path = OUTPUT_DIR / "versionA_predictions.parquet"
    metrics_path = OUTPUT_DIR / "versionA_participant_metrics.csv"
    results_path = OUTPUT_DIR / "versionA_results.csv"
    hyper_path = OUTPUT_DIR / "hyperparameters_by_fold_ratio_embedding.csv"
    selected_path = OUTPUT_DIR / "selected_items_by_fold_ratio_embedding.json"

    required = [predictions_path, metrics_path, results_path, hyper_path, selected_path]
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
    }
    assert required_pred_cols <= set(pred.columns), sorted(required_pred_cols - set(pred.columns))
    assert set(pred["version"]) == {"A1_fixed"}, "smoke must run A1 only"
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

    hyper = pd.read_csv(hyper_path)
    assert hyper["version"].eq("A1_fixed").all(), "smoke hyperparameters should be A1 only"
    assert hyper["is_tuned"].eq(False).all(), "A1 must not tune"
    assert hyper["K"].eq(7).all(), "A1 m=10 fixed K should be 7"
    assert np.allclose(hyper["tau"], 0.1), "A1 m=10 fixed tau should be 0.1"

    selected = json.loads(selected_path.read_text())
    assert selected["selection_scope"] == "fold_ratio", "selected-items scope missing"
    assert selected["reused_across_embeddings"] is True, "selected-items embedding invariant missing"
    assert selected["reused_across_versions"] is True, "selected-items version invariant missing"
    assert selected["records"], "selected-items audit records missing"


def main() -> int:
    run_smoke()
    check_outputs()
    print("F013 smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

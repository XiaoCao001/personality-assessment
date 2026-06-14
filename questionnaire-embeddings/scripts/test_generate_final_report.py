#!/usr/bin/env python3
"""Smoke checks for F015 final report generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "final_report_smoke"


def fresh_output_dir(base: Path) -> Path:
    """Return a non-existing output directory without deleting old artifacts."""
    if not base.exists():
        return base
    for i in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find fresh smoke output directory near {base}")


def run_report() -> Path:
    output_dir = fresh_output_dir(BASE_OUTPUT_DIR)
    subprocess.run(
        [sys.executable, "scripts/generate_final_report.py", "--output-dir", str(output_dir)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return output_dir


def check_outputs(output_dir: Path) -> None:
    required = [
        output_dir / "final_report.md",
        output_dir / "final_report.txt",
        output_dir / "final_report.pdf",
        output_dir / "final_summary.csv",
        output_dir / "statistical_summary.csv",
        output_dir / "best_pipeline_table.csv",
        output_dir / "report_manifest.json",
        output_dir / "figures" / "table5_phase4_integrated_synthesis.csv",
        output_dir / "figures" / "figure6_phase4_selection_vs_performance.pdf",
        output_dir / "figures" / "figure6_phase4_selection_vs_performance.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"Missing final report outputs: {missing}")

    pdf_path = output_dir / "final_report.pdf"
    assert pdf_path.stat().st_size > 10_000, "final_report.pdf is unexpectedly small"
    assert pdf_path.read_bytes()[:4] == b"%PDF", "final_report.pdf is not a PDF"

    summary = pd.read_csv(output_dir / "final_summary.csv")
    required_summary_cols = {
        "section",
        "phase",
        "ratio",
        "version",
        "embedding_key",
        "embedding_label",
        "selection_strategy",
        "selection_scope",
        "predictor",
        "item_r_mean",
        "trait_r_mean",
        "source_artifact",
    }
    assert required_summary_cols <= set(summary.columns), sorted(required_summary_cols - set(summary.columns))
    assert {"phase1", "phase2", "phase3", "phase4", "final"} <= set(summary["phase"]), "missing phase coverage"
    assert {"A1_fixed", "A2_tuned", "B1_fixed", "B2_tuned"} <= set(summary["version"].dropna()), "missing Phase 4 versions"
    assert "recommended_pipeline" in set(summary["section"]), "missing recommended pipeline rows"
    recommended = summary[summary["section"] == "recommended_pipeline"]
    assert recommended["K"].notna().all(), "recommended rows must populate normalized K"
    assert recommended["tau"].notna().all(), "recommended rows must populate normalized tau"
    assert recommended["source_artifact"].notna().all(), "recommended rows must populate source_artifact"

    best = pd.read_csv(output_dir / "best_pipeline_table.csv")
    assert set(best["ratio"]) == {10, 30, 50, 90}, "best pipeline table must cover all ratios"
    required_best_cols = {
        "ratio",
        "recommended_version",
        "embedding_key",
        "selection_scope",
        "predictor",
        "K_mode",
        "tau_mean",
        "item_r_mean",
        "trait_r_mean",
        "profile_r_mean",
        "item_mae_mean",
    }
    assert required_best_cols <= set(best.columns), sorted(required_best_cols - set(best.columns))

    stats = pd.read_csv(output_dir / "statistical_summary.csv")
    assert {"phase4_vs_sbert", "phase4_selection_contribution"} <= set(stats["comparison_family"]), "missing Phase 4 statistical families"
    assert {"p_raw", "p_holm", "p_bh"} <= set(stats.columns), "missing p-value columns"

    synth = pd.read_csv(output_dir / "figures" / "table5_phase4_integrated_synthesis.csv")
    assert len(synth) == 40, "Table 5 must cover 2 families × 5 embeddings × 4 ratios"
    assert set(synth["family"]) == {"fixed_params", "tuned_params"}, "missing synthesis families"
    assert set(synth["ratio"]) == {10, 30, 50, 90}, "missing synthesis ratios"
    assert {"A1_fixed", "A2_tuned"} == set(synth["a_version"]), "missing A versions in synthesis"
    assert {"B1_fixed", "B2_tuned"} == set(synth["b_version"]), "missing B versions in synthesis"

    report = (output_dir / "final_report.md").read_text(encoding="utf-8")
    required_text = [
        "# Final Report",
        "Phase 1",
        "Phase 2",
        "Phase 3",
        "Phase 4",
        "item-level Pearson r",
        "A1_fixed",
        "A2_tuned",
        "B1_fixed",
        "B2_tuned",
        "Table 5",
        "Figure 6",
        "NEO-PI-R",
        "cross-questionnaire generalization remains to be tested",
    ]
    missing_text = [text for text in required_text if text not in report]
    if missing_text:
        raise AssertionError(f"final_report.md missing required narrative text: {missing_text}")
    text_report = (output_dir / "final_report.txt").read_text(encoding="utf-8")
    assert "Scope and deliverables" in text_report, "text report should strip markdown heading markers"
    assert "#Scope" not in text_report and "##" not in text_report, "text report has malformed heading markers"

    manifest = json.loads((output_dir / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_id"] == "F015", "manifest feature_id mismatch"
    assert manifest["no_experiments_rerun"] is True, "manifest must record no experiment rerun"
    assert manifest["limitation_cross_questionnaire_not_run"] is True, "manifest must record generalization limitation"
    assert manifest["row_counts"]["phase4_integrated_synthesis"] == 40, "manifest row count mismatch"
    source_keys = {entry["key"] for entry in manifest["source_artifacts"]}
    assert "phase4_table4" in source_keys and "phase4_contribution" in source_keys, "manifest missing key Phase 4 sources"

    copied_figures = [
        output_dir / "figures" / "phase1_figure1_learning_curve.png",
        output_dir / "figures" / "phase2_figure3_delta_r.png",
        output_dir / "figures" / "phase3_figure5.png",
        output_dir / "figures" / "phase4_figure4.png",
    ]
    for figure in copied_figures:
        assert figure.exists() and figure.stat().st_size > 0, f"missing copied figure: {figure}"


def main() -> int:
    output_dir = run_report()
    check_outputs(output_dir)
    print(f"F015 final-report smoke checks passed: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

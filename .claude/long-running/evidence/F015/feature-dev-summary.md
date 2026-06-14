# F015 feature-dev summary

## Changed files
- `questionnaire-embeddings/scripts/generate_final_report.py`
- `questionnaire-embeddings/scripts/test_generate_final_report.py`
- `questionnaire-embeddings/results/final_report/` generated canonical report artifacts

## What was built
- Deterministic F015 report aggregation script that consumes accepted Phase 1–4 artifacts only.
- Canonical outputs under `results/final_report/`: `final_report.pdf`, `final_report.md`, `final_report.txt`, `final_summary.csv`, `statistical_summary.csv`, `best_pipeline_table.csv`, `report_manifest.json`, copied upstream figures, Table 5 synthesis, and Figure 6 synthesis.
- Standalone smoke test that runs the real generator into a fresh scratch directory and validates files, schemas, phase coverage, Phase 4 A1/A2/B1/B2 coverage, key narrative text, PDF signature/size, manifest provenance, and normalized recommended-pipeline rows.

## Verification
- `python3 scripts/generate_final_report.py` passed.
- `python3 scripts/test_generate_final_report.py` passed.

## Scope notes
- No Phase 1–4 experiment is rerun.
- Cross-questionnaire generalization experiment is not added; final report explicitly states the NEO-PI-R limitation.
- Table 5 joins/cross-validates `table4.csv`, `versionB_selection_contribution.csv`, and `versionB_selection_overlap.csv`.

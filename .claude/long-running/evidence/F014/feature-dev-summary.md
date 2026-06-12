# F014 feature-dev summary

## Feature
F014 — Phase 4: Embedding 对比实验 — 版本 B1/B2（重新选题）与完整对比

## What was built
- Added `questionnaire-embeddings/scripts/run_phase4_versionB.py`.
  - Implements Version B1/B2 with embedding-specific Coverage re-selection (`S_new`).
  - B1 uses fixed Phase 2 SoftmaxKNN hyperparameters.
  - B2 tunes K/τ using the same outer-train / train-inner discipline as F013 A2.
  - Preserves F013 Phase 4 scoring: continuous clip-only predictions, selected items as observed inputs, metrics on held-out/unselected items only.
  - Writes Version-B-specific canonical artifacts without overwriting F013 `versionA_*` files.
- Added `questionnaire-embeddings/scripts/test_phase4_versionB.py`.
  - Hermetic smoke test: generates a fresh Version A quick fixture, then runs Version B smoke against that fixture.
  - Avoids deleting prior smoke artifacts by allocating fresh output directories.
- Generated full Version B artifacts under `questionnaire-embeddings/results/phase4/`:
  - `versionB_predictions.parquet`
  - `versionB_participant_metrics.csv`
  - `versionB_results.csv`
  - `versionB_aggregate_metrics.csv`
  - `versionB_summary.csv`
  - `versionB_hyperparameters_by_fold_ratio_embedding.csv`
  - `versionB_selected_items_by_fold_ratio_embedding.json`
  - `versionB_statistical_tests.csv`
  - `versionB_selection_contribution.csv`
  - `versionB_selection_overlap.csv`
  - `versionB_outer_folds_subject_ids.json`
  - `figures/table4.csv`
  - `figures/figure4.pdf`
  - `figures/figure4.png`

## Key implementation decisions
- Output naming follows the user-approved Version-B-specific convention. F013 `versionA_*` artifacts are not overwritten.
- The unsupervised `CoverageSelector` is preserved as-is. `S_new` is effectively embedding×ratio, then materialized by fold for downstream joins and overlap audits.
- Jaccard overlap is computed against the fold-scoped historical `S_old` for every embedding×ratio×fold and saved to `versionB_selection_overlap.csv`.
- B−A selection contribution is paired by subject×outer_fold×embedding×ratio and tuning condition:
  - `B1_minus_A1`
  - `B2_minus_A2`
- Bootstrap inference reuses F013 participant-within-fold paired bootstrap over per-subject item_r differences.
- `figures/table4.csv` is a reader-friendly joined summary over A1/A2/B1/B2, while `versionB_statistical_tests.csv` and `versionB_selection_contribution.csv` remain the long-form audit sources of truth.
- Figure 4 is a faceted A/B learning curve:
  - fixed params panel: A1 vs B1
  - tuned params panel: A2 vs B2
  - x-axis ratio/m, y-axis primary item-level Pearson r, color by embedding, style/marker by A vs B.

## Verification summary
- Smoke test: PASS (`.claude/long-running/evidence/F014/smoke-test-output.txt`).
- Full run: PASS (`.claude/long-running/evidence/F014/test-output.txt`).
- Artifact checks: PASS (`.claude/long-running/evidence/F014/artifact-checks.txt`).

## Full-run artifact counts
- `versionB_predictions.parquet`: 109,960 rows.
- `versionB_participant_metrics.csv`: 109,960 rows.
- `versionB_results.csv`: 200 fold-level rows.
- `versionB_summary.csv`: 40 summary rows.
- `versionB_hyperparameters_by_fold_ratio_embedding.csv`: 200 rows.
- `versionB_statistical_tests.csv`: 32 vs-SBERT rows.
- `versionB_selection_contribution.csv`: 40 B−A rows.
- `versionB_selection_overlap.csv`: 100 embedding×ratio×fold rows.
- `versionB_selected_items_by_fold_ratio_embedding.json`: 200 records.
- `figures/table4.csv`: 80 A/B summary rows.

## Warnings / risks
- Full run emitted expected constant/near-constant Pearson warnings and mean-of-empty-slice warnings for degenerate metric cases. Artifact validation passed.
- Recomputed SBERT Coverage matched historical `S_old` for ratios 10 and 90 but showed documented tie/numerical drift at ratios 30 and 50 (Jaccard 0.935 and 0.923). This is recorded in `versionB_selection_overlap.csv`.
- Generic non-prefixed selected-items/hyperparameter/fold filenames in canonical `results/phase4` may still refer to F013 because the user explicitly requested not to overwrite F013 audit artifacts. F014 canonical artifacts are the `versionB_*` files listed above.

## Incomplete criteria
None known after artifact validation. Final completion remains pending independent evaluator review.

# F013 feature-dev summary

Implemented Phase 4 Version A fixed-S_old embedding comparison.

## Changed files
- `scripts/phase4_common.py` — Phase 4 loaders, fixed S_old parsing, embedding registry, bootstrap/p-value helpers, output paths.
- `scripts/phase4_predictors.py` — continuous clip-only SoftmaxKNN implementation, separate from historical rounded predictors.
- `scripts/run_phase4_versionA.py` — A1/A2 runner with fixed Phase 1 Coverage S_old, A1 fixed params, nested A2 tuning, participant-level predictions and bootstrap outputs.
- `scripts/test_phase4_versionA.py` — isolated smoke test writing to `results/phase4_smoke`.

## Main outputs
- `results/phase4/versionA_predictions.parquet`
- `results/phase4/versionA_participant_metrics.csv`
- `results/phase4/versionA_results.csv`
- `results/phase4/versionA_summary.csv`
- `results/phase4/hyperparameters_by_fold_ratio_embedding.csv`
- `results/phase4/selected_items_by_fold_ratio_embedding.json`
- `results/phase4/versionA_statistical_tests.csv`
- `results/phase4/outer_folds_subject_ids.json`

## Safeguards implemented
- Fixed S_old is loaded by fold×ratio from Phase 1 Coverage rows and reused across embeddings/versions.
- Prediction rows include `heldout_item_ids`/`predicted_item_ids` alongside `y_true` and `y_pred_continuous` vectors.
- Primary scoring follows Phase 2 held-out/unselected-only convention; selected items remain observed inputs and are not scored.
- A1 uses fixed SoftmaxKNN params: m=10 K=7 τ=0.1; m=30 K=7 τ=0.1; m=50 K=10 τ=0.1; m=90 K=3 τ=0.03.
- A2 tunes only on outer-train inner validation splits; hyperparameter CSV records train/valid/test sizes and grid scores.
- Bootstrap compares new embeddings vs SBERT by paired subject-level item_r, resampling subjects within each outer fold; raw, Holm, and BH p-values are reported.
- Continuous clip-only predictions are primary; rounded accuracy/MAE are supplemental only.
- Smoke test validates schema/vector alignment/no NaNs/A1 no tuning in an isolated output directory.

## Verification
- Full run completed successfully: `python scripts/run_phase4_versionA.py --all`.
- Smoke validation passed: `python scripts/test_phase4_versionA.py`.
- Artifact checks passed for row counts, schemas, selected item invariants, prediction vector alignment, A1/A2 hyperparameter rules, and bootstrap columns.

## Risks / notes
- Existing Phase 1/2 result CSVs store fold labels but not participant subject IDs. F013 reuses the same `participant_cv_split(n_subjects, n_folds=5, seed=0)` convention and writes the regenerated subject IDs to `outer_folds_subject_ids.json` for audit.

# Handoff — Next Session

## Immediate action
Run `/long-running-coding F014` to execute **Phase 4-B1/B2: embedding-specific re-selection**.

F013 is complete and should be treated as background context. Do not spend the next feature on F013 cleanup unless explicitly requested.

## Completed latest feature
### F013 — Version A: fixed original selected items
- Implemented `scripts/run_phase4_versionA.py` plus Phase 4 helper modules.
- A1: fixed Phase 2 SoftmaxKNN hyperparameters with fixed Phase 1 SBERT Coverage `S_old`.
- A2: fixed `S_old` plus nested train-inner K/τ tuning for each embedding/fold/ratio.
- Main scoring convention: continuous clip-only prediction, held-out/unselected items only; rounded metrics are supplemental.
- Outputs live in `questionnaire-embeddings/results/phase4/`:
  - `versionA_predictions.parquet`
  - `versionA_participant_metrics.csv`
  - `versionA_results.csv`
  - `versionA_summary.csv`
  - `hyperparameters_by_fold_ratio_embedding.csv`
  - `selected_items_by_fold_ratio_embedding.json`
  - `versionA_statistical_tests.csv`
  - `outer_folds_subject_ids.json`
- Evaluator verdict: PASS.

## Next feature: F014 — Version B: re-select items per embedding
- **B1:** embedding-specific Coverage `S_new` + fixed Phase 2 recommended SoftmaxKNN hyperparameters.
- **B2:** embedding-specific Coverage `S_new` + embedding-specific train-inner K/τ tuning.
- Compare against F013 A1/A2 using the same participant folds and prediction/scoring convention.
- Report `Jaccard(S_new, S_old)` for every embedding/fold/ratio.
- Define selection contribution within the same embedding and tuning regime:
  - `Δ_selection_fixed = performance(B1: S_new, E_new) - performance(A1: S_old, E_new)`
  - `Δ_selection_tuned = performance(B2: S_new, E_new) - performance(A2: S_old, E_new)`

## Pre-registered Phase 4 analysis rules
- Primary metric: **item-level Pearson r**.
- Key secondary metrics: `trait_r_mean`, `profile_r`, `MAE`.
- Main comparison: each new embedding vs `sbert_original`.
- Selection contribution comparison: B−A within the same embedding and same tuning regime.
- Statistical inference: paired bootstrap over participants, preserving outer-fold pairing.
- Multiple-comparison correction: Holm or Benjamini-Hochberg.
- Prediction mode: primary analysis uses continuous predictions clipped to `[1,5]`, **without rounding**.
- Rounded accuracy / rounded MAE are supplemental only.

## Useful files for F014
1. `.claude/long-running/features.json` — F014 acceptance criteria and dependencies.
2. `.claude/long-running/decisions.md` — Phase 4 design decisions.
3. `questionnaire-embeddings/scripts/phase4_common.py` — loaders, embedding registry, fixed S_old, bootstrap helpers.
4. `questionnaire-embeddings/scripts/phase4_predictors.py` — continuous SoftmaxKNN.
5. `questionnaire-embeddings/scripts/run_phase4_versionA.py` — A1/A2 loop and output schemas to mirror.
6. `questionnaire-embeddings/results/phase4/` — F013 A outputs used for B−A comparisons.
7. `questionnaire-embeddings/scripts/diagnose_embeddings.py` — CoverageSelector usage and embedding diagnostics patterns.

## F015 final-report requirement
If no cross-questionnaire generalization experiment is added before the final report, explicitly state the limitation: current conclusions are primarily for NEO-PI-R and cross-questionnaire generalization remains to be tested.

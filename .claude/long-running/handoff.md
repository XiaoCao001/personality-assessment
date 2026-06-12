# Handoff — Next Session

## Immediate action
Run `/long-running-coding F015` to execute **Phase 4 final integrated report**.

F013 and F014 are complete and should be treated as background context. Do not spend the next feature on Phase 4 A/B cleanup unless explicitly requested.

## Completed latest feature
### F014 — Version B: re-select items per embedding
- Implemented `scripts/run_phase4_versionB.py` and `scripts/test_phase4_versionB.py`.
- **B1:** embedding-specific Coverage `S_new` + fixed Phase 2 SoftmaxKNN hyperparameters.
- **B2:** embedding-specific Coverage `S_new` + embedding-specific train-inner K/τ tuning.
- Full run reused F013 outer folds and held-out-only continuous clip-only scoring.
- Canonical F014 outputs use `versionB_*` names so F013 generic audit artifacts remain intact:
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
  - `figures/figure4.pdf` / `figure4.png`
- B−A contribution is paired within the same subject×fold×embedding×ratio×tuning regime:
  - `B1_minus_A1 = B1(S_new,E_new) − A1(S_old,E_new)`
  - `B2_minus_A2 = B2(S_new,E_new) − A2(S_old,E_new)`
- Evaluator verdict: PASS.

## Completed previous feature
### F013 — Version A: fixed original selected items
- Implemented `scripts/run_phase4_versionA.py` plus Phase 4 helper modules.
- A1: fixed Phase 2 SoftmaxKNN hyperparameters with fixed Phase 1 SBERT Coverage `S_old`.
- A2: fixed `S_old` plus nested train-inner K/τ tuning for each embedding/fold/ratio.
- Main scoring convention: continuous clip-only prediction, held-out/unselected items only; rounded metrics are supplemental.
- Evaluator verdict: PASS.

## Next feature: F015 — final integrated report
- Summarize Phase 1–4 results, including A1/A2/B1/B2 embedding comparison and selection contribution attribution.
- Use F014 `figures/table4.csv`, `versionB_selection_overlap.csv`, and `versionB_selection_contribution.csv` as key Phase 4-B inputs.

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

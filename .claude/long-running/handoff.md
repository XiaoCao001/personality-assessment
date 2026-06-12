# Handoff — Next Session

## Immediate action
Run `/long-running-coding F013` to execute **Phase 4-A1/A2: fixed SBERT-Coverage selected items**.

F012 is complete and should be treated as background context. Do not spend the next feature on F012 cleanup unless explicitly requested.

## Phase 4 design update (2026-06-12)
The Phase 4 route has been revised before implementation to avoid confounding embedding quality, hyperparameter tuning, and item-selection effects.

### F013 — Version A: fixed original selected items
- **A1 (main analysis):** fixed SBERT-Coverage `S_old` + fixed Phase 2 recommended SoftmaxKNN hyperparameters.
  - Purpose: isolate whether the new embedding space improves nearest-neighbor geometry and prediction.
- **A2 (supplemental):** fixed `S_old` + embedding-specific K/τ tuning on train-inner only.
  - Purpose: estimate calibrated prediction upper bound for each embedding.

### F014 — Version B: re-select items per embedding
- **B1:** embedding-specific Coverage `S_new` + fixed Phase 2 recommended hyperparameters.
- **B2:** embedding-specific Coverage `S_new` + embedding-specific train-inner K/τ tuning.
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

## Required Phase 4 outputs
Each A/B script should save:
- `per-participant predictions`
- `selected_items_by_fold_ratio_embedding.json`
- `hyperparameters_by_fold_ratio_embedding.csv`
- metrics table with `item_r`, `MAE`, `trait_r_mean`, `profile_r`
- paired bootstrap Δ/CI/p-value vs SBERT original
- corrected p-values for embedding×ratio comparisons
- for B: Jaccard overlap and B−A `Δ_selection`

## Useful files for F013
1. `.claude/long-running/features.json` — updated F013/F014/F015 acceptance criteria.
2. `.claude/long-running/decisions.md` — updated Phase 4 design decisions.
3. `questionnaire-embeddings/scripts/run_softmax_kernel.py` and `scripts/predictors.py` — Phase 2 SoftmaxKNN implementation.
4. `questionnaire-embeddings/results/phase1/semantic_selection_detail.csv` — historical SBERT Coverage selected sets for fixed `S_old`.
5. `questionnaire-embeddings/scripts/diagnose_embeddings.py` — embedding registry and output conventions.
6. `questionnaire-embeddings/results/phase3/embedding_diagnostics_selected_sets.csv` and `embedding_diagnostics_global_space.csv` — diagnostic context if available locally.

## F015 final-report requirement
If no cross-questionnaire generalization experiment is added before the final report, explicitly state the limitation: current conclusions are primarily for NEO-PI-R and cross-questionnaire generalization remains to be tested.

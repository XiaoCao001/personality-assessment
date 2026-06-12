# Feature-dev handoff for F014

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F014
- Title: Phase 4: Embedding 对比实验 — 版本 B1/B2（重新选题）与完整对比
- Description: 版本 B 对每个 embedding 重新运行 Coverage 选题，测试新 embedding 是否同时改善选题和预测。为避免混淆，拆成 B1 = S_new + 固定 Phase 2 超参数，B2 = S_new + embedding-specific train-inner tuning。报告同一 embedding 下 B−A 的 Δ_selection = performance(S_new, E_new) − performance(S_old, E_new)，并报告 S_new 与 SBERT S_old 的 Jaccard overlap，严格分离选题贡献和预测空间贡献。
- Priority: high
- Dependencies: [F013]

## Acceptance criteria
1. [AC001] 换 embedding 后确实重新选题：selected_items_by_fold_ratio_embedding.json 记录每个 embedding/fold/ratio 的 S_new，且至少部分 embedding 与 SBERT S_old 不同。
2. [AC002] 每个 embedding 与 SBERT 原选题 S_old 的 Jaccard overlap 已按 fold/ratio 输出，用于解释重新选题幅度。
3. [AC003] 重新选题净贡献以同一 embedding 内 B−A 定义：Δ_selection = performance(S_new,E_new) − performance(S_old,E_new)，分别报告 B1−A1 与 B2−A2。
4. [AC004] B1 使用固定 Phase 2 超参数，B2 的 K/τ 调优完全在 train-inner 内完成，并由 hyperparameters 输出验证。
5. [AC005] Table 4 包含所有 embedding×ratio×version 的 item-level Pearson r、MAE、trait_r_mean、profile_r，以及 vs SBERT 和 B−A 的 bootstrap CI/校正 p 值。
6. [AC006] Figure 4 展示各 embedding 的 learning curve，并能区分固定 S/重新选题与固定超参数/重新调参版本。

## Test plan
- `python scripts/run_phase4_versionB.py --all`
- `python scripts/run_phase4_versionB.py --quick`
- `ls results/phase4/versionB_results.csv results/phase4/selection_contribution.csv results/phase4/selected_items_by_fold_ratio_embedding.json results/phase4/figures/table4.csv results/phase4/figures/figure4.pdf`
- `python -c "import pandas as pd; df=pd.read_csv('results/phase4/selection_contribution.csv'); assert {'B1_minus_A1','B2_minus_A2'} <= set(df['comparison'])"`

## Context from completed F013
- F013 is complete and provides Version A outputs under `questionnaire-embeddings/results/phase4/`:
  - `versionA_predictions.parquet`
  - `versionA_participant_metrics.csv`
  - `versionA_results.csv`
  - `versionA_summary.csv`
  - `versionA_statistical_tests.csv`
  - `hyperparameters_by_fold_ratio_embedding.csv`
  - `selected_items_by_fold_ratio_embedding.json`
  - `outer_folds_subject_ids.json`
- Useful implementation files:
  - `questionnaire-embeddings/scripts/phase4_common.py`
  - `questionnaire-embeddings/scripts/phase4_predictors.py`
  - `questionnaire-embeddings/scripts/run_phase4_versionA.py`
  - `questionnaire-embeddings/scripts/test_phase4_versionA.py`
  - `questionnaire-embeddings/scripts/diagnose_embeddings.py`
- F014 should mirror F013 output schemas where practical while keeping Version B outputs distinguishable.

## Phase 4 pre-registered analysis rules
- Primary metric: item-level Pearson r.
- Key secondary metrics: `trait_r_mean`, `profile_r`, `MAE`.
- Main comparison: each new embedding vs `sbert_original`.
- Selection contribution comparison: B−A within the same embedding and same tuning regime.
- Statistical inference: paired bootstrap over participants, preserving outer-fold pairing.
- Multiple-comparison correction: Holm or Benjamini-Hochberg.
- Prediction mode: primary analysis uses continuous predictions clipped to `[1,5]`, without rounding.
- Rounded accuracy / rounded MAE are supplemental only.

## Constraints
- Implement only F014. Do not touch unrelated code or start F015.
- Do NOT mark this feature `completed` in `features.json` — that is the orchestrator's job after evaluator PASS.
- Save verification artifacts under `.claude/long-running/evidence/F014/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

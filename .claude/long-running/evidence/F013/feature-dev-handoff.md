# Feature-dev handoff for F013

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F013
- Title: Phase 4: Embedding 对比实验 — 版本 A1/A2（固定原选题）
- Description: 版本 A 固定原 SBERT Coverage 选出的题目集合 S_old，只替换预测阶段 embedding，分成两个预注册子版本：A1 = 固定 S_old + 固定 Phase 2 推荐 SoftmaxKNN 超参数，用作主分析，纯测试 embedding 邻居几何是否更好；A2 = 固定 S_old + 每个 embedding 在 train-inner 上重新调 K/τ，用作补充分析，估计 embedding 在合理校准后的预测上限。主分析使用 continuous prediction（只 clip 到 [1,5]，不 round），rounded accuracy/rounded MAE 仅作为补充。
- Priority: high
- Dependencies: [F007, F010, F011]

## Acceptance criteria
1. [AC001] A1/A2 中所有 5 个 embedding 均在同一批 S_old、outer folds 和 test participants 上评估，固定原选题的可比性可由 selected_items_by_fold_ratio_embedding.json 验证。
2. [AC002] Version A1 使用固定 Phase 2 推荐超参数，不进行 embedding-specific K/τ 调参；输出 hyperparameters 文件可验证固定设置。
3. [AC003] Version A2 的 K/τ 调优完全在 train-inner 内完成，test participants 不参与调参。
4. [AC004] 版本 A 结果表包含 embedding×ratio×version 的 item-level Pearson r（主指标）、MAE、trait_r_mean、profile_r，并保存 per-participant predictions。
5. [AC005] 新 embedding vs SBERT original 的主比较使用 paired bootstrap over participants，报告 Δ、95% CI、原始 p 值和 Holm/BH 校正后的 p 值。
6. [AC006] 主分析预测为 continuous clip-only（不 round）；rounded accuracy/rounded MAE 仅作为补充输出。

## Test plan
- `cd /workspace/questionnaire-embeddings && python scripts/run_phase4_versionA.py --all`
- `cd /workspace/questionnaire-embeddings && python scripts/run_phase4_versionA.py --quick`
- `cd /workspace/questionnaire-embeddings && ls results/phase4/versionA_results.csv results/phase4/versionA_summary.csv results/phase4/selected_items_by_fold_ratio_embedding.json results/phase4/hyperparameters_by_fold_ratio_embedding.csv`
- `cd /workspace/questionnaire-embeddings && python -c "import pandas as pd; df=pd.read_csv('results/phase4/versionA_results.csv'); assert {'A1_fixed_hyperparams','A2_tuned_hyperparams'} <= set(df['version']); assert df['rounded'].eq(False).any()"`

## Constraints
- Implement only F013. Do not touch unrelated code or implement F014/F015.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job after evaluator PASS.
- Save verification artifacts under `.claude/long-running/evidence/F013/`.
- Primary analysis prediction mode must be continuous clip-only to [1,5], without rounding.
- A1 must use fixed Phase 2 recommended SoftmaxKNN hyperparameters and must not tune per embedding.
- A2 tuning must use train-inner only; test participants must be used only for final evaluation.
- Use the same fixed SBERT Coverage S_old, outer folds, and test participants for all embeddings and both A versions.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Useful context
- `questionnaire-embeddings/scripts/run_softmax_kernel.py` and `questionnaire-embeddings/scripts/predictors.py` contain Phase 2 SoftmaxKNN and evaluation patterns.
- `questionnaire-embeddings/results/phase1/semantic_selection_detail.csv` contains historical SBERT Coverage selected sets for fixed S_old.
- `questionnaire-embeddings/scripts/diagnose_embeddings.py` contains the embedding registry and output conventions.
- `questionnaire-embeddings/results/phase3/embedding_diagnostics_selected_sets.csv` and `embedding_diagnostics_global_space.csv` provide diagnostic context if available locally.

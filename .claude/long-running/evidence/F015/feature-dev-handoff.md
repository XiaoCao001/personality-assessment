# Feature-dev handoff for F015

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F015
- Title: Phase 4: 最终综合分析 — 完整结果汇总与论文级输出
- Description: 汇总全部四个阶段的实验结果，生成最终综合分析报告。包含：(1) 三层次贡献总结（选题策略、预测算法、测量效度）；(2) A1/A2/B1/B2 的 embedding 对比和选题贡献归因；(3) 所有 Table 和 Figure 的最终版本；(4) 统计检验汇总表，明确主指标、关键次指标和多重比较校正；(5) 最佳 pipeline 推荐；(6) 可复现性说明；(7) 若未完成跨问卷泛化实验，明确将结论限定为 NEO-PI-R 并列为 limitation。
- Priority: medium
- Dependencies: [F007, F010, F014]

## Acceptance criteria
1. [AC001] 最终报告包含三层次贡献的清晰论述，并明确区分 A1/A2 的预测空间贡献与 B1/B2 的重新选题贡献。
2. [AC002] 所有 Table 和 Figure 可复现生成（单一脚本或 notebook），并覆盖 Phase 4 A1/A2/B1/B2、Jaccard overlap、B−A selection contribution。
3. [AC003] 最佳 pipeline 有明确的 embedding、选题策略、预测器、K/τ 超参数、是否 tuned，以及 10/30/50/90% 下的预期性能。
4. [AC004] 统计检验预先固定主指标 item-level Pearson r，报告 trait_r_mean/profile_r/MAE 为关键次指标，并包含 paired bootstrap 与 Holm/BH 多重比较校正说明。
5. [AC005] 可复现性说明覆盖所有关键随机因素、fold 配对、inner tuning、continuous clip-only 主分析和 rounded 补充分析。
6. [AC006] 最终报告包含跨问卷泛化小实验结果，或明确写入 limitation：当前结论主要针对 NEO-PI-R，跨问卷泛化仍需验证。

## Test plan
- `cd /workspace/questionnaire-embeddings && python scripts/generate_final_report.py`
- `cd /workspace/questionnaire-embeddings && python scripts/test_generate_final_report.py`
- Verify outputs exist under `questionnaire-embeddings/results/final_report/`, especially `final_summary.csv` and `final_report.pdf`.

## Important context and inputs
- Phase 1 outputs: `questionnaire-embeddings/results/phase1/figures/`.
- Phase 2 outputs: `questionnaire-embeddings/results/phase2/figures/`.
- Phase 3 outputs: `questionnaire-embeddings/results/phase3/` and `questionnaire-embeddings/results/phase3/figures/`.
- Phase 4 Version A outputs: `questionnaire-embeddings/results/phase4/versionA_*`, `hyperparameters_by_fold_ratio_embedding.csv`, `selected_items_by_fold_ratio_embedding.json`, `versionA_statistical_tests.csv`.
- Phase 4 Version B outputs: `questionnaire-embeddings/results/phase4/versionB_*`, `versionB_selection_overlap.csv`, `versionB_selection_contribution.csv`, `versionB_statistical_tests.csv`, `figures/table4.csv`, `figures/figure4.pdf`, `figures/figure4.png`.
- Pre-registered Phase 4 rules: main metric is item-level Pearson r; key secondary metrics are `trait_r_mean`, `profile_r`, `MAE`; new embedding vs `sbert_original` and B−A comparisons use paired bootstrap over participants with Holm/BH correction; main predictions are continuous clip-only without rounding; rounded accuracy/MAE are supplemental only.
- If no cross-questionnaire generalization experiment is added, explicitly state the limitation: current conclusions are primarily for NEO-PI-R and cross-questionnaire generalization remains to be tested.

## Constraints
- Implement only F015. Do not touch unrelated code or long-running state completion fields.
- Do NOT mark this feature `completed` in `features.json` — that is the orchestrator's job after evaluator PASS.
- Save verification artifacts under `.claude/long-running/evidence/F015/` when practical.
- After implementation, summarize: changed files, commands run, generated artifacts, test results, risks, and any incomplete criteria.

# Feature-dev handoff for F010

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F010
- Title: Phase 2: 预测器消融评估 — 表格与图表
- Description: 固定最佳选题策略 Coverage（Phase 1 推荐），固定原 SBERT embedding，比较所有 5 种预测器（UniformKNN, CosineWeightedKNN, SoftmaxKNN, KernelSmoothing，以及 F002 原文 KNN K=5 baseline）。读取已有 Phase 2 实验数据（weighted_knn_detail.csv, softmax_kernel_detail.csv），不需要重新跑实验。生成 Table 3（预测器×比例消融表，含 item_r, item MAE, mean Big Five r, profile correlation）、Figure 3（加权方法相对原文 UniformKNN 的 Δr，分比例 bar/line chart）。通过 paired bootstrap test 确定最佳预测器，用于 Phase 4。
- Priority: high
- Dependencies: [F007, F008, F009] — all completed

## Acceptance criteria
1. [AC001] Table 3 包含所有 5 种预测器 × 4 种比例的完整指标（item_r ± 95% CI, item_mae, trait_r_mean, profile_r）。注意需要从 F008 和 F009 的 detail CSV 中汇总所有 5 种预测器：UniformKNN, CosineWeightedKNN, SoftmaxKNN, KernelSmoothing，如果原 baseline 数据可用则加入原文 KNN K=5 作为第 0 列对照。如果原文 baseline 不可用，则最少包含 F008+F009 的 4 种预测器。
2. [AC002] Figure 3 清楚展示加权方法在不同比例下的提升幅度（Δr = r_predictor - r_UniformKNN），分比例（10/30/50/90%）展示，柱状图或折线图。
3. [AC003] 最佳预测器有统计显著的提升证据——paired bootstrap test（N=10,000），95% CI 不跨 0。
4. [AC004] 所有预测器在同一批 Coverage S 和 test participants 上比较（公平对照）。F008 和 F009 均使用 Coverage 选题 + 相同 5-fold split（seed=0），Coverage 是确定性的，所以 S 相同，可直接对比。如果 S 不同则需在 evaluate_phase2.py 中重新统一计算。

## Test plan (verification commands)
- `cd /workspace/questionnaire-embeddings && python scripts/evaluate_phase2.py` — 全量运行
- `ls results/phase2/figures/table3_predictor_ablation.csv results/phase2/figures/figure3_delta_r.pdf`

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F010/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Data sources (pre-existing — do NOT re-run experiments)

### Phase 2 predictor results
- `results/phase2/weighted_knn_detail.csv` — 40 rows (2 predictors × 4 ratios × 5 folds), with columns: predictor, ratio, fold, best_K, item_r, item_r_ci_lower, item_r_ci_upper, item_mae, item_rmse, item_rounded_accuracy, trait_r_O/C/E/A/N, trait_r_mean, profile_r, coverage, redundancy, selected_S, inner_val_K_scores
- `results/phase2/weighted_knn_summary.csv` — 8 rows, aggregated mean ± CI
- `results/phase2/softmax_kernel_detail.csv` — 40 rows (2 predictors × 4 ratios × 5 folds), same columns + best_tau, inner_val_scores
- `results/phase2/softmax_kernel_summary.csv` — 8 rows, aggregated
- `results/phase2/softmax_kernel_sensitivity.csv` — τ sensitivity data

### Reference code
- `scripts/evaluate_phase1.py` (F007) — **primary template**: follow its structure for data loading, Table building, Figure generation, bootstrap CI, paired bootstrap tests, and recommendation output
- `scripts/predictors.py` — 5 predictor classes: UniformKNN, CosineWeightedKNN, SoftmaxKNN, KernelSmoothing
- `scripts/cv_framework.py` — evaluate_predictions, participant_cv_split, inner_validation_split
- `scripts/selection.py` — CoverageSelector (deterministic greedy, used by both F008 and F009)

### Phase 1 reference
- `results/phase1/figures/phase1_recommendation.txt` — Phase 1 conclusion: Coverage is best strategy
- `results/phase1/figures/statistical_tests.csv` — format reference for paired tests

## Implementation plan (for evaluate_phase2.py)

### 1. Data loading
- Load `weighted_knn_detail.csv` (UniformKNN, CosineWeightedKNN)
- Load `softmax_kernel_detail.csv` (SoftmaxKNN, KernelSmoothing)
- Concatenate into unified DataFrame — both use same Coverage S and fold splits
- Optionally load F002 baseline if available: `results/baseline/original_10fold_itemcv_results.csv`

### 2. Table 3 — Predictor Ablation
- Rows: 5 predictors × 4 ratios = 20 rows (or 4 predictors × 4 ratios = 16 if no F002)
- Columns: predictor, ratio, item_r (± 95% CI from bootstrap across folds), item_mae, trait_r_mean, profile_r, best_K, best_tau
- Use `bootstrap_ci()` pattern from evaluate_phase1.py (N_BOOTSTRAP=10_000)
- Print formatted table to stdout
- Save to `results/phase2/figures/table3_predictor_ablation.csv`

### 3. Figure 3 — Δr (Weighted − UniformKNN) by Ratio
- x-axis: 4 ratios (10, 30, 50, 90)
- y-axis: Δr (item_r relative to UniformKNN baseline)
- One bar/line per weighted predictor: CosineWeightedKNN, SoftmaxKNN, KernelSmoothing
- Add error bars (95% CI from paired bootstrap)
- Color scheme: SoftmaxKNN=#E63946 (red, best), KernelSmoothing=#457B9D (blue), CosineWeightedKNN=#2A9D8F (green)
- Save as PDF+PNG to `results/phase2/figures/figure3_delta_r.{pdf,png}`

### 4. Statistical Tests
- Paired bootstrap (N=10,000): each predictor vs UniformKNN at each ratio
- Paired bootstrap: SoftmaxKNN vs CosineWeightedKNN at each ratio
- Report Δr, 95% CI, p-value, significance marker (*/**/***)
- Save to `results/phase2/figures/statistical_tests_phase2.csv`

### 5. Recommendation
- Determine best predictor (highest mean item_r across ratios)
- Report per-ratio best predictor
- Report statistical significance vs UniformKNN
- Save to `results/phase2/figures/phase2_recommendation.txt`

### 6. Also compute trait-level comparison (optional but recommended)
- Short-form trait r: trait scores from administered items only, correlated against full-item trait scores
- Imputed trait r: from existing CSV data (trait_r_mean column)
- Compare predictors on trait-level performance

## Code style
- Follow `evaluate_phase1.py` conventions: same helper functions, same matplotlib/seaborn style, same output path pattern
- Use `PROJECT_ROOT = Path(__file__).resolve().parent.parent`
- Use `RANDOM_STATE = 0`, `N_BOOTSTRAP = 10_000`
- Support `--quick` mode (fewer bootstrap iterations) for faster testing

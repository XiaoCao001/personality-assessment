# Feature-dev handoff for F007

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F007
- Title: Phase 1: 选题策略完整评估 — 指标、表格与图表
- Description: 汇总所有 8 种选题策略（Random, Balanced Random, Coverage, Coverage+Diversity, TraitPredictiveness, Hybrid-A/B/C）× 4 种比例（10/30/50/90%）的完整评估结果。生成 Table 1（题项预测表现）、Table 2（人格总分预测表现，含 Short-form/Imputed/Held-out 三面板）、Figure 1（learning curve）、Figure 2（选中题项维度分布）。报告 95% CI 和 paired comparison 统计检验（bootstrap test）。确定 Phase 2 使用的最佳选题策略。
- Priority: high
- Dependencies: [F004, F005, F006] — all completed

## Key results to aggregate (already computed)

The following CSV files contain the raw 5-fold CV results:

1. **`results/phase1/random_baseline_summary.csv`** (F004): Random + BalancedRandom, columns: strategy, ratio, item_r, trait_r_mean, trait_r_O/C/E/A/N
2. **`results/phase1/semantic_selection_summary.csv`** (F005): Coverage + Coverage+Div(λ∈{0.25,0.5,1.0}), columns: strategy, ratio, item_r, item_mae, item_rmse, trait_r_O/C/E/A/N, trait_r_mean, profile_r, coverage, redundancy
3. **`results/phase1/trait_hybrid_selection_summary.csv`** (F006): Hybrid-A/B/C + TraitPredictiveness, columns: strategy, ratio, item_r, item_mae, item_rmse, item_rounded_accuracy, trait_r_O/C/E/A/N, trait_r_mean, profile_r, coverage, redundancy, trait_max_dev, dir_imbalance, fwd_ratio

Also available with per-fold detail:
- `results/phase1/random_baseline_detail.csv`
- `results/phase1/semantic_selection_detail.csv`
- `results/phase1/trait_hybrid_selection_detail.csv`

And aggregated (mean±std over folds):
- `results/phase1/random_baseline_aggregated.csv`
- `results/phase1/semantic_selection_aggregated.csv`
- `results/phase1/trait_hybrid_selection_aggregated.csv`

## Current codebase

- **`scripts/selection.py`**: All 8 selector classes (RandomSelector, BalancedRandomSelector, CoverageSelector, CoverageDiversitySelector, TraitPredictivenessSelector, HybridSelector with variants A/B/C)
- **`scripts/cv_framework.py`**: 5-fold participant CV, inner validation, evaluation functions (item-level r/MAE/RMSE, trait-level r/MAE/RMSE, profile correlation), reverse scoring
- **`data/processed/Y.npy`**: (2749, 100) response matrix
- **`data/processed/E_old.npy`**: (100, 1024) L2-normalized SBERT embeddings
- **`data/processed/metadata.parquet`**: item metadata (item_text, trait_id, reverse_id)

## Current strategy performance ranking (m=30)

1. Coverage: item_r=0.256 (F005)
2. Coverage+Div(λ=1.0): item_r≈0.25 (F005)
3. Hybrid-C: item_r=0.228 (F006)
4. BalancedRandom: item_r=0.214 (F004)
5. Hybrid-B: item_r=0.214 (F006)
6. Hybrid-A: item_r=0.200 (F006)
7. Random: item_r=0.198 (F004)
8. TraitPredictiveness: item_r=0.123 (F006)

Coverage is the clear winner across all ratios. F007 produces formal tables and figures.

## Acceptance criteria
1. [AC001] Table 1 contains all 8×4=32 cells with item-level r and 95% CI
2. [AC002] Table 2 has separate Short-form/Imputed/Held-out personality score panels (3 sub-tables)
3. [AC003] Figure 1 learning curve clearly shows strategy differences across ratios
4. [AC004] Statistical test conclusions clear: whether best strategy significantly outperforms random baseline (paired bootstrap test)

## Test plan
- `python scripts/evaluate_phase1.py --all`
- Verify outputs: `results/phase1/figures/table1.csv`, `results/phase1/figures/table2_*.csv`, `results/phase1/figures/figure1.pdf`, `results/phase1/figures/figure2.pdf`

## Implementation plan

Create **`scripts/evaluate_phase1.py`** that:

### 1. Load & Merge
- Load all 3 summary CSVs (random_baseline, semantic_selection, trait_hybrid_selection)
- Also load detail CSVs for per-fold CI computation
- Normalize strategy names for display

### 2. Table 1 — Item-level prediction (AC001)
- 8 strategies × 4 ratios = 32 cells
- Each cell: item_r (mean across 5 folds) ± 95% CI (bootstrap over folds or across 50 repeats for Random/BalancedRandom)
- Rows = strategies, columns = ratio (10/30/50/90)
- Save to `results/phase1/figures/table1_item_level.csv`

### 3. Table 2 — Personality trait-level prediction (AC002)
- Three sub-tables corresponding to three personality score types:
  - **Short-form**: trait scores computed from the m administered items only (reverse-scored, trait means)
  - **Imputed**: full 100-item trait scores after KNN imputation of held-out items
  - **Held-out**: trait scores computed from the (100−m) held-out items only (upper-bound check)
- Each sub-table: strategies × ratios, trait_r_mean ± 95% CI
- Save to `results/phase1/figures/table2_shortform.csv`, `table2_imputed.csv`, `table2_heldout.csv`

**Important**: The existing data has `trait_r_mean` (imputed). Short-form and held-out scores may need to be recomputed or estimated from the detail-level data. If the raw predictions aren't available, compute short-form scores as: for each fold, take the m administered items, reverse-score them, compute per-trait means, correlate with true per-trait means. For held-out: same but on the held-out items.

### 4. Figure 1 — Learning curve (AC003)
- x-axis: ratio (10, 30, 50, 90)
- y-axis: item-level Pearson r
- One line per strategy, with error bands (95% CI)
- Use matplotlib/seaborn, save as PDF (vector) + PNG
- Recommended styling: Coverage line highlighted (bold/red), baselines (Random, BalancedRandom) dashed
- Save to `results/phase1/figures/figure1_learning_curve.{pdf,png}`

### 5. Figure 2 — Trait balance distribution (AC004)
- Bar chart showing how many items each strategy selects per trait
- One panel per ratio (10/30/50/90), or one grouped bar chart
- Show deviation from uniform distribution (5 traits, ideally 20% each)
- Color-code traits (O/C/E/A/N) consistently
- Save to `results/phase1/figures/figure2_trait_distribution.{pdf,png}`

### 6. Statistical tests (AC004)
- Paired bootstrap test: for each non-random strategy at each ratio, compare item_r against Random baseline
- Compute: mean difference Δr, 95% bootstrap CI (10,000 resamples), p-value
- Also: paired test Coverage vs BalancedRandom (best baseline)
- Output: `results/phase1/figures/statistical_tests.csv`
- One row per comparison (strategy vs baseline, ratio)

### 7. Summary & recommendation
- Print a clear recommendation: which strategy is best, at which ratio, with statistical justification
- Output to stdout and save to `results/phase1/figures/phase1_recommendation.txt`

## Constraints
- Implement only this feature (F007). Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F007/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.
- Use `RANDOM_STATE=0` or `seed=0` for reproducibility.
- All output tables and figures go under `results/phase1/figures/`.
- Use matplotlib + seaborn for figures; pandas for tables. No heavy new dependencies.

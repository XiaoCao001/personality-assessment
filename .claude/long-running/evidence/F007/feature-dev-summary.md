# F007 Feature-dev Summary

## What was built
Created `scripts/evaluate_phase1.py` (~520 lines) — a comprehensive Phase 1 evaluation script that aggregates all 8 selection strategies × 4 ratios into tables, figures, and statistical tests.

## Key decisions
1. **Coverage+Div aggregation**: Uses best-λ per fold (from `is_best_lam` column in detail CSV), then averages across folds. This gives a single "Coverage+Div" row per ratio using the optimal λ chosen via inner validation.
2. **CI computation**: Bootstrap percentile CI with 10,000 resamples.
3. **Paired bootstrap tests**: Per-fold item_r values paired between strategy and Random baseline.
4. **Short-form/Held-out computation**: Computed from Y matrix + selected_S per fold (from detail CSVs).
5. **F005/F006 CSVs regenerated**: The original CSVs were truncated (only 4 rows). Re-ran `run_semantic_selection.py` and `run_trait_hybrid_selection.py` to regenerate full 80-row detail CSVs.

## Files changed
- **Created**: `scripts/evaluate_phase1.py` (520 lines)
- **Regenerated**: `results/phase1/semantic_selection_detail.csv` (80 rows), `semantic_selection_aggregated.csv` (16 rows), `semantic_selection_summary.csv` (16 rows)
- **Regenerated**: `results/phase1/trait_hybrid_selection_detail.csv` (80 rows), `trait_hybrid_selection_aggregated.csv` (16 rows), `trait_hybrid_selection_summary.csv` (16 rows)

## Outputs (under results/phase1/figures/)
- `table1_item_level.csv` — 32 cells (8 strategies × 4 ratios), item_r ± 95% CI
- `table2_shortform.csv` — trait-level r for short-form scores
- `table2_imputed.csv` — trait-level r for imputed-full scores (per-trait + mean)
- `table2_heldout.csv` — trait-level r for held-out scores
- `figure1_learning_curve.{pdf,png}` — learning curve with error bands
- `figure2_trait_distribution.{pdf,png}` — trait balance bar charts
- `statistical_tests.csv` — 32 paired bootstrap tests
- `phase1_recommendation.txt` — recommendation: Coverage is best strategy for Phase 2

## Key results
- **Coverage** dominates at all ratios: m=10: 0.084, m=30: 0.255, m=50: 0.304, m=90: 0.484
- Coverage is significantly better than Random at all ratios (p<0.001, paired bootstrap)
- Coverage is significantly better than BalancedRandom at all ratios (p<0.05)
- Coverage+Div penalty never improves prediction over pure Coverage
- TraitPredictiveness is the worst performer (selects from only 2-3 traits)
- **Recommendation**: Use Coverage for Phase 2

## Commands run
```bash
python scripts/run_semantic_selection.py       # Regenerate F005 CSVs
python scripts/run_trait_hybrid_selection.py   # Regenerate F006 CSVs
python scripts/evaluate_phase1.py              # Run F007 evaluation
```

## Risks
- Short-form/Held-out trait r computation regenerates selections for Random/BalancedRandom using deterministic seeds — the regenerated selections may not exactly match the original run if seed handling differs
- Figures generated with basic matplotlib/seaborn styling; could be improved for publication quality

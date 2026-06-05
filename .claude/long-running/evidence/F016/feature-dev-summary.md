# F016 Feature-dev Summary

## What was built
Reporting-only fix to `scripts/evaluate_phase2.py` for cross-phase baseline alignment.

## Key decisions
- **Option A chosen**: Rename "UniformKNN" → "Tuned UniformKNN" everywhere, add Phase 1 Coverage K=5 as separate "UniformKNN K=5 (原文 baseline)" row
- **No re-running experiments**: All data comes from existing F008/F009 detail CSVs and Phase 1 aggregated CSV
- **Data rename at load time**: `load_data()` replaces "UniformKNN" → "Tuned UniformKNN" in the DataFrame so all downstream functions automatically use the new name
- **Phase 1 baseline with best_K=5**: Reflects the original paper's fixed K=5, no inner validation tuning

## Files modified
1. `scripts/evaluate_phase2.py` — 6 sections modified:
   - Module docstring: updated figure description
   - PREDICTOR_ORDER/COLORS: renamed + added Phase 1 baseline
   - load_data(): added predictor rename step
   - build_table3(): added Phase 1 Coverage K=5 rows
   - build_figure3(): updated title/text
   - run_statistical_tests(): updated comparison labels
   - write_recommendation(): added cross-phase baseline note
2. `.claude/long-running/progress.md` — F008/F009/F010 session logs + ranking + status

## Files generated
- `results/phase2/figures/table3_predictor_ablation.csv` (updated)
- `results/phase2/figures/figure3_delta_r.pdf` (updated)
- `results/phase2/figures/figure3_delta_r.png` (updated)
- `results/phase2/figures/statistical_tests_phase2.csv` (updated)
- `results/phase2/figures/phase2_recommendation.txt` (updated)

## Verification
- Table 3: 20 rows (5 predictors × 4 ratios) — "Tuned UniformKNN" first + "UniformKNN K=5 (原文 baseline)" last
- Figure 3: Title uses "Tuned UniformKNN"
- Statistical tests: comparison column uses "Tuned UniformKNN"
- Recommendation: Includes Cross-Phase Baseline Alignment Note
- progress.md: F008/F009/F010/F016 session logs corrected

## Risks / incomplete criteria
- None. All 5 acceptance criteria met.

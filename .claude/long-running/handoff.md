# Handoff — Next Session

## Immediate action
Run `/long-running-coding F007` to start **Phase 1 完整评估：指标、表格与图表**，汇总所有 8 种策略，选出 Phase 2 最佳策略。

也可并行推进：
- `/long-running-coding F008`（Cosine Weighted KNN 预测器，depends on F003 ✓）

## Files to read first
1. `.claude/long-running/progress.md` — 项目总览（F001–F006 已完成）
2. `.claude/long-running/features.json` — 完整 15 个 feature 定义
3. `questionnaire-embeddings/scripts/selection.py` — 8 种选择器（Random, BalancedRandom, Coverage, CoverageDiversity, TraitPredictiveness, Hybrid A/B/C）
4. `questionnaire-embeddings/scripts/cv_framework.py` — F003 CV 框架
5. `questionnaire-embeddings/results/phase1/random_baseline_summary.csv` — F004 baselines
6. `questionnaire-embeddings/results/phase1/semantic_selection_summary.csv` — F005 results
7. `questionnaire-embeddings/results/phase1/trait_hybrid_selection_summary.csv` — F006 results

## Current state
- **F001 completed** — Y.npy (2749×100), E_old.npy (100×1024), metadata.parquet
- **F002 completed** — 原文基线复现: Mean r = 0.4628
- **F003 completed** — CV 框架: 7 functions, evaluator PASS
- **F004 completed** — Random & BalancedRandom baselines
- **F005 completed** — Coverage & Coverage+Diversity: Coverage beats all
- **F006 completed** — TraitPredictiveness & Hybrid A/B/C: Coverage still dominates
- **Phase 1 strategies complete (8/8).** F007 aggregates and selects best for Phase 2.

## F006 completion details
- **Module**: `scripts/selection.py` — added TraitPredictivenessSelector (+80 lines), HybridSelector (+160 lines)
- **Runner**: `scripts/run_trait_hybrid_selection.py` (320 lines) — 5-fold CV × 4 ratios × 4 strategies
- **Key results**:
  - Hybrid-C best at m=10/30/50 (item_r = 0.075/0.228/0.295) with near-perfect trait balance (max_dev=1)
  - Hybrid-A best at m=90 (item_r = 0.370)
  - Pure TraitPredictiveness worst: selects from only 2-3 traits (item_r = 0.052/0.123/0.120/0.197)
  - **All F006 strategies underperform F005 Coverage** (0.084/0.256/0.304/0.484) — Coverage remains the recommended semantic strategy
- **Evaluator**: PASS, 4/4 acceptance criteria met (attempt 1)
- **AC001**: max |r| = 0.7143 < 0.95 (self-exclusion confirmed) ✓
- **AC002**: Hybrid-C max trait dev = 1.0 vs TraitPredictiveness dev = 4-9 ✓
- **AC003**: selection on y_train only, never touches y_test ✓
- **AC004**: A/B/C variants produce distinct selections (A ≠ C confirmed) ✓
- **Results**: `results/phase1/trait_hybrid_selection_{detail,aggregated,summary}.csv`

## Phase 1 strategy ranking (by item_r at m=30)
1. **Coverage** (F005): 0.256
2. **Coverage+Div(λ=1.0)** (F005): ~0.25
3. **Hybrid-C** (F006): 0.228
4. **BalancedRandom** (F004): 0.214
5. **Hybrid-B** (F006): 0.214
6. **Hybrid-A** (F006): 0.200
7. **Random** (F004): 0.198
8. **TraitPredictiveness** (F006): 0.123

Coverage is the clear winner across all ratios. F007 will produce the formal tables and figures.

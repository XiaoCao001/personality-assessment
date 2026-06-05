# Feature-dev summary for F006

## What was built

### 1. TraitPredictivenessSelector (`scripts/selection.py`)
- Computes corrected item-total correlation for each item (within-trait, excludes self — AC001)
- Selects top-m items by |r_i|
- Exposes `.get_correlations()` and `.get_abs_correlations()` for diagnostics

### 2. HybridSelector (`scripts/selection.py`)
- Single class with variant='A'|'B'|'C'
- Greedy selection with z-score normalised multi-criteria scoring:
  - **A**: Coverage + TraitPredictiveness
  - **B**: Coverage + TraitPredictiveness − Redundancy
  - **C**: Coverage + TraitPredictiveness − Redundancy − TraitImbalancePenalty − DirectionImbalancePenalty
- Balances item-trait representation and forward/reverse ratio (variant C)
- All criteria z-scored across candidates at each greedy step for commensurability

### 3. Smoke test (`scripts/selection.py` `_demo()` extended)
- AC001: verifies max |r| < 0.95 (self-exclusion confirmed)
- AC002: verifies Hybrid-C trait distribution more balanced than TraitPredictiveness
- AC004: verifies Hybrid A ≠ C (independent operation)

### 4. Runner (`scripts/run_trait_hybrid_selection.py`)
- 5-fold participant-level CV following run_semantic_selection.py pattern
- Evaluates 4 strategies × 4 ratios = 16 conditions
- Saves detail/aggregated/summary CSVs to `results/phase1/`
- Supports --quick and --smoke flags

## Key results (5-fold CV)

| Strategy | m=10 | m=30 | m=50 | m=90 |
|---|---|---|---|---|
| TraitPredictiveness | 0.052 | 0.123 | 0.120 | 0.197 |
| Hybrid-A | 0.065 | 0.200 | 0.286 | 0.370 |
| Hybrid-B | 0.071 | 0.214 | 0.285 | 0.335 |
| Hybrid-C | **0.075** | **0.228** | **0.295** | 0.335 |

- Hybrid-C best at m=10/30/50 (balanced trait coverage + predictiveness)
- Hybrid-A best at m=90 (Coverage dominates when many items selected)
- Pure TraitPredictiveness worst: selects from only 2-3 traits, misses personality facets
- **However**: all F006 strategies underperform F005 pure Coverage (0.084/0.256/0.304/0.484)
  - Coverage-only remains the recommended semantic strategy
  
## Acceptance criteria status

- **AC001** ✓: corrected item-total excludes self (max |r|=0.7143 < 0.95)
- **AC002** ✓: Hybrid-C trait distribution balanced (max_dev=1 vs TP max_dev=4-9)
- **AC003** ✓: all selection on train participants only (y_train passed at selector init)
- **AC004** ✓: all three Hybrid variants run independently (A ≠ C confirmed)

## Files modified/created
- `scripts/selection.py` — added TraitPredictivenessSelector (+80 lines), HybridSelector (+160 lines), smoke tests (+110 lines), docstring update
- `scripts/run_trait_hybrid_selection.py` — new, 320 lines
- `results/phase1/trait_hybrid_selection_detail.csv` — 80 rows (5 folds × 4 ratios × 4 strategies)
- `results/phase1/trait_hybrid_selection_aggregated.csv` — 16 rows
- `results/phase1/trait_hybrid_selection_summary.csv` — 16 rows

## Risks / notes
- TraitPredictiveness uses the same response data as the target — inherently leaks item-level information across traits
- Hybrid weights (α=1, β=1, γ=0.5, δ=0.5) are fixed; could be tuned via inner validation in future work
- All greedy selection is deterministic (no random component with these weights)

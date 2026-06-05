# F008: Feature-dev Implementation Summary

## What was built

Two new files created:
1. **`scripts/predictors.py`** (298 lines) — Reusable predictor module
   - `_BaseKNN` abstract base class with vectorised prediction pipeline
   - `UniformKNN(K)` — baseline uniform-weight KNN
   - `CosineWeightedKNN(K)` — weighted by sim+(i,j) = (cos(e_i,e_j)+1)/2
   - Standalone smoke test with synthetic + real data

2. **`scripts/run_weighted_knn.py`** (480 lines) — Full evaluation runner
   - 5-fold participant CV with CoverageSelector
   - Inner validation: 80/20 split on train participants, tunes K∈{3,5,7,10,15}
   - Per-predictor K tuning (UniformKNN and CosineWeightedKNN independently)
   - 3 CSV outputs: detail, aggregated, summary → `results/phase2/`
   - `--quick` and `--smoke` flags

## Key results

| Predictor | m=10 | m=30 | m=50 | m=90 |
|---|---|---|---|---|
| UniformKNN | 0.1246 | 0.2856 | 0.3411 | 0.5582 |
| CosineWeightedKNN | **0.1511** | **0.2884** | **0.3422** | **0.5583** |
| Δr | **+0.0265** | +0.0028 | +0.0011 | +0.0001 |
| p-value | 0.0224 * | <0.001 *** | <0.001 *** | 0.7003 ns |

## Acceptance criteria status

| AC | Status | Notes |
|---|---|---|
| AC001 (Weighted KNN significantly better) | ✅ PASS | Paired t-test: m=10 p=0.022, m=30 p<0.001, m=50 p<0.001 |
| AC002 (K tuned on train only) | ✅ PASS | Inner validation uses 80/20 train-inner/valid-inner split |
| AC003 (K_eff = min(K, |S|)) | ✅ PASS | Verified in smoke test: K=10 > |S|=4 → K_eff=4 |
| AC004 (Predictions in [1,5]) | ✅ PASS | Round + clip in _BaseKNN.predict(); smoke test verified |

## Commands run
- `python scripts/predictors.py` — smoke test
- `python scripts/run_weighted_knn.py --smoke` — smoke runner
- `python scripts/run_weighted_knn.py` — full 5-fold CV

## Changed files
- `questionnaire-embeddings/scripts/predictors.py` (new, 298 lines)
- `questionnaire-embeddings/scripts/run_weighted_knn.py` (new, 480 lines)
- `questionnaire-embeddings/results/phase2/weighted_knn_detail.csv` (40 rows)
- `questionnaire-embeddings/results/phase2/weighted_knn_aggregated.csv` (8 rows)
- `questionnaire-embeddings/results/phase2/weighted_knn_summary.csv` (8 rows)

## Risks / notes
- Weighted KNN at m=10 uses K=10 (vs Uniform K=3) — the weighting allows more neighbours without noise penalty
- At m≥90, the advantage is negligible (too much information to benefit from weighting)
- CosineWeightedKNN shows 21% improvement over UniformKNN at m=10 (the most practical low-item scenario)

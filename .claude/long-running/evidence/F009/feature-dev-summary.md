# F009 Feature-dev Summary

## What was built

### Modified: `scripts/predictors.py`
- Extracted `_weighted_average()` — shared per-subject weighted-average helper used by both `_BaseKNN.predict()` and `KernelSmoothing.predict()`
- Added `SoftmaxKNN(_BaseKNN)` — KNN with numerically-stable softmax-normalised weights and temperature τ. Grid: K ∈ {3,5,7,10,15} × τ ∈ {0.03,0.05,0.1,0.2,0.5}
- Added `KernelSmoothing` (standalone) — Nadaraya-Watson kernel regression using ALL |S| items. Grid: τ ∈ {0.03,0.05,0.1,0.2,0.5}. No K parameter needed.
- Updated smoke test with 12 additional checks covering both new predictors

### Created: `scripts/run_softmax_kernel.py`
- 5-fold participant-level CV runner (430 lines)
- Coverage selector for item selection (Phase 1 recommendation)
- Inner validation: 2D grid search (K×τ) for SoftmaxKNN, 1D (τ) for KernelSmoothing
- τ sensitivity analysis (150 rows saved)
- Side-by-side comparison with F008 results
- Outputs: softmax_kernel_detail.csv, softmax_kernel_aggregated.csv, softmax_kernel_summary.csv, softmax_kernel_sensitivity.csv

## Key results (5-fold CV, Coverage selector, SBERT embedding)

| Predictor | m=10 | m=30 | m=50 | m=90 |
|---|---|---|---|---|
| SoftmaxKNN | 0.2587 | 0.3542 | 0.4037 | 0.5995 |
| KernelSmoothing | 0.2571 | 0.3368 | 0.3893 | 0.6001 |
| CosineWeightedKNN (F008) | 0.1511 | 0.2884 | 0.3422 | 0.5583 |
| UniformKNN (F008) | 0.1246 | 0.2856 | 0.3411 | 0.5582 |

**SoftmaxKNN vs F008 CosineWeightedKNN improvement:**
- m=10: +71.2% (+0.1076)
- m=30: +22.8% (+0.0658)
- m=50: +18.0% (+0.0615)
- m=90: +7.4% (+0.0412)

## Best hyperparameters
- SoftmaxKNN: K=7, τ=0.1 (m=10/30), K=10, τ=0.1 (m=50), K=3, τ=0.038 (m=90)
- KernelSmoothing: τ=0.1 (m=10/30), τ=0.09 (m=50), τ=0.034 (m=90)

## τ sensitivity (AC001)
- τ=0.1 is the sweet spot for low-item scenarios (m=10/30)
- As τ→0.03, SoftmaxKNN needs larger K (K=15) to compensate for peaky weights
- As τ→∞, weights become uniform (K=3 preferred, matching UniformKNN)
- Sensitivity is most pronounced at low m, confirming AC001

## Changed files
- `scripts/predictors.py` — +140 lines (2 new classes + shared helper)
- `scripts/run_softmax_kernel.py` — new file (430 lines)

## Risks
- Full 5-fold CV takes ~10 minutes (614s) — acceptable for research code
- KernelSmoothing is slightly slower than SoftmaxKNN due to using all |S| neighbours
- At m=90, all predictors converge (ceiling effect)

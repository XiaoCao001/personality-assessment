# F010 Feature-Dev Summary

## What was built

Created `scripts/evaluate_phase2.py` (~340 lines) — Phase 2 predictor ablation evaluation script.

## Architecture

Single self-contained Python script that:
1. Loads F008 and F009 detail CSVs (weighted_knn_detail.csv, softmax_kernel_detail.csv)
2. Verifies AC004: all predictors evaluated on identical Coverage S sets
3. Generates Table 3: 4 predictors × 4 ratios with item_r ± 95% CI, item_mae, trait_r_mean, profile_r
4. Generates Figure 3: Δr bar chart (weighted − UniformKNN) per ratio with 95% CI error bars
5. Runs paired bootstrap tests (N=10,000): each predictor vs UniformKNN, SoftmaxKNN vs CosineWeightedKNN, SoftmaxKNN vs KernelSmoothing
6. Outputs recommendation for Phase 4

## Key findings

### Predictor ranking (mean item_r across m=10,30,50,90)
1. **SoftmaxKNN**: 0.4040
2. **KernelSmoothing**: 0.3958
3. **CosineWeightedKNN**: 0.3350
4. **UniformKNN**: 0.3274

### Best predictor by ratio
- m=10: SoftmaxKNN (r=0.2587, K=7, τ=0.1)
- m=30: SoftmaxKNN (r=0.3542, K=7, τ=0.1)
- m=50: SoftmaxKNN (r=0.4037, K=10, τ=0.1)
- m=90: KernelSmoothing (r=0.6001, τ=0.034) — marginally better than SoftmaxKNN (0.5995, ns)

### Statistical significance
- All weighted predictors significantly (p<0.001) better than UniformKNN at all ratios except CosineWeightedKNN at m=90 (ns)
- SoftmaxKNN significantly better than CosineWeightedKNN at all ratios (p<0.001)
- SoftmaxKNN vs KernelSmoothing: significant at m=10/30/50 (p<0.001) but ns at m=90 (p=0.697)

### AC004 verification
- Coverage S sets are identical across F008 and F009 — fair comparison confirmed.

## Changed files
- `scripts/evaluate_phase2.py` (new, ~340 lines)

## Output files
- `results/phase2/figures/table3_predictor_ablation.csv`
- `results/phase2/figures/figure3_delta_r.pdf`
- `results/phase2/figures/figure3_delta_r.png`
- `results/phase2/figures/statistical_tests_phase2.csv`
- `results/phase2/figures/phase2_recommendation.txt`

## Test results
- Quick mode (2k bootstrap): PASS — all outputs generated
- Full mode (10k bootstrap): PASS — all outputs generated
- All 4 acceptance criteria satisfied

## Risks / caveats
- Bootstrap CIs with only 5 folds have limited granularity — per-fold values used as analysis units
- The "5 predictors" mentioned in the original spec was reduced to 4 since F002 uses a different CV paradigm (10-fold item CV) incompatible with the 5-fold participant CV framework
- CosineWeightedKNN shows diminishing returns at high ratios (m=90: Δr=+0.0001, ns) — confirms that weighting matters most when few items are available

## Recommendation for Phase 4
- Use **SoftmaxKNN** as the default predictor
- In very high-item scenarios (m≈90), KernelSmoothing is a viable alternative
- CosineWeightedKNN is not recommended for Phase 4 — SoftmaxKNN dominates it at all ratios

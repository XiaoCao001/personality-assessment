# F002 Feature-Dev Summary

## What was built
`scripts/run_baseline.py` — standalone script (214 lines) reproducing the original paper's per-participant 10-fold item-level CV.

## Key decisions
1. **Non-reversed data**: Uses `big5_responses_nonReversed.csv` (matching `modelPerformance(R=2)` in original paper). F001's Y.npy is the reversed version — not suitable for baseline reproduction.
2. **Exact pipeline**: StandardScaler → PCA(0.9) → KFold(10, random_state=0) → KNeighborsRegressor(K=5) → round + clamp [1,5] → per-participant Pearson r.

## Results
| Metric | F002 | Original modelPerformance() |
|---|---|---|
| Mean Pearson r | 0.4628 | 0.453 |
| 95% CI | [0.4571, 0.4686] | [0.447, 0.458] |
| Subjects evaluated | 2747/2749 | 2748/2749 |
| Mean MAE | 0.8913 | — |

## Files
- Created: `scripts/run_baseline.py`
- Output: `results/baseline/original_10fold_itemcv_results.csv`, `results/baseline/predictions.npy`

## Notes
- 2 subjects NaN r due to constant predictions (expected edge case)
- Small ~0.01 r difference from original `modelPerformance()` due to numpy vs pandas implementation
- PCA retains 46 components from 1024 dims (90.1% variance)

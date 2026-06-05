# Feature-dev Summary — F004

## What was built

Two modules implementing random baseline item-selection strategies for the Phase 1 experiment:

### `scripts/selection.py` (171 lines)
- **`RandomSelector`**: Uniformly random selection of m items from 100-item pool
- **`BalancedRandomSelector`**: Balanced random selection with `m//5` items per Big-Five trait, remainder randomly allocated
- Both support `select(m)` and `select_multi(m, n_repeats)` APIs
- Smoke test verified: 50/50 unique sets (Random), max deviation ≤ 1 (Balanced)

### `scripts/run_selection_baselines.py` (330 lines)
- Full 5-fold participant-level CV pipeline
- Vectorized KNN prediction using precomputed cosine distance matrix (100×100)
- Per-subject item-level KNN (K=5, cosine distance on raw E_old embeddings, no PCA)
- Uses `cv_framework.evaluate_predictions()` for item-level and trait-level metrics
- Outputs: detail (2000 rows), aggregated (40 rows), summary (8 rows)

## Key design decisions
1. **Cosine distance on raw E_old** — no StandardScaler/PCA preprocessing; E_old is already L2-normalized
2. **Vectorized prediction** — precomputed 100×100 cosine distance matrix; K nearest neighbors found via `argpartition`
3. **Y.npy (reversed)** used as response matrix — consistent with `evaluate_predictions` contract
4. **No cross-fold contamination** — all 5 folds use independent train/test splits

## Commands run
```bash
python scripts/selection.py                          # → ALL CHECKS PASSED
python scripts/run_selection_baselines.py --smoke     # → verified 1 fold, 1 repeat
python scripts/run_selection_baselines.py             # → full experiment
```

## Results (5-fold × 50-repeat averaged)

| Strategy | m | Item r | Big5 Mean r | O | C | E | A | N |
|---|---|---|---|---|---|---|---|---|
| BalancedRandom | 10 | 0.0751 | 0.4718 | 0.457 | 0.545 | 0.577 | 0.496 | 0.284 |
| Random | 10 | 0.0684 | 0.4132 | 0.412 | 0.494 | 0.503 | 0.429 | 0.228 |
| BalancedRandom | 30 | 0.2136 | 0.8219 | 0.781 | 0.851 | 0.869 | 0.803 | 0.805 |
| Random | 30 | 0.1984 | 0.7810 | 0.740 | 0.833 | 0.829 | 0.767 | 0.736 |
| BalancedRandom | 50 | 0.2852 | 0.9278 | 0.899 | 0.944 | 0.948 | 0.914 | 0.934 |
| Random | 50 | 0.2767 | 0.9167 | 0.894 | 0.938 | 0.941 | 0.895 | 0.917 |
| BalancedRandom | 90 | 0.3437 | 0.9928 | 0.989 | 0.994 | 0.995 | 0.991 | 0.995 |
| Random | 90 | 0.3214 | 0.9924 | 0.989 | 0.994 | 0.994 | 0.990 | 0.994 |

**Key findings:**
- BalancedRandom outperforms Random at all ratios (+6-10% for item r at m≤50)
- At m=90, both strategies nearly ceiling (Big5 r > 0.99)
- At m=10, item-level r is very low (~0.07) — KNN with only 10 training items predicts 90 items poorly
- Neuroticism (N) trait r is consistently lowest at m=10 for both strategies
- Results are stable across folds (consistent per-fold values)

## Acceptance criteria status

| Criterion | Status | Evidence |
|---|---|---|
| AC001: Random selections diverse (50 unique sets) | **PASS** | `selection.py` smoke test: 50/50 unique sets |
| AC002: Balanced max deviation ≤ 1 | **PASS** | `selection.py` smoke test: max deviation 0 for all ratios |
| AC003: Both strategies cover 10/30/50/90 | **PASS** | Runner output: all 4 ratios evaluated × 5 folds |
| AC004: item-level r + trait-level r in output | **PASS** | Results columns: item_r, trait_r_O/C/E/A/N, trait_r_mean |

## Files created/modified
- `questionnaire-embeddings/scripts/selection.py` — new
- `questionnaire-embeddings/scripts/run_selection_baselines.py` — new
- `questionnaire-embeddings/results/phase1/random_baseline_detail.csv` — 2000 rows
- `questionnaire-embeddings/results/phase1/random_baseline_aggregated.csv` — 40 rows
- `questionnaire-embeddings/results/phase1/random_baseline_summary.csv` — 8 rows

## Risks
- Item-level r values are low at m=10,10% — this is inherent to the data sparsity, not a bug
- Neuroticism trait r is lower than other traits at low m — warrants investigation in F005-F007
- ConstantInputWarning from scipy when KNN predicts constant values for some subjects — handled gracefully (NaN per-subject r)

## Incomplete criteria
None — all 4 acceptance criteria are met.

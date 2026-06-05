# Feature-dev handoff for F008

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- **ID**: F008
- **Title**: Phase 2: Cosine Weighted KNN 预测器
- **Priority**: high
- **Dependencies**: F003 (completed ✓) — cv_framework.py providing `participant_cv_split`, `inner_validation_split`, `reverse_score`, `compute_trait_scores`, `evaluate_predictions`

## Description

Implement cosine weighted KNN predictor that weights neighbour responses by semantic similarity `sim+(i,j) = (cos(e_i,e_j)+1)/2` instead of uniform averaging. Tune K on inner validation, and compare against the original uniform-weight KNN on the same outer folds and item sets S (selected by Phase 1's best strategy: Coverage).

## Project structure context

The project lives in `/workspace/questionnaire-embeddings/`. Key existing files:

| File | Purpose |
|---|---|
| `scripts/cv_framework.py` | F003 — 5-fold participant CV, inner validation, reverse scoring, evaluate_predictions |
| `scripts/selection.py` | F004–F006 — 8 selectors including CoverageSelector |
| `scripts/run_selection_baselines.py` | F004 — pattern to follow: `_precompute_cosine_dist`, `_predict_held_out_batch` (uniform KNN) |
| `scripts/run_semantic_selection.py` | F005 — inner validation pattern for hyperparameter tuning |
| `data/processed/Y.npy` | (2749, 100) float64, reverse-scored responses |
| `data/processed/E_old.npy` | (100, 1024) float32, L2-normalised SBERT embeddings |
| `data/processed/metadata.parquet` | trait_id, reverse_id, item_text |

Prediction pattern (from `run_selection_baselines.py` lines 90-168):
- `_precompute_cosine_dist(E)` → `dist[i,j] = 1 - dot(e_i,e_j)`, diagonal = inf
- `_predict_held_out_batch(y_test, dist, S, k)` → vectorised: for each held-out item j, find k nearest neighbours in S, average their responses, round+clamp to [1,5]

## Acceptance criteria

1. **AC001**: Weighted KNN 在至少一个比例上 item-level r 显著高于原文 KNN (paired test)
2. **AC002**: K 调优完全在 train participants 内完成
3. **AC003**: 当 |S|<K 时自动使用 K'=min(K,|S|)
4. **AC004**: 预测值范围在 [1,5] 内

## Implementation plan

### 1. Create `scripts/predictors.py` — predictor module

Two predictor classes (or functions), following the same conventions as `selection.py`:

```python
# CosineWeightedKNN:
#   K: int               — number of neighbours
#   predict(y_train_subjects, dist, S, T) → y_pred for held-out items T
#   Weight: w_j = sim+(neighbour_j, target_item) = (cos(e_nj, e_t) + 1) / 2
#   Prediction: ŷ = Σ(w_j × y_j) / Σ(w_j)   (weighted mean of neighbour responses)
#
# UniformKNN (baseline):
#   K: int
#   Same interface, uniform weights
```

Key design decisions:
- **Vectorised** prediction over all test subjects at once (like `_predict_held_out_batch`)
- Use precomputed cosine similarity matrix: `sim = E @ E.T` (clamped to [-1,1])
- sim+(i,j) = (sim[i,j] + 1) / 2
- Self-similarity exclusion: mask diagonal before neighbour search
- K_eff = min(K, |S|) when |S| < K (AC003)
- Round and clamp predictions to [1,5] (AC004)

### 2. Create `scripts/run_weighted_knn.py` — evaluation runner

Pattern: follow `run_semantic_selection.py` structure, adapted for predictor comparison.

Pipeline per outer fold:
1. **Outer CV**: `participant_cv_split(2749, 5, seed=0)` → train/test participant indices
2. **Item selection**: Use `CoverageSelector(E_old)` on train participants' data to select S for each ratio m∈{10,30,50,90}
3. **Inner validation** (AC002): Split train participants 80/20 via `inner_validation_split`. For each K∈{3,5,7,10,15}:
   - On train-inner: select S, predict held-out items with both WeightedKNN and UniformKNN
   - On valid-inner: compute item-level Pearson r
   - Pick best K per (predictor, ratio) by highest valid-inner item_r
4. **Test evaluation** (AC001): On test participants, with best-K per predictor, predict held-out items. Evaluate via `cv_framework.evaluate_predictions` → item_r, trait_r, profile_correlation
5. **Save**: detailed per-fold results and aggregated summary

Use `--quick` (1 fold, reduced ratios) and `--smoke` (1 fold, 1 ratio) flags for fast iteration.

### 3. Output structure

```
results/phase2/
├── weighted_knn_detail.csv      # per-fold × ratio × predictor rows
├── weighted_knn_aggregated.csv  # mean ± 95% CI across folds
├── weighted_knn_summary.csv     # top-level summary
└── inner_validation_k.csv       # best K per fold/ratio/predictor
```

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F008/`.
- Use `RANDOM_STATE = 0` and `N_FOLDS = 5` throughout for reproducibility.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Test plan
1. `python scripts/run_weighted_knn.py` — full run (expected ~5-10 min)
2. `python scripts/run_weighted_knn.py --smoke` — quick smoke test first
3. `python -c "from predictors import CosineWeightedKNN, UniformKNN; print('Import OK')"`

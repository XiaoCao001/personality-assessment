# Feature-dev handoff for F006

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- **ID**: F006
- **Title**: Phase 1: 心理测量选题策略 — Trait Predictiveness 与 Hybrid (A/B/C)
- **Description**: 实现使用被试作答数据的选题策略。(1) Trait Predictiveness: 计算 corrected item-total correlation r_i，选 |r_i| 最大的题。(2) Hybrid-A: Coverage + TraitPredictiveness。(3) Hybrid-B: Coverage + TraitPredictiveness - Redundancy。(4) Hybrid-C: 全部四项组合（Coverage + TraitPredictiveness - Redundancy - ImbalancePenalty），含维度均衡和正反向均衡惩罚。所有指标 z-score 标准化后组合。
- **Priority**: high
- **Dependencies**: F003 (completed), F005 (completed)

## Acceptance criteria
1. **[AC001]** corrected item-total correlation 不包含题项自身（无泄漏）
2. **[AC002]** Hybrid-C 选出的题目维度分布比纯 TraitPredictiveness 更均衡
3. **[AC003]** 所有选题决策仅在 train participants 上完成
4. **[AC004]** 三种 Hybrid 变体可单独运行和比较

## Codebase context

### Existing selection.py patterns (in `scripts/selection.py`)
- `RandomSelector(n_items, seed)` → `.select(m)` / `.select_multi(m, n_repeats)`
- `BalancedRandomSelector(trait_ids, trait_order, seed)` → `.select(m)` / `.select_multi(m, n_repeats)`
- `CoverageSelector(embeddings)` → `.select(m)`, `.compute_coverage(S)`, `.compute_redundancy(S)`, `.sim_plus_matrix`
- `CoverageDiversitySelector(embeddings, lam)` → `.select(m)`, uses z-score normalization across candidates
- Helper: `_zscore(arr)` already defined
- Constants: `RANDOM_STATE=0`, `N_ITEMS=100`, `TRAIT_ORDER=("O","C","E","A","N")`, `RATIOS=(10,30,50,90)`

### Existing cv_framework.py functions (in `scripts/cv_framework.py`)
- `participant_cv_split(n_subjects, n_folds=5, seed=0)` → list of (train_idx, test_idx)
- `inner_validation_split(train_indices, val_ratio=0.2, seed=0)` → (train_inner, valid_inner)
- `reverse_score(responses, reverse_ids)` → scored responses
- `compute_trait_scores(responses, trait_ids, trait_order)` → (n_subjects, 5)
- `evaluate_predictions(y_true, y_pred, trait_ids, reverse_ids)` → {item_level, trait_level, profile_correlation}
- `simulate_real_testing(y_test, S)` → (y_observed, y_held_out)

### Existing runner pattern (from `scripts/run_semantic_selection.py`)
- Load data: `Y = np.load(DATA_DIR / "Y.npy")` (2749×100, reverse-scored), `E_old` (100×1024, L2-normed), `metadata.parquet`
- Precompute cosine distance: `dist = 1.0 - E @ E.T`, fill diagonal with `inf`
- Vectorised KNN prediction: `_predict_held_out_batch(y_test, dist, S, k=5)`
- Per-fold evaluation with inner validation for tuning
- Save detail/aggregated/summary CSVs to `results/phase1/`

### Data files
- `data/processed/Y.npy` — (2749, 100) float32, REVERSE-SCORED responses (range [1,5])
- `data/processed/E_old.npy` — (100, 1024) float32, L2-normalised SBERT embeddings
- `data/processed/metadata.parquet` — columns: item_text, trait_id (O/C/E/A/N), forward_id, reverse_id

### Key design decisions
- Y matrix is already reverse-scored (confirmed in F001). Trait Predictiveness should use reverse-scored data.
- Embedding space used: E_old (SBERT, dim=1024). Coverage/Redundancy computations on E_old.
- Prediction: KNN K=5 with cosine distance, round+clamp to [1,5]
- item-total correlation: per-dimension, within the 20 items of that trait

## Implementation plan

### 1. Add selectors to `scripts/selection.py`

#### 1a. `TraitPredictivenessSelector` (extends pattern of CoverageSelector)

```python
class TraitPredictivenessSelector:
    """Select m items with the largest |corrected item-total correlation|.
    
    For each item i, compute r_i = Pearson correlation between the item's 
    response vector and the sum of OTHER items in the same trait (corrected 
    item-total — excludes item i itself).  Select the m items with largest |r_i|.
    
    Parameters
    ----------
    y_train : np.ndarray  shape (n_subjects, n_items)
        Response matrix (reverse-scored). Used to compute item-total correlations.
    trait_ids : np.ndarray  shape (n_items,)
        Trait label per item.
    trait_order : tuple of str
        Ordered trait labels (default OCEAN).
    """
    
    def __init__(self, y_train, trait_ids, trait_order=TRAIT_ORDER):
        # Compute corrected item-total correlations per trait
        self.n_items = y_train.shape[1]
        self.trait_ids = np.asarray(trait_ids)
        self._r_values = np.zeros(self.n_items)
        
        for trait in trait_order:
            mask = trait_ids == trait
            trait_items = np.where(mask)[0]
            n_trait = len(trait_items)
            for idx in trait_items:
                # Other items in same trait (exclude self)
                other = [j for j in trait_items if j != idx]
                # Item-total = sum of other items
                total_other = y_train[:, other].sum(axis=1)
                r, _ = sp_stats.pearsonr(y_train[:, idx], total_other)
                self._r_values[idx] = r
    
    def select(self, m):
        # Select m items with largest |r|
        abs_r = np.abs(self._r_values)
        top = np.argpartition(abs_r, -m)[-m:]
        # Sort by |r| descending for determinism
        top = top[np.argsort(-abs_r[top])]
        return np.sort(top)
```

**AC001 check**: Verify that for each trait item, the "other" list excludes the item itself. Add a smoke test assertion.

#### 1b. `HybridSelector` (single class with variant parameter)

```python
class HybridSelector:
    """Greedy hybrid selection combining semantic and psychometric criteria.
    
    Variants:
    - 'A': Coverage + TraitPredictiveness
    - 'B': Coverage + TraitPredictiveness - Redundancy
    - 'C': Coverage + TraitPredictiveness - Redundancy - ImbalancePenalty
    
    All criteria are z-score normalised across candidates at each greedy step.
    
    Parameters
    ----------
    embeddings : np.ndarray  shape (n_items, d)
    y_train : np.ndarray  shape (n_subjects, n_items)
    trait_ids : np.ndarray  shape (n_items,)
    reverse_ids : np.ndarray  shape (n_items,)
    variant : str  {'A', 'B', 'C'}
    alpha : float  TraitPredictiveness weight (default 1.0)
    beta : float   Redundancy penalty weight (default 1.0)
    gamma : float  Trait imbalance penalty weight (default 0.5)
    delta : float  Direction imbalance penalty weight (default 0.5)
    trait_order : tuple of str
    """
```

**Hybrid scoring per candidate:**

At each greedy step, for each remaining candidate i:
- `cov_z` = z-score of Coverage(S ∪ {i})
- `pred_z` = z-score of TraitPredictiveness (precomputed |r_i|)
- `red_z` = z-score of Redundancy(S ∪ {i})  [variants B, C]
- `imbalance_trait_z` = z-score of trait imbalance penalty [variant C]
- `imbalance_dir_z` = z-score of direction (forward/reverse) imbalance penalty [variant C]

Scores:
- Hybrid-A: `score = cov_z + α × pred_z`
- Hybrid-B: `score = cov_z + α × pred_z − β × red_z`
- Hybrid-C: `score = cov_z + α × pred_z − β × red_z − γ × imbalance_trait_z − δ × imbalance_dir_z`

**ImbalancePenalty_trait**: For S ∪ {i}, compute the deviation of per-trait counts from the ideal (|S|+1)/5. Use sum of squared deviations or max deviation.

**ImbalancePenalty_direction**: For S ∪ {i}, compute the deviation of forward/reverse ratio from 50/50. Use |fwd_pct - 0.5|.

The `select(m)` method follows the greedy pattern from `CoverageDiversitySelector.select()`:
```python
def select(self, m):
    S = []
    remaining = list(range(self.n_items))
    for _ in range(m):
        # Score all remaining candidates
        scores = np.empty(len(remaining))
        for k, i in enumerate(remaining):
            scores[k] = self._score_candidate(tuple(S), i)
        best_k = int(np.argmax(scores))
        S.append(remaining.pop(best_k))
    return np.sort(np.array(S, dtype=np.intp))
```

### 2. Smoke test in `_demo()` of `selection.py`

Add sections for TraitPredictiveness and HybridSelector to the existing `_demo()` function:
- Load Y.npy and metadata.parquet
- Instantiate TraitPredictivenessSelector, verify |r| values are in [0, 1]
- Verify Hybrid-A/B/C run without error for m=10,30,50,90
- **AC001**: Verify corrected item-total excludes self
- **AC002**: Compare trait distribution balance between Hybrid-C and pure TraitPredictiveness

### 3. Create runner script `scripts/run_trait_hybrid_selection.py`

Follow the pattern of `run_semantic_selection.py`:

```
Strategies evaluated:
- TraitPredictiveness
- Hybrid-A (Coverage + TraitPredictiveness)
- Hybrid-B (Coverage + TraitPredictiveness − Redundancy)  
- Hybrid-C (Coverage + TraitPredictiveness − Redundancy − ImbalancePenalty)

For Hybrid variants, z-score weights: α=1.0, β=1.0, γ=0.5, δ=0.5
```

The runner should:
1. Load data (Y, E_old, metadata)
2. Precompute cosine distance matrix (reuse `_precompute_cosine_dist`)
3. For each outer fold:
   a. Split train/test participants
   b. For each ratio m:
      - TraitPredictivenessSelector on y_train → select m items → predict → evaluate
      - HybridSelector(variant='A') on y_train, E_old → select m items → predict → evaluate
      - HybridSelector(variant='B') on y_train, E_old → select m items → predict → evaluate
      - HybridSelector(variant='C') on y_train, E_old → select m items → predict → evaluate
4. Aggregate results across folds
5. Save detail/aggregated/summary CSV to `results/phase1/`

Use `--quick` and `--smoke` flags like `run_semantic_selection.py`.

### 4. Output format

Results CSV columns (matching existing pattern + new fields):
```
strategy, ratio, fold, item_r, item_r_ci_lower, item_r_ci_upper, item_mae, item_rmse,
trait_r_O, trait_r_C, trait_r_E, trait_r_A, trait_r_N, trait_r_mean, profile_r,
coverage, redundancy, trait_imbalance, dir_imbalance, selected_S
```

Summary file: `results/phase1/trait_hybrid_selection_{detail,aggregated,summary}.csv`

## Test plan
1. `python scripts/selection.py` — smoke test (should pass all existing + new checks)
2. `python scripts/run_trait_hybrid_selection.py --smoke` — quick single-fold validation
3. `python scripts/run_trait_hybrid_selection.py` — full 5-fold CV run

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F006/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.
- Use existing patterns: same import style, same docstring format, same constant names (RANDOM_STATE, etc.)
- All greedy selection runs on train participants only (y_train passed at init time)
- Prediction phase uses KNN K=5 cosine distance (reuse `_predict_held_out_batch` from `run_semantic_selection.py`)
- Evaluation uses `cv_framework.evaluate_predictions`

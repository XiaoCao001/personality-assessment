# Feature-dev handoff for F009

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F009
- Title: Phase 2: Softmax Weighted KNN 与 Kernel Smoothing 预测器
- Description: 实现两个更灵活的预测器。(1) Softmax Weighted KNN: w_ij = exp(sim(i,j)/τ) / Σexp(sim/τ)，网格 K∈{3,5,7,10,15} × τ∈{0.03,0.05,0.1,0.2,0.5}。(2) Kernel Smoothing: 所有已作答题参与预测，w_ij = exp(sim(i,j)/τ)，网格 τ∈{0.03,0.05,0.1,0.2,0.5}。参数在 train participants inner validation 上选择。
- Priority: high
- Dependencies: [F003] (completed ✓)

## Acceptance criteria
1. [AC001] Softmax KNN 在低题量（10%, 30%）时 τ 参数敏感度合理
2. [AC002] Kernel Smoothing 正确使用所有 |S| 个已作答题（不限于 K 个邻居）
3. [AC003] 网格搜索不涉及 test participants
4. [AC004] 所有预测器输出与 F008 格式一致，可直接比较

## Test plan
- `python scripts/run_softmax_kernel.py`
- `python -c "from predictors import SoftmaxKNN, KernelSmoothing; print('OK')"`

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F009/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Technical context

### Existing code to extend: `scripts/predictors.py`
This file (298 lines) already contains:
- `_BaseKNN` — base class with vectorised neighbour lookup and weighted average prediction
- `UniformKNN(_BaseKNN)` — uniform weights (baseline)
- `CosineWeightedKNN(_BaseKNN)` — weights = (cos+1)/2

The two new predictor classes should follow the same `_BaseKNN` pattern:

**SoftmaxKNN(_BaseKNN)**: extend `_BaseKNN`, adding a `tau` parameter (temperature). Override `_compute_weights` to:
```python
w_ij = exp(similarities / tau) / sum(exp(similarities / tau), axis=0)  # softmax over K neighbours
```
- similarities are raw cosine values in [-1, 1]
- tau controls sharpness: low tau → peaky (closer to max), high tau → flat (closer to uniform)

**KernelSmoothing**: This one is different — it uses ALL items in S, not just K nearest neighbours. Two approaches:
1. Extend `_BaseKNN` with K=infinity (use all neighbours) and softmax/exp weights with temperature
2. Write a standalone predictor that loops over all S items

Simpler approach: make `KernelSmoothing` override `predict()` to use all |S| neighbours (ignore K), then apply exp weights:
```python
w_ij = exp(similarities / tau)  # raw exp, no normalisation needed (denominator handles it)
```
Actually, the kernel smoothing weight is simply `exp(sim(i,j) / tau)`. No softmax normalisation — the denominator in the weighted average handles normalisation:
```
ŷ_j = Σ_i exp(sim(i,j)/τ) × y_i / Σ_i exp(sim(i,j)/τ)
```
This is Nadaraya-Watson kernel regression with an exponential kernel on cosine similarity.

### Existing runner pattern: `scripts/run_weighted_knn.py`
The F008 runner (480 lines) provides the template. F009's runner `scripts/run_softmax_kernel.py` should:
- Load data the same way (Y, E_old, metadata)
- Precompute cosine similarity matrix
- Use Coverage selector for item selection (from Phase 1 recommendation)
- Inner validation: grid search over K × τ for SoftmaxKNN, and τ only for KernelSmoothing
- Evaluate on test participants with tuned parameters
- Save results to `results/phase2/softmax_kernel_detail.csv`, `softmax_kernel_aggregated.csv`, `softmax_kernel_summary.csv`

### Key differences from F008
1. **SoftmaxKNN** has 2D grid: K ∈ {3,5,7,10,15} × τ ∈ {0.03,0.05,0.1,0.2,0.5} = 25 combinations
2. **KernelSmoothing** has 1D grid: τ ∈ {0.03,0.05,0.1,0.2,0.5} = 5 combinations (all |S| neighbours)
3. Results should includes `best_K` AND `best_tau` columns (NaN for KernelSmoothing's K)
4. Add τ sensitivity analysis: for a fixed fold/ratio, evaluate all τ values and show how item_r varies with τ

### Data files
- `data/processed/Y.npy` — (2749, 100) response matrix
- `data/processed/E_old.npy` — (100, 1024) L2-normalised SBERT embeddings
- `data/processed/metadata.parquet` — item metadata (trait_id, reverse_id, item_text)

### Output format
Results CSV should match F008's columns plus `best_tau` and `tau`:
- predictor, ratio, fold, best_K, best_tau, item_r, item_r_ci_lower, item_r_ci_upper, item_mae, item_rmse, item_rounded_accuracy, trait_r_O/C/E/A/N, trait_r_mean, profile_r, coverage, redundancy, selected_S, inner_val_scores

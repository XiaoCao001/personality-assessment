# Feature-dev handoff for F003

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- **ID**: F003
- **Title**: Phase 1: 外层被试级交叉验证框架
- **Description**: 实现 5-fold participant-level CV 框架。将 2,749 名被试分成 5 折，每折 train(80%)/test(20%)。在 train participants 上确定题目集合 S 和调参，在 test participants 上模拟真实作答并评估。同时实现 inner validation 机制用于超参数选择。包含人格总分计算的完整 pipeline（反向计分→维度分数→profile correlation）。
- **Priority**: high
- **Dependencies**: F001 (completed — data ready in data/processed/)

## Acceptance criteria
1. [AC001] 5-fold split 每折 train:test ≈ 80:20，被试不跨折泄漏
2. [AC002] 反向计分正确：正向题 score=y，反向题 score=6-y
3. [AC003] 维度分数 = 该维度所有题项 score 的均值
4. [AC004] 评估指标输出格式统一，可通过脚本批量调用

## Test plan
- `python scripts/test_cv_framework.py`
- `python -c "from cv_framework import participant_cv_split; folds=participant_cv_split(2749,5,seed=0); assert all(abs(len(t)/2749-0.8)<0.01 for t,_ in folds)"`

## Implementation details

### What to build
Create a Python module `scripts/cv_framework.py` that provides the shared CV infrastructure for F004–F010.

### Key functions to implement

1. **`participant_cv_split(n_subjects, n_folds=5, seed=0)`**
   - Returns list of `(train_indices, test_indices)` tuples
   - Each fold ~80/20 split
   - Fixed random seed for reproducibility
   - Optional stratification by gender (metadata if available)

2. **`inner_validation_split(train_indices, val_ratio=0.2, seed=0)`**
   - Within a train set, further split into train-inner (80%) / valid-inner (20%)
   - Used for hyperparameter selection

3. **`reverse_score(responses, reverse_ids)`**
   - Forward items: score = raw_response
   - Reverse items: score = 6 - raw_response
   - `reverse_ids`: binary array where 1=reverse, 0=forward

4. **`compute_trait_scores(responses, trait_ids)`**
   - For each trait (O, C, E, A, N), compute mean of all items belonging to that trait
   - Returns (n_subjects, 5) array of trait scores

5. **`compute_profile_correlation(trait_scores_true, trait_scores_pred)`**
   - Per-subject Pearson r across the 5 trait dimensions
   - Returns mean profile correlation across subjects

6. **`evaluate_predictions(y_true, y_pred, trait_ids, reverse_ids)`**
   - Returns dict of all metrics:
     - item-level: pearson_r, mae, rmse, rounded_accuracy
     - trait-level: per-trait r, mae, rmse; mean Big Five r; profile correlation
   - All metrics computed per-subject then averaged

7. **`simulate_real_testing(y_test, S)`**
   - Given selected item set S, retain real responses for S, mark remaining as to-be-predicted
   - Returns masked y for test participants

### Data context
- **Y.npy**: (2749, 100) float32 — raw 1-5 responses (reverse-scored version from F001)
- **E_old.npy**: (100, 1024) float32 — L2-normalised SBERT embeddings
- **metadata.parquet**: columns = [question_id, item_text, trait_id, reverse_id]
- All paths relative to `questionnaire-embeddings/data/processed/`

### Code style (match existing codebase)
- Use numpy, scipy, sklearn (no heavy frameworks)
- `RANDOM_STATE = 0` at module level
- Use type hints in docstrings (numpy style)
- Print progress with `[OK]` / `[INFO]` prefixes
- All paths use `pathlib.Path`
- Import style: `import numpy as np`, `from scipy import stats`, `from sklearn.model_selection import KFold`

### Key design decisions (from decisions.md)
1. **Decision 1**: Outer CV is participant-level (between-subjects), not item-level (within-subject). This prevents data leakage for strategies that use trait-item correlations.
2. **Decision 2**: Predict raw 1-5 responses first. Reverse scoring ONLY during trait score computation, not before prediction.
3. **Decision 5**: Three personality scores per test subject: (a) Short-form (real items only), (b) Imputed full (real + predicted), (c) Held-out (predicted items only). Plus Profile Correlation.

### Output format
The module should be importable and also runnable as a script:
```bash
python scripts/cv_framework.py  # runs self-test/demo
python scripts/test_cv_framework.py  # runs unit tests
```

Unit tests should verify:
- Split ratios are correct (80/20 ± 1%)
- No subject appears in both train and test of any fold
- Reverse scoring correctness on known examples
- Trait score computation correctness
- All folds together cover all subjects exactly once

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F003/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.
- Work in `/workspace/questionnaire-embeddings/` directory.

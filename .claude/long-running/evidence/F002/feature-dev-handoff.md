# Feature-dev handoff for F002

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- **ID**: F002
- **Title**: Phase 0: 原文基线复现 — 10 折题项交叉验证
- **Description**: 用原 SBERT embedding + KNN K=5，严格复现原文的 per-participant 10-fold item CV。目标不是精确复现到小数点，而是确认平均题项预测 Pearson r 接近原文 ~.45 水平。如果偏差过大，优先排查 item 顺序、embedding 对应、反向计分时机、KNN 距离度量等问题。
- **Priority**: high
- **Dependencies**: [F001] (completed)

## Acceptance criteria
1. **[AC001]** 10-fold item CV 运行无错误，覆盖全部 2749 名被试
2. **[AC002]** Mean item-level Pearson r 在 .40–.50 范围内（或偏差有合理解释）
3. **[AC003]** 结果保存到 `results/baseline/original_10fold_itemcv_results.csv`
4. **[AC004]** 输出包含 per-participant r 值和汇总统计量

## Test plan
- `python scripts/run_baseline.py`
- `python -c "import pandas as pd; df=pd.read_csv('results/baseline/original_10fold_itemcv_results.csv'); print(f'Mean r: {df[\"pearson_r\"].mean():.4f}')"`

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F002/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Technical context

### Input data (from F001)
All data is in `data/processed/`:
- **Y.npy**: shape (2749, 100), float32, raw responses range [1, 5], NO reverse scoring
- **E_old.npy**: shape (100, 1024), float32, L2-normalized SBERT embeddings (`roberta-large-nli-stsb-mean-tokens`)
- **metadata.parquet**: 100 rows × 4 cols (question_id, item_text, trait_id, reverse_id)
- **subject_ids.txt**: 2749 subject IDs

### Original paper methodology (from `scripts/functions.py`)

The existing `modelPerformance(m=4, par=5, d="BIG5", e="sentencebert")` in `scripts/functions.py` is the reference implementation. Key details:

1. **10-fold CV over items** (not participants): `KFold(n_splits=10, random_state=0, shuffle=True)` splits the 100 items into 10 folds
2. **Per-participant training**: For each participant, train on 90 items (9 folds), predict the 10 held-out items (1 fold)
3. **KNN Regressor**: `KNeighborsRegressor(n_neighbors=5)` with default `metric='minkowski'` (Euclidean distance)
4. **Preprocessing**: StandardScaler → PCA(0.9 variance) on embeddings before KNN
5. **Prediction**: `np.round(predictions, 0)`, clamped to [1, 5]
6. **Evaluation**: `corrUserBased(preds, responses)` — per-participant Pearson r, then mean ± 95% CI via `mean_confidence_interval()`

### Implementation requirements

Create a standalone script `scripts/run_baseline.py` that:

1. Loads Y.npy and E_old.npy from `data/processed/`
2. Runs the exact same pipeline as `modelPerformance()` but using the new data format:
   - StandardScaler + PCA(0.9) on E_old
   - 10-fold item CV with KFold(n_splits=10, random_state=0, shuffle=True)
   - KNN K=5 regressor
   - Per-participant prediction for each fold
   - Round predictions to integers, clamp to [1, 5]
3. Computes per-participant Pearson r using `scipy.stats.pearsonr`
4. Computes mean r, 95% CI via `stats.sem` + `stats.t.ppf`
5. Saves per-participant results CSV to `results/baseline/original_10fold_itemcv_results.csv` with columns: `subject_id`, `pearson_r`, `p_value`, `mae`
6. Prints summary: mean r, 95% CI, p-value

### Important notes
- Use `random_state=0` everywhere for reproducibility
- The existing code uses Euclidean distance (default for KNeighborsRegressor), NOT cosine distance
- Keep cosine distance as a diagnostic if results deviate from ~.45
- The existing `functions.py` applies PCA to standardized embeddings — retain this
- Response scale clamping: `y_pred[y_pred < 1] = 1; y_pred[y_pred > 5] = 5`

### Project root
Script should use `Path(__file__).resolve().parent.parent` to locate project root (the script lives in `scripts/`, project root is `questionnaire-embeddings/`).

### Output directory
Create `results/baseline/` if it doesn't exist.

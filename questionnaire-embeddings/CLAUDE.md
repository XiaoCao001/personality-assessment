# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for the paper "A Deep Language Approach to Personality Assessment: Generalizing Across Items and Expanding the Reach of Survey-Based Research." The project uses text embeddings (Sentence-BERT, Word2Vec, LIWC) of personality questionnaire items to predict individual participant responses, and compares model performance against human raters.

## Environment setup

```bash
conda env create --file questionnaire.yaml
conda activate questionnaire
```

Key dependencies: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, statsmodels, sentence-transformers (for S-BERT extraction), tensorflow-hub.

## Architecture

### Data flow

1. **Extract embeddings** (`scripts/extract_questions_embeddings.ipynb`): Questionnaire item texts → Sentence-BERT embeddings (using `roberta-large-nli-stsb-mean-tokens`), saved as CSVs to `embeddings/{DATASET}/`.
2. **Train models** (`scripts/Study1.ipynb` through `Study4.ipynb`): Embeddings → 10-fold cross-validation over questions → per-user Ridge/KNN/SVC/KnnReg prediction → results saved to `results/{DATASET}/nonReversed/`.
3. **Human comparison** (within Study notebooks): Model predictions compared against human raters from Qualtrics experiments (data in `human_studies/`).

### Core module: `scripts/functions.py`

All notebooks import from this shared module. It provides:

- **Data loading**: `chooseData(dataset)`, `chooseEmb(type)`, `getResponses(folder, data, R)` — load questionnaire embeddings and response matrices from `embeddings/`.
- **Model selection**: `predModel(nr, par)` — returns model by numeric code: 0=Ridge, 1=RidgeClassifier, 2=KNN, 3=SVC, 4=KNNRegressor.
- **Cross-validation**: `compareModels()`, `modelPerformance()` — 10-fold CV over questions, training per-user. `compareModelsConstructs()`, `compareModelsKey()` — construct/key direction prediction variants.
- **Evaluation**: `corrUserBased(x, y)` — per-user Pearson correlation (primary metric), returns mean correlation, CI, p-value. `accuracy_constr()`, `accuracy_keys()` — classification accuracy for construct/key tasks.
- **Human comparison**: `regCorr()`, `predictionPerformance()`, `welch_t_test()` — compare model vs. human rater correlations.

### Data directory conventions

`embeddings/{DATASET}/` contains:
- `{dataset}_questions_text.csv` — item texts with construct labels and keying/encoding directions
- `{dataset}_questions_embeddings_{TYPE}.csv` — embeddings (SENTENCEBERT, WORD2VEC, or LIWC)
- `{dataset}_responses.csv` — raw participant responses (reversed-coded)
- `{dataset}_responses_nonReversed.csv` — non-reversed responses

`results/{DATASET}/nonReversed/` contains prediction outputs named `{ModelName}_{param}_{EmbeddingType}_{task}.csv`.

### Datasets

BIG5 (NEO-PI-R), IPIP (full), IPIP2 (assigned items only), RIASEC, HSQ, 16PF.

### Embedding types

SENTENCEBERT (primary), WORD2VEC (baseline), LIWC (baseline). The best-performing model across studies is typically **KNN Regressor (k=5) with Sentence-BERT embeddings**.

### Fixed parameters

- Random state: `0` throughout for reproducibility
- 10-fold CV over questions, shuffled
- PCA retaining 90% variance applied before training
- StandardScaler applied to embeddings before PCA
- Response scale clamped to 1–5 after prediction

## Running the notebooks

Execute notebooks in `scripts/` in order (extract embeddings first, then Study1–Study4). All paths are relative to the `scripts/` directory. The `compareModels(verbose=1)` call in Study1 runs a full grid search over all model × embedding × parameter combinations (takes significant time).

## Plotting

Aggregated performance data is cached in `plot_data/` as CSVs. Plots are saved to `plot_data/` as PDF/TIFF/SVG.


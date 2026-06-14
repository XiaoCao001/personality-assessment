# A Deep Learning Approach to Personality Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains source code and data for the project **"A Deep Language Approach to Personality Assessment: Generalizing Across Items and Expanding the Reach of Survey-Based Research."**

The project uses text embeddings (Sentence-BERT, Word2Vec, LIWC) of personality questionnaire items to predict individual participant responses, and compares model performance against human raters across multiple personality frameworks.

## 📁 Repository Structure

```
.
├── questionnaire-embeddings/     # Main research codebase
│   ├── scripts/                  # Jupyter notebooks & Python modules
│   │   ├── functions.py          # Core ML pipeline (data loading, CV, evaluation)
│   │   ├── Study1.ipynb          # Main model evaluation
│   │   ├── Study2.ipynb          # Construct-level analysis
│   │   ├── Study3.ipynb          # Key direction prediction
│   │   ├── Study4.ipynb          # Human rater comparison
│   │   ├── extract_questions_embeddings.ipynb  # SBERT embedding extraction
│   │   ├── preprocess_human_data.ipynb         # Human rater preprocessing
│   │   ├── target_selection.ipynb              # Target selection for human studies
│   │   ├── evaluate_phase1.py    # Phase 1 evaluation scripts
│   │   └── evaluate_phase2.py    # Phase 2 evaluation scripts
│   ├── embeddings/               # Item texts, embeddings & participant responses
│   │   ├── BIG5/
│   │   ├── IPIP/
│   │   ├── IPIP2/
│   │   ├── RIASEC/
│   │   ├── HSQ/
│   │   └── 16PF/
│   ├── results/                  # Model prediction outputs & evaluation tables
│   ├── human_studies/            # Human rater data (Qualtrics experiments)
│   ├── plot_data/                # Figures & aggregated performance CSVs
│   ├── data/                     # Processed data files
│   ├── questionnaire.yaml        # Conda environment specification
│   └── CLAUDE.md                 # Developer guide for Claude Code
├── .claude/                      # Claude Code long-running harness
│   ├── skills/                   # Harness skills (init, coding, status, repair)
│   ├── hooks/                    # Git hooks & validation scripts
│   ├── agents/                   # Evaluator agent definition
│   └── long-running/             # Feature state, evidence & progress tracking
├── scripts/                      # Harness management scripts
│   ├── check-harness.sh
│   └── install-into-project.sh
└── results/                      # Top-level results directory
```


## ✅ Completed Extension Study and Final Report

The long-running extension study is complete: all planned features **F001–F016** have been implemented, independently evaluated, and committed. The final integrated Phase 1–4 report is available in:

- `questionnaire-embeddings/results/final_report/final_report.pdf` — canonical PDF deliverable
- `questionnaire-embeddings/results/final_report/final_report.md` — reviewable Markdown source
- `questionnaire-embeddings/results/final_report/final_summary.csv` — normalized Phase 1–4 summary table
- `questionnaire-embeddings/results/final_report/statistical_summary.csv` — consolidated statistical tests
- `questionnaire-embeddings/results/final_report/best_pipeline_table.csv` — recommended pipeline by administered-item ratio

To regenerate and smoke-test the final report from existing accepted artifacts:

```bash
cd questionnaire-embeddings
python scripts/generate_final_report.py
python scripts/test_generate_final_report.py
```

The F015 final report is an aggregation/synthesis layer: it consumes accepted Phase 1–4 artifacts and does **not** rerun the expensive experiments. It explicitly notes that conclusions are primarily based on NEO-PI-R and that cross-questionnaire generalization remains future work.

## 🚀 Getting Started

### Environment Setup

Using conda (recommended):

```bash
conda env create --file questionnaire-embeddings/questionnaire.yaml
conda activate questionnaire
```

Or manually install the required packages:

```bash
pip install jupyterlab matplotlib numpy pandas scikit-learn scipy seaborn statsmodels sentence-transformers
```

### Running the Analysis

All code is located in `questionnaire-embeddings/scripts/`. Run the notebooks in the following order:

1. **`extract_questions_embeddings.ipynb`** — Extract Sentence-BERT embeddings from questionnaire items
2. **`Study1.ipynb`** — Main model evaluation across all datasets and embedding types
3. **`Study2.ipynb`** — Construct-level performance analysis
4. **`Study3.ipynb`** — Key direction prediction tasks
5. **`Study4.ipynb`** — Comparison with human raters

Additional utilities:
- **`preprocess_human_data.ipynb`** — Preprocess Qualtrics human rater data
- **`target_selection.ipynb`** — Select targets for human rater studies
- **`evaluate_phase1.py`** — Phase 1 statistical evaluation
- **`evaluate_phase2.py`** — Phase 2 predictor ablation evaluation
- **`generate_final_report.py`** — F015 final integrated Phase 1–4 report generator
- **`test_generate_final_report.py`** — standalone smoke test for the final report package

## 📊 Datasets

The project covers six personality frameworks:

| Dataset | Description |
|---------|-------------|
| **BIG5** | NEO-PI-R Big Five personality inventory |
| **IPIP** | International Personality Item Pool (full) |
| **IPIP2** | IPIP with assigned items only |
| **RIASEC** | Holland's RIASEC vocational interests |
| **HSQ** | Humor Styles Questionnaire |
| **16PF** | Cattell's 16 Personality Factors |

## 🔬 Methods

### Embedding Types
- **Sentence-BERT** (primary) — using `roberta-large-nli-stsb-mean-tokens`
- **Word2Vec** (baseline)
- **LIWC** (baseline)

### Models
The core ML pipeline (`scripts/functions.py`) implements:
- **Ridge Regression** — L2-regularized linear regression
- **Ridge Classifier** — For construct/key classification tasks
- **KNN Regressor** (k=5) — Best-performing model overall
- **SVC** — Support Vector Classifier
- **Weighted KNN** — Cosine & softmax-weighted variants

### Evaluation
- 10-fold cross-validation over questions (random state: 0)
- PCA retaining 90% variance
- StandardScaler on embeddings
- Per-user Pearson correlation (primary metric)
- Statistical significance testing (Welch's t-test, confidence intervals)

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{deepLanguagePersonality,
  title   = {A Deep Language Approach to Personality Assessment: Generalizing Across Items and Expanding the Reach of Survey-Based Research},
  journal = {To appear},
  year    = {2025}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🛠️ Development Harness

This repository uses a [Claude Code](https://claude.ai/code) long-running development harness for AI-assisted feature development. See `.claude/` for the harness implementation, including:

- **Skills**: `/long-running-init`, `/long-running-coding`, `/long-running-status`, `/long-running-repair`
- **State tracking**: `.claude/long-running/features.json`, `progress.md`
- **Evidence**: `.claude/long-running/evidence/<FEATURE_ID>/`
- **Evaluator**: Independent verification agent per feature

For setup instructions, run `./scripts/install-into-project.sh /path/to/target-project`.

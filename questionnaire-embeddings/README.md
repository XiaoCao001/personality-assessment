# A Deep Learning Approach to Personality Assessment
This repository contains source codes and data for the "A Deep Language Approach to Personality Assessment: Generalizing Across Items and Expanding the Reach of Survey-Based Research" project.

# Instructions

If using conda/anaconda, run `conda env create --file questionnaire.yaml` to recreate the exact environment/packages versions as in this project. Otherwise, manually load compatible versions of the necessary packages.
Then, run the following programs in the desired order of replication (e.g., extract embeddings -> replicate Study 1 -> ...)

# Code files
All codes are located in the `scripts` folder:
- Codes for extracting S-BERT embeddings can be found at `extract_questions_embeddings.ipynb`.
- Codes to replicate analyses and visualizations from the main publication can be found in `Study1.ipynb` to `Study4.ipynb`
- Codes to preprocess the human rater data (behavioral experiment) can be found in `preprocess_human_data.ipynb`
- Codes to replicate the target selection for the human rater studies can be found in `target_selection.ipynb`
- `generate_final_report.py` builds the completed Phase 1–4 final report from existing artifacts.
- `test_generate_final_report.py` runs a standalone smoke test for the final report package.

# Data files
- Item texts, embeddings (SBERT, Word2Vec and LIWC), and original participant responses can be found under `/embeddings` in the respective questionnaire's folder.
- Human rater data (raw; Qualtrics export) can be found found under `/human_studies` in the respective questionnaire's folder.

# Final integrated report

The completed extension study includes a deterministic final-report generator that aggregates accepted Phase 1–4 artifacts without rerunning the underlying experiments. The canonical deliverables are stored in `results/final_report/`:

- `final_report.pdf` — final integrated PDF report
- `final_report.md` / `final_report.txt` — reviewable narrative sources
- `final_summary.csv` — normalized Phase 1–4 summary table
- `statistical_summary.csv` — consolidated inferential results
- `best_pipeline_table.csv` — best observed pipeline for m = 10, 30, 50, 90 administered items
- `figures/table5_phase4_integrated_synthesis.csv` and `figures/figure6_phase4_selection_vs_performance.*` — Phase 4 A/B synthesis outputs

Regenerate and validate the final report with:

```bash
python scripts/generate_final_report.py
python scripts/test_generate_final_report.py
```

The final report explicitly limits its conclusions to the current NEO-PI-R evidence base; cross-questionnaire generalization remains future work.


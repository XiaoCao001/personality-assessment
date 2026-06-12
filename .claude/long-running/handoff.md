# Handoff — Next Session

## Immediate action
Run `/long-running-coding F013` to execute **Phase 4: Embedding 对比实验 — 版本 A（固定原选题）**.

F012 is now complete — embedding-space diagnostics for original SBERT + MiniLM/MPNet/E5/BGE were generated and independently evaluated as PASS.

## What was done in F012
- Created `questionnaire-embeddings/scripts/diagnose_embeddings.py`
- Compared five embedding spaces:
  - `sbert_original` — `data/processed/E_old.npy`
  - `minilm_l6_v2` — `embeddings/neo_minilm_l6_v2.npy`
  - `mpnet_base_v2` — `embeddings/neo_mpnet_base_v2.npy`
  - `e5_base_v2` — `embeddings/neo_e5_base_v2.npy`
  - `bge_base_en_v15` — `embeddings/neo_bge_base_en_v15.npy`
- Generated selected-set diagnostics for `m ∈ {10,30,50,90}`:
  - `coverage_shifted_cosine`
  - `redundancy_shifted_cosine`
  - `selected_S`
- Generated full-space/global diagnostics:
  - all-pair raw cosine mean/std
  - within-trait raw cosine
  - between-trait raw cosine
  - within-minus-between raw cosine
  - full-100 shifted Coverage/Redundancy summary columns
- Generated Figure 5:
  - `questionnaire-embeddings/results/phase3/figures/figure5.pdf`
  - `questionnaire-embeddings/results/phase3/figures/figure5.png`
- Generated Phase 3 explanatory text:
  - `questionnaire-embeddings/results/phase3/figures/phase3_embedding_diagnostics.txt`

## Key F012 interpretation for Phase 4
- E5-base-v2 has the highest selected-set Coverage across all m values.
- MPNet-base-v2 has the largest within-minus-between raw cosine trait-separation gap.
- These are hypotheses only; F013/F014 must test whether the geometry differences improve prediction.
- F012 warns that current deterministic SBERT Coverage sets differ slightly from historical Phase 1 CSV at m=30/50, likely tie/numerical drift. Do not assume byte-for-byte selected_S equivalence unless explicitly loading Phase 1 saved S.

## Current state
- **F001–F012 completed** — Phase 1, Phase 2, and Phase 3 diagnostics done
- **F016 completed** — Phase 2 baseline alignment fix done
- **F013–F015 pending** — Phase 4 embedding comparison and final analysis

## Files to read for F013
1. `.claude/long-running/features.json` — F013 steps and acceptance criteria
2. `questionnaire-embeddings/scripts/diagnose_embeddings.py` — F012 embedding registry and output conventions
3. `questionnaire-embeddings/results/phase3/embedding_diagnostics_selected_sets.csv` — model-specific Coverage-selected S diagnostics
4. `questionnaire-embeddings/results/phase3/embedding_diagnostics_global_space.csv` — raw cosine trait-structure diagnostics
5. `questionnaire-embeddings/results/phase3/figures/phase3_embedding_diagnostics.txt` — interpretation hypotheses for Phase 4
6. `questionnaire-embeddings/scripts/run_softmax_kernel.py` and `scripts/predictors.py` — Phase 2 best predictor implementation
7. `questionnaire-embeddings/results/phase1/semantic_selection_detail.csv` — Phase 1 original SBERT Coverage-selected S if F013 fixes original S

## Evidence
- F012 evidence directory: `.claude/long-running/evidence/F012/`
- Evaluator report: `.claude/long-running/evidence/F012/evaluator-report.json` (PASS)

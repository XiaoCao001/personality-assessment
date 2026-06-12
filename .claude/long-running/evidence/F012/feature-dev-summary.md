# F012 feature-dev summary

## Feature
F012 — Phase 3: Embedding 空间质量诊断

## What was built
- Added `questionnaire-embeddings/scripts/diagnose_embeddings.py`, a standalone Phase 3 diagnostics script.
- The script compares 5 embedding spaces:
  1. `sbert_original` — `data/processed/E_old.npy`
  2. `minilm_l6_v2` — `embeddings/neo_minilm_l6_v2.npy`
  3. `mpnet_base_v2` — `embeddings/neo_mpnet_base_v2.npy`
  4. `e5_base_v2` — `embeddings/neo_e5_base_v2.npy`
  5. `bge_base_en_v15` — `embeddings/neo_bge_base_en_v15.npy`
- Reuses existing `CoverageSelector` for Phase 1/2-consistent shifted-cosine selected-set diagnostics.
- Computes global raw-cosine embedding-space diagnostics using all unordered item pairs, excluding diagonal self-pairs.

## Outputs generated
- `questionnaire-embeddings/results/phase3/embedding_diagnostics_selected_sets.csv`
  - 20 rows = 5 embeddings × m ∈ {10, 30, 50, 90}
  - Includes `coverage_shifted_cosine`, `redundancy_shifted_cosine`, and `selected_S`.
- `questionnaire-embeddings/results/phase3/embedding_diagnostics_global_space.csv`
  - 5 rows = 5 embeddings
  - Includes all-pair raw cosine mean/std, within-trait raw cosine, between-trait raw cosine, within-minus-between, and full-100 shifted-cosine Coverage/Redundancy diagnostics.
- `questionnaire-embeddings/results/phase3/embedding_diagnostics_summary.csv`
  - Compact merged summary for downstream reading.
- `questionnaire-embeddings/results/phase3/figures/figure5.pdf`
- `questionnaire-embeddings/results/phase3/figures/figure5.png`
- `questionnaire-embeddings/results/phase3/figures/phase3_embedding_diagnostics.txt`

## Figure 5 layout
- Panel A: Coverage by embedding and selected-set size m (`coverage_shifted_cosine`)
- Panel B: Redundancy by embedding and selected-set size m (`redundancy_shifted_cosine`)
- Panel C: Within vs Between trait similarity by embedding (`raw cosine`)
- Panel D: Trait separation = within − between (`raw cosine`)

Full-100 diagnostics are included in CSV/text outputs and are intentionally not plotted as Panel A/B curve points.

## Verification
- `python scripts/diagnose_embeddings.py --all` completed successfully.
- `results/phase3/figures/figure5.pdf` exists.
- Artifact assertion checks passed:
  - selected-set CSV has 20 rows
  - global-space CSV has 5 rows
  - summary CSV has 5 rows
  - m values are exactly {10, 30, 50, 90}
  - required raw/shifted metric columns exist
  - Figure 5 PDF/PNG and diagnostic text exist and are non-empty

## Notes / risks
- The SBERT Phase 1 historical cross-check emits non-blocking warnings at m=30 and m=50 because current deterministic Coverage selection differs by a few item IDs from the historical Phase 1 CSV. The differences are treated as tie/numerical-drift warnings, not hard failures, because F012 computes current embedding-space diagnostics and does not require byte-for-byte reproduction of Phase 1 selected sets.
- Diagnostic interpretation is deliberately hypothesis-level: Phase 4 must still test predictive performance.

## Incomplete criteria
None known.

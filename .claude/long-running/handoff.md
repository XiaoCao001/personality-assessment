# Handoff — Next Session

## Immediate action
Run `/long-running-coding F012` to execute **Phase 3: Embedding 空间质量诊断** (depends on F011 ✓).

F011 is now complete — MiniLM/MPNet/E5/BGE embeddings and combined metadata manifest were generated and validated.

## What was done in F011
- Created `questionnaire-embeddings/scripts/generate_embeddings.py`
- Generated four new L2-normalized embedding matrices:
  - `questionnaire-embeddings/embeddings/neo_minilm_l6_v2.npy` — shape (100,384)
  - `questionnaire-embeddings/embeddings/neo_mpnet_base_v2.npy` — shape (100,768)
  - `questionnaire-embeddings/embeddings/neo_e5_base_v2.npy` — shape (100,768), E5 `query: ` prefix
  - `questionnaire-embeddings/embeddings/neo_bge_base_en_v15.npy` — shape (100,768)
- Added `questionnaire-embeddings/embeddings/neo_embeddings_metadata.json` with source item-order/text hashes, package versions, device info, model metadata, and artifact SHA256 hashes
- Added `pyarrow`, `sentence-transformers`, and `torch` to `questionnaire-embeddings/questionnaire.yaml`
- Verification passed:
  - `python3 scripts/generate_embeddings.py --validate`
  - `HF_HUB_OFFLINE=1 python3 scripts/generate_embeddings.py --validate`
  - artifact shape/norm/order assertion script
- Evaluator verdict: PASS (4/4 criteria)

## Current state
- **F001–F011 completed** — Phase 1, Phase 2, and Phase 3 embedding generation done
- **F016 completed** — Phase 2 baseline alignment fix done
- **F012–F015 pending** — Phase 3 diagnostics and Phase 4/final analysis

## Files to read for F012
1. `.claude/long-running/features.json` — F012 steps and acceptance criteria
2. `questionnaire-embeddings/scripts/generate_embeddings.py` — F011 manifest/output conventions
3. `questionnaire-embeddings/embeddings/neo_embeddings_metadata.json` — new embedding registry and provenance
4. `questionnaire-embeddings/data/processed/E_old.npy` and `metadata.parquet` — original SBERT and item trait metadata
5. `questionnaire-embeddings/scripts/selection.py` — existing Coverage/Redundancy logic that F012 can reuse

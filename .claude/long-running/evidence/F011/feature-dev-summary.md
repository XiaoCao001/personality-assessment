# F011 feature-dev summary

## Built
- Added `questionnaire-embeddings/scripts/generate_embeddings.py`, a self-contained CLI for generating and validating new NEO-PI-R sentence-transformer embeddings.
- Generated four L2-normalized embedding matrices under `questionnaire-embeddings/embeddings/`:
  - `neo_minilm_l6_v2.npy` — `sentence-transformers/all-MiniLM-L6-v2`, shape `(100, 384)`
  - `neo_mpnet_base_v2.npy` — `sentence-transformers/all-mpnet-base-v2`, shape `(100, 768)`
  - `neo_e5_base_v2.npy` — `intfloat/e5-base-v2`, shape `(100, 768)`, with `query: ` prefix
  - `neo_bge_base_en_v15.npy` — `BAAI/bge-base-en-v1.5`, shape `(100, 768)`
- Added combined metadata manifest `questionnaire-embeddings/embeddings/neo_embeddings_metadata.json`.
- Updated `questionnaire-embeddings/questionnaire.yaml` with F011 runtime dependencies: `pyarrow`, `sentence-transformers`, and `torch`.

## Key implementation decisions
- Canonical source is `data/processed/metadata.parquet`; row `i` is generated from `source.question_ids[i]` / `source.item_texts[i]`.
- Metadata records stable hashes for ordered question IDs, ordered item texts, and paired question-id/text provenance.
- Default generation refuses to overwrite existing selected `.npy` files or metadata entries; `--overwrite` is required for replacement.
- `--validate` is pure local artifact validation. It does not import `sentence_transformers`, instantiate models, or trigger Hugging Face downloads.
- `--models` supports subset generation and merges selected metadata entries without deleting existing entries for other models.
- Artifact writes are deferred until all selected models generate successfully, avoiding partial output files after mid-run failures.

## Commands run
- `python3 -m py_compile scripts/generate_embeddings.py`
- `python3 scripts/generate_embeddings.py --validate` — initially failed as expected before artifacts existed.
- `python3 scripts/generate_embeddings.py --all` — generated initial artifacts on CUDA/Tesla T4.
- `python3 scripts/generate_embeddings.py --validate` — validation passed.
- `HF_HUB_OFFLINE=1 python3 scripts/generate_embeddings.py --validate` — offline/local validation passed.
- Shape/norm/order metadata Python check — passed.
- `python3 scripts/generate_embeddings.py --models minilm_l6_v2` without `--overwrite` — failed as expected, confirming overwrite guard.
- After review fixes: `python3 -m py_compile scripts/generate_embeddings.py && python3 scripts/generate_embeddings.py --all --overwrite` — regenerated all artifacts and validation passed.

## Test results
- Generation used auto device resolution: `cuda` on Tesla T4.
- Package versions recorded in metadata: Python 3.10.11, NumPy 1.26.4, pandas 2.2.2, pyarrow 16.1.0, torch 2.4.0, sentence-transformers 3.1.0.
- All four matrices validate with correct shapes and float32 dtype.
- All row L2 norms are within `1e-5` of 1.0.
- Metadata source provenance matches current `data/processed/metadata.parquet`.
- Metadata file SHA256 values match the saved `.npy` artifacts.

## Risks / notes
- The generated `.npy` files are model artifacts and may be large relative to source code, but F011 acceptance explicitly requires saved embeddings.
- `questionnaire.yaml` still contains the pre-existing `panda==0.3.1` entry; this was not changed because it is outside F011 scope.

## Incomplete criteria
- None known from the builder perspective. Orchestrator still needs to collect evidence, run evaluator, update long-running state, and commit if evaluator returns PASS.

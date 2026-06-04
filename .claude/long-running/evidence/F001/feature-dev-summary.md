# Feature-dev Summary for F001

## Feature
- **ID**: F001
- **Title**: Phase 0: Data Preparation & Matrix Standardisation
- **Date**: 2026-06-04

## What was built

Two new scripts and three data output files:

### Scripts
1. **`scripts/prepare_data.py`** — Main data pipeline:
   - Reads 4 CSV files from `embeddings/BIG5/`
   - Builds Y matrix [2749 × 100] (raw 1-5 responses, transposed from items×subjects)
   - Builds E_old matrix [100 × 1024] (SBERT embeddings, L2 normalised)
   - Builds metadata table (question_id, item_text, trait_id, reverse_id)
   - Runs integrity validation before saving

2. **`scripts/validate_data.py`** — Standalone validation:
   - Checks output files exist and are loadable
   - Verifies dimensions, value ranges, L2 normalisation, trait distribution

### Output files (`data/processed/`)
- `Y.npy` — float32, (2749, 100), range [1, 5]
- `E_old.npy` — float32, (100, 1024), L2-normalised
- `metadata.parquet` — 100 rows × 4 columns

## Key decisions
- Used raw responses (NOT non-reversed) for Y matrix — reverse scoring is handled later during trait score computation
- L2 normalisation applied directly to raw embeddings (no StandardScaler/PCA — those are for the prediction pipeline)
- Float32 precision to conserve memory (~1.1 MB for Y, ~400 KB for E_old)
- Used the actual dataset path `BIG5` (not `NEO_PI_R` as written in features.json)

## Acceptance criteria status
| Criterion | Status | Notes |
|-----------|--------|-------|
| AC001: Y shape (2749,100), range [1,5], no NaN | PASS | Verified |
| AC002: E_old shape (100, d≥384), L2 normalised | PASS | d=1024, norms within 1e-5 of 1.0 |
| AC003: 5 traits × 20 items, reverse ratio reasonable | PASS | 50/50 forward/reverse |
| AC004: Output files loadable | PASS | numpy.load and pandas.read_parquet work |

## Files changed
- `scripts/prepare_data.py` (new, 278 lines)
- `scripts/validate_data.py` (new, 66 lines)
- `data/processed/Y.npy` (new)
- `data/processed/E_old.npy` (new)
- `data/processed/metadata.parquet` (new)

## Commands run
```
python scripts/prepare_data.py    # Main pipeline — PASS
python scripts/validate_data.py   # Validation — All passed
```

## Risks / Notes
- features.json references `NEO_PI_R` as dataset name but actual directory is `BIG5`. The handoff corrected this.
- SBERT embedding dimension is 1024 (roberta-large-nli-stsb-mean-tokens), which matches the original project's embedding model.
- The data/processed directory should be added to .gitignore if binary artifacts aren't tracked.

## Next steps for F001
Ready for evaluator review. Then F002 (baseline reproduction) and F003 (CV framework) can proceed.

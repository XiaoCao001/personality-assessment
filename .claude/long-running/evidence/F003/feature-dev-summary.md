# F003 Feature-dev Summary

## What was built
Created `scripts/cv_framework.py` — a Python module providing shared CV infrastructure for the full experiment pipeline (F004–F010). Created `scripts/test_cv_framework.py` with 112 unit tests covering all acceptance criteria.

## Files created
1. **`scripts/cv_framework.py`** (397 lines) — 7 public functions + internal helper + stand-alone demo
2. **`scripts/test_cv_framework.py`** (315 lines) — 112 tests across 10 test functions

## Functions implemented
1. `participant_cv_split(n_subjects, n_folds, seed)` — 5-fold participant-level CV via sklearn KFold
2. `inner_validation_split(train_indices, val_ratio, seed)` — inner validation split for hyperparameter tuning
3. `reverse_score(responses, reverse_ids)` — forward=y, reverse=6-y transform
4. `compute_trait_scores(responses, trait_ids, trait_order)` — mean per trait (O/C/E/A/N) per subject
5. `compute_profile_correlation(trait_true, trait_pred)` — per-subject Pearson r across 5 traits
6. `evaluate_predictions(y_true, y_pred, trait_ids, reverse_ids)` — comprehensive metrics dict with item-level, trait-level, and profile correlation
7. `simulate_real_testing(y_test, S)` — mask held-out items given selected set S

## Key design decisions
- All functions are pure (no side effects, no global state) for composability
- Trait order fixed as OCEAN (O, C, E, A, N) across the module
- `evaluate_predictions` computes three trait score types (short-form, imputed-full, held-out) per Decision 5
- Reverse scoring is applied only during trait score computation (Decision 2), not to prediction inputs
- All random operations accept explicit seed for reproducibility
- Match existing codebase conventions: numpy/scipy/sklearn, RANDOM_STATE=0, pathlib, [OK] prefixes

## Test results
- **112/112 tests pass** (0 failures)
- Smoke test with real F001 data: all 7 function checks pass
- Coverage: unit tests + real-data integration test
- Edge cases tested: empty traits, NaN handling, single subjects, constant predictions, reproducibility

## Acceptance criteria status
- AC001 ✓ — 5-fold split: 80/20 ratios, no leakage, deterministic
- AC002 ✓ — Reverse scoring: forward=y, reverse=6-y, double-reverse=identity, matches F001 Y.npy
- AC003 ✓ — Trait scores: mean per trait, (2749, 5) shape, no NaN, values in [1,5]
- AC004 ✓ — evaluate_predictions returns structured dict with item_level, trait_level, profile_correlation

## Commands run
- `python scripts/cv_framework.py` — smoke test (ALL CHECKS PASSED)
- `python scripts/test_cv_framework.py` — unit tests (112/112 passed)

## Risks / Notes
- Constant prediction edge case: if all predictions are identical, Pearson r is undefined (scipy returns NaN with ConstantInputWarning). Callers should ensure predictions have variance.
- The module assumes reverse scoring is applied AFTER prediction (Decision 2). Y.npy from F001 is already reverse-scored; if raw responses are needed, load from source CSV or use `reverse_score()` to un-reverse.
- No GPU or heavy dependencies required — pure numpy/scipy/sklearn.

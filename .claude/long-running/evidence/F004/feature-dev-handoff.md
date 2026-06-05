# Feature-dev handoff for F004

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F004
- Title: Phase 1: 随机选题策略 — Random 与 Balanced Random
- Description: 实现两种随机基线选题策略。(1) Random: 每个比例随机选题，每个 outer fold 内重复 50 次取平均。(2) Balanced Random: 按 Big Five 维度均衡随机。Random baseline 是证明策略有效的关键对照。
- Priority: high
- Dependencies: [F003]

## Acceptance criteria
1. [AC001] Random 选题在 50 次重复间题目集合不同（验证随机性）
2. [AC002] Balanced Random 每维度题数偏差 ≤1
3. [AC003] 两种方法均覆盖 10/30/50/90 四种比例
4. [AC004] 评估指标包含 item-level r 和 trait-level r（至少 Big Five mean）

## Test plan
- `python scripts/run_selection_baselines.py`
- `python -c "from selection import BalancedRandomSelector; s=BalancedRandomSelector().select(10); print(len(s), 'items selected')"`

## Technical context

### CV Framework (F003) — available imports
The CV framework module at `scripts/cv_framework.py` provides these functions:
- `participant_cv_split(n_subjects, n_folds=5, seed=0)` — returns list of `(train_idx, test_idx)` tuples
- `evaluate_predictions(y_true, y_pred, trait_ids, reverse_ids)` — returns dict with `item_level` and `trait_level` metrics
- `simulate_real_testing(y, S)` — given selected items S, returns y_seen (only S columns) and y_unseen (T columns)
- `compute_trait_scores(responses, trait_ids, reverse_ids)` — reverse score then compute per-trait means
- `compute_profile_correlation(traits_pred, traits_true)` — Big Five profile correlation

### Data files (F001) — available
- `data/processed/Y.npy` — (2749, 100) float32, raw responses 1-5
- `data/processed/E_old.npy` — (100, 1024) float32, L2-normalized SBERT embeddings
- `data/processed/metadata.parquet` — columns: item_id, item_text, trait_id (O/C/E/A/N), reverse_id (0=forward, 1=reverse)
- `data/processed/subject_ids.txt` — 2749 subject IDs

### Trait ID mapping
- O=0: Openness (items 0-19)
- C=1: Conscientiousness (items 20-39)
- E=2: Extraversion (items 40-59)
- A=3: Agreeableness (items 60-79)
- N=4: Neuroticism (items 80-99)

### Experiment ratios
- 10% (m=10 items), 30% (m=30 items), 50% (m=50 items), 90% (m=90 items)

## Implementation requirements

1. Create `scripts/selection.py` — module with:
   - `RandomSelector` class: `select(m, seed=None)` returns list of m item indices
   - `BalancedRandomSelector` class: `select(m, seed=None)` returns list of m item indices, balanced across 5 traits
   - Both should accept the trait_id mapping and number of items (100 total)

2. Create `scripts/run_selection_baselines.py` — runner script that:
   - Loads Y.npy, metadata.parquet
   - Runs 5-fold participant CV (from cv_framework)
   - For each fold, for each ratio (10, 30, 50, 90):
     - Random: 50 repeats of random selection → per-repeat predictions → average
     - Balanced Random: 50 repeats of balanced random → per-repeat predictions → average
   - Uses item-level KNN prediction (K=5, cosine distance on E_old)
   - Collects and saves item-level r and trait-level r (Big Five mean)
   - Saves results to `results/phase1/`

3. Key implementation notes:
   - FIXED random seed = 0 throughout
   - The KNN predictor should use cosine similarity on E_old embeddings
   - Per-repeat: select S items, predict remaining T items using KNN, evaluate
   - Item-level Pearson r is computed per subject then averaged
   - Trait-level: compute trait scores (reverse scored), then Pearson r per trait
   - Output CSV files with columns: strategy, ratio, fold, repeat, item_r, trait_r_O, trait_r_C, trait_r_E, trait_r_A, trait_r_N, trait_r_mean

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F004/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

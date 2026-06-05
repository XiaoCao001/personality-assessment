# Feature-dev handoff for F005

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F005
- Title: Phase 1: 语义选题策略 — Coverage 与 Coverage+Diversity
- Description: 实现基于语义 embedding 的选题策略。(1) Semantic Coverage: greedy facility-location，每次加入使 Coverage(S) 最大的题。(2) Coverage+Diversity: 在 Coverage 基础上减去 Redundancy 惩罚项，λ∈{0.25,0.5,1.0}。两个策略都只使用 embedding 空间信息，不使用被试作答数据。
- Priority: high
- Dependencies: [F003]

## Acceptance criteria
1. [AC001] Coverage 贪心选择的题目集合 Coverage 值 ≥ 随机选择的 95% 上限
2. [AC002] Coverage+Diversity 相比纯 Coverage 降低 Redundancy ≥ 10%
3. [AC003] λ 选择过程完全在 train participants 内完成
4. [AC004] 所有比例和 λ 值的结果可复现（固定 random seed）

## Implementation steps (from features.json)
1. 实现 sim+(i,j) = (cos(e_i,e_j)+1)/2 语义相似度矩阵
2. 实现 Coverage(S) = mean_j max_{i∈S} sim+(i,j)
3. 实现 greedy facility-location selection 算法
4. 实现 Redundancy(S) = mean pairwise sim+ within S
5. 实现 Coverage+Diversity: Score = Coverage_z - λ×Redundancy_z
6. 在 train participants 上用 inner validation 选最佳 λ
7. 对所有比例运行评估，保存结果

## Existing infrastructure
- `scripts/selection.py`: Contains `RandomSelector` and `BalancedRandomSelector` from F004. Add `CoverageSelector` and `CoverageDiversitySelector` to this file.
- `scripts/cv_framework.py`: Provides `participant_cv_split`, `inner_validation_split`, `reverse_score`, `compute_trait_scores`, `evaluate_predictions`, `simulate_real_testing`.
- `scripts/run_selection_baselines.py`: Full 5-fold CV pipeline from F004 — can be adapted for `run_semantic_selection.py`.
- `data/processed/E_old.npy`: Original SBERT embeddings (100×1024, L2-normalized).
- `data/processed/Y.npy`: Response matrix (2749×100, reverse-scored).
- `data/processed/metadata.parquet`: item_text, trait_id, reverse_id.

## Key design notes
- `sim+(i,j)` maps cosine similarity from [-1,1] to [0,1].
- Coverage measures how well S "covers" the full item set in semantic space.
- The greedy algorithm picks one item at a time: argmax_i Coverage(S∪{i}).
- Coverage(S) is monotone and submodular — greedy gives a (1-1/e) approximation guarantee.
- Redundancy penalizes selecting items that are too similar to each other.
- λ selection: grid search {0.25, 0.5, 1.0} on inner validation splits within train folds.
- Z-score normalization: Coverage_z and Redundancy_z are computed against all candidates in each greedy step.

## Test plan
1. `python scripts/run_semantic_selection.py`
2. `python -c "from selection import CoverageSelector; import numpy as np; e=np.load('data/processed/E_old.npy'); s=CoverageSelector(e).select(10); print(s)"`

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F005/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Evidence directory
`.claude/long-running/evidence/F005/`

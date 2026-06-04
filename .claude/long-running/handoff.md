# Handoff — Next Session

## Immediate action
Run `/long-running-coding F004` to start **随机选题策略（Random & Balanced Random）**，这是 Phase 1 的第一个选题策略实验。

也可并行推进：
- `/long-running-coding F005`（语义选题策略 Coverage & Coverage+Diversity）
- `/long-running-coding F011`（新 Embedding 模型生成，独立于 Phase 1）

## Files to read first
1. `.claude/long-running/progress.md` — 项目总览（F001/F002/F003 已完成）
2. `.claude/long-running/features.json` — 完整 15 个 feature 定义
3. `.claude/long-running/decisions.md` — 7 个关键设计决策
4. `questionnaire-embeddings/scripts/cv_framework.py` — F003 CV 框架（F004–F010 的基础模块）
5. `questionnaire-embeddings/scripts/functions.py` — 原有共享模块

## Current state
- **F001 completed** — Y.npy (2749×100), E_old.npy (100×1024), metadata.parquet
- **F002 completed** — 原文基线复现: Mean r = 0.4628 [0.4571, 0.4686]
- **F003 completed** — CV 框架: 7 functions, 112 tests, evaluator PASS
- 12 features 待处理，F004/F005 是 Phase 1 的下两个 feature
- CV 框架 API: `from cv_framework import participant_cv_split, evaluate_predictions, ...`

## F003 completion details
- **Module**: `scripts/cv_framework.py` (397 lines) — 7 public functions
- **Tests**: `scripts/test_cv_framework.py` (315 lines) — 112/112 passed
- **Functions**: participant_cv_split, inner_validation_split, reverse_score, compute_trait_scores, compute_profile_correlation, evaluate_predictions, simulate_real_testing
- **Evaluator**: PASS, 4/4 acceptance criteria met
- **AC001**: 5-fold split 80/20, no leakage, deterministic ✓
- **AC002**: Reverse scoring forward=y, reverse=6-y, verified against F001 Y.npy ✓
- **AC003**: Trait scores = mean per trait, (2749, 5), no NaN ✓
- **AC004**: evaluate_predictions returns structured dict, importable and batch-callable ✓

## Next feature: F004 — Random & Balanced Random 选题策略
- **依赖**: F001 ✓, F003 ✓
- **关键任务**: 实现 RandomSelector 和 BalancedRandomSelector，在 cv_framework 上运行评估
- **使用**: cv_framework.participant_cv_split, cv_framework.evaluate_predictions, cv_framework.simulate_real_testing
- **输出**: results/phase1/ 下选题和评估结果

## Key constraints
- 所有实验在 NEO-PI-R 数据上运行（2749 被试 × 100 题）
- 预测阶段使用 raw response (1-5)，反向计分仅在人格总分计算时处理
- T4 GPU 可用（16GB 显存）
- 务必固定 random seed (0)
- 每次只处理一个 feature，通过 `/long-running-coding <id>` 选择
- 无 git 仓库 — 无法 commit，状态仅存在 features.json/progress.md 中

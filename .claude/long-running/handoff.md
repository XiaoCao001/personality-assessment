# Handoff — Next Session

## Immediate action
Run `/long-running-coding F005` to start **语义选题策略（Coverage & Coverage+Diversity）**，这是 Phase 1 的下一个选题策略实验。

也可并行推进：
- `/long-running-coding F011`（新 Embedding 模型生成，独立于 Phase 1）

## Files to read first
1. `.claude/long-running/progress.md` — 项目总览（F001–F004 已完成）
2. `.claude/long-running/features.json` — 完整 15 个 feature 定义
3. `.claude/long-running/decisions.md` — 7 个关键设计决策
4. `questionnaire-embeddings/scripts/selection.py` — F004 Random/BalancedRandom 选择器（可参考模式）
5. `questionnaire-embeddings/scripts/run_selection_baselines.py` — F004 runner（可参考管道）
6. `questionnaire-embeddings/scripts/cv_framework.py` — F003 CV 框架
7. `questionnaire-embeddings/results/phase1/random_baseline_summary.csv` — F004 random baselines（F005 可对比）

## Current state
- **F001 completed** — Y.npy (2749×100), E_old.npy (100×1024), metadata.parquet
- **F002 completed** — 原文基线复现: Mean r = 0.4628 [0.4571, 0.4686]
- **F003 completed** — CV 框架: 7 functions, 112 tests, evaluator PASS
- **F004 completed** — 随机选题策略: BalancedRandom > Random at all ratios, evaluator PASS
- 11 features 待处理，F005/F006 是 Phase 1 的下两个 feature

## F004 completion details
- **Module**: `scripts/selection.py` (171 lines) — RandomSelector, BalancedRandomSelector
- **Runner**: `scripts/run_selection_baselines.py` (330 lines) — 5-fold CV × 4 ratios × 2 strategies × 50 repeats
- **Key results**: BalancedRandom m=30 item_r=0.2136, big5_r=0.8219; m=50 item_r=0.2852, big5_r=0.9278
- **Evaluator**: PASS, 4/4 acceptance criteria met
- **AC001**: 50/50 unique random sets ✓
- **AC002**: Balanced per-trait deviation ≤ 1 (all 0) ✓
- **AC003**: Both strategies cover 10/30/50/90 ratios ✓
- **AC004**: Output includes item_r + trait_r (O/C/E/A/N + mean) ✓

## Next feature: F005 — Coverage & Coverage+Diversity 语义选题策略
- **依赖**: F001 ✓, F003 ✓
- **关键任务**: 实现 Coverage 贪心选择、Coverage+Diversity (with λ penalty)、inner validation 选 λ
- **使用**: E_old 语义空间、cv_framework CV pipeline、selection.py 中的模式
- **输出**: results/phase1/ 下语义策略结果
- **对比基线**: F004 random baselines (results/phase1/random_baseline_summary.csv)

## Key constraints
- 所有实验在 NEO-PI-R 数据上运行（2749 被试 × 100 题）
- 预测阶段使用 raw response (1-5)，反向计分仅在人格总分计算时处理
- T4 GPU 可用（16GB 显存）
- 务必固定 random seed (0)
- 每次只处理一个 feature，通过 `/long-running-coding <id>` 选择
- 无 git 仓库 — 无法 commit，状态仅存在 features.json/progress.md 中

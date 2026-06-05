# Handoff — Next Session

## Immediate action
Run `/long-running-coding F006` to start **心理测量选题策略（Trait Predictiveness & Hybrid A/B/C）**，完成 Phase 1 的 8 种策略集合。

也可并行推进：
- `/long-running-coding F011`（新 Embedding 模型生成，独立于 Phase 1）

## Files to read first
1. `.claude/long-running/progress.md` — 项目总览（F001–F005 已完成）
2. `.claude/long-running/features.json` — 完整 15 个 feature 定义
3. `.claude/long-running/decisions.md` — 关键设计决策
4. `questionnaire-embeddings/scripts/selection.py` — F004/F005 选择器（已含 Coverage/CoverageDiversity）
5. `questionnaire-embeddings/scripts/run_selection_baselines.py` — F004 runner pattern
6. `questionnaire-embeddings/scripts/run_semantic_selection.py` — F005 runner (inner validation for λ tuning)
7. `questionnaire-embeddings/scripts/cv_framework.py` — F003 CV 框架
8. `questionnaire-embeddings/results/phase1/random_baseline_summary.csv` — F004 baselines
9. `questionnaire-embeddings/results/phase1/semantic_selection_summary.csv` — F005 results

## Current state
- **F001 completed** — Y.npy (2749×100), E_old.npy (100×1024), metadata.parquet
- **F002 completed** — 原文基线复现: Mean r = 0.4628
- **F003 completed** — CV 框架: 7 functions, evaluator PASS
- **F004 completed** — Random & BalancedRandom baselines
- **F005 completed** — Coverage & Coverage+Diversity: Coverage beats all baselines
- 10 features 待处理，F006 是 Phase 1 最后一个 strategy feature

## F005 completion details
- **Module**: `scripts/selection.py` — added CoverageSelector (+175 lines), CoverageDiversitySelector (+120 lines)
- **Runner**: `scripts/run_semantic_selection.py` (355 lines) — 5-fold CV × 4 ratios × 4 strategies with inner validation λ tuning
- **Key results**: Pure Coverage dominates — item_r = 0.084/0.256/0.304/0.484 at m=10/30/50/90
- **Coverage vs BalancedRandom**: +11%/+20%/+6%/+41% at m=10/30/50/90
- **Coverage+Diversity never beats pure Coverage** — λ penalty hurts prediction
- **Evaluator**: PASS, 4/4 acceptance criteria met (attempt 2)
- **AC001**: greedy ≥ random 95% upper bound for all 4 ratios ✓
- **AC002**: λ=1.0, m=10 → 10.2% redundancy reduction ✓
- **AC003**: λ selection via inner validation on train only ✓
- **AC004**: deterministic algorithms, RANDOM_STATE=0 ✓
- **Results**: `results/phase1/semantic_selection_{detail,aggregated,summary}.csv`
- **Design insight**: Coverage naturally favors diversity (near-duplicates don't improve coverage), so explicit redundancy penalty provides no benefit

## Next feature: F006 — Trait Predictiveness & Hybrid (A/B/C) 心理测量选题策略
- **依赖**: F001 ✓, F003 ✓, F005 ✓
- **关键任务**: corrected item-total correlation, ImbalancePenalty (维度+正反向), z-score 标准化组合, 3 种 Hybrid 变体
- **使用**: Y.npy (subject responses), E_old.npy (embeddings), cv_framework, selection.py patterns
- **输出**: results/phase1/ 下 trait/hybrid 策略结果
- **对比基线**: F004 random baselines + F005 semantic strategies

## Key constraints
- 所有实验在 NEO-PI-R 数据上运行（2749 被试 × 100 题）
- 预测阶段使用 raw response (1-5)，反向计分仅在人格总分计算时处理
- T4 GPU 可用（16GB 显存）
- 务必固定 random seed (0)
- 每次只处理一个 feature，通过 `/long-running-coding <id>` 选择

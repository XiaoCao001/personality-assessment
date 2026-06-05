# Project Progress

## Project overview
基于原论文 "A Deep Language Approach to Personality Assessment" 的 questionnaire-embeddings 代码库进行改进研究。三个核心改进方向：(1) 用语义覆盖+心理测量策略选择最具代表性的真实作答题，替代原文随机 90/10 题项划分；(2) 用加权 KNN/Softmax KNN/Kernel Smoothing 替代原文简单 KNN 预测器；(3) 用更新的本地开源 embedding 模型（MiniLM, MPNet, E5, BGE）替代原 SBERT。全程增加人格总分预测维度。主实验数据：NEO-PI-R (2749 被试 × 100 题项 × Big Five 5 维度)。

## Current status
- Phase: F001–F007 completed — Phase 1 complete (8 strategies evaluated, Coverage recommended for Phase 2)
- Last updated: 2026-06-05T06:15:00Z
- Completed features: 7 / 15
- Active feature: none

## Completed work
- 2026-06-04T16:00:00Z Harness initialized with 15 features across 5 phases.
- **2026-06-04T17:30:00Z F001 completed** — 数据准备与矩阵标准化
  - Created `scripts/prepare_data.py` (data pipeline) and `scripts/validate_data.py` (validation)
  - Outputs: Y.npy (2749×100, float32), E_old.npy (100×1024, L2-normalised), metadata.parquet (100×4), subject_ids.txt
  - Evaluator verdict: PASS — all 4 acceptance criteria met
  - Evidence: evaluator-report.json, test-output.txt, git-diff.patch, commands.log

## Current risks / blockers
- 原文 SBERT embedding 可能使用较旧的 sentence-transformers 版本，baseline 复现时需注意版本兼容性
- T4 GPU 显存有限（~16GB），E5-large 和 BGE-large 可能需要 batch 处理
- E_old SBERT embedding dim=1024 (roberta-large-nli-stsb-mean-tokens)，与 features.json 初始设计一致

## Next recommended feature
- **[F008] Phase 2: Cosine Weighted KNN 预测器** — Phase 1 done, Phase 2 starts here (depends on F003 ✓)
- [F011] Phase 3: 新 Embedding 模型生成 (depends on F001 ✓) — independently parallel
- Phase 1 全部完成 (F001–F007)，Coverage 被推荐为 Phase 2 最佳选题策略

- **2026-06-05T06:10:00Z F007 completed** — 选题策略完整评估（指标、表格与图表）
  - Created `scripts/evaluate_phase1.py` (520 lines) — Phase 1 comprehensive evaluation
  - Generated Table 1 (item-level r ± 95% CI, 8×4=32 cells), Table 2 (short-form/imputed/held-out trait r), Figure 1 (learning curve), Figure 2 (trait distribution)
  - 32 paired bootstrap tests: Coverage significantly better than Random at all ratios (p<0.001)
  - **Recommendation: Coverage is the best strategy for Phase 2** (dominates at m=10: 0.084, m=30: 0.255, m=50: 0.304, m=90: 0.484)
  - Evaluator verdict: PASS — all 4 acceptance criteria met
  - Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
  - Outputs: results/phase1/figures/{table1,table2_*.csv,figure1/2.pdf,statistical_tests.csv,recommendation.txt}

- **2026-06-05T05:25:00Z F006 completed** — 心理测量选题策略（Trait Predictiveness & Hybrid A/B/C）
  - Added TraitPredictivenessSelector and HybridSelector to `scripts/selection.py` (+350 lines)
  - Created `scripts/run_trait_hybrid_selection.py` (320 lines) — full 5-fold CV runner
  - Hybrid-C dominates at m=10/30/50 (item_r = 0.075/0.228/0.295); Hybrid-A best at m=90 (0.370)
  - Pure TraitPredictiveness worst performer (0.052/0.123/0.120/0.197) — selects from only 2-3 traits
  - All F006 strategies underperform F005 Coverage (0.084/0.256/0.304/0.484) — Coverage remains recommended
  - Evaluator verdict: PASS — all 4 acceptance criteria met (attempt 1)
  - Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
  - Results: results/phase1/trait_hybrid_selection_{detail,aggregated,summary}.csv

- **2026-06-05T04:30:00Z F004 completed** — 随机选题策略（Random & Balanced Random）
  - Created `scripts/selection.py` (171 lines) — RandomSelector + BalancedRandomSelector
  - Created `scripts/run_selection_baselines.py` (330 lines) — full 5-fold CV pipeline with vectorized cosine-distance KNN
  - Results: 5 folds × 4 ratios × 2 strategies × 50 repeats = 2000 evaluations
  - BalancedRandom consistently outperforms Random (e.g., m=30: item_r=0.2136 vs 0.1984, +7.7%)
  - At m=90, both strategies nearly ceiling (Big5 r > 0.99)
  - Evaluator verdict: PASS — all 4 acceptance criteria met
  - Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, git-diff.patch, commands.log

- **2026-06-05T04:58:00Z F005 completed** — 语义选题策略（Coverage & Coverage+Diversity）
  - Added CoverageSelector and CoverageDiversitySelector to `scripts/selection.py` (+320 lines)
  - Created `scripts/run_semantic_selection.py` (355 lines) — 5-fold CV with inner validation for λ tuning
  - Coverage uses greedy facility-location: Coverage(S) = mean_j max_{i∈S} sim+(i,j)
  - Coverage+Diversity: Score = Coverage_z − λ×Redundancy_z (λ∈{0.25,0.5,1.0})
  - Key result: Pure Coverage dominates — beats BalancedRandom by +11% at m=10, +20% at m=30, +6% at m=50, +41% at m=90
  - λ penalty never improves prediction; pure Coverage is the recommended semantic strategy
  - 2 attempts (first: NEEDS_WORK on AC001 evidence gap; second: PASS)
  - Evaluator verdict: PASS — all 4 acceptance criteria met
  - Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
  - Results: results/phase1/semantic_selection_{detail,aggregated,summary}.csv

## Session log
### 2026-06-04T16:00:00Z Initialization
- 用户提供了完整的 4 阶段实验设计方案，覆盖选题策略、预测器、embedding 和人格总分四个维度
- 将实验方案分解为 15 个可独立完成的功能特征
- 创建 features.json（通过 schema 验证）、progress.md、decisions.md、handoff.md
- 建立 evidence/ 子目录结构
- 确认 questionnaire-embeddings 项目为 git 仓库，基线完好

### 2026-06-04T17:30:00Z F001 Implementation
- 实现了两个 Python 脚本：prepare_data.py（278行）和 validate_data.py（122行）
- 数据特征：100 题项（5 traits × 20 题），2749 被试，SBERT embedding dim=1024
- Forward/reverse ratio: 50/50
- 所有输出保存到 data/processed/
- Code review 通过（3 个 code-reviewer agents：bugs/dry/conventions）
- Evaluator verdict: PASS（4/4 acceptance criteria）
- 提交: pending（等待 orchestrator 提交）

### 2026-06-04T18:20:00Z F002 Implementation — 原文基线复现
- 实现了 `scripts/run_baseline.py`（214行），严格复现原文 KNN K=5 10 折题项交叉验证
- 使用非反向计分数据（`big5_responses_nonReversed.csv`），匹配原文 `modelPerformance(R=2)` 的基准线
- 结果：Mean Pearson r = 0.4628 [0.4571, 0.4686]，接近原文 0.453 [0.447, 0.458]
- 覆盖 2747/2749 名被试（2 名因恒定预测产生 NaN r，属预期边界情况）
- Pipeline: StandardScaler → PCA(0.9, 46 PCs) → KFold(10) → KNeighborsRegressor(K=5) → round+clamp
- Evaluator verdict: PASS（4/4 acceptance criteria）
- 输出: results/baseline/original_10fold_itemcv_results.csv (2749 rows)

### 2026-06-04T19:00:00Z F003 Implementation — 外层被试级交叉验证框架
- 实现了 `scripts/cv_framework.py`（397行）— 7 个公共函数 + smoke test demo
- 实现了 `scripts/test_cv_framework.py`（315行）— 112 个单元测试，全部通过
- 7 个函数：participant_cv_split, inner_validation_split, reverse_score, compute_trait_scores, compute_profile_correlation, evaluate_predictions, simulate_real_testing
- Evaluator verdict: PASS（4/4 acceptance criteria met）
- 证据: evaluator-report.json, test-output.txt (112/112 passed), commands.log, feature-dev-summary.md
- F003 是 F004–F010 的前置依赖，现已就绪；Phase 1-2 可全面推进

### 2026-06-05T06:10:00Z F007 Implementation — Phase 1 完整评估
- 实现了 `scripts/evaluate_phase1.py`（520行）— Phase 1 汇总评估脚本
- 汇总全部 8 种策略 × 4 比例，生成 Table 1、Table 2、Figure 1、Figure 2
- 32 个 paired bootstrap tests：Coverage 在所有比例上显著优于 Random (p<0.001)
- **Phase 1 结论：Coverage 为最佳选题策略**，推荐用于 Phase 2
- Evaluator verdict: PASS（4/4 acceptance criteria）
- 输出: results/phase1/figures/ 下完整图表和统计检验

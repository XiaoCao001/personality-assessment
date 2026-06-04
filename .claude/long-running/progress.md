# Project Progress

## Project overview
基于原论文 "A Deep Language Approach to Personality Assessment" 的 questionnaire-embeddings 代码库进行改进研究。三个核心改进方向：(1) 用语义覆盖+心理测量策略选择最具代表性的真实作答题，替代原文随机 90/10 题项划分；(2) 用加权 KNN/Softmax KNN/Kernel Smoothing 替代原文简单 KNN 预测器；(3) 用更新的本地开源 embedding 模型（MiniLM, MPNet, E5, BGE）替代原 SBERT。全程增加人格总分预测维度。主实验数据：NEO-PI-R (2749 被试 × 100 题项 × Big Five 5 维度)。

## Current status
- Phase: F001, F002, F003 completed — CV framework ready
- Last updated: 2026-06-04T19:00:00Z
- Completed features: 3 / 15
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
- [F004] Phase 1: 随机选题策略（依赖 F003 ✓ + F001 ✓）— 可立即推进
- [F005] Phase 1: 语义选题策略（依赖 F003 ✓ + F001 ✓）— 可与 F004 并行
- [F011] Phase 3: 新 Embedding 模型生成（依赖 F001 ✓）— 独立可并行
- F003 完成，Phase 1 的选题策略实验 (F004–F007) 可全面推进

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

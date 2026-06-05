# Handoff — Next Session

## Immediate action
Run `/long-running-coding F010` to continue **Phase 2: 预测器消融评估 — 表格与图表** (depends on F007 ✓, F008 ✓, F009 ✓).

也可并行推进：
- `/long-running-coding F011`（新 Embedding 模型生成，depends on F001 ✓）— 独立并行

## Files to read first
1. `.claude/long-running/progress.md` — 项目总览（F001–F009 已完成）
2. `.claude/long-running/features.json` — 完整 15 个 feature 定义
3. `scripts/predictors.py` — 5 种预测器（UniformKNN, CosineWeightedKNN, SoftmaxKNN, KernelSmoothing）
4. `scripts/run_weighted_knn.py` — F008 runner, inner validation 调参范式
5. `scripts/run_softmax_kernel.py` — F009 runner
6. `scripts/selection.py` — 8 种选择器
7. `scripts/cv_framework.py` — F003 CV 框架
8. `results/phase1/figures/phase1_recommendation.txt` — Phase 1 推荐
9. `results/phase2/weighted_knn_summary.csv` — F008 结果
10. `results/phase2/softmax_kernel_summary.csv` — F009 结果

## Current state
- **F001–F009 completed** — Phase 1 全部完成，Phase 2 前两个 feature 完成
- **F008 completed** — CosineWeightedKNN 显著优于 UniformKNN
- **F009 completed** — SoftmaxKNN 在所有比例上大幅优于所有其他预测器

## F009 key findings
- **SoftmaxKNN is the best predictor so far**, dominating at all ratios:
  - m=10: item_r=0.2587 vs CosineWeightedKNN 0.1511 (+71%)
  - m=30: item_r=0.3542 vs CosineWeightedKNN 0.2884 (+23%)
  - m=50: item_r=0.4037 vs CosineWeightedKNN 0.3422 (+18%)
  - m=90: item_r=0.5995 vs CosineWeightedKNN 0.5583 (+7%)
- Best params: K=7, τ=0.1 (m=10/30); K=10, τ=0.1 (m=50); K=3, τ=0.038 (m=90)
- KernelSmoothing slightly behind SoftmaxKNN but still much better than F008 predictors
- τ sensitivity confirmed: bell-shaped curve, τ=0.1 is sweet spot

## Predictor ranking (mean item_r across m=10,30,50,90)
1. **SoftmaxKNN (F009)**: 0.4040
2. **KernelSmoothing (F009)**: 0.3958
3. **CosineWeightedKNN (F008)**: 0.3350
4. **UniformKNN (F008)**: 0.3274

## Phase 1 final results (F007)
- **Coverage** is the recommended strategy for Phase 2
- Full ranking: Coverage > Coverage+Div > Hybrid-C > Hybrid-A > BalancedRandom > Hybrid-B > Random > TraitPredictiveness
- Outputs: results/phase1/figures/{table1,table2_*.csv,figure1/2.pdf,statistical_tests.csv,recommendation.txt}

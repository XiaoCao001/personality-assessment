# Handoff — Next Session

## Immediate action
Run `/long-running-coding F011` to continue **Phase 3: 新 Embedding 模型生成** (depends on F001 ✓).

Phase 2 is now complete (F010 PASS). F011 is the next recommended feature — independently parallel.

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
- **F001–F010 completed** — Phase 1 全部完成，Phase 2 全部完成
- **F010 completed** — Phase 2 predictor ablation: SoftmaxKNN 为最佳预测器，Phase 4 推荐已就绪
- **Next**: F011 (Phase 3: 新 Embedding 模型生成, depends on F001 ✓)

## F010 key findings (Phase 2 Final)
- **Phase 2 预测器消融评估完成** — Table 3, Figure 3, statistical tests, recommendation 全部输出
- 4 个预测器 × 4 种比例，AC004 验证通过（F008/F009 使用相同 Coverage S）
- Predictor ranking (mean item_r across m=10,30,50,90):
  1. **SoftmaxKNN**: 0.4040
  2. **KernelSmoothing**: 0.3958
  3. **CosineWeightedKNN**: 0.3350
  4. **UniformKNN**: 0.3274
- SoftmaxKNN significantly better than UniformKNN at all ratios (p<0.001); CosWeightedKNN ns at m=90
- SoftmaxKNN vs KernelSmoothing: ns at m=90 (p=0.697) but significant at m=10/30/50
- Recommended Phase 4 params: SoftmaxKNN K=7 τ=0.1 (low m), K=10 τ=0.1 (mid m), KernelSmoothing τ≈0.034 (high m)
- Output: results/phase2/figures/{table3_predictor_ablation.csv, figure3_delta_r.{pdf,png}, statistical_tests_phase2.csv, phase2_recommendation.txt}

## Phase 1 final results (F007)
- **Coverage** is the recommended strategy for Phase 2
- Full ranking: Coverage > Coverage+Div > Hybrid-C > Hybrid-A > BalancedRandom > Hybrid-B > Random > TraitPredictiveness
- Outputs: results/phase1/figures/{table1,table2_*.csv,figure1/2.pdf,statistical_tests.csv,recommendation.txt}

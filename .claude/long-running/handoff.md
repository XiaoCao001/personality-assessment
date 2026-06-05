# Handoff — Next Session

## Immediate action
Run `/long-running-coding F009` to continue **Phase 2: Softmax Weighted KNN & Kernel Smoothing** (depends on F003 ✓, F008 ✓).

也可并行推进：
- `/long-running-coding F011`（新 Embedding 模型生成，depends on F001 ✓）— 独立并行

## Files to read first
1. `.claude/long-running/progress.md` — 项目总览（F001–F008 已完成）
2. `.claude/long-running/features.json` — 完整 15 个 feature 定义
3. `questionnaire-embeddings/scripts/predictors.py` — F008 预测器模块（UniformKNN + CosineWeightedKNN）
4. `questionnaire-embeddings/scripts/run_weighted_knn.py` — F008 runner（inner validation 调参范式）
5. `questionnaire-embeddings/scripts/selection.py` — 8 种选择器
6. `questionnaire-embeddings/scripts/cv_framework.py` — F003 CV 框架
7. `questionnaire-embeddings/results/phase1/figures/phase1_recommendation.txt` — Phase 1 推荐
8. `questionnaire-embeddings/results/phase2/weighted_knn_summary.csv` — F008 结果

## Current state
- **F001–F008 completed** — Phase 1 全部完成，Phase 2 首个 feature 完成
- **F008 completed** — CosineWeightedKNN 显著优于 UniformKNN (m=10: +21%, p=0.022)
- **Next**: F009 (Softmax KNN + Kernel Smoothing) or F011 (Phase 3 embedding)

## F008 key findings
- CosineWeightedKNN vs UniformKNN:
  - m=10: Δr=+0.0265 (+21%), p=0.022 *
  - m=30: Δr=+0.0028, p<0.001 ***
  - m=50: Δr=+0.0011, p<0.001 ***
  - m=90: Δr=+0.0001, p=0.678 ns
- Best K: CosineWeightedKNN=10 at m=10 (weighting allows more neighbours), K=3 at higher ratios
- UniformKNN always K=3
- Recommendation: Use weighted KNN for Phase 4, especially at low item counts

## Phase 1 final results (F007)
- **Coverage** is the recommended strategy for Phase 2
- Full ranking: Coverage > Coverage+Div > Hybrid-C > Hybrid-A > BalancedRandom > Hybrid-B > Random > TraitPredictiveness
- Outputs: results/phase1/figures/{table1,table2_*.csv,figure1/2.pdf,statistical_tests.csv,recommendation.txt}

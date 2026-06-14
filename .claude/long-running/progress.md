# Project Progress

## Project overview
基于原论文 "A Deep Language Approach to Personality Assessment" 的 questionnaire-embeddings 代码库进行改进研究。三个核心改进方向：(1) 用语义覆盖+心理测量策略选择最具代表性的真实作答题，替代原文随机 90/10 题项划分；(2) 用加权 KNN/Softmax KNN/Kernel Smoothing 替代原文简单 KNN 预测器；(3) 用更新的本地开源 embedding 模型（MiniLM, MPNet, E5, BGE）替代原 SBERT。全程增加人格总分预测维度。主实验数据：NEO-PI-R (2749 被试 × 100 题项 × Big Five 5 维度)。

## Current status
- Phase: Phase 1 全部完成 (F001–F007)，Phase 2 全部完成 (F008–F010, F016)，Phase 3 F011–F012 完成，Phase 4 F013–F015 全部完成
- Last updated: 2026-06-14T07:19:11Z
- Completed features: 16 / 16
- Active feature: none

## Completed work

### F001 — Phase 0: 数据准备与矩阵标准化 (completed 2026-06-04T17:30:00Z)
- Created `scripts/prepare_data.py` (data pipeline) and `scripts/validate_data.py` (validation)
- Outputs: Y.npy (2749×100, float32), E_old.npy (100×1024, L2-normalised), metadata.parquet (100×4), subject_ids.txt
- Evaluator verdict: PASS — all 4 acceptance criteria met
- Evidence: evaluator-report.json, test-output.txt, git-diff.patch, commands.log

### F002 — Phase 0: 原文基线复现 — 10 折题项交叉验证 (completed 2026-06-04T18:20:00Z)
- 实现了 `scripts/run_baseline.py`（214行），严格复现原文 KNN K=5 10 折题项交叉验证
- 结果：Mean Pearson r = 0.4628 [0.4571, 0.4686]，接近原文 0.453 [0.447, 0.458]
- Pipeline: StandardScaler → PCA(0.9, 46 PCs) → KFold(10) → KNeighborsRegressor(K=5) → round+clamp
- Evaluator verdict: PASS — all 4 acceptance criteria met
- Evidence: evaluator-report.json, test-output.txt, git-diff.patch, commands.log

### F003 — Phase 1: 外层被试级交叉验证框架 (completed 2026-06-04T19:00:00Z)
- 实现了 `scripts/cv_framework.py`（397行）— 7 个公共函数 + smoke test demo
- 实现了 `scripts/test_cv_framework.py`（315行）— 112 个单元测试，全部通过
- 7 个函数：participant_cv_split, inner_validation_split, reverse_score, compute_trait_scores, compute_profile_correlation, evaluate_predictions, simulate_real_testing
- Evaluator verdict: PASS — all 4 acceptance criteria met
- Evidence: evaluator-report.json, test-output.txt (112/112 passed), commands.log, feature-dev-summary.md
- F003 是 F004–F010 的前置依赖，现已就绪

### F004 — Phase 1: 随机选题策略 — Random 与 Balanced Random (completed 2026-06-05T04:30:00Z)
- Created `scripts/selection.py` (171 lines) — RandomSelector + BalancedRandomSelector
- Created `scripts/run_selection_baselines.py` (330 lines) — full 5-fold CV pipeline with vectorized cosine-distance KNN
- Results: 5 folds × 4 ratios × 2 strategies × 50 repeats = 2000 evaluations
- BalancedRandom consistently outperforms Random (e.g., m=30: item_r=0.2136 vs 0.1984, +7.7%)
- At m=90, both strategies nearly ceiling (Big5 r > 0.99)
- Evaluator verdict: PASS — all 4 acceptance criteria met
- Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, git-diff.patch, commands.log

### F005 — Phase 1: 语义选题策略 — Coverage 与 Coverage+Diversity (completed 2026-06-05T04:58:00Z)
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

### F006 — Phase 1: 心理测量选题策略 — Trait Predictiveness 与 Hybrid A/B/C (completed 2026-06-05T05:25:00Z)
- Added TraitPredictivenessSelector and HybridSelector to `scripts/selection.py` (+350 lines)
- Created `scripts/run_trait_hybrid_selection.py` (320 lines) — full 5-fold CV runner
- Hybrid-C dominates at m=10/30/50 (item_r = 0.075/0.228/0.295); Hybrid-A best at m=90 (0.370)
- Pure TraitPredictiveness worst performer (0.052/0.123/0.120/0.197) — selects from only 2-3 traits
- All F006 strategies underperform F005 Coverage — Coverage remains recommended
- Evaluator verdict: PASS — all 4 acceptance criteria met (attempt 1)
- Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- Results: results/phase1/trait_hybrid_selection_{detail,aggregated,summary}.csv

### F007 — Phase 1: 选题策略完整评估 — 指标、表格与图表 (completed 2026-06-05T06:10:00Z)
- Created `scripts/evaluate_phase1.py` (520 lines) — Phase 1 comprehensive evaluation
- Generated Table 1 (item-level r ± 95% CI, 8 strategies × 4 ratios = 32 cells)
- Generated Table 2 (short-form / imputed / held-out trait-level r, 3 sub-tables)
- Generated Figure 1 (learning curve with 95% CI error bands) and Figure 2 (trait distribution bar chart)
- 32 paired bootstrap tests: Coverage significantly better than Random at all ratios (p<0.001)
- Coverage significantly better than BalancedRandom at all ratios (p<0.05)
- **Recommendation: Coverage is the best strategy for Phase 2**
- Evaluator verdict: PASS — all 4 acceptance criteria met (attempt 1)
- Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- Outputs: results/phase1/figures/{table1,table2_*.csv,figure1/2.pdf,statistical_tests.csv,recommendation.txt}

### F008 — Phase 2: Cosine Weighted KNN 预测器 (completed 2026-06-05T07:25:00Z)
- Created `scripts/predictors.py`（298 行）— 统一的向量化 KNN 预测模块：Tuned UniformKNN K=3 + CosineWeightedKNN
- Created `scripts/run_weighted_knn.py`（480 行）— 5-fold CV + Coverage 选题 + inner validation K 调优
- CosineWeightedKNN 权重公式：w_ij = sim+(i,j) = (cos(e_i,e_j)+1)/2
- K 调优范围 {3,5,7,10,15}，per predictor/per fold/ratio，在 train-inner 80/20 上选择
- Tuned UniformKNN inner validation 在所有 fold/ratio 均选中 K=3（非原文 fixed K=5）
- 关键结果：m=10 时 CosineWeightedKNN 显著优于 Tuned UniformKNN（Δr=+0.0265, p=0.022, +21%）
- m=30（Δr=+0.0028, p<0.001）、m=50（Δr=+0.0011, p<0.001）也有统计显著优势
- m=90 差距可忽略（Δr=+0.0001, p=0.678）— 题量充足时加权无额外收益
- 最佳 K：CosineWeightedKNN 在 m=10 用 K=10（权重使更多邻居可行），更高比例用 K=3；Tuned UniformKNN 始终选择 K=3
- Evaluator verdict: PASS — all 4 acceptance criteria met（attempt 1）
- Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- Outputs: results/phase2/weighted_knn_{detail,aggregated,summary}.csv
- 建议：F009 SoftmaxKNN 已全面超越 CosineWeightedKNN 和 Tuned UniformKNN K=3，Phase 4 应优先使用 SoftmaxKNN

### F009 — Phase 2: Softmax Weighted KNN 与 Kernel Smoothing 预测器 (completed 2026-06-05T08:15:00Z)
- Refactored `scripts/predictors.py`（+140 行）— extracted `_weighted_average()` shared helper；新增 SoftmaxKNN 和 KernelSmoothing
- SoftmaxKNN：softmax 归一化权重 + 温度 τ，网格 K∈{3,5,7,10,15} × τ∈{0.03,0.05,0.1,0.2,0.5}
- KernelSmoothing：Nadaraya-Watson kernel regression，使用全部 |S| 题项，无需 K 参数
- Created `scripts/run_softmax_kernel.py`（430 行）— 5-fold CV + Coverage 选题 + inner validation K×τ 网格搜索
- **SoftmaxKNN 在所有比例上大幅领先 F008 CosineWeightedKNN**：
  - m=10: item_r=0.2587 vs 0.1511（+71%）、m=30: 0.3542 vs 0.2884（+23%）
  - m=50: 0.4037 vs 0.3422（+18%）、m=90: 0.5995 vs 0.5583（+7%）
- KernelSmoothing 紧随其后（m=10: 0.2571, m=30: 0.3368, m=50: 0.3893, m=90: 0.6001）
- τ 敏感度确认：τ=0.1 为低题量最优（bell-shaped curve）；τ→0 需更大 K，τ→∞ 趋近 uniform
- 最佳参数：SoftmaxKNN K=7, τ=0.1 (m=10/30)；K=10, τ=0.1 (m=50)；K=3, τ≈0.035 (m=90)
- Evaluator verdict: PASS — all 4 acceptance criteria met（attempt 1）
- Evidence: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- Outputs: results/phase2/softmax_kernel_{detail,aggregated,summary,sensitivity}.csv
- **建议：Phase 4 所有比例使用 SoftmaxKNN（低题量 K=7, τ=0.1；高题量 K=3, τ≈0.035）**

### F011 — 第三阶段：新嵌入模型生成（2026-06-12T06:20:00Z 完成）
- 实现了 `scripts/generate_embeddings.py`：统一的新嵌入生成脚本，支持模型清单、自动选择运行设备（优先使用 Tesla T4 上的 CUDA，必要时回退到 CPU）、指定模型、覆盖已有文件，以及纯本地校验模式。
- 基于 `data/processed/metadata.parquet` 中固定的 100 道 NEO-PI-R 题项顺序，生成并校验了 4 个新嵌入矩阵，均已逐行 L2 归一化：
  - `neo_minilm_l6_v2.npy`：all-MiniLM-L6-v2，矩阵形状为 (100,384)
  - `neo_mpnet_base_v2.npy`：all-mpnet-base-v2，矩阵形状为 (100,768)
  - `neo_e5_base_v2.npy`：intfloat/e5-base-v2，矩阵形状为 (100,768)，题项文本使用 `query: ` 前缀
  - `neo_bge_base_en_v15.npy`：BAAI/bge-base-en-v1.5，矩阵形状为 (100,768)
- 新增 `embeddings/neo_embeddings_metadata.json` 综合元数据文件，记录模型名称、维度、池化方式、软件包版本、运行设备、文件 SHA256，以及题项编号和题项文本顺序的来源校验哈希。
- `--validate` 校验模式不加载 SentenceTransformer、不加载模型、不触发 Hugging Face 下载；离线环境校验（`HF_HUB_OFFLINE=1`）通过。
- 更新 `questionnaire.yaml`，加入 F011 所需依赖：pyarrow、sentence-transformers、torch。
- 独立验收结论：通过；4 条验收标准全部满足（第 1 次尝试）。
- 证据文件：evaluator-report.json、generation-output.txt、test-output.txt、offline-validate-output.txt、artifact-checks.txt、feature-dev-summary.md、commands.log。

### F012 — 第三阶段：嵌入空间质量诊断（2026-06-12T07:10:13Z 完成）
- 实现了 `scripts/diagnose_embeddings.py`：独立的第三阶段诊断脚本，用统一的 5 模型清单比较原始 SBERT、MiniLM、MPNet、E5 和 BGE 嵌入空间。
- 对每个嵌入模型分别使用 `CoverageSelector` 重新选择题项集合 S，在 m∈{10,30,50,90} 下计算已选题集合诊断指标：`coverage_shifted_cosine` 和 `redundancy_shifted_cosine`，相似度尺度为 sim+ = (cos+1)/2。
- 对完整 100 题空间计算全局诊断指标：全部题项对的原始余弦均值/标准差、同维度题项原始余弦、跨维度题项原始余弦、同维度减跨维度的原始余弦差值，并额外记录完整 100 题的 Coverage/Redundancy；完整 100 题指标只进入表格和文字总结，不作为 Figure 5 的曲线点。
- 生成 Figure 5（2×2 面板）：Coverage 曲线、Redundancy 曲线、同维度与跨维度原始余弦对比、维度分离差值。
- 输出文件：`results/phase3/embedding_diagnostics_{selected_sets,global_space,summary}.csv`、`results/phase3/figures/figure5.{pdf,png}`、`results/phase3/figures/phase3_embedding_diagnostics.txt`。
- 关键诊断假设：E5-base-v2 在已选题集合 Coverage 上最高；MPNet-base-v2 的同维度减跨维度原始余弦差值最大；第四阶段版本 A/B 需要进一步验证这些空间结构差异是否能转化为预测性能收益。
- 独立验收结论：通过；3 条验收标准全部满足（第 1 次尝试）。
- 证据文件：evaluator-report.json、test-output.txt、artifact-checks.txt、feature-dev-summary.md、commands.log、git-diff.patch。

### F013 — 第四阶段：Embedding 对比实验版本 A1/A2（固定原选题）（2026-06-12T09:45:00Z 完成）
- 新增 `scripts/phase4_common.py`：集中管理 Phase 4 数据加载、F011/F012 风格 embedding registry、按 fold×ratio 读取 Phase 1 Coverage `S_old`、participant-level paired bootstrap、Holm/BH p 值校正和输出路径。
- 新增 `scripts/phase4_predictors.py`：实现 Phase 4 专用 `ContinuousSoftmaxKNN`，保持 Phase 2 SoftmaxKNN 邻居几何和 softmax 权重，但主分析只 clip 到 `[1,5]`，不 round；历史 Phase 1/2 predictor 行为未修改。
- 新增 `scripts/run_phase4_versionA.py`：完成 A1/A2 固定原选题实验，覆盖 5 个 embedding（SBERT original、MiniLM、MPNet、E5、BGE）× 4 个比例 × 5 个 outer folds × 2 个版本。
- A1 固定预注册 SoftmaxKNN 参数：m=10 K=7 τ=0.1；m=30 K=7 τ=0.1；m=50 K=10 τ=0.1；m=90 K=3 τ=0.03，不做 embedding-specific tuning。
- A2 在每个 embedding/fold/ratio 内仅用 outer-train 的 train-inner/valid-inner 进行 K/τ 网格搜索（K∈{3,5,7,10,15}，τ∈{0.03,0.05,0.1,0.2,0.5}），outer-test 只用于最终评价。
- 主评分明确复用 Phase 2 可比性约定：只预测并评分 unselected/held-out items；selected items 作为 observed inputs 保留，预测矩阵中为 NaN。
- 输出 `results/phase4/versionA_predictions.parquet`（109,960 participant-level rows，含 heldout/predicted item IDs、y_true、continuous y_pred 向量）、`versionA_participant_metrics.csv`、`versionA_results.csv`（200 rows）、`versionA_summary.csv`（40 rows）、`hyperparameters_by_fold_ratio_embedding.csv`（200 rows）、`selected_items_by_fold_ratio_embedding.json`（200 rows）和 `versionA_statistical_tests.csv`（32 paired bootstrap comparisons）。
- 新 embedding vs SBERT original 比较使用 subject-level item_r 配对差值，并在每个 outer fold 内重采样 subjects；输出 Δ、95% CI、raw p、Holm p、BH p。
- 新增 `scripts/test_phase4_versionA.py` 作为轻量 smoke test，默认写入 `results/phase4_smoke`，避免覆盖 canonical full-run artifacts。
- 独立验收结论：通过（6/6 条验收标准，第 1 次尝试）。
- 证据文件：evaluator-report.json、test-output.txt、smoke-test-output.txt、artifact-checks.txt、feature-dev-summary.md、commands.log、git-diff.patch。

### F014 — 第四阶段：Embedding 对比实验版本 B1/B2（重新选题）（2026-06-12T16:39:22Z 完成）
- 新增 `scripts/run_phase4_versionB.py`：对 5 个 embedding 分别用 `CoverageSelector` 重新选题 `S_new`，并在与 F013 相同的 outer folds 和 held-out-only continuous clip-only 评分约定下运行 B1/B2。
- B1 使用固定 Phase 2 SoftmaxKNN 参数；B2 在每个 embedding/fold/ratio 内仅用 outer-train 的 train-inner/valid-inner 调 K/τ。
- 新增 `scripts/test_phase4_versionB.py`：hermetic smoke test，先生成 Version A quick fixture，再运行 Version B smoke，验证预测向量、Jaccard、B−A join、B1/B2 hyperparameter flags 和 Figure 4/Table 4。
- F014 canonical artifacts 使用 `versionB_*` 命名，避免覆盖 F013 generic audit artifacts：`versionB_predictions.parquet`、`versionB_participant_metrics.csv`、`versionB_results.csv`、`versionB_summary.csv`、`versionB_hyperparameters_by_fold_ratio_embedding.csv`、`versionB_selected_items_by_fold_ratio_embedding.json`、`versionB_statistical_tests.csv`、`versionB_selection_contribution.csv`、`versionB_selection_overlap.csv`。
- Full run 成功完成：109,960 prediction/participant rows，200 fold rows，200 hyperparameter rows，40 summary rows，32 vs-SBERT bootstrap tests，40 B−A selection contribution tests，100 Jaccard overlap rows。
- `versionB_selection_overlap.csv` 显示新 embedding 的 `S_new` 与 `S_old` 明显不同（例如低题量 Jaccard 约 0.11–0.25）；SBERT 在 m=30/50 与历史 S_old 有轻微 tie/numerical drift（Jaccard 0.935/0.923），已记录为解释 caveat。
- `figures/table4.csv` 汇总 A1/A2/B1/B2 × embedding × ratio 的主指标、关键次指标、vs-SBERT bootstrap 和 B−A selection contribution；`figures/figure4.pdf/png` 以 fixed/tuned 两面板展示 A/B learning curve，并用 on-canvas legend/annotation 区分 embedding 与 A 固定 S_old / B 重新选题。
- 独立验收结论：PASS（6/6 条验收标准）。
- 证据文件：evaluator-report.json、test-output.txt、smoke-test-output.txt、artifact-checks.txt、feature-dev-summary.md、commands.log、git-diff.patch。

### F015 — Phase 4: 最终综合分析与论文级输出（completed 2026-06-14T07:19:11Z）
- 新增 `scripts/generate_final_report.py`：确定性聚合已验收的 Phase 1–4 artifacts，不重新运行实验、不重新生成 embedding、不重新估计 bootstrap。
- 新增 `scripts/test_generate_final_report.py`：真实运行 final-report generator 到 fresh smoke 输出目录，并检查输出文件、schema、phase coverage、A1/A2/B1/B2 覆盖、PDF 签名/大小、manifest provenance、limitation 文本和 recommended-pipeline 规范化字段。
- 生成 canonical `results/final_report/` 包：`final_report.pdf`（真实 Matplotlib PdfPages 生成）、`final_report.md`、`final_report.txt`、`final_summary.csv`、`statistical_summary.csv`、`best_pipeline_table.csv`、`report_manifest.json`。
- 新增 Phase 4 综合产物：`figures/table5_phase4_integrated_synthesis.csv` 和 `figures/figure6_phase4_selection_vs_performance.pdf/png`，联合/校验 `table4.csv`、`versionB_selection_contribution.csv`、`versionB_selection_overlap.csv`。
- 最终报告明确三层贡献：Phase 1 选题策略、Phase 2 预测算法、Phase 3 embedding 空间诊断，并在 Phase 4 区分 A1/A2 固定 `S_old` 的 embedding 邻居几何/调参贡献与 B1/B2 重新选题 `S_new` 的 selection contribution。
- Best pipeline table 对 10/30/50/90% 分别给出 embedding、Coverage 选题策略、SoftmaxKNN、K/τ、fixed/tuned 策略和预期 item_r/trait_r/profile_r/MAE。
- 统计说明固定主指标为 item-level Pearson r，关键次指标为 `trait_r_mean`、`profile_r`、MAE；保留 paired bootstrap、Holm 和 BH 校正 p 值。
- 可复现性说明覆盖 random state、outer-fold pairing、inner tuning、continuous clip-only 主分析以及 rounded supplemental metrics。
- 未新增跨问卷泛化实验；报告明确 limitation：当前结论主要基于 NEO-PI-R，cross-questionnaire generalization remains to be tested。
- 独立 evaluator verdict: PASS（6/6 acceptance criteria）。
- 证据文件：evaluator-report.json、generate-output.txt、test-output.txt、artifact-checks.txt、feature-dev-summary.md、commands.log、git-diff.patch。

## Phase 1 final ranking (by mean item_r across m=10,30,50,90)
1. **Coverage** (F005): 0.2818
2. **Coverage+Div** (F005): 0.2620
3. **Hybrid-C** (F006): 0.2332
4. **Hybrid-A** (F006): 0.2301
5. **BalancedRandom** (F004): 0.2294
6. **Hybrid-B** (F006): 0.2262
7. **Random** (F004): 0.2163
8. **TraitPredictiveness** (F006): 0.1228

## Phase 2 predictor ranking (by mean item_r across m=10,30,50,90)
1. **SoftmaxKNN** (F009): 0.4040
2. **KernelSmoothing** (F009): 0.3958
3. **CosineWeightedKNN** (F008): 0.3350
4. **Tuned UniformKNN K=3** (F008): 0.3274
5. **UniformKNN K=5 (原文 baseline)**: 0.2818 (from Phase 1 Coverage K=5)

> **Note (F016):** Phase 2 uses inner-validation tuned K=3 for UniformKNN (renamed "Tuned UniformKNN"), not the original paper's fixed K=5. The original K=5 baseline values are from Phase 1 Coverage K=5.

## Current risks / blockers
- 原文 SBERT embedding 可能使用较旧的 sentence-transformers 版本，baseline 复现时需注意版本兼容性
- T4 GPU 显存有限（~16GB），E5-large 和 BGE-large 可能需要 batch 处理
- E_old SBERT embedding dim=1024 (roberta-large-nli-stsb-mean-tokens)，与 features.json 初始设计一致

## Next recommended feature
- 所有 16 个 feature 均已完成。若继续研究，建议人工审阅 `questionnaire-embeddings/results/final_report/final_report.pdf` 并决定是否开展跨问卷泛化实验。

## Session log
### 2026-06-04T16:00:00Z Initialization
- 用户提供了完整的 4 阶段实验设计方案，覆盖选题策略、预测器、embedding 和人格总分四个维度
- 将实验方案分解为 15 个可独立完成的功能特征
- 创建 features.json（通过 schema 验证）、progress.md、decisions.md、handoff.md
- 建立 evidence/ 子目录结构
- 确认 questionnaire-embeddings 项目为 git 仓库，基线完好

### 2026-06-04T17:30:00Z F001 Implementation — 数据准备与矩阵标准化
- 实现了两个 Python 脚本：prepare_data.py（278行）和 validate_data.py（122行）
- 数据特征：100 题项（5 traits × 20 题），2749 被试，SBERT embedding dim=1024
- Forward/reverse ratio: 50/50
- 所有输出保存到 data/processed/
- Evaluator verdict: PASS（4/4 acceptance criteria）

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

### 2026-06-05T04:30:00Z F004 Implementation — 随机选题策略
- 实现了 RandomSelector 和 BalancedRandomSelector（`scripts/selection.py`，171 行）
- 实现了 `scripts/run_selection_baselines.py`（330 行）— 5-fold CV × 4 ratios × 2 strategies × 50 repeats
- 2000 次评估完整运行；BalancedRandom 在所有比例上稳定优于 Random (+7.7% at m=30)
- Evaluator verdict: PASS（4/4 acceptance criteria）
- 输出: results/phase1/random_baseline_{detail,aggregated,summary}.csv

### 2026-06-05T04:58:00Z F005 Implementation — 语义选题策略
- 实现了 CoverageSelector 和 CoverageDiversitySelector（`scripts/selection.py`，+320 行）
- 实现了 `scripts/run_semantic_selection.py`（355 行）— 5-fold CV + inner validation λ 调优
- Coverage 在所有比例上远优于所有 baseline；λ penalty 从不提升预测性能
- 第 1 次尝试 NEEDS_WORK（AC001 evidence gap），第 2 次 PASS
- Evaluator verdict: PASS（4/4 acceptance criteria）
- 输出: results/phase1/semantic_selection_{detail,aggregated,summary}.csv

### 2026-06-05T05:25:00Z F006 Implementation — 心理测量选题策略
- 实现了 TraitPredictivenessSelector 和 HybridSelector（`scripts/selection.py`，+350 行）
- 实现了 `scripts/run_trait_hybrid_selection.py`（320 行）— 5-fold CV × 4 ratios × 4 strategies
- Hybrid-C 在低题量下平衡最佳；TraitPredictiveness 最差（仅从 2-3 个 trait 选题）
- 所有 F006 策略均不如 F005 Coverage
- Evaluator verdict: PASS（4/4 acceptance criteria, attempt 1）
- 输出: results/phase1/trait_hybrid_selection_{detail,aggregated,summary}.csv

### 2026-06-05T06:10:00Z F007 Implementation — Phase 1 完整评估
- 实现了 `scripts/evaluate_phase1.py`（520 行）— Phase 1 汇总评估脚本
- 汇总全部 8 种策略 × 4 比例，生成 Table 1（item-level r ± CI）、Table 2（short-form/imputed/held-out trait r）
- Figure 1（learning curve)、Figure 2（trait distribution bar chart）
- 32 个 paired bootstrap tests：Coverage 在所有比例上显著优于 Random (p<0.001)
- **Phase 1 结论：Coverage 为最佳选题策略**，推荐用于 Phase 2
- F005/F006 CSV 数据曾被截断，运行完整脚本重新生成（80 行 detail + 16 行 aggregated/summary）
- Evaluator verdict: PASS（4/4 acceptance criteria）
- 输出: results/phase1/figures/ 下完整图表和统计检验

### 2026-06-05T07:25:00Z F008 Implementation — Weighted KNN Predictor
- 实现了 `scripts/predictors.py`（298行）— Tuned UniformKNN K=3 + CosineWeightedKNN + 共享向量化预测 pipeline
- 实现了 `scripts/run_weighted_knn.py`（480行）— 5-fold CV + Coverage 选题 + inner validation K 调参
- CosineWeightedKNN 在 m=10 显著优于 Tuned UniformKNN K=3（Δ=+0.0265, p=0.022, +21%）
- m=30 和 m=50 也有统计显著优势（p<0.001）
- 最佳 K：CosineWeightedKNN 在 m=10 用 K=10（权重让更多邻居可行），更高比例用 K=3
- Tuned UniformKNN K=3 在所有 fold/ratio 均被 inner validation 选中
- Evaluator verdict: PASS（4/4 acceptance criteria, attempt 1）
- 证据: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- 输出: results/phase2/weighted_knn_{detail,aggregated,summary}.csv
- 建议: Phase 4 低题量场景使用 CosineWeightedKNN

### 2026-06-05T08:15:00Z F009 Implementation — Softmax KNN & Kernel Smoothing
- Refactored `scripts/predictors.py`（+140 行）— extracted `_weighted_average()` helper；新增 SoftmaxKNN 和 KernelSmoothing
- SoftmaxKNN：softmax 归一化权重 + 温度 τ；KernelSmoothing：Nadaraya-Watson kernel regression（全部 |S| 题项）
- Created `scripts/run_softmax_kernel.py`（430 行）— 5-fold CV + Coverage 选题 + inner validation K×τ 网格搜索
- **SoftmaxKNN 在所有比例上大幅领先 F008 CosineWeightedKNN**：
  - m=10: 0.2587 vs 0.1511 (+71%)、m=30: 0.3542 vs 0.2884 (+23%)
  - m=50: 0.4037 vs 0.3422 (+18%)、m=90: 0.5995 vs 0.5583 (+7%)
- KernelSmoothing 紧随其后；τ=0.1 为低题量最优（bell-shaped 敏感度曲线确认）
- Evaluator verdict: PASS（4/4 acceptance criteria, attempt 1）
- 证据: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- 输出: results/phase2/softmax_kernel_{detail,aggregated,summary,sensitivity}.csv
- 建议: Phase 4 所有比例使用 SoftmaxKNN（低题量 K=7 τ=0.1；高题量 K=3 τ≈0.035）

### 2026-06-05T08:35:00Z F010 Implementation — Phase 2 预测器消融评估
- Created `scripts/evaluate_phase2.py`（340 行）— Phase 2 predictor ablation evaluation
- 读取已有 F008/F009 detail CSV，不重新跑实验
- 生成 Table 3（4 个预测器 × 4 种比例 + Phase 1 K=5 baseline，item_r ± 95% CI, item_mae, trait_r_mean, profile_r）
- 生成 Figure 3（Δr bar chart，加权方法 vs Tuned UniformKNN K=3，分比例展示 95% CI）
- AC004 验证通过：F008/F009 使用相同 Coverage S 集合，公平对照
- 20 个 paired bootstrap tests（N=10,000），含 3 组比较
- **Phase 2 最终结论：SoftmaxKNN 为最佳预测器**（mean item_r = 0.4040）
- Predictor ranking: SoftmaxKNN (0.4040) > KernelSmoothing (0.3958) > CosineWeightedKNN (0.3350) > Tuned UniformKNN K=3 (0.3274)
- SoftmaxKNN 在所有比例上显著优于 Tuned UniformKNN K=3（p<0.001），CI 均不跨 0
- m=90 时 KernelSmoothing 略优于 SoftmaxKNN（0.6001 vs 0.5995, ns）
- Evaluator verdict: PASS（4/4 acceptance criteria, attempt 1）
- 证据: evaluator-report.json, test-output.txt, feature-dev-summary.md, commands.log, git-diff.patch
- 输出: results/phase2/figures/{table3_predictor_ablation.csv, figure3_delta_r.{pdf,png}, statistical_tests_phase2.csv, phase2_recommendation.txt}
- **建议: Phase 4 使用 SoftmaxKNN（低题量 K=7 τ=0.1；中间题量 K=10 τ=0.1；高题量 KernelSmoothing τ≈0.034）**

### 2026-06-05T09:00:00Z Cross-Phase Baseline Audit
- **审计发现: Phase 2 "UniformKNN" 不是原文 K=5 baseline！**
- Phase 1 (F005/F007): Coverage + KNN **K=5 (fixed)**，如 m=10 时 item_r=0.0836
- Phase 2 (F008/F010): "UniformKNN" 实际使用 **inner validation tuned K=3**，m=10 时 item_r=0.1246 (+49%)
- 根因: run_weighted_knn.py 对 UniformKNN 也做了 inner validation K 调优，K∈{3,5,7,10,15}，所有 fold/ratio 均选中 K=3
- 后果: Phase 2 baseline 被系统性高估，Δr (weighted − baseline) 被低估
- 验证: S 集合逐 fold/ratio 一致 ✓、fold splits 一致 ✓、evaluation mask 一致 ✓、aggregation 一致 ✓、预测实现数学等价 ✓
- **唯一差异: K value (3 vs 5)**

### 2026-06-05T09:00:00Z F016 Created — Cross-Phase Baseline Alignment Fix
- 新增 F016: Phase 2 跨 Phase Baseline 对齐修正（Option A）
- 计划: (1) UniformKNN → Tuned UniformKNN 重命名；(2) 新增 Phase 1 Coverage K=5 原文 baseline 行；(3) 更新 Figure 3/统计检验/推荐文本
- Depends on: F007 (Phase 1 results), F010 (Phase 2 evaluation script)
- 不重新跑实验，纯报告修正

### 2026-06-05T10:00:00Z F016 Implementation — 跨 Phase Baseline 对齐修正
- 修改 `scripts/evaluate_phase2.py`：
  - PREDICTOR_ORDER: "UniformKNN" → "Tuned UniformKNN"
  - COLORS: 同步更新 key + 新增 Phase 1 baseline 颜色
  - load_data(): 运行时重命名 df["predictor"] = replace(UniformKNN → Tuned UniformKNN)
  - build_table3(): 从 Phase 1 semantic_selection_aggregated.csv 加载 Coverage 行，追加为 "UniformKNN K=5 (原文 baseline)"
  - build_figure3(): 标题更新为 "over Tuned UniformKNN"
  - run_statistical_tests(): comparison/predictor_b 更新为 "Tuned UniformKNN"
  - write_recommendation(): 新增 Cross-Phase Baseline Alignment Note 段落
- 重新运行 evaluate_phase2.py（quick 模式，2k bootstrap）
- Table 3 验证通过：20 rows (5 predictors × 4 ratios)
  - Tuned UniformKNN K=3: item_r=0.125/0.286/0.341/0.558
  - UniformKNN K=5 (原文 baseline): item_r=0.084/0.256/0.304/0.484 (K=5, best_K=5)
- Statistical tests 验证通过：comparison 列使用 "vs Tuned UniformKNN"
- Recommendation 包含 baseline 对齐说明
- progress.md F008/F009/F010 session log 已完成 UniformKNN → Tuned UniformKNN K=3 修正
- Phase 2 predictor ranking 新增 UniformKNN K=5 (原文 baseline) 行及说明注释
- Evaluator verdict: pending

### 2026-06-12T06:20:00Z F011 实现 — 新嵌入模型生成
- 新增 `scripts/generate_embeddings.py`，提供统一的新嵌入生成流程：模型清单管理、自动选择运行设备（Tesla T4 上使用 CUDA，必要时回退 CPU）、按需指定模型、覆盖已有输出，以及纯本地校验模式。
- 使用 `data/processed/metadata.parquet` 中固定顺序的 100 道 NEO-PI-R 题项，生成并校验 MiniLM、MPNet、E5、BGE 四个新嵌入矩阵。
- 输出产物：
  - `embeddings/neo_minilm_l6_v2.npy`（100×384）
  - `embeddings/neo_mpnet_base_v2.npy`（100×768）
  - `embeddings/neo_e5_base_v2.npy`（100×768，E5 使用 `query: ` 前缀）
  - `embeddings/neo_bge_base_en_v15.npy`（100×768）
  - `embeddings/neo_embeddings_metadata.json`
- 元数据文件记录题项编号顺序、题项文本顺序、题项编号/文本配对顺序的稳定哈希，同时记录模型信息、软件包版本、运行设备和输出文件 SHA256。
- 校验结果：所有矩阵均为 float32、形状符合预期、逐行 L2 范数在 1e-5 容差内；元数据来源顺序与 `metadata.parquet` 一致；离线校验（`HF_HUB_OFFLINE=1`）通过。
- 更新 `questionnaire.yaml`，加入 pyarrow、sentence-transformers、torch。
- 独立验收结论：通过（4/4 条验收标准，第 1 次尝试）。

### 2026-06-12T07:10:13Z F012 实现 — 嵌入空间质量诊断
- 新增 `scripts/diagnose_embeddings.py`，用统一的 5 模型清单比较原始 SBERT、MiniLM、MPNet、E5、BGE，并校验 F011 元数据顺序、嵌入矩阵形状和 L2 范数。
- 已选题集合诊断：共 20 行结果（5 个嵌入 × m∈{10,30,50,90}），Coverage/Redundancy 使用 shifted cosine 指标列：`coverage_shifted_cosine`、`redundancy_shifted_cosine`。
- 全局空间诊断：共 5 行结果，包含全部题项对原始余弦均值/标准差、同维度题项原始余弦、跨维度题项原始余弦、同维度减跨维度原始余弦差值，以及完整 100 题的 shifted Coverage/Redundancy 汇总列。
- 生成 `results/phase3/figures/figure5.pdf` 和 `.png`，采用 2×2 面板展示 Coverage、Redundancy、同维度/跨维度原始余弦、维度分离差值。
- 写入 `results/phase3/figures/phase3_embedding_diagnostics.txt`，给出第四阶段解释假设：E5 的已选题 Coverage 最高；MPNet 的维度分离差值最大；版本 A/B 应分别检验邻居几何结构与重新选题贡献。
- 验证：`python scripts/diagnose_embeddings.py --all` 通过；产物检查确认 20 行已选题集合结果、5 行全局结果、必需指标列和非空图表/文字输出均存在。
- 说明：与第一阶段历史 CSV 对比时，SBERT 在 m=30/50 的 selected_S 和 Redundancy 有非阻塞轻微漂移，可能来自贪心选择并列项或数值差异；F012 独立验收确认该问题不影响本功能验收。
- 独立验收结论：通过（3/3 条验收标准，第 1 次尝试）。


### 2026-06-12T08:00:00Z Phase 4 Design Revision — A1/A2/B1/B2 and analysis preregistration
- 根据用户反馈，在进入 F013 前修订 Phase 4 设计，避免 embedding 空间、K/τ 重新调参和重新选题贡献混淆。
- F013 从单一 Version A 改为：A1 固定 SBERT-Coverage S_old + 固定 Phase 2 推荐 SoftmaxKNN 超参数（主分析）；A2 固定 S_old + 每个 embedding 在 train-inner 上重新调 K/τ（补充上限分析）。
- F014 从单一 Version B 改为：B1 每个 embedding 重新 Coverage 选题 + 固定超参数；B2 重新选题 + embedding-specific train-inner tuning。
- 新增归因输出要求：每个 embedding/fold/ratio 的 selected_items JSON、hyperparameters CSV、per-participant predictions、S_new vs S_old Jaccard overlap，以及同一 embedding 下的 B−A selection contribution。
- 统计分析预注册：主指标 item-level Pearson r；关键次指标 trait_r_mean、profile_r、MAE；新 embedding vs SBERT original 和 B−A 使用 paired bootstrap over participants，并报告 Holm/BH 多重比较校正。
- Phase 4 主分析预测改为 continuous clip-only（只裁剪到 [1,5]，不 round）；rounded accuracy/rounded MAE 仅作为补充分析。
- F015 最终报告新增要求：若不做 IPIP/其他问卷的小型泛化实验，必须明确写入 limitation：当前结论主要针对 NEO-PI-R，跨问卷泛化仍需验证。

### 2026-06-12T09:45:00Z F013 实现 — Phase 4 Version A 固定原选题 Embedding 对比
- 实现 `scripts/phase4_common.py`、`scripts/phase4_predictors.py`、`scripts/run_phase4_versionA.py`、`scripts/test_phase4_versionA.py`。
- `S_old` 由 Phase 1 `semantic_selection_detail.csv` 中 Coverage rows 按 fold×ratio 读取，不重新 Coverage 选题；在每个 fold×ratio 内所有 embedding 和 A1/A2 共享完全相同 selected item indices 和 question IDs。
- 主评分遵循 Phase 2 held-out-only convention：只预测和评分未选题项；selected items 作为 observed responses，不参与 item_r/MAE 主指标。
- A1 固定 SoftmaxKNN 参数；A2 只在 outer-train 内做 nested train-inner tuning，test subjects 不参与调参。
- Full run `python scripts/run_phase4_versionA.py --all` 成功完成：participant rows=109,960；fold rows=200；hyperparameter rows=200。
- Artifact checks 通过：Parquet prediction row vectors 与 heldout item IDs 长度一致，无 NaN predictions；A1 fixed params exact；A2 inner grid scores recorded；bootstrap rows=32 且包含 raw/Holm/BH p-values。
- Evaluator verdict: PASS（6/6 acceptance criteria）。

# Design Decisions

## Decision 1: 外层被试级交叉验证替代原文 per-participant item CV
**Date:** 2026-06-04
**Context:** 原文使用每个被试内部 10-fold item CV（随机 90 题 train / 10 题 test）。新实验模拟真实测量场景，需要在被试间泛化选题策略。
**Decision:** 使用 5-fold participant-level outer CV，将 2749 名被试分成 80/20 train/test。选题策略在 train participants 上确定，在 test participants 上评估。
**Rationale:** 避免选题策略的数据泄漏（尤其是涉及题项-人格分数相关性的策略），同时保证统计稳定性。
**Alternatives considered:** Leave-one-out CV（计算成本过高）、10-fold（与原文对齐但训练集更小）。
**Impact:** 所有阶段的评估必须使用相同的 outer folds 以保证可比性。

## Decision 2: 先预测 raw response，再反向计分
**Date:** 2026-06-04
**Context:** NEO-PI-R 约 1/3 题目为反向题。原文在数据预处理阶段即反向编码。
**Decision:** 预测阶段使用原始 1-5 作答（raw response），反向计分仅在人格总分计算阶段处理。
**Rationale:** KNN 预测器在 embedding 空间找语义近邻，语义近邻关系与正向/反向编码无关。提前反向会破坏题项原始作答分布。
**Impact:** 人格总分计算需要自己的反向处理逻辑，不能直接使用原文预处理的 nonReversed 分数。

## Decision 3: 执行顺序 — 固定 embedding 验证方法，再更新 embedding
**Date:** 2026-06-04
**Context:** 有选题策略、预测器和 embedding 三个可改进维度。
**Decision:** 先用原 SBERT embedding 验证选题策略（Phase 1）和预测器（Phase 2），确认方法改进有效后，再生成新 embedding（Phase 3）并进行完整实验（Phase 4）。
**Rationale:** 如果方法本身不 work，换 embedding 不会拯救结果。先验证方法可以避免在新 embedding 上浪费计算资源。
**Alternatives considered:** 全排列实验（8 策略 × 5 预测器 × 5 embedding × 4 比例 = 800 组合，计算不可行）。

## Decision 4: Hybrid-C 作为首选方案，A/B 作为消融
**Date:** 2026-06-04
**Context:** Hybrid 策略组合了 Coverage、TraitPredictiveness、Redundancy 和 ImbalancePenalty 四个量。
**Decision:** 第一版只比较 Hybrid-A (Coverage+Trait), Hybrid-B (+Redundancy), Hybrid-C (+Imbalance) 三种变体，不做完整网格搜索。
**Rationale:** 消融式比较比网格搜索更容易解释"哪个成分贡献了什么"，同时大幅减少计算量。
**Impact:** 权重 α/β/γ/δ 使用初始建议值 (1,1,0.5,0.5)，后续可单独调优。

## Decision 5: 人格总分三级评估
**Date:** 2026-06-04
**Context:** 需要评估选题+预测对人格测量的实际影响。
**Decision:** 对每个测试被试计算三种人格分数：(1) Short-form score（仅真实作答题），(2) Imputed full score（真实+预测合成），(3) Held-out score（仅预测题）。再加 Profile Correlation 作为画像级指标。
**Rationale:** Short-form 回答"这组题本身能不能当短量表"；Imputed full 是核心应用场景；Held-out 避免高比例条件下虚高。Profile Correlation 捕捉五维画像恢复质量。
**Impact:** 所有阶段都需要同时报告 item-level 和 trait-level 指标，增加了评估代码复杂度但大幅提升论证力。

## Decision 6: Phase 4 拆分 A1/A2/B1/B2 以分离 embedding、调参和选题贡献
**Date:** 2026-06-12
**Context:** Phase 4 同时涉及 embedding 空间、SoftmaxKNN 超参数 K/τ、以及 Coverage 重新选题。若只做一个版本，性能差异可能被“新 embedding 本身”“重新调参”和“重新选题”混淆。
**Decision:** 将 Phase 4 拆为四个预注册版本：A1 = 固定 SBERT-Coverage S_old + 固定 Phase 2 推荐超参数；A2 = 固定 S_old + embedding-specific train-inner tuning；B1 = 每个 embedding 重新 Coverage 选题 + 固定 Phase 2 推荐超参数；B2 = 重新选题 + embedding-specific train-inner tuning。A1 作为主分析，A2/B1/B2 作为补充和归因分析。
**Rationale:** A1 最接近“纯 embedding 邻居几何”检验；A2 估计校准后预测上限；B−A 在同一 embedding 和同一 tuning regime 下定义重新选题净贡献。
**Impact:** Phase 4 输出必须记录 selected items、hyperparameters、per-participant predictions、Jaccard overlap 和 Δ_selection，避免后验混合解释。

## Decision 7: Phase 4 预先固定主指标与多重比较校正
**Date:** 2026-06-12
**Context:** Phase 4 有 embedding × ratio × version 多个比较，若事后挑选指标或不做校正，会增加偶然显著结果风险。
**Decision:** Phase 4 主指标固定为 item-level Pearson r。关键次指标为 trait_r_mean、profile_r、MAE。主比较为每个新 embedding vs SBERT original，以及同一 embedding 下 B−A 的重新选题净贡献。统计检验使用 paired bootstrap over participants，保持同一 outer fold 配对，并报告 Holm 或 Benjamini-Hochberg 校正后的 p 值。
**Rationale:** item-level r 与原论文主评估一致；trait/profile/MAE 保留测量效度解释但不替代主结论；配对 bootstrap 与 fold 配对能减少方差并避免泄漏。
**Impact:** Phase 4/F015 的结果表必须同时包含 Δ、95% CI、原始 p 值和校正 p 值，并在报告中标注主/次指标。

## Decision 8: Phase 4 主分析使用 continuous clip-only prediction
**Date:** 2026-06-12
**Context:** Phase 2 predictors historically used round + clip to [1,5]. 对 SoftmaxKNN/KernelSmoothing 这类连续加权模型，rounding 会损失连续预测信息，并可能影响 Pearson r 与 MAE。
**Decision:** Phase 4 主分析使用连续预测值，仅 clip 到 [1,5]，不 round。Rounded accuracy / rounded MAE 作为补充分析输出，不作为主结论依据。
**Rationale:** 连续预测更公平地反映加权模型输出，也更适合作为 Pearson r/MAE 主分析输入；补充 rounded 指标仍能回应 Likert 离散作答解释。
**Impact:** Phase 4 实现若复用 Phase 2 predictor，需要新增或启用 no-round/continuous 模式，并在 hyperparameters/results 中记录 prediction_mode。

## Decision 9: 跨问卷泛化作为最终报告 limitation 或小型补充实验
**Date:** 2026-06-12
**Context:** 当前改进实验主要围绕 NEO-PI-R，但原仓库还包含 IPIP/IPIP2/RIASEC/HSQ/16PF 等问卷。
**Decision:** F015 最终报告前若时间允许，添加一个小型跨问卷泛化实验（优先 IPIP 或另一个 Big Five 数据集，30%/50% 两个比例）。若不执行，则必须明确写入 limitation：当前结论主要针对 NEO-PI-R，跨问卷泛化仍需验证。
**Rationale:** 避免将 NEO-PI-R 单数据集结论过度推广；即使没有额外实验，也应在论文级输出中清楚界定外推范围。
**Impact:** F015 acceptance criteria 包含跨问卷泛化结果或 limitation 文本二选一。

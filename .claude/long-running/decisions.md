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

## Decision 6: 新 embedding 必须重新选题
**Date:** 2026-06-04
**Context:** Phase 4 需要比较不同 embedding 的完整 pipeline 表现。
**Decision:** 做两个版本：版本 A 固定原 SBERT 选出的 S（只换预测 embedding），版本 B 每个 embedding 重新选题。
**Rationale:** 版本 A vs B 的差异可以分离"embedding 改善邻居关系"和"embedding 改善选题"两个贡献来源。
**Impact:** Phase 4 计算量加倍，但分析更透彻。

## Decision 7: 主指标固定，辅助指标不做过多样本检验
**Date:** 2026-06-04
**Context:** 多指标 + 多比例 + 多策略会导致大量统计检验，增加 Type I error 风险。
**Decision:** 主指标固定两个：(1) item-level per-person Pearson r，(2) mean Big Five Pearson r for imputed full score。其他 MAE/RMSE/Profile Correlation 作为辅助。
**Rationale:** 减少多重比较问题，让论文叙述聚焦。主指标对应原文（item-level r）和新贡献（trait-level r）。
**Impact:** 统计检验只对主指标做 paired bootstrap test，辅助指标仅报告均值和 CI。

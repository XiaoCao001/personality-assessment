# Feature-dev handoff for F012

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F012
- Title: Phase 3: Embedding 空间质量诊断
- Description: 对各 embedding 模型进行质量诊断分析。计算 Coverage of selected S、Redundancy of selected S、within-trait similarity（同维度题项平均相似度）、between-trait similarity（跨维度题项平均相似度）。生成 Figure 5（embedding 质量诊断对比图）。为 Phase 4 的 embedding 性能差异提供解释基础。
- Priority: medium
- Dependencies: [F011]

## Acceptance criteria
1. [AC001] 所有 5 个 embedding 模型的四项诊断指标均已计算
2. [AC002] Figure 5 清楚展示各模型在语义空间结构上的差异
3. [AC003] 诊断结果能够为 Phase 4 的性能差异提供解释假设

## Test plan
- `cd /workspace/questionnaire-embeddings && python scripts/diagnose_embeddings.py --all`
- `cd /workspace/questionnaire-embeddings && ls results/phase3/figures/figure5.pdf`

## Relevant context
- F011 generated modern embedding artifacts and manifest:
  - `questionnaire-embeddings/embeddings/neo_minilm_l6_v2.npy`
  - `questionnaire-embeddings/embeddings/neo_mpnet_base_v2.npy`
  - `questionnaire-embeddings/embeddings/neo_e5_base_v2.npy`
  - `questionnaire-embeddings/embeddings/neo_bge_base_en_v15.npy`
  - `questionnaire-embeddings/embeddings/neo_embeddings_metadata.json`
- Original SBERT embedding: `questionnaire-embeddings/data/processed/E_old.npy`
- Item metadata: `questionnaire-embeddings/data/processed/metadata.parquet`
- Existing Coverage/Redundancy selection logic may be reusable from `questionnaire-embeddings/scripts/selection.py`.
- Existing result/figure style examples: `questionnaire-embeddings/scripts/evaluate_phase1.py` and `questionnaire-embeddings/scripts/evaluate_phase2.py`.

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F012/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

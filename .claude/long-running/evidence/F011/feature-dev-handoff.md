# Feature-dev handoff for F011

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F011
- Title: Phase 3: 新 Embedding 模型生成
- Description: 使用 sentence-transformers 在本地 T4 GPU 上为 NEO-PI-R 100 道题项生成新 embedding。模型列表：all-MiniLM-L6-v2, all-mpnet-base-v2, intfloat/e5-base-v2, BAAI/bge-base-en-v1.5。统一后处理：L2 normalize，保存为 .npy。记录 metadata（模型名、维度、pooling 方式、日期、包版本）。E5 模型使用 'query: ' prefix。
- Priority: high
- Dependencies: [F001]

## Acceptance criteria
1. [AC001] 所有 4 个模型的 embedding 均已生成并保存
2. [AC002] 每个 embedding 矩阵形状为 (100, d)，已 L2 normalize
3. [AC003] embedding 与 item_text 顺序一致（第 i 行对应第 i 题）
4. [AC004] metadata JSON 包含所有要求字段

## Test plan
- `python scripts/generate_embeddings.py --all`
- `python -c "import numpy as np; [print(f'{m}: {np.load(chr(39)+f'embeddings/neo_{m}.npy'+chr(39)).shape}') for m in ['minilm_l6_v2','mpnet_base_v2','e5_base_v2','bge_base_en_v15']]"`

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F011/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Context from prior phases
- F001 prepared `data/processed/Y.npy`, `data/processed/E_old.npy`, `data/processed/metadata.parquet`, and `data/processed/subject_ids.txt`.
- F011 should create a reusable script, likely `scripts/generate_embeddings.py`, that loads item text from processed metadata and writes the four new embedding matrices plus metadata JSON.
- Expected output file names from the feature test plan: `embeddings/neo_minilm_l6_v2.npy`, `embeddings/neo_mpnet_base_v2.npy`, `embeddings/neo_e5_base_v2.npy`, `embeddings/neo_bge_base_en_v15.npy`.
- Preserve row order: row i in every new embedding matrix must correspond exactly to row i / item i in `data/processed/metadata.parquet`.

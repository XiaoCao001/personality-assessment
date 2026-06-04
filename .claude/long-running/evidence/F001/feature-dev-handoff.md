# Feature-dev handoff for F001

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F001
- Title: Phase 0: 数据准备与矩阵标准化
- Description: 从 NEO-PI-R (BIG5) 原始数据中提取并标准化 Y 矩阵 (2749×100)、E_old 原 SBERT embedding (100×d)、item_text、trait_id (O/C/E/A/N)、reverse_id。输出为 .npy 和 .parquet 文件，作为后续所有实验的数据基础。包含数据完整性验证（行列对应、无缺失值、embedding 维度检查）。
- Priority: high
- Dependencies: []

## Actual data paths

The dataset is located under `BIG5` (not `NEO_PI_R` as written in features.json). The project root is `/workspace/questionnaire-embeddings/`.

- Item texts: `embeddings/BIG5/big5_questions_text.csv`
- SBERT embeddings: `embeddings/BIG5/big5_questions_embeddings_SENTENCEBERT.csv`
- Raw responses: `embeddings/BIG5/big5_responses.csv`
- Non-reversed responses: `embeddings/BIG5/big5_responses_nonReversed.csv`

Output directory: `data/processed/` (under the project root; create if needed)

## Acceptance criteria
1. [AC001] Y 矩阵形状为 (2749, 100)，值域 [1,5]，无缺失值
2. [AC002] E_old 矩阵形状为 (100, d)，d≥384，已 L2 normalize
3. [AC003] trait_id 包含 5 个维度各 20 题，reverse_id 正反比合理
4. [AC004] 所有输出文件可通过 numpy.load / pandas.read_parquet 读取

## Implementation steps
1. Read the 4 raw CSV files from `embeddings/BIG5/`
2. Validate data integrity: 100 items, 2749 subjects, embedding dimension consistent, no NaN
3. Build Y matrix [2749, 100] (raw 1-5 responses, no reverse scoring)
4. Build E_old matrix [100, d] (original SBERT embeddings, L2 normalize)
5. Extract item_text list, trait_id mapping (each item → O/C/E/A/N), reverse_id flag
6. Save as .npy (Y, E_old) and .parquet (metadata) to `data/processed/`
7. Generate data report: dimension shapes, items per trait distribution, forward/reverse item ratio

## Test plan
```bash
python -c "import numpy as np; y=np.load('data/processed/Y.npy'); assert y.shape==(2749,100); assert y.min()>=1 and y.max()<=5"
python -c "import numpy as np; e=np.load('data/processed/E_old.npy'); assert e.shape[0]==100; norms=np.linalg.norm(e,axis=1); assert np.allclose(norms,1.0,atol=1e-5)"
python scripts/validate_data.py  # if exists, otherwise create a validation script
```

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F001/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.
- Work inside the `/workspace/questionnaire-embeddings/` project directory.
- Use the git repo at `/workspace/questionnaire-embeddings/`.
- The dataset directory is `BIG5`, not `NEO_PI_R` — the features.json has a naming mismatch with the actual filesystem.

# Feature-dev handoff for F016

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F016
- Title: Phase 2: 跨 Phase Baseline 对齐修正 — UniformKNN 重命名 + 原文 K=5 baseline
- Description: 修正 Phase 2 报告中的 baseline 命名和跨 Phase 对齐问题。审计发现 Phase 2 的 "UniformKNN" 并非原文 K=5，而是 inner validation tuned K=3。本 feature 实施 Option A：(1) 将 Phase 2 Table 3/Figure 3/statistical tests/recommendation 中的 "UniformKNN" 重命名为 "Tuned UniformKNN"; (2) 新增一行 Phase 1 Coverage K=5 作为原文 baseline; (3) Δr 比较改为 "加权方法 vs Tuned UniformKNN"; (4) 在报告中添加跨 Phase baseline 差异说明。不重新跑实验，纯报告修正。
- Priority: high
- Dependencies: [F007, F010]

## Acceptance criteria
1. [AC001] Table 3 第一行显示为 "Tuned UniformKNN"（原 UniformKNN），并新增一行 "UniformKNN K=5 (原文 baseline)" 引用 Phase 1 Coverage K=5 结果
2. [AC002] Figure 3 标题和 legend 使用 "Tuned UniformKNN"，不再暗示其为原文 K=5 baseline
3. [AC003] Statistical tests 表格中 comparison 列使用 "Tuned UniformKNN" 名称
4. [AC004] Recommendation 文件包含跨 Phase baseline 差异说明：Phase 2 tuned K=3 系统性优于 Phase 1 fixed K=5
5. [AC005] progress.md F008/F009/F010 session log 中 UniformKNN 相关描述已修正为 Tuned UniformKNN，并注明 K 参数

## Test plan
- cd /workspace/questionnaire-embeddings && python scripts/evaluate_phase2.py
- grep 'Tuned' results/phase2/figures/table3_predictor_ablation.csv
- grep '原文' results/phase2/figures/table3_predictor_ablation.csv
- grep 'Tuned' results/phase2/figures/phase2_recommendation.txt

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F016/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.

## Detailed implementation plan

### File to modify: `scripts/evaluate_phase2.py`

#### 1. Rename predictor labels (lines 47-59)
- `PREDICTOR_ORDER`: change `"UniformKNN"` → `"Tuned UniformKNN"`
- `COLORS`: change key `"UniformKNN"` → `"Tuned UniformKNN"`

#### 2. Update Figure 3 (lines 302-387)
- Line 375: change `"Figure 3: Predictor Improvement over UniformKNN"` → `"Figure 3: Predictor Improvement over Tuned UniformKNN"`
- Line 309: update print statement to `"Figure 3: Δr over Tuned UniformKNN (Weighted − Tuned Uniform)"`
- Line 325: change `df["predictor"] == "UniformKNN"` → `df["predictor"] == "Tuned UniformKNN"`
- Line 374-375: update the title to use "Tuned UniformKNN"

#### 3. Add Phase 1 Coverage K=5 baseline row in build_table3()
After building the existing table rows, append a new row from Phase 1 data. Read from `results/phase1/figures/table1_item_level.csv`:
- Coverage m=10: item_r=0.0836, ci_lower=0.0764, ci_upper=0.0911
- Coverage m=30: item_r=0.2555, ci_lower=0.2493, ci_upper=0.2617
- Coverage m=50: item_r=0.3037, ci_lower=0.2967, ci_upper=0.3109
- Coverage m=90: item_r=0.4842, ci_lower=0.4709, ci_upper=0.4973

For `item_mae`, `trait_r_mean`, and `profile_r`, read from the Phase 1 detail data or set to NaN with a note. Best to load Phase 1 detail CSVs if available, otherwise mark as NaN.

The predictor name should be: `"UniformKNN K=5 (原文 baseline)"`

#### 4. Update statistical tests (lines 392-500)
- Line 407: change `df["predictor"] == "UniformKNN"` → `df["predictor"] == "Tuned UniformKNN"`
- Line 421: change `f"{pred_name} vs UniformKNN"` → `f"{pred_name} vs Tuned UniformKNN"`
- Line 423: change `"UniformKNN"` → `"Tuned UniformKNN"` in `predictor_b`
- Line 433: change `"vs UniformKNN"` → `"vs Tuned UniformKNN"` in print
- Add a new test section: SoftmaxKNN vs Phase 1 Coverage K=5 (read from Phase 1 detail CSV if available)

#### 5. Update recommendation (lines 506-586)
- Line 540: change `"Statistical significance vs UniformKNN:"` → `"Statistical significance vs Tuned UniformKNN:"`
- Line 545: change `"vs UniformKNN"` → `"vs Tuned UniformKNN"`
- Line 563: change `"improvement over UniformKNN"` → `"improvement over Tuned UniformKNN"`
- Add a section explaining: Phase 2 "Tuned UniformKNN" uses K=3 (tuned via inner validation), while Phase 1 "UniformKNN K=5" uses fixed K=5 as in the original paper. The tuned K=3 systematically outperforms fixed K=5 (e.g., m=10: 0.125 vs 0.084, +49%).

#### 6. Update progress.md
Edit `.claude/long-running/progress.md` to:
- F008 session log: change "UniformKNN" → "Tuned UniformKNN", note tuned K=3
- F009 session log: change "UniformKNN" → "Tuned UniformKNN" where it references Phase 2 baseline
- F010 session log: change "UniformKNN" → "Tuned UniformKNN" throughout, update ranking descriptions
- Phase 2 predictor ranking: add note that UniformKNN values are Tuned UniformKNN K=3

### Key data reference
Phase 1 Coverage K=5 baseline (from `results/phase1/figures/table1_item_level.csv`):
```
Coverage,10,0.0836368102444791,0.07641782862883724,0.09107782707319227
Coverage,30,0.2555322634425815,0.24932602307190205,0.2617385038132609
Coverage,50,0.3037373382247258,0.2966510044882892,0.3109040761066599
Coverage,90,0.4842084651344137,0.47092607740883824,0.4973374026695311
```

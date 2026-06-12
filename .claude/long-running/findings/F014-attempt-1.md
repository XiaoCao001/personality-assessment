# F014 attempt 1 evaluator findings

Evaluator verdict: NEEDS_WORK

## Required fix
- Regenerate `questionnaire-embeddings/results/phase4/figures/figure4.pdf` and `.png` with visible on-canvas labeling or a legend that clearly identifies each embedding and explicitly distinguishes A (fixed S_old) vs B (re-selected S_new) within the fixed-parameter and tuned-parameter panels.

## Notes
- AC001–AC005 passed after updating F014 canonical artifact names to versionB-specific files.
- AC006 failed because the legend was placed outside the saved PNG canvas (`bbox_to_anchor=(1.0, 0.5)`), leaving the right margin blank and clipping/omitting the legend.

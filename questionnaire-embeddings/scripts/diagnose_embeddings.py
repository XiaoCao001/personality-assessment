#!/usr/bin/env python3
"""
F012: Embedding-space quality diagnostics for NEO-PI-R item embeddings.

Compares the original SBERT embedding space against the four F011 modern
sentence-transformer spaces.  Selected-set Coverage/Redundancy are computed
with the Phase 1 shifted-cosine definitions, while global trait-structure
metrics use raw cosine similarities.

Outputs::

    results/phase3/embedding_diagnostics_selected_sets.csv
    results/phase3/embedding_diagnostics_global_space.csv
    results/phase3/embedding_diagnostics_summary.csv
    results/phase3/figures/figure5.pdf
    results/phase3/figures/figure5.png
    results/phase3/figures/phase3_embedding_diagnostics.txt

Usage::

    python scripts/diagnose_embeddings.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from selection import CoverageSelector

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3"
FIGURES_DIR = RESULTS_DIR / "figures"

SOURCE_METADATA = DATA_DIR / "metadata.parquet"
MODERN_MANIFEST = EMBEDDINGS_DIR / "neo_embeddings_metadata.json"
PHASE1_SEMANTIC_DETAIL = PROJECT_ROOT / "results" / "phase1" / "semantic_selection_detail.csv"

EXPECTED_N_ITEMS = 100
TRAIT_ORDER = ("O", "C", "E", "A", "N")
RATIOS = (10, 30, 50, 90)
NORM_ATOL = 1e-5

MODEL_ORDER = (
    "sbert_original",
    "minilm_l6_v2",
    "mpnet_base_v2",
    "e5_base_v2",
    "bge_base_en_v15",
)

COLORS = {
    "sbert_original": "#666666",
    "minilm_l6_v2": "#2A9D8F",
    "mpnet_base_v2": "#457B9D",
    "e5_base_v2": "#E63946",
    "bge_base_en_v15": "#F4A261",
}


@dataclass(frozen=True)
class EmbeddingSpec:
    """One embedding artifact included in F012 diagnostics."""

    key: str
    label: str
    path: Path
    source_type: str
    model_name: str
    dimension: int | None = None


# ---------------------------------------------------------------------------
# CLI & validation helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose semantic structure of NEO-PI-R item embeddings."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full F012 diagnostics over all five approved embeddings.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        help=(
            "Optional subset of embedding keys for validation-only debugging. "
            "Canonical outputs require --all."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate inputs without writing results or figures.",
    )
    return parser.parse_args()


def relpath(path: Path) -> str:
    """Return project-root-relative POSIX path when possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _fail(message: str) -> None:
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_item_metadata() -> pd.DataFrame:
    """Load and validate canonical F001 item metadata."""
    if not SOURCE_METADATA.exists():
        _fail(f"Missing source metadata: {SOURCE_METADATA}")

    metadata = pd.read_parquet(SOURCE_METADATA)
    required = {"question_id", "item_text", "trait_id", "reverse_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        _fail(f"metadata.parquet missing required columns: {missing}")
    if len(metadata) != EXPECTED_N_ITEMS:
        _fail(f"metadata.parquet has {len(metadata)} rows, expected {EXPECTED_N_ITEMS}")

    trait_counts = metadata["trait_id"].value_counts().to_dict()
    expected_counts = {trait: 20 for trait in TRAIT_ORDER}
    if trait_counts != expected_counts:
        _fail(f"Unexpected trait counts: {trait_counts}; expected {expected_counts}")

    _ok(
        "Loaded canonical item metadata: "
        f"{len(metadata)} items, traits={trait_counts}"
    )
    return metadata


def load_modern_manifest() -> dict[str, Any]:
    """Load F011 manifest for the four modern embedding artifacts."""
    if not MODERN_MANIFEST.exists():
        _fail(f"Missing modern embedding manifest: {MODERN_MANIFEST}")
    with MODERN_MANIFEST.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "models" not in manifest or not isinstance(manifest["models"], dict):
        _fail("neo_embeddings_metadata.json missing 'models' object")
    return manifest


def validate_manifest_source(manifest: dict[str, Any], metadata: pd.DataFrame) -> None:
    """Check that the F011 manifest describes the same canonical 100 item order."""
    source = manifest.get("source", {})
    manifest_ids = source.get("question_ids")
    if manifest_ids is None:
        _fail("neo_embeddings_metadata.json source.question_ids is missing")

    current_ids = [str(x) for x in metadata["question_id"].tolist()]
    if [str(x) for x in manifest_ids] != current_ids:
        _fail("F011 manifest question_ids do not match current metadata.parquet order")
    _ok("F011 manifest source order matches metadata.parquet")


def build_embedding_registry(manifest: dict[str, Any]) -> list[EmbeddingSpec]:
    """Build the unified 5-model registry for F012."""
    specs: list[EmbeddingSpec] = [
        EmbeddingSpec(
            key="sbert_original",
            label="SBERT (original)",
            path=DATA_DIR / "E_old.npy",
            source_type="legacy_f001",
            model_name="roberta-large-nli-stsb-mean-tokens",
            dimension=1024,
        )
    ]

    modern_labels = {
        "minilm_l6_v2": "MiniLM-L6-v2",
        "mpnet_base_v2": "MPNet-base-v2",
        "e5_base_v2": "E5-base-v2",
        "bge_base_en_v15": "BGE-base-en-v1.5",
    }
    for key in MODEL_ORDER[1:]:
        info = manifest["models"].get(key)
        if info is None:
            _fail(f"F011 manifest missing model entry: {key}")
        output_file = info.get("output_file")
        if not output_file:
            _fail(f"F011 manifest model {key} missing output_file")
        specs.append(
            EmbeddingSpec(
                key=key,
                label=modern_labels[key],
                path=PROJECT_ROOT / output_file,
                source_type="modern_f011",
                model_name=info.get("model_name", key),
                dimension=int(info.get("dimension", 0)) or None,
            )
        )
    return specs


def load_embedding_matrix(spec: EmbeddingSpec) -> np.ndarray:
    """Load, validate, and return a float64 L2-normalised computation matrix."""
    if not spec.path.exists():
        _fail(f"Missing embedding matrix for {spec.key}: {spec.path}")

    E = np.load(spec.path)
    if E.ndim != 2 or E.shape[0] != EXPECTED_N_ITEMS:
        _fail(f"{spec.key} shape is {E.shape}; expected ({EXPECTED_N_ITEMS}, d)")
    if spec.dimension is not None and E.shape[1] != spec.dimension:
        _fail(f"{spec.key} dimension is {E.shape[1]}; expected {spec.dimension}")
    if not np.issubdtype(E.dtype, np.number):
        _fail(f"{spec.key} dtype is not numeric: {E.dtype}")
    if not np.isfinite(E).all():
        _fail(f"{spec.key} contains NaN or infinite values")

    E = np.asarray(E, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1)
    if np.any(norms == 0):
        _fail(f"{spec.key} contains zero-norm rows")
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > NORM_ATOL:
        _warn(
            f"{spec.key} row norms deviate from 1 by up to {max_norm_error:.2e}; "
            "renormalising computation copy"
        )
        E = E / norms[:, None]
    else:
        # Normalize a computation copy anyway to remove tiny float32 drift.
        E = E / norms[:, None]

    _ok(
        f"Loaded {spec.label}: shape={E.shape}, "
        f"max norm error before copy-normalization={max_norm_error:.2e}"
    )
    return E


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def pairwise_raw_cosine(E: np.ndarray) -> np.ndarray:
    """Return clipped raw-cosine matrix for an L2-normalised embedding matrix."""
    C = E @ E.T
    return np.clip(C, -1.0, 1.0)


def compute_global_raw_cosine_stats(
    spec: EmbeddingSpec,
    E: np.ndarray,
    trait_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute full-space raw cosine diagnostics, excluding diagonal self-pairs."""
    C = pairwise_raw_cosine(E)
    iu, ju = np.triu_indices_from(C, k=1)
    all_pairs = C[iu, ju]
    same_trait = trait_ids[iu] == trait_ids[ju]

    within = all_pairs[same_trait]
    between = all_pairs[~same_trait]

    expected_all = EXPECTED_N_ITEMS * (EXPECTED_N_ITEMS - 1) // 2
    expected_within = len(TRAIT_ORDER) * (20 * 19 // 2)
    expected_between = expected_all - expected_within
    if len(all_pairs) != expected_all or len(within) != expected_within or len(between) != expected_between:
        _fail(
            f"Unexpected pair counts for {spec.key}: "
            f"all={len(all_pairs)}, within={len(within)}, between={len(between)}"
        )

    full_selector = CoverageSelector(E)
    full_S = np.arange(EXPECTED_N_ITEMS, dtype=np.intp)

    row: dict[str, Any] = {
        "embedding_key": spec.key,
        "embedding_label": spec.label,
        "embedding_path": relpath(spec.path),
        "source_type": spec.source_type,
        "model_name": spec.model_name,
        "dimension": E.shape[1],
        "full100_coverage_shifted_cosine": full_selector.compute_coverage(full_S),
        "full100_redundancy_shifted_cosine": full_selector.compute_redundancy(full_S),
        "n_all_pairs": len(all_pairs),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
        "allpair_raw_cosine_mean": float(np.mean(all_pairs)),
        "allpair_raw_cosine_std": float(np.std(all_pairs, ddof=1)),
        "within_trait_raw_cosine": float(np.mean(within)),
        "within_trait_raw_cosine_std": float(np.std(within, ddof=1)),
        "between_trait_raw_cosine": float(np.mean(between)),
        "between_trait_raw_cosine_std": float(np.std(between, ddof=1)),
        "within_minus_between_raw_cosine": float(np.mean(within) - np.mean(between)),
    }

    for trait in TRAIT_ORDER:
        idx = np.where(trait_ids == trait)[0]
        sub = C[np.ix_(idx, idx)]
        ti, tj = np.triu_indices_from(sub, k=1)
        vals = sub[ti, tj]
        row[f"within_trait_{trait}_raw_cosine"] = float(np.mean(vals))

    return row


def compute_selected_set_stats(spec: EmbeddingSpec, E: np.ndarray) -> list[dict[str, Any]]:
    """Compute Coverage-selected-set diagnostics for all approved m values."""
    selector = CoverageSelector(E)
    rows: list[dict[str, Any]] = []
    for m in RATIOS:
        S = selector.select(m)
        coverage = selector.compute_coverage(S)
        redundancy = selector.compute_redundancy(S)
        rows.append(
            {
                "embedding_key": spec.key,
                "embedding_label": spec.label,
                "embedding_path": relpath(spec.path),
                "source_type": spec.source_type,
                "model_name": spec.model_name,
                "dimension": E.shape[1],
                "m": m,
                "n_selected": len(S),
                "coverage_shifted_cosine": coverage,
                "redundancy_shifted_cosine": redundancy,
                "coverage_minus_redundancy_shifted_cosine": coverage - redundancy,
                "selected_S": ",".join(str(i) for i in S),
            }
        )
    return rows


def validate_sbert_against_phase1(selected_df: pd.DataFrame) -> None:
    """Cross-check deterministic old-SBERT Coverage rows against Phase 1 output."""
    if not PHASE1_SEMANTIC_DETAIL.exists():
        _warn(f"Skipping SBERT Phase 1 cross-check; missing {PHASE1_SEMANTIC_DETAIL}")
        return

    phase1 = pd.read_csv(PHASE1_SEMANTIC_DETAIL)
    required = {"strategy", "ratio", "selected_S", "coverage", "redundancy"}
    if not required.issubset(phase1.columns):
        _warn("Skipping SBERT Phase 1 cross-check; semantic detail schema differs")
        return

    ref = phase1[phase1["strategy"] == "Coverage"]
    if ref.empty:
        _warn("Skipping SBERT Phase 1 cross-check; no Coverage rows found")
        return

    sbert_rows = selected_df[selected_df["embedding_key"] == "sbert_original"]
    for m in RATIOS:
        ref_m = ref[ref["ratio"] == m]
        cur_m = sbert_rows[sbert_rows["m"] == m]
        if ref_m.empty or cur_m.empty:
            _fail(f"Missing SBERT cross-check row for m={m}")

        selected_values = set(str(x) for x in ref_m["selected_S"].dropna().unique())
        coverage_values = ref_m["coverage"].astype(float).to_numpy()
        redundancy_values = ref_m["redundancy"].astype(float).to_numpy()
        cur = cur_m.iloc[0]

        if selected_values and cur["selected_S"] not in selected_values:
            ref_first = sorted(selected_values)[0]
            cur_set = {int(x) for x in str(cur["selected_S"]).split(",") if x != ""}
            ref_set = {int(x) for x in ref_first.split(",") if x != ""}
            only_current = sorted(cur_set - ref_set)
            only_phase1 = sorted(ref_set - cur_set)
            _warn(
                f"SBERT selected_S differs from Phase 1 at m={m}; "
                f"current-only={only_current}, phase1-only={only_phase1}. "
                "Continuing because selected-set ties/numerical drift can change item IDs."
            )
        phase1_coverage = float(np.mean(coverage_values))
        phase1_redundancy = float(np.mean(redundancy_values))
        if not np.isclose(cur["coverage_shifted_cosine"], phase1_coverage, atol=1e-4):
            _warn(
                f"SBERT coverage differs from Phase 1 at m={m}: "
                f"current={cur['coverage_shifted_cosine']:.6f}, "
                f"phase1_mean={phase1_coverage:.6f}"
            )
        if not np.isclose(cur["redundancy_shifted_cosine"], phase1_redundancy, atol=1e-4):
            _warn(
                f"SBERT redundancy differs from Phase 1 at m={m}: "
                f"current={cur['redundancy_shifted_cosine']:.6f}, "
                f"phase1_mean={phase1_redundancy:.6f}"
            )

    _ok("SBERT Phase 1 cross-check completed; warnings above indicate non-blocking drift")


# ---------------------------------------------------------------------------
# Outputs: Figure 5 and text summary
# ---------------------------------------------------------------------------
def build_figure5(global_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    """Build Figure 5 as a 2×2 embedding diagnostics panel."""
    print("\n" + "-" * 50)
    print("Figure 5: Embedding Space Diagnostics")
    print("-" * 50)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        _fail(f"matplotlib/seaborn unavailable; cannot build Figure 5: {exc}")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # Panel A: Coverage curves (selected-set, shifted cosine)
    for key in MODEL_ORDER:
        sub = selected_df[selected_df["embedding_key"] == key].sort_values("m")
        if sub.empty:
            continue
        ax_a.plot(
            sub["m"],
            sub["coverage_shifted_cosine"],
            marker="o",
            lw=2.2,
            color=COLORS[key],
            label=sub["embedding_label"].iloc[0],
        )
    ax_a.set_title("A. Coverage of selected S (shifted cosine)", fontsize=12)
    ax_a.set_xlabel("Number of administered items (m)")
    ax_a.set_ylabel("coverage_shifted_cosine")
    ax_a.set_xticks(list(RATIOS))
    ax_a.legend(fontsize=8, loc="lower right")

    # Panel B: Redundancy curves (selected-set, shifted cosine)
    for key in MODEL_ORDER:
        sub = selected_df[selected_df["embedding_key"] == key].sort_values("m")
        if sub.empty:
            continue
        ax_b.plot(
            sub["m"],
            sub["redundancy_shifted_cosine"],
            marker="o",
            lw=2.2,
            color=COLORS[key],
            label=sub["embedding_label"].iloc[0],
        )
    ax_b.set_title("B. Redundancy within selected S (shifted cosine)", fontsize=12)
    ax_b.set_xlabel("Number of administered items (m)")
    ax_b.set_ylabel("redundancy_shifted_cosine")
    ax_b.set_xticks(list(RATIOS))

    # Panel C: Global raw cosine within vs between trait structure
    ordered = global_df.set_index("embedding_key").loc[list(MODEL_ORDER)].reset_index()
    x = np.arange(len(ordered))
    width = 0.36
    ax_c.bar(
        x - width / 2,
        ordered["within_trait_raw_cosine"],
        width,
        yerr=ordered["within_trait_raw_cosine_std"],
        capsize=3,
        label="Within trait",
        color="#457B9D",
        alpha=0.85,
    )
    ax_c.bar(
        x + width / 2,
        ordered["between_trait_raw_cosine"],
        width,
        yerr=ordered["between_trait_raw_cosine_std"],
        capsize=3,
        label="Between trait",
        color="#E9C46A",
        alpha=0.85,
    )
    ax_c.set_title("C. Full-space trait similarity (raw cosine)", fontsize=12)
    ax_c.set_ylabel("raw cosine mean ± SD")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(ordered["embedding_label"], rotation=25, ha="right")
    ax_c.legend(fontsize=9)

    # Panel D: Within-minus-between raw cosine separation
    ax_d.bar(
        x,
        ordered["within_minus_between_raw_cosine"],
        color=[COLORS[k] for k in ordered["embedding_key"]],
        alpha=0.9,
    )
    ax_d.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax_d.set_title("D. Trait separation: within − between (raw cosine)", fontsize=12)
    ax_d.set_ylabel("within_minus_between_raw_cosine")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(ordered["embedding_label"], rotation=25, ha="right")

    fig.suptitle("Figure 5: Embedding Space Quality Diagnostics", fontsize=15, y=1.01)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = FIGURES_DIR / f"figure5.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def write_text_summary(global_df: pd.DataFrame, selected_df: pd.DataFrame) -> str:
    """Write Phase 1/2-style interpretation artifact for Phase 4 hypotheses."""
    print("\n" + "=" * 70)
    print("Phase 3 Embedding Diagnostics Summary")
    print("=" * 70)

    lines: list[str] = [
        "=" * 70,
        "Phase 3: Embedding Space Diagnostics — Phase 4 Hypotheses",
        "=" * 70,
        "",
        "Metric scales:",
        "  - coverage_shifted_cosine and redundancy_shifted_cosine use sim+ = (raw cosine + 1) / 2.",
        "  - allpair/within/between trait metrics use raw cosine and exclude diagonal self-pairs.",
        "  - Coverage/Redundancy curves use Coverage-selected S at m = 10, 30, 50, 90 only.",
        "  - Full-100/global diagnostics are reported here and in the global-space CSV, not as Panel A/B curve points.",
        "",
    ]

    lines.append("Full-100 shifted-cosine diagnostics (not plotted as selected-set curve points):")
    for row in global_df.sort_values("embedding_key", key=lambda s: s.map({k: i for i, k in enumerate(MODEL_ORDER)})).itertuples(index=False):
        lines.append(
            f"  {row.embedding_label:18s}: "
            f"full100_coverage_shifted_cosine={row.full100_coverage_shifted_cosine:.4f}, "
            f"full100_redundancy_shifted_cosine={row.full100_redundancy_shifted_cosine:.4f}"
        )
    lines.append("")

    lines.append("Best selected-set coverage by m:")
    for m in RATIOS:
        sub = selected_df[selected_df["m"] == m].sort_values(
            "coverage_shifted_cosine", ascending=False
        )
        best = sub.iloc[0]
        lines.append(
            f"  m={m:3d}: {best['embedding_label']:18s} "
            f"coverage_shifted_cosine={best['coverage_shifted_cosine']:.4f}, "
            f"redundancy_shifted_cosine={best['redundancy_shifted_cosine']:.4f}"
        )
    lines.append("")

    lines.append("Lowest selected-set redundancy by m:")
    for m in RATIOS:
        sub = selected_df[selected_df["m"] == m].sort_values(
            "redundancy_shifted_cosine", ascending=True
        )
        best = sub.iloc[0]
        lines.append(
            f"  m={m:3d}: {best['embedding_label']:18s} "
            f"redundancy_shifted_cosine={best['redundancy_shifted_cosine']:.4f}, "
            f"coverage_shifted_cosine={best['coverage_shifted_cosine']:.4f}"
        )
    lines.append("")

    mean_coverage = (
        selected_df.groupby(["embedding_key", "embedding_label"])["coverage_shifted_cosine"]
        .mean()
        .sort_values(ascending=False)
    )
    lines.append("Overall ranking by mean selected-set coverage (m=10,30,50,90):")
    for rank, ((_, label), value) in enumerate(mean_coverage.items(), 1):
        lines.append(f"  {rank}. {label:18s}: {value:.4f}")
    lines.append("")

    gap_rank = global_df.sort_values("within_minus_between_raw_cosine", ascending=False)
    lines.append("Global trait-structure ranking by within-minus-between raw cosine:")
    for rank, row in enumerate(gap_rank.itertuples(index=False), 1):
        lines.append(
            f"  {rank}. {row.embedding_label:18s}: "
            f"gap={row.within_minus_between_raw_cosine:+.4f} "
            f"(within={row.within_trait_raw_cosine:.4f}, "
            f"between={row.between_trait_raw_cosine:.4f}, "
            f"all-pair mean={row.allpair_raw_cosine_mean:.4f})"
        )
    lines.append("")

    best_cov_label = mean_coverage.index[0][1]
    best_gap = gap_rank.iloc[0]
    lines.append("RECOMMENDATION / PHASE 4 HYPOTHESES:")
    lines.append(
        f"  Coverage diagnostic: {best_cov_label} has the highest average selected-set"
    )
    lines.append(
        "  coverage across the administered-item ratios, suggesting stronger semantic"
    )
    lines.append(
        "  coverage when each embedding is allowed to choose its own Coverage set S."
    )
    lines.append(
        f"  Trait-structure diagnostic: {best_gap['embedding_label']} has the largest"
    )
    lines.append(
        "  within-minus-between raw cosine gap, suggesting the clearest Big Five"
    )
    lines.append(
        "  clustering signal in the embedding space."
    )
    lines.append("")
    lines.append(
        "  These are explanatory hypotheses, not final predictive-performance claims."
    )
    lines.append(
        "  Phase 4 Version A should test whether improved neighbor geometry helps when"
    )
    lines.append(
        "  S is fixed; Phase 4 Version B should test whether improved Coverage selection"
    )
    lines.append(
        "  and trait separation improve the full selection + prediction pipeline."
    )
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    path = FIGURES_DIR / "phase3_embedding_diagnostics.txt"
    path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[SAVE] {path}")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_diagnostics(model_keys: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all diagnostics and return (global_df, selected_df)."""
    metadata = load_item_metadata()
    trait_ids = metadata["trait_id"].astype(str).to_numpy()

    manifest = load_modern_manifest()
    validate_manifest_source(manifest, metadata)
    registry = build_embedding_registry(manifest)
    if model_keys is not None:
        registry = [spec for spec in registry if spec.key in model_keys]
    if not registry:
        _fail("No embedding models selected")

    global_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for spec in registry:
        print("\n" + "-" * 70)
        print(f"Diagnosing {spec.label} ({spec.key})")
        print("-" * 70)
        E = load_embedding_matrix(spec)
        global_rows.append(compute_global_raw_cosine_stats(spec, E, trait_ids))
        selected_rows.extend(compute_selected_set_stats(spec, E))

    global_df = pd.DataFrame(global_rows)
    selected_df = pd.DataFrame(selected_rows)
    if model_keys is None or "sbert_original" in model_keys:
        validate_sbert_against_phase1(selected_df)
    return global_df, selected_df


def main() -> int:
    args = parse_args()
    model_keys = set(args.models) if args.models else None

    if model_keys is not None and not args.validate_only:
        _fail("--models is validation-only; use --validate-only or run --all for canonical outputs")
    if not args.all and model_keys is None:
        _warn("No --all/--models supplied; defaulting to --all for F012 diagnostics")

    global_df, selected_df = run_diagnostics(model_keys=model_keys)

    if args.validate_only:
        print("\n[OK] validate-only completed; no outputs written")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    global_path = RESULTS_DIR / "embedding_diagnostics_global_space.csv"
    selected_path = RESULTS_DIR / "embedding_diagnostics_selected_sets.csv"
    summary_path = RESULTS_DIR / "embedding_diagnostics_summary.csv"

    global_df.to_csv(global_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    # A compact all-in-one summary table joins global diagnostics to mean selected-set diagnostics.
    selected_summary = (
        selected_df.groupby(["embedding_key", "embedding_label"])[
            ["coverage_shifted_cosine", "redundancy_shifted_cosine"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "coverage_shifted_cosine": "mean_selected_set_coverage_shifted_cosine",
                "redundancy_shifted_cosine": "mean_selected_set_redundancy_shifted_cosine",
            }
        )
    )
    summary_df = global_df.merge(selected_summary, on=["embedding_key", "embedding_label"])
    summary_df.to_csv(summary_path, index=False)

    print("\n[SAVE] Diagnostic CSVs")
    print(f"  {global_path}")
    print(f"  {selected_path}")
    print(f"  {summary_path}")

    build_figure5(global_df, selected_df)
    write_text_summary(global_df, selected_df)

    print("\n" + "=" * 70)
    print("F012 diagnostics complete")
    print("=" * 70)
    print(f"Global-space rows: {len(global_df)}")
    print(f"Selected-set rows: {len(selected_df)}")
    print(f"Figure 5: {FIGURES_DIR / 'figure5.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

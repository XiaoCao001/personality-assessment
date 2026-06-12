#!/usr/bin/env python3
"""Shared helpers for Phase 4 embedding comparison experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE1_DETAIL = RESULTS_DIR / "phase1" / "semantic_selection_detail.csv"
PHASE2_DETAIL = RESULTS_DIR / "phase2" / "softmax_kernel_detail.csv"
MODERN_MANIFEST = PROJECT_ROOT / "embeddings" / "neo_embeddings_metadata.json"
PHASE4_DIR = RESULTS_DIR / "phase4"

RANDOM_STATE = 0
N_FOLDS = 5
RATIOS = (10, 30, 50, 90)
K_CANDIDATES = (3, 5, 7, 10, 15)
TAU_CANDIDATES = (0.03, 0.05, 0.1, 0.2, 0.5)
A1_FIXED_PARAMS = {
    10: (7, 0.1),
    30: (7, 0.1),
    50: (10, 0.1),
    90: (3, 0.03),
}
MODEL_ORDER = (
    "sbert_original",
    "minilm_l6_v2",
    "mpnet_base_v2",
    "e5_base_v2",
    "bge_base_en_v15",
)
TRAIT_ORDER = ("O", "C", "E", "A", "N")


@dataclass(frozen=True)
class CoreData:
    Y: np.ndarray
    metadata: pd.DataFrame
    subject_ids: list[str]
    trait_ids: np.ndarray
    reverse_ids: np.ndarray
    question_ids: list[str]


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    label: str
    path: Path
    source_type: str
    model_name: str
    dimension: int | None


@dataclass(frozen=True)
class FixedSelectedSet:
    fold: int
    ratio: int
    selected_indices: np.ndarray
    selected_question_ids: list[str]
    coverage: float
    redundancy: float
    source_path: Path

    @property
    def selected_csv(self) -> str:
        return ",".join(str(int(i)) for i in self.selected_indices)

    @property
    def selected_hash(self) -> str:
        return stable_hash(self.selected_csv)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def relpath(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_selected_s(value: Any) -> np.ndarray:
    text = str(value).strip().strip('"')
    if not text:
        return np.array([], dtype=np.intp)
    arr = np.array([int(part) for part in text.split(",") if part != ""], dtype=np.intp)
    if len(arr) != len(set(arr.tolist())):
        raise ValueError(f"selected_S contains duplicates: {text}")
    if np.any(arr < 0) or np.any(arr >= 100):
        raise ValueError(f"selected_S out of range [0,99]: {text}")
    return arr


def load_core_data() -> CoreData:
    Y = np.load(DATA_DIR / "Y.npy").astype(np.float64)
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    subject_path = DATA_DIR / "subject_ids.txt"
    subject_ids = [line.strip() for line in subject_path.read_text().splitlines() if line.strip()]
    if len(subject_ids) != Y.shape[0]:
        raise RuntimeError(f"subject_ids length {len(subject_ids)} != Y rows {Y.shape[0]}")
    required = {"question_id", "trait_id", "reverse_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise RuntimeError(f"metadata.parquet missing columns: {sorted(missing)}")
    if len(metadata) != Y.shape[1]:
        raise RuntimeError(f"metadata rows {len(metadata)} != Y columns {Y.shape[1]}")
    return CoreData(
        Y=Y,
        metadata=metadata,
        subject_ids=subject_ids,
        trait_ids=metadata["trait_id"].astype(str).to_numpy(),
        reverse_ids=metadata["reverse_id"].astype(float).to_numpy(),
        question_ids=[str(x) for x in metadata["question_id"].tolist()],
    )


def load_fixed_s_old_by_fold_ratio(question_ids: list[str]) -> dict[tuple[int, int], FixedSelectedSet]:
    """Load Phase 1 Coverage S_old per fold and ratio from the historical CSV."""
    if not PHASE1_DETAIL.exists():
        raise RuntimeError(f"Missing Phase 1 semantic detail artifact: {PHASE1_DETAIL}")
    df = pd.read_csv(PHASE1_DETAIL)
    required = {"strategy", "ratio", "fold", "selected_S", "coverage", "redundancy"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{PHASE1_DETAIL} missing columns: {sorted(missing)}")
    cov = df[df["strategy"] == "Coverage"].copy()
    if cov.empty:
        raise RuntimeError("No Coverage rows found in Phase 1 semantic detail artifact")

    selected: dict[tuple[int, int], FixedSelectedSet] = {}
    for fold in range(N_FOLDS):
        for ratio in RATIOS:
            sub = cov[(cov["fold"] == fold) & (cov["ratio"] == ratio)]
            if len(sub) != 1:
                raise RuntimeError(f"Expected exactly one Coverage row for fold={fold}, ratio={ratio}; found {len(sub)}")
            row = sub.iloc[0]
            S = parse_selected_s(row["selected_S"])
            if len(S) != ratio:
                raise RuntimeError(f"fold={fold}, ratio={ratio}: selected {len(S)} items")
            selected[(fold, ratio)] = FixedSelectedSet(
                fold=fold,
                ratio=ratio,
                selected_indices=S,
                selected_question_ids=[question_ids[int(i)] for i in S],
                coverage=float(row["coverage"]),
                redundancy=float(row["redundancy"]),
                source_path=PHASE1_DETAIL,
            )

    # Version A requires fixed S_old across embeddings within each fold×ratio.
    # We also report whether the historical Coverage sets are fold-invariant.
    return selected


def load_modern_manifest() -> dict[str, Any]:
    if not MODERN_MANIFEST.exists():
        raise RuntimeError(f"Missing F011 manifest: {MODERN_MANIFEST}")
    return json.loads(MODERN_MANIFEST.read_text())


def validate_manifest_source(manifest: dict[str, Any], question_ids: list[str]) -> None:
    manifest_ids = manifest.get("source", {}).get("question_ids")
    if manifest_ids is None:
        raise RuntimeError("neo_embeddings_metadata.json missing source.question_ids")
    if [str(x) for x in manifest_ids] != [str(x) for x in question_ids]:
        raise RuntimeError("F011 manifest source.question_ids does not match metadata.parquet order")


def build_embedding_registry(core: CoreData) -> list[EmbeddingSpec]:
    manifest = load_modern_manifest()
    validate_manifest_source(manifest, core.question_ids)
    labels = {
        "sbert_original": "SBERT (original)",
        "minilm_l6_v2": "MiniLM-L6-v2",
        "mpnet_base_v2": "MPNet-base-v2",
        "e5_base_v2": "E5-base-v2",
        "bge_base_en_v15": "BGE-base-en-v1.5",
    }
    specs = [
        EmbeddingSpec(
            key="sbert_original",
            label=labels["sbert_original"],
            path=DATA_DIR / "E_old.npy",
            source_type="legacy_f001",
            model_name="roberta-large-nli-stsb-mean-tokens",
            dimension=1024,
        )
    ]
    for key in MODEL_ORDER[1:]:
        info = manifest.get("models", {}).get(key)
        if info is None:
            raise RuntimeError(f"F011 manifest missing model entry: {key}")
        specs.append(
            EmbeddingSpec(
                key=key,
                label=labels[key],
                path=PROJECT_ROOT / info["output_file"],
                source_type="modern_f011",
                model_name=info.get("model_name", key),
                dimension=int(info.get("dimension", 0)) or None,
            )
        )
    return specs


def load_embedding_matrix(spec: EmbeddingSpec) -> np.ndarray:
    if not spec.path.exists():
        raise RuntimeError(f"Missing embedding matrix for {spec.key}: {spec.path}")
    E = np.load(spec.path)
    if E.ndim != 2 or E.shape[0] != 100:
        raise RuntimeError(f"{spec.key} shape {E.shape}; expected (100,d)")
    if spec.dimension is not None and E.shape[1] != spec.dimension:
        raise RuntimeError(f"{spec.key} dimension {E.shape[1]}; expected {spec.dimension}")
    E = np.asarray(E, dtype=np.float64)
    if not np.isfinite(E).all():
        raise RuntimeError(f"{spec.key} contains non-finite values")
    norms = np.linalg.norm(E, axis=1)
    if np.any(norms == 0):
        raise RuntimeError(f"{spec.key} contains zero-norm rows")
    return E / norms[:, None]


def precompute_similarity(E: np.ndarray) -> np.ndarray:
    return np.clip(E @ E.T, -1.0, 1.0)


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return (mean, mean, mean)
    se = sp_stats.sem(vals)
    h = se * sp_stats.t.ppf((1 + confidence) / 2.0, len(vals) - 1)
    return (mean, float(mean - h), float(mean + h))


def pearson_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 2:
        return np.nan
    av = a[valid]
    bv = b[valid]
    if np.nanstd(av) == 0 or np.nanstd(bv) == 0:
        return np.nan
    return float(sp_stats.pearsonr(av, bv)[0])


def adjust_pvalues_holm(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, original_idx in enumerate(order):
        value = min(1.0, (m - rank) * p[original_idx])
        running = max(running, value)
        adjusted[rank] = running
    out[order] = adjusted
    return out.tolist()


def adjust_pvalues_bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 1.0
    for rev_rank, original_idx in enumerate(order[::-1], start=1):
        rank = m - rev_rank + 1
        value = min(1.0, p[original_idx] * m / rank)
        running = min(running, value)
        adjusted_sorted[rank - 1] = running
    out[order] = adjusted_sorted
    return out.tolist()


def paired_bootstrap_by_fold(
    paired: pd.DataFrame,
    diff_col: str = "diff",
    fold_col: str = "outer_fold",
    n_bootstrap: int = 10_000,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Bootstrap paired subject diffs by resampling subjects within each fold."""
    work = paired[[fold_col, diff_col]].dropna().copy()
    if work.empty:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
    observed = float(work[diff_col].mean())
    rng = np.random.RandomState(seed)
    fold_values = [grp[diff_col].to_numpy(dtype=np.float64) for _, grp in work.groupby(fold_col)]
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled_parts = []
        for vals in fold_values:
            sampled_parts.append(rng.choice(vals, size=len(vals), replace=True))
        boot[b] = float(np.mean(np.concatenate(sampled_parts)))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    p_val = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return {"delta": observed, "ci_low": ci_low, "ci_high": ci_high, "p": min(p_val, 1.0), "n": int(len(work))}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_predictions_parquet(records: list[dict[str, Any]], path: Path) -> None:
    """Write prediction rows with native list columns when pyarrow is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - fallback for unusual envs
        fallback = path.with_suffix(".jsonl")
        with fallback.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise RuntimeError(f"pyarrow unavailable; wrote JSONL fallback to {fallback}: {exc}")

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def output_paths(output_dir: Path = PHASE4_DIR) -> dict[str, Path]:
    return {
        "predictions": output_dir / "versionA_predictions.parquet",
        "participant_metrics": output_dir / "versionA_participant_metrics.csv",
        "results": output_dir / "versionA_results.csv",
        "summary": output_dir / "versionA_summary.csv",
        "selected_items": output_dir / "selected_items_by_fold_ratio_embedding.json",
        "hyperparameters": output_dir / "hyperparameters_by_fold_ratio_embedding.csv",
        "stats": output_dir / "versionA_statistical_tests.csv",
        "folds": output_dir / "outer_folds_subject_ids.json",
    }

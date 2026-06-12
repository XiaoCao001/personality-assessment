#!/usr/bin/env python3
"""
F011: Generate modern sentence-transformer embeddings for NEO-PI-R items.

Reads the canonical item order/texts from ``data/processed/metadata.parquet``
and writes L2-normalised embedding matrices plus one combined metadata manifest.

Usage::

    python scripts/generate_embeddings.py --all
    python scripts/generate_embeddings.py --models minilm_l6_v2 e5_base_v2
    python scripts/generate_embeddings.py --all --device cpu --overwrite
    python scripts/generate_embeddings.py --validate
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
SOURCE_METADATA = DATA_DIR / "metadata.parquet"
OUTPUT_METADATA_JSON = EMBEDDINGS_DIR / "neo_embeddings_metadata.json"
ENV_FILE = PROJECT_ROOT / "questionnaire.yaml"

EXPECTED_N_ITEMS = 100
SCHEMA_VERSION = "1.0"
FEATURE_ID = "F011"
DATASET = "BIG5"
QUESTIONNAIRE = "NEO-PI-R"
NORM_ATOL = 1e-5


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one approved F011 embedding model."""

    key: str
    hf_model_name: str
    output_filename: str
    expected_dim: int
    pooling: str = "mean"
    text_prefix: str = ""
    default_batch_size: int = 16


MODEL_SPECS: dict[str, ModelSpec] = {
    "minilm_l6_v2": ModelSpec(
        key="minilm_l6_v2",
        hf_model_name="sentence-transformers/all-MiniLM-L6-v2",
        output_filename="neo_minilm_l6_v2.npy",
        expected_dim=384,
        default_batch_size=32,
    ),
    "mpnet_base_v2": ModelSpec(
        key="mpnet_base_v2",
        hf_model_name="sentence-transformers/all-mpnet-base-v2",
        output_filename="neo_mpnet_base_v2.npy",
        expected_dim=768,
    ),
    "e5_base_v2": ModelSpec(
        key="e5_base_v2",
        hf_model_name="intfloat/e5-base-v2",
        output_filename="neo_e5_base_v2.npy",
        expected_dim=768,
        text_prefix="query: ",
    ),
    "bge_base_en_v15": ModelSpec(
        key="bge_base_en_v15",
        hf_model_name="BAAI/bge-base-en-v1.5",
        output_filename="neo_bge_base_en_v15.npy",
        expected_dim=768,
    ),
}


# ---------------------------------------------------------------------------
# Logging & validation helpers
# ---------------------------------------------------------------------------
def _check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    """Print [PASS]/[FAIL] and return True iff the check passed."""
    if condition:
        print(f"[PASS] {pass_msg} ✓")
        return True
    print(f"[FAIL] {fail_msg}")
    return False


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _fail(message: str) -> None:
    raise RuntimeError(message)


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relpath(path: Path) -> str:
    """Return project-root-relative POSIX path when possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_utc_timestamp(value: Any) -> bool:
    """Return True iff *value* is an ISO-8601 UTC timestamp ending in Z."""
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Canonical source metadata
# ---------------------------------------------------------------------------
def sha256_text(values: list[str]) -> str:
    """Hash an ordered list of strings with explicit separators."""
    h = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        h.update(len(encoded).to_bytes(8, byteorder="big"))
        h.update(encoded)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_item_hash(question_ids: list[str], item_texts: list[str]) -> str:
    """Hash ordered question-id/text pairs to prove row-order provenance."""
    pairs = [f"{qid}\t{text}" for qid, text in zip(question_ids, item_texts)]
    return sha256_text(pairs)


def load_item_metadata() -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """Load and validate the F001 canonical item metadata."""
    if not SOURCE_METADATA.exists():
        _fail(f"Missing source metadata: {SOURCE_METADATA}")

    metadata = pd.read_parquet(SOURCE_METADATA)
    required = {"question_id", "item_text", "trait_id", "reverse_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        _fail(f"metadata.parquet missing required columns: {missing}")
    if len(metadata) != EXPECTED_N_ITEMS:
        _fail(f"metadata.parquet has {len(metadata)} rows, expected {EXPECTED_N_ITEMS}")

    question_ids = [str(x) for x in metadata["question_id"].tolist()]
    item_texts = [str(x) for x in metadata["item_text"].tolist()]

    if len(set(question_ids)) != len(question_ids):
        _fail("metadata.parquet contains duplicate question_id values")
    if any(text.strip() == "" or text.lower() == "nan" for text in item_texts):
        _fail("metadata.parquet contains blank item_text values")

    source = {
        "metadata_path": relpath(SOURCE_METADATA),
        "metadata_sha256": sha256_file(SOURCE_METADATA),
        "item_count": len(metadata),
        "question_ids": question_ids,
        "question_ids_sha256": sha256_text(question_ids),
        "item_text_sha256": sha256_text(item_texts),
        "canonical_item_sha256": canonical_item_hash(question_ids, item_texts),
        "provenance": (
            "Canonical item ids and texts loaded from F001 data/processed/metadata.parquet; "
            "embedding row i is generated from question_ids[i] and item_texts[i]."
        ),
    }

    _ok(f"Loaded canonical item metadata: {len(metadata)} items from {relpath(SOURCE_METADATA)}")
    return metadata, question_ids, item_texts, source


# ---------------------------------------------------------------------------
# CLI & model selection
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate F011 NEO-PI-R sentence-transformer embeddings."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all approved F011 models. This is the default if --models is omitted.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        help="Generate or validate a subset of approved model keys.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing local .npy files and metadata only; no model loading or downloads.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override for generation: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-model encoding batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of selected existing embedding files and metadata entries.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional sentence-transformers/Hugging Face cache directory for generation.",
    )
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.all and args.models:
        parser.error("Use either --all or --models, not both")

    return args


def selected_specs(args: argparse.Namespace) -> list[ModelSpec]:
    """Return selected models in deterministic registry order."""
    if args.models:
        selected = set(args.models)
    else:
        selected = set(MODEL_SPECS)  # default and --all both mean all models
    return [spec for key, spec in MODEL_SPECS.items() if key in selected]


# ---------------------------------------------------------------------------
# Runtime dependency and device handling (generation mode only)
# ---------------------------------------------------------------------------
def import_generation_dependencies():
    """Import heavy/model dependencies only in generation mode."""
    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        _fail(
            "Missing generation dependency. Install/update the environment with "
            "`conda env update --file questionnaire.yaml` or install `torch` and "
            f"`sentence-transformers`. Original error: {exc}"
        )
    return torch, SentenceTransformer


def package_version(package_name: str) -> str | None:
    """Return installed package version without importing the package when possible."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_package_versions() -> dict[str, str | None]:
    """Capture runtime versions used for generation metadata."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": package_version("pyarrow"),
        "torch": package_version("torch"),
        "sentence_transformers": package_version("sentence-transformers"),
    }


def resolve_device(requested: str, torch_module) -> tuple[str, dict[str, Any]]:
    """Resolve generation device from auto/cpu/cuda override."""
    requested = requested.lower()
    cuda_available = bool(torch_module.cuda.is_available())

    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif requested == "cpu":
        resolved = "cpu"
    elif requested.startswith("cuda"):
        if not cuda_available:
            _fail("--device requested CUDA, but torch.cuda.is_available() is False")
        resolved = requested
    else:
        _fail(f"Unsupported --device value: {requested!r}; use auto, cpu, cuda, or cuda:N")

    info: dict[str, Any] = {
        "requested": requested,
        "resolved": resolved,
        "cuda_available": cuda_available,
    }
    if resolved.startswith("cuda"):
        try:
            info["cuda_device_name"] = torch_module.cuda.get_device_name(resolved)
        except Exception:
            info["cuda_device_name"] = torch_module.cuda.get_device_name()
    else:
        info["cuda_device_name"] = None

    _ok(f"Device requested={requested}, resolved={resolved}, cuda_available={cuda_available}")
    if info["cuda_device_name"]:
        _ok(f"CUDA device: {info['cuda_device_name']}")
    return resolved, info


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------
def build_model_inputs(item_texts: list[str], spec: ModelSpec) -> list[str]:
    """Apply model-specific input prefix while preserving canonical order."""
    if spec.text_prefix:
        return [f"{spec.text_prefix}{text}" for text in item_texts]
    return list(item_texts)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Manually L2-normalise every embedding row."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        zero_rows = np.where(norms.ravel() == 0)[0].tolist()
        _fail(f"Zero-norm embedding rows detected: {zero_rows}")
    return x / norms


def validate_embedding_matrix(x: np.ndarray, spec: ModelSpec) -> dict[str, Any]:
    """Validate shape, finite values, and L2 norms for one embedding matrix."""
    if x.shape != (EXPECTED_N_ITEMS, spec.expected_dim):
        _fail(
            f"{spec.key} shape {x.shape} != ({EXPECTED_N_ITEMS}, {spec.expected_dim})"
        )
    if not np.issubdtype(x.dtype, np.floating):
        _fail(f"{spec.key} dtype {x.dtype} is not floating")
    if not np.all(np.isfinite(x)):
        _fail(f"{spec.key} contains NaN or Inf values")

    norms = np.linalg.norm(x, axis=1)
    if not np.allclose(norms, 1.0, atol=NORM_ATOL):
        _fail(
            f"{spec.key} L2 normalisation failed: "
            f"norm range [{norms.min():.8f}, {norms.max():.8f}]"
        )

    return {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "dtype": str(x.dtype),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "l2_norm_atol": NORM_ATOL,
    }


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Atomically save an .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def ensure_no_overwrite_conflicts(
    specs: list[ModelSpec], manifest: dict[str, Any] | None, overwrite: bool
) -> None:
    """Prevent replacing selected files/metadata unless --overwrite is set."""
    if overwrite:
        return

    model_entries = (manifest or {}).get("models", {})
    conflicts: list[str] = []
    for spec in specs:
        out_path = EMBEDDINGS_DIR / spec.output_filename
        if out_path.exists():
            conflicts.append(f"existing file {relpath(out_path)}")
        if spec.key in model_entries:
            conflicts.append(f"existing metadata entry models.{spec.key}")

    if conflicts:
        joined = "; ".join(conflicts)
        _fail(f"Refusing to overwrite without --overwrite: {joined}")


def generate_one_model(
    spec: ModelSpec,
    item_texts: list[str],
    device: str,
    batch_size: int,
    cache_dir: str | None,
    SentenceTransformer,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode, normalise, validate, and return metadata for one model."""
    print("\n" + "-" * 60)
    print(f"Generating {spec.key}: {spec.hf_model_name}")
    print("-" * 60)

    model_kwargs: dict[str, Any] = {"device": device}
    if cache_dir:
        model_kwargs["cache_folder"] = cache_dir

    model = SentenceTransformer(spec.hf_model_name, **model_kwargs)
    inputs = build_model_inputs(item_texts, spec)
    raw = model.encode(
        inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(raw, dtype=np.float32)
    embeddings = l2_normalize_rows(embeddings).astype(np.float32)
    stats = validate_embedding_matrix(embeddings, spec)

    out_path = EMBEDDINGS_DIR / spec.output_filename
    _ok(
        f"Generated {spec.key} shape={tuple(stats['shape'])} "
        f"norm_range=[{stats['l2_norm_min']:.6f}, {stats['l2_norm_max']:.6f}]"
    )

    entry = {
        "key": spec.key,
        "model_name": spec.hf_model_name,
        "huggingface_id": spec.hf_model_name,
        "output_file": relpath(out_path),
        "file_sha256": None,
        "shape": stats["shape"],
        "dimension": spec.expected_dim,
        "dtype": stats["dtype"],
        "pooling": spec.pooling,
        "text_prefix": spec.text_prefix,
        "l2_normalized": True,
        "normalization": "manual row-wise L2",
        "l2_norm_min": stats["l2_norm_min"],
        "l2_norm_max": stats["l2_norm_max"],
        "l2_norm_atol": stats["l2_norm_atol"],
        "batch_size": batch_size,
        "device_used": device,
        "generated_at_utc": utc_now(),
        "row_order_source": "source.question_ids and source.canonical_item_sha256",
    }
    return embeddings, entry


def cleanup_model(torch_module, device: str) -> None:
    """Release model memory between encoders."""
    gc.collect()
    if device.startswith("cuda"):
        torch_module.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------
def empty_manifest(source: dict[str, Any]) -> dict[str, Any]:
    """Create a new combined metadata manifest skeleton."""
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_id": FEATURE_ID,
        "dataset": DATASET,
        "questionnaire": QUESTIONNAIRE,
        "created_at_utc": now,
        "updated_at_utc": now,
        "script": relpath(Path(__file__).resolve()),
        "source": source,
        "models": {},
    }


def load_existing_manifest() -> dict[str, Any] | None:
    """Load existing metadata manifest if present."""
    if not OUTPUT_METADATA_JSON.exists():
        return None
    with OUTPUT_METADATA_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        _fail(f"Metadata manifest is not a JSON object: {OUTPUT_METADATA_JSON}")
    if not isinstance(payload.get("models"), dict):
        _fail("Metadata manifest missing object field: models")
    return payload


def assert_manifest_source_compatible(manifest: dict[str, Any] | None, source: dict[str, Any]) -> None:
    """Ensure subset metadata updates do not mix different source item orders/texts."""
    if manifest is None:
        return
    existing_source = manifest.get("source", {})
    for key in ("question_ids_sha256", "item_text_sha256", "canonical_item_sha256"):
        if existing_source.get(key) != source.get(key):
            _fail(
                f"Existing metadata source hash {key} differs from current "
                "data/processed/metadata.parquet; refusing to merge metadata."
            )


def merge_manifest(
    manifest: dict[str, Any] | None,
    source: dict[str, Any],
    model_entries: dict[str, dict[str, Any]],
    device_info: dict[str, Any],
    package_versions: dict[str, str | None],
    overwrite: bool,
) -> dict[str, Any]:
    """Merge selected generated models into existing metadata safely."""
    if manifest is None:
        manifest = empty_manifest(source)
    else:
        assert_manifest_source_compatible(manifest, source)

    existing_models = manifest.setdefault("models", {})
    if not overwrite:
        for key in model_entries:
            if key in existing_models:
                _fail(f"Refusing to overwrite metadata entry models.{key} without --overwrite")

    for key, entry in model_entries.items():
        entry = dict(entry)
        entry["package_versions"] = package_versions
        entry["device"] = device_info
        existing_models[key] = entry

    manifest["schema_version"] = SCHEMA_VERSION
    manifest["feature_id"] = FEATURE_ID
    manifest["dataset"] = DATASET
    manifest["questionnaire"] = QUESTIONNAIRE
    manifest["script"] = relpath(Path(__file__).resolve())
    manifest["source"] = source
    manifest["updated_at_utc"] = utc_now()
    manifest["package_versions_last_run"] = package_versions
    manifest["device_last_run"] = device_info
    return manifest


# ---------------------------------------------------------------------------
# Validation-only mode (pure local; no model imports)
# ---------------------------------------------------------------------------
def validate_environment_file() -> bool:
    """Check questionnaire.yaml records F011 generation dependencies."""
    all_ok = True
    all_ok &= _check(ENV_FILE.exists(), f"Environment file exists: {relpath(ENV_FILE)}", f"Missing {ENV_FILE}")
    if not ENV_FILE.exists():
        return False
    text = ENV_FILE.read_text(encoding="utf-8")
    all_ok &= _check(
        "pyarrow" in text or "fastparquet" in text,
        "questionnaire.yaml includes a parquet engine",
        "questionnaire.yaml missing a parquet engine (pyarrow or fastparquet)",
    )
    all_ok &= _check(
        "sentence-transformers" in text,
        "questionnaire.yaml includes sentence-transformers",
        "questionnaire.yaml missing sentence-transformers",
    )
    all_ok &= _check(
        "torch" in text,
        "questionnaire.yaml includes torch",
        "questionnaire.yaml missing torch",
    )
    return all_ok


def validate_manifest_schema(payload: dict[str, Any], source: dict[str, Any]) -> bool:
    """Validate top-level manifest fields and source provenance."""
    all_ok = True
    all_ok &= _check(
        payload.get("schema_version") == SCHEMA_VERSION,
        f"metadata schema_version = {SCHEMA_VERSION}",
        f"metadata schema_version is {payload.get('schema_version')!r}",
    )
    all_ok &= _check(
        payload.get("feature_id") == FEATURE_ID,
        f"metadata feature_id = {FEATURE_ID}",
        f"metadata feature_id is {payload.get('feature_id')!r}",
    )
    all_ok &= _check(
        payload.get("dataset") == DATASET,
        f"metadata dataset = {DATASET}",
        f"metadata dataset is {payload.get('dataset')!r}",
    )
    all_ok &= _check(
        payload.get("questionnaire") == QUESTIONNAIRE,
        f"metadata questionnaire = {QUESTIONNAIRE}",
        f"metadata questionnaire is {payload.get('questionnaire')!r}",
    )
    all_ok &= _check(
        payload.get("script") == "scripts/generate_embeddings.py",
        "metadata script path matches generator",
        f"metadata script path is {payload.get('script')!r}",
    )
    all_ok &= _check(
        isinstance(payload.get("models"), dict),
        "metadata models object exists",
        "metadata missing models object",
    )

    manifest_source = payload.get("source", {})
    for key in ("question_ids", "question_ids_sha256", "item_text_sha256", "canonical_item_sha256"):
        all_ok &= _check(
            manifest_source.get(key) == source.get(key),
            f"source provenance matches for {key}",
            f"source provenance mismatch for {key}",
        )
    return all_ok


def validate_saved_artifact(spec: ModelSpec, model_entry: dict[str, Any] | None) -> bool:
    """Validate one saved .npy artifact and corresponding metadata entry."""
    all_ok = True
    out_path = EMBEDDINGS_DIR / spec.output_filename
    all_ok &= _check(
        out_path.exists(),
        f"File exists: {relpath(out_path)}",
        f"Missing file: {out_path}",
    )
    if not out_path.exists():
        return False

    try:
        arr = np.load(out_path, allow_pickle=False)
        stats = validate_embedding_matrix(arr, spec)
        print(
            f"  {spec.key}: shape={tuple(stats['shape'])}, dtype={stats['dtype']}, "
            f"norm_range=[{stats['l2_norm_min']:.6f}, {stats['l2_norm_max']:.6f}]"
        )
        all_ok &= True
    except Exception as exc:
        print(f"[FAIL] {spec.key} matrix validation failed: {exc}")
        return False

    all_ok &= _check(
        model_entry is not None,
        f"metadata entry exists: models.{spec.key}",
        f"metadata missing entry: models.{spec.key}",
    )
    if model_entry is None:
        return False

    expected_fields = {
        "model_name",
        "output_file",
        "file_sha256",
        "shape",
        "dimension",
        "pooling",
        "text_prefix",
        "l2_normalized",
        "normalization",
        "device_used",
        "generated_at_utc",
        "row_order_source",
        "package_versions",
    }
    missing = sorted(expected_fields - set(model_entry))
    all_ok &= _check(
        not missing,
        f"metadata fields complete for {spec.key}",
        f"metadata fields missing for {spec.key}: {missing}",
    )
    all_ok &= _check(
        model_entry.get("model_name") == spec.hf_model_name,
        f"metadata model_name matches for {spec.key}",
        f"metadata model_name mismatch for {spec.key}",
    )
    all_ok &= _check(
        model_entry.get("output_file") == relpath(out_path),
        f"metadata output_file matches for {spec.key}",
        f"metadata output_file mismatch for {spec.key}: {model_entry.get('output_file')}",
    )
    all_ok &= _check(
        model_entry.get("shape") == [EXPECTED_N_ITEMS, spec.expected_dim],
        f"metadata shape matches for {spec.key}",
        f"metadata shape mismatch for {spec.key}: {model_entry.get('shape')}",
    )
    all_ok &= _check(
        model_entry.get("dimension") == spec.expected_dim,
        f"metadata dimension matches for {spec.key}",
        f"metadata dimension mismatch for {spec.key}: {model_entry.get('dimension')}",
    )
    all_ok &= _check(
        model_entry.get("pooling") == spec.pooling,
        f"metadata pooling matches for {spec.key}",
        f"metadata pooling mismatch for {spec.key}: {model_entry.get('pooling')}",
    )
    all_ok &= _check(
        model_entry.get("text_prefix") == spec.text_prefix,
        f"metadata text_prefix matches for {spec.key}",
        f"metadata text_prefix mismatch for {spec.key}",
    )
    all_ok &= _check(
        model_entry.get("l2_normalized") is True,
        f"metadata l2_normalized true for {spec.key}",
        f"metadata l2_normalized not true for {spec.key}",
    )
    all_ok &= _check(
        model_entry.get("normalization") == "manual row-wise L2",
        f"metadata normalization matches for {spec.key}",
        f"metadata normalization mismatch for {spec.key}: {model_entry.get('normalization')}",
    )
    all_ok &= _check(
        model_entry.get("row_order_source") == "source.question_ids and source.canonical_item_sha256",
        f"metadata row_order_source matches for {spec.key}",
        f"metadata row_order_source mismatch for {spec.key}",
    )
    all_ok &= _check(
        is_utc_timestamp(model_entry.get("generated_at_utc")),
        f"metadata generated_at_utc is UTC for {spec.key}",
        f"metadata generated_at_utc invalid for {spec.key}: {model_entry.get('generated_at_utc')}",
    )
    versions = model_entry.get("package_versions", {})
    all_ok &= _check(
        isinstance(versions, dict) and all(versions.get(k) for k in ("python", "numpy", "pandas", "pyarrow", "torch", "sentence_transformers")),
        f"metadata package_versions complete for {spec.key}",
        f"metadata package_versions incomplete for {spec.key}: {versions}",
    )
    all_ok &= _check(
        model_entry.get("file_sha256") == sha256_file(out_path),
        f"metadata file_sha256 matches for {spec.key}",
        f"metadata file_sha256 mismatch for {spec.key}",
    )
    return all_ok


def validate_outputs_only(specs: list[ModelSpec]) -> int:
    """Validate local outputs without loading models or sentence-transformers."""
    print("=" * 60)
    print("F011: Validate existing NEO-PI-R embedding artifacts")
    print("=" * 60)
    print("[INFO] Pure local validation: no sentence-transformers imports, model loads, or downloads.")

    all_ok = validate_environment_file()

    try:
        _, _, _, source = load_item_metadata()
    except Exception as exc:
        print(f"[FAIL] Source metadata validation failed: {exc}")
        return 1

    all_ok &= _check(
        OUTPUT_METADATA_JSON.exists(),
        f"Metadata JSON exists: {relpath(OUTPUT_METADATA_JSON)}",
        f"Missing metadata JSON: {OUTPUT_METADATA_JSON}",
    )
    if not OUTPUT_METADATA_JSON.exists():
        return 1

    try:
        with OUTPUT_METADATA_JSON.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[FAIL] Could not read metadata JSON: {exc}")
        return 1

    all_ok &= validate_manifest_schema(payload, source)
    models = payload.get("models", {}) if isinstance(payload, dict) else {}

    print("\n--- Embedding artifacts ---")
    for spec in specs:
        all_ok &= validate_saved_artifact(spec, models.get(spec.key))

    print("\n" + "=" * 60)
    if all_ok:
        print("F011 validation PASSED.")
        return 0
    print("F011 validation FAILED. See [FAIL] messages above.")
    return 1


# ---------------------------------------------------------------------------
# Main generation flow
# ---------------------------------------------------------------------------
def run_generation(args: argparse.Namespace, specs: list[ModelSpec]) -> int:
    print("=" * 60)
    print("F011: Generate NEO-PI-R sentence-transformer embeddings")
    print("=" * 60)

    _, _, item_texts, source = load_item_metadata()
    manifest = load_existing_manifest()
    assert_manifest_source_compatible(manifest, source)
    ensure_no_overwrite_conflicts(specs, manifest, overwrite=args.overwrite)

    torch_module, SentenceTransformer = import_generation_dependencies()
    device, device_info = resolve_device(args.device, torch_module)
    package_versions = collect_package_versions()
    _ok(f"Package versions: {package_versions}")

    generated_arrays: dict[str, np.ndarray] = {}
    model_entries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        batch_size = args.batch_size or spec.default_batch_size
        try:
            embeddings, entry = generate_one_model(
                spec=spec,
                item_texts=item_texts,
                device=device,
                batch_size=batch_size,
                cache_dir=args.cache_dir,
                SentenceTransformer=SentenceTransformer,
            )
            generated_arrays[spec.key] = embeddings
            model_entries[spec.key] = entry
        finally:
            cleanup_model(torch_module, device)

    # Defer final artifact writes until every selected model has generated
    # successfully.  This avoids leaving partial output files after download/OOM
    # failures in later models.
    for spec in specs:
        out_path = EMBEDDINGS_DIR / spec.output_filename
        save_npy_atomic(out_path, generated_arrays[spec.key])
        file_sha = sha256_file(out_path)
        model_entries[spec.key]["file_sha256"] = file_sha
        _ok(f"Saved {relpath(out_path)}")

    merged = merge_manifest(
        manifest=manifest,
        source=source,
        model_entries=model_entries,
        device_info=device_info,
        package_versions=package_versions,
        overwrite=args.overwrite,
    )
    write_json_atomic(OUTPUT_METADATA_JSON, merged)
    _ok(f"Saved combined metadata → {relpath(OUTPUT_METADATA_JSON)}")

    print("\n[Post-save validation]")
    return validate_outputs_only([MODEL_SPECS[key] for key in model_entries])


def main() -> int:
    args = parse_args()
    specs = selected_specs(args)

    try:
        if args.validate:
            return validate_outputs_only(specs)
        return run_generation(args, specs)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

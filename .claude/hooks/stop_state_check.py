#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from _harness_lib import (
    active_path,
    block,
    features_path,
    git_short_status,
    has_pass_evaluator,
    lr_dir,
    progress_path,
    read_json,
    read_stdin_json,
)


def has_any_file(directory: Path) -> bool:
    try:
        return any(p.is_file() and p.name != ".gitkeep" for p in directory.rglob("*"))
    except Exception:
        return False


def main() -> None:
    hook = read_stdin_json()
    if not features_path().exists() or not active_path().exists():
        return
    if hook.get("stop_hook_active"):
        return

    active = read_json(active_path(), {}) or {}
    feature_id = active.get("featureId")
    if not feature_id:
        return

    dirty = bool(active.get("dirty"))
    git_dirty = bool(git_short_status())
    evidence_dir = lr_dir() / "evidence" / feature_id
    missing = []

    if dirty or git_dirty:
        if not progress_path().exists():
            missing.append("progress.md missing")
        if not features_path().exists():
            missing.append("features.json missing")
        if not has_any_file(evidence_dir):
            missing.append(f"evidence directory has no artifacts: {evidence_dir}")

    # If active feature claims completed, require a PASS evaluator and clean git.
    status = active.get("status")
    if status == "completed":
        if not has_pass_evaluator(feature_id):
            missing.append("completed active feature lacks evaluator PASS report")
        if git_dirty:
            missing.append("git status is dirty after completed feature; update state and commit or explicitly mark needs_work/blocked")

    if missing:
        block(
            "Long-running harness stop gate: finish handoff before stopping for "
            + feature_id
            + "\n- "
            + "\n- ".join(missing)
            + "\nRequired: update progress.md/features.json, save evidence, and commit completed work or mark needs_work/blocked."
        )

if __name__ == "__main__":
    main()

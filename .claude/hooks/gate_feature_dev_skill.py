#!/usr/bin/env python3
"""PreToolUse hook: when Claude invokes Skill(feature-dev, ...), ensure active-feature.json is synced."""

from __future__ import annotations

import re
from _harness_lib import (
    active_path,
    features_path,
    lr_dir,
    now_iso,
    read_json,
    read_stdin_json,
    write_json,
)


def extract_feature_id(text: str) -> str | None:
    m = re.search(r"\b(F\d{3,})\b", text)
    return m.group(1) if m else None


def main() -> None:
    hook = read_stdin_json()
    tool_name = hook.get("tool_name", "")

    if tool_name != "Skill":
        return

    # Only handle feature-dev skill invocations
    skill_name = hook.get("tool_input", {}).get("skill", "")
    if skill_name != "feature-dev":
        return

    if not features_path().exists():
        return

    # Extract feature ID from the skill arguments
    args = str(hook.get("tool_input", {}).get("args", ""))
    feature_id = extract_feature_id(args)

    if not feature_id:
        # Try to infer from active-feature.json
        active = read_json(active_path(), {}) or {}
        feature_id = active.get("featureId")

    if not feature_id:
        return

    # Sync active-feature.json
    active = read_json(active_path(), {}) or {}
    if active.get("featureId") and active.get("featureId") != feature_id:
        # Feature mismatch — user might be switching. Allow but update.
        pass

    active.update({
        "featureId": feature_id,
        "phase": "feature-dev",
        "status": active.get("status") or "in_progress",
        "startedAt": active.get("startedAt") or now_iso(),
        "dirty": active.get("dirty", False),
        "changedFiles": active.get("changedFiles", []),
        "evidenceDir": f".claude/long-running/evidence/{feature_id}",
    })
    write_json(active_path(), active)
    (lr_dir() / "evidence" / feature_id).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
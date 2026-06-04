#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from _harness_lib import (
    active_path,
    deny_pretool,
    features_path,
    has_pass_evaluator,
    is_business_code_path,
    is_harness_path,
    read_json,
    read_stdin_json,
    tool_file_paths,
)


def incoming_text(hook: dict) -> str:
    ti = hook.get("tool_input") or {}
    parts = []
    for key in ("content", "new_string", "old_string"):
        val = ti.get(key)
        if isinstance(val, str):
            parts.append(val)
    edits = ti.get("edits")
    if isinstance(edits, list):
        for e in edits:
            if isinstance(e, dict):
                for key in ("new_string", "old_string", "replacement"):
                    val = e.get(key)
                    if isinstance(val, str):
                        parts.append(val)
    return "\n".join(parts)


def main() -> None:
    hook = read_stdin_json()
    tool = hook.get("tool_name", "")
    paths = tool_file_paths(hook)
    text = incoming_text(hook)

    # Before init, do not block normal setup work.
    if not features_path().exists():
        return

    # Gate code edits: business files require an active feature.
    if tool in {"Edit", "Write", "MultiEdit"}:
        business_paths = [p for p in paths if is_business_code_path(p)]
        if business_paths:
            active = read_json(active_path(), None)
            if not active:
                deny_pretool(
                    "Long-running harness blocked this edit: business code changes require an active feature. Run /long-running-coding or create .claude/long-running/active-feature.json first."
                )
            phase = str(active.get("phase", ""))
            if phase not in {"delegated_to_feature_dev", "feature-dev", "implementation", "in_progress"}:
                deny_pretool(
                    f"Long-running harness blocked this edit: active feature phase is {phase!r}, not a feature-dev implementation phase."
                )

    # Gate completion edits to features.json.
    features_rel = ".claude/long-running/features.json"
    touches_features = any(p.replace("\\", "/").endswith(features_rel) or p.replace("\\", "/") == features_rel for p in paths)
    if touches_features:
        if '"acceptanceCriteria"' in text and re.search(r'"acceptanceCriteria"\s*:\s*\[\s*\]', text):
            deny_pretool("Long-running harness blocked this edit: acceptanceCriteria must not be deleted or emptied.")

        wants_completed = bool(re.search(r'"status"\s*:\s*"completed"', text))
        if wants_completed:
            active = read_json(active_path(), {}) or {}
            feature_id = active.get("featureId")
            if not feature_id:
                # Fallback: try to infer from nearby text.
                m = re.search(r'"id"\s*:\s*"(F\d{3,})"', text)
                feature_id = m.group(1) if m else ""
            if not feature_id:
                deny_pretool("Long-running harness blocked completed status: no active featureId was found.")
            if not has_pass_evaluator(feature_id):
                deny_pretool(
                    f"Long-running harness blocked completed status for {feature_id}: missing evaluator-report.json with verdict PASS."
                )

        # Retry limit gate: prevent needs_work → in_progress if attempts >= 3.
        wants_in_progress = bool(re.search(r'"status"\s*:\s*"in_progress"', text))
        if wants_in_progress:
            m = re.search(r'"id"\s*:\s*"(F\d{3,})"', text)
            fid = m.group(1) if m else ""
            if fid:
                features_data = read_json(features_path(), {}) or {}
                for feat in features_data.get("features", []) or []:
                    if feat.get("id") == fid:
                        attempts = feat.get("attempts") or []
                        attempt_count = len(attempts) if isinstance(attempts, list) else 0
                        if attempt_count >= 3 and feat.get("status") == "needs_work":
                            deny_pretool(
                                f"Long-running harness blocked status change for {fid}: "
                                f"reached {attempt_count} attempts (limit 3). "
                                f"Either escalate to blocked or manually intervene with a plan before retrying."
                            )
                        break

if __name__ == "__main__":
    main()

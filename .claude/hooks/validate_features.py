#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from _harness_lib import block, features_path, has_pass_evaluator, project_dir, read_json

ALLOWED = {"pending", "in_progress", "ready_for_eval", "needs_work", "completed", "blocked"}
MAX_ATTEMPTS = 3


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["features.json must be a JSON object"]
    features = data.get("features")
    if not isinstance(features, list):
        return ["features.json must contain a features array"]
    seen = set()
    for i, f in enumerate(features):
        if not isinstance(f, dict):
            errors.append(f"features[{i}] must be an object")
            continue
        fid = f.get("id")
        if not fid:
            errors.append(f"features[{i}] missing id")
            continue
        if fid in seen:
            errors.append(f"duplicate feature id: {fid}")
        seen.add(fid)
        if not f.get("title") and not f.get("description"):
            errors.append(f"{fid}: missing title/description")
        status = f.get("status")
        if status not in ALLOWED:
            errors.append(f"{fid}: invalid status {status!r}")
        ac = f.get("acceptanceCriteria")
        if not isinstance(ac, list) or not ac:
            errors.append(f"{fid}: acceptanceCriteria must be a non-empty array")
        if status == "completed":
            if not f.get("completedAt"):
                errors.append(f"{fid}: completed feature missing completedAt")
            ev = f.get("evidence") or {}
            report = ev.get("evaluatorReport") if isinstance(ev, dict) else None
            if not report:
                errors.append(f"{fid}: completed feature missing evidence.evaluatorReport")
            if not has_pass_evaluator(fid):
                errors.append(f"{fid}: completed feature missing evaluator PASS report")

        # Retry limit: features should not stay needs_work or in_progress with >= MAX_ATTEMPTS
        attempts = f.get("attempts") or []
        attempt_count = len(attempts) if isinstance(attempts, list) else 0
        if attempt_count >= MAX_ATTEMPTS and status in ("needs_work", "in_progress"):
            errors.append(
                f"{fid}: has {attempt_count} attempts (limit {MAX_ATTEMPTS}) with status {status!r}. "
                "Manual intervention required — escalate to blocked or adjust acceptance criteria."
            )
        if status == "needs_work" and f.get("blockedReason") and not errors:
            # needs_work with a blockedReason is inconsistent — should be blocked
            errors.append(
                f"{fid}: status is needs_work but blockedReason is set. "
                "If truly blocked, set status=blocked. If fixable, clear blockedReason."
            )
    return errors


def main() -> None:
    # If not initialized, no-op.
    path = features_path()
    if not path.exists():
        return
    data = read_json(path, None)
    if data is None:
        block("Long-running harness: .claude/long-running/features.json is invalid JSON.")
    errors = validate(data)
    if errors:
        block("Long-running harness features.json validation failed:\n- " + "\n- ".join(errors))
    if "--hook" not in sys.argv:
        print("features.json OK")

if __name__ == "__main__":
    main()

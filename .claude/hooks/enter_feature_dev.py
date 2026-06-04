#!/usr/bin/env python3
from __future__ import annotations

import re
from _harness_lib import active_path, lr_dir, now_iso, read_json, read_stdin_json, write_json


def main() -> None:
    hook = read_stdin_json()
    args = " ".join(str(hook.get(k, "")) for k in ("command_args", "prompt", "expanded_prompt"))
    m = re.search(r"\b(F\d{3,})\b", args)
    feature_id = m.group(1) if m else None
    if not feature_id:
        return
    active = read_json(active_path(), {}) or {}
    if active.get("featureId") and active.get("featureId") != feature_id:
        # Do not silently switch active features.
        return
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

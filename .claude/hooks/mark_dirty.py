#!/usr/bin/env python3
from __future__ import annotations

from _harness_lib import active_path, is_business_code_path, now_iso, read_json, read_stdin_json, tool_file_paths, write_json


def main() -> None:
    hook = read_stdin_json()
    paths = tool_file_paths(hook)
    business = [p for p in paths if is_business_code_path(p)]
    if not business or not active_path().exists():
        return
    active = read_json(active_path(), {}) or {}
    active["dirty"] = True
    active["lastModifiedAt"] = now_iso()
    existing = list(active.get("changedFiles") or [])
    for p in business:
        if p not in existing:
            existing.append(p)
    active["changedFiles"] = existing
    write_json(active_path(), active)

if __name__ == "__main__":
    main()

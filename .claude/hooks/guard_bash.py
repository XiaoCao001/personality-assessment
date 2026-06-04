#!/usr/bin/env python3
from __future__ import annotations

import re
from _harness_lib import deny_pretool, features_path, read_stdin_json

DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+(/|~|\$HOME)(\s|$)",
    r"\bsudo\b",
    r"\bchmod\s+-R\s+777\b",
    r"\bgit\s+push\s+--force\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-fdx\b",
]


def main() -> None:
    hook = read_stdin_json()
    cmd = ((hook.get("tool_input") or {}).get("command") or "").strip()
    if not cmd:
        return
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            deny_pretool(f"Long-running harness blocked dangerous Bash command: {cmd}")

    # If initialized, discourage committing a completed feature without evidence.
    # Detailed completion validation is handled by pre_tool_gate.py and validate_features.py.
    if features_path().exists() and re.match(r"git\s+commit\b", cmd):
        return

if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 .claude/hooks/validate_features.py || true

printf '\nHarness files:\n'
find .claude -maxdepth 4 -type f | sort

printf '\nHook scripts executable check:\n'
find .claude/hooks -type f -name '*.py' -maxdepth 1 -print -exec test -x {} \; -exec echo '  ok: {}' \;

printf '\nGit status:\n'
git status --short || true

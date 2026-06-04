#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/target-project" >&2
  exit 2
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$1"
mkdir -p "$DST"

copy_safe() {
  local item="$1"
  if [ -e "$DST/$item" ]; then
    echo "SKIP existing $item (merge manually if needed)"
  else
    cp -R "$SRC/$item" "$DST/$item"
    echo "COPIED $item"
  fi
}

copy_safe CLAUDE.md
copy_safe README.md
copy_safe MIGRATION.md

mkdir -p "$DST/.claude"
for item in settings.json agents hooks skills long-running; do
  if [ -e "$DST/.claude/$item" ]; then
    echo "SKIP existing .claude/$item (merge manually if needed)"
  else
    cp -R "$SRC/.claude/$item" "$DST/.claude/$item"
    echo "COPIED .claude/$item"
  fi
done

mkdir -p "$DST/scripts"
for f in "$SRC"/scripts/*.sh; do
  base="$(basename "$f")"
  if [ -e "$DST/scripts/$base" ]; then
    echo "SKIP existing scripts/$base"
  else
    cp "$f" "$DST/scripts/$base"
    echo "COPIED scripts/$base"
  fi
done

chmod +x "$DST"/.claude/hooks/*.py 2>/dev/null || true
chmod +x "$DST"/scripts/*.sh 2>/dev/null || true

echo "Installed. Start Claude Code from: $DST"
echo "Then run: /long-running-init"

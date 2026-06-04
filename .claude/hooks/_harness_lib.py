#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()


def lr_dir() -> Path:
    return project_dir() / ".claude" / "long-running"


def features_path() -> Path:
    return lr_dir() / "features.json"


def active_path() -> Path:
    return lr_dir() / "active-feature.json"


def progress_path() -> Path:
    return lr_dir() / "progress.md"


def read_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"_raw": raw}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel_to_project(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(str(path))
    try:
        if p.is_absolute():
            return str(p.resolve().relative_to(project_dir()))
    except Exception:
        pass
    return str(p).replace("\\", "/")


def tool_file_paths(hook: dict[str, Any]) -> list[str]:
    ti = hook.get("tool_input") or {}
    paths: list[str] = []
    for key in ("file_path", "path"):
        val = ti.get(key)
        if isinstance(val, str):
            paths.append(rel_to_project(val))
    # MultiEdit sometimes has file_path once and edits array.
    return [p for p in paths if p]


def is_harness_path(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    allowed_prefixes = (
        ".claude/long-running/",
        ".claude/hooks/",
        ".claude/skills/",
        ".claude/agents/",
    )
    allowed_files = {"CLAUDE.md", "README.md", "MIGRATION.md"}
    return rel in allowed_files or rel.startswith(allowed_prefixes)


def is_business_code_path(rel: str) -> bool:
    if not rel:
        return False
    rel = rel.replace("\\", "/")
    if is_harness_path(rel):
        return False
    if rel.startswith(".git/"):
        return False
    return True


def deny_pretool(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    raise SystemExit(0)


def block(reason: str) -> None:
    print(reason, file=sys.stderr)
    raise SystemExit(2)


def git(args: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(["git"] + args, cwd=project_dir(), text=True, capture_output=True, timeout=20)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def git_short_status() -> str:
    code, out, err = git(["status", "--short"])
    return out if code == 0 else ""


def feature_by_id(data: dict[str, Any], feature_id: str) -> dict[str, Any] | None:
    for f in data.get("features", []) or []:
        if f.get("id") == feature_id:
            return f
    return None


def evaluator_report_for(feature_id: str) -> dict[str, Any] | None:
    data = read_json(features_path(), {}) or {}
    feature = feature_by_id(data, feature_id) if feature_id else None
    paths: list[Path] = []
    if feature:
        ev = feature.get("evidence") or {}
        report = ev.get("evaluatorReport")
        if report:
            paths.append(project_dir() / report)
    paths.append(lr_dir() / "evidence" / feature_id / "evaluator-report.json")
    for p in paths:
        report = read_json(p, None)
        if isinstance(report, dict):
            return report
    return None


def has_pass_evaluator(feature_id: str) -> bool:
    report = evaluator_report_for(feature_id)
    return bool(report and str(report.get("verdict", "")).upper() == "PASS")


def file_mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone().isoformat(timespec="seconds")
    except Exception:
        return None

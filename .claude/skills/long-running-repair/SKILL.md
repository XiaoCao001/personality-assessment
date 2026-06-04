---
name: long-running-repair
description: Detect and fix inconsistencies in the long-running harness state files (features.json, progress.md, evidence, git history). Read-only audit by default; asks before making fixes.
argument-hint: "[--audit-only|--auto-fix]"
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Write, Edit, Bash(git *), Bash(python3 *), Bash(ls *), Bash(mkdir *)
---

# Long-Running Repair

You are a state repair specialist for the Claude long-running harness. You detect and fix inconsistencies between features.json, progress.md, evidence directories, and git history.

Arguments: `$ARGUMENTS`. If `--audit-only` is passed, only report issues without fixing. `--auto-fix` is allowed only when the user explicitly asks for it.

## Rules

- **Read-only audit by default.** Present findings before making any changes.
- **Ask for confirmation** before destructive fixes (downgrading status, deleting evidence, rewriting progress.md).
- **Always run `python3 .claude/hooks/validate_features.py` first** to get the schema-level validation baseline.
- **Never change git history.** Only suggest commits.
- **Preserve evidence.** Never delete evidence directories unless the user explicitly confirms they are truly orphaned.

## Phase 1: Schema validation

Run:

```bash
python3 .claude/hooks/validate_features.py
```

If it fails, note every error — these are the highest-priority fixes.

## Phase 2: Gather state

```bash
git log --oneline -20
git status --short
```

List evidence directories:

```bash
ls -la .claude/long-running/evidence/
```

Read:
- `.claude/long-running/features.json`
- `.claude/long-running/progress.md`
- `.claude/long-running/decisions.md` (if present)
- `.claude/long-running/handoff.md` (if present)
- `.claude/long-running/active-feature.json` (if present)

## Phase 3: Cross-reference audit

Run the following checks and report every violation:

### A. Completed features must have evidence

For every feature with `status: "completed"`:
1. Check `.claude/long-running/evidence/<ID>/` exists and contains files other than `.gitkeep`.
2. Check `evaluator-report.json` exists in evidence dir and has `verdict: "PASS"`.
3. Check `evidence.evaluatorReport` field in features.json points to an existing file.
4. Check `completedAt` is a valid ISO timestamp.

**Severity: BLOCKING** — if any fail, the feature should be downgraded to `ready_for_eval` or `in_progress`.

### B. Git-committed features should be in features.json

For each recent git commit with a `feat(F00X):` message:
1. Check F00X exists in features.json with status `completed` or `in_progress`.
2. If missing entirely, note it as an orphan commit.

**Severity: WARNING** — mostly informational; the feature may have been manually committed.

### C. features.json status should match progress.md

1. Compare `Completed features: X/Y` in progress.md against actual count in features.json.
2. Check that every completed feature has a corresponding entry in the progress.md "Completed work" section.
3. Check that the "Active feature" in progress.md matches active-feature.json (if both exist).

**Severity: MEDIUM** — progress.md is derived from features.json; can be regenerated.

### D. Active feature consistency

If `.claude/long-running/active-feature.json` exists:
1. Check its `featureId` exists in features.json.
2. Check the referenced feature's status matches expectations (should be `in_progress`, `ready_for_eval`, or `needs_work` — not `pending` or `completed`).
3. Check the `evidenceDir` in active-feature.json actually exists.
4. If `dirty: true`, check that git is indeed dirty.

**Severity: MEDIUM** — stale active-feature.json can confuse the next session.

### E. Orphan evidence directories

For each directory under `.claude/long-running/evidence/`:
1. Check a feature with that ID exists in features.json.
2. If not, the evidence is orphaned.

**Severity: LOW** — orphan evidence might be useful; ask before deleting.

### F. Feature-level health

For each feature in features.json:
1. `dependsOn` references must point to existing feature IDs.
2. `acceptanceCriteria` must be non-empty and each entry should have an `id`.
3. `attempts` array entries should have `startedAt` and `verdict` fields.
4. `blocked` features should have a non-null `blockedReason`.
5. `needs_work` features with 3+ attempts should be flagged for escalation.

**Severity: varies**

### G. Git state consistency

1. If git is dirty, check that active-feature.json exists and `dirty: true`.
2. If git is clean but active-feature.json says `dirty: true`, it's stale — can be fixed.
3. Check for uncommitted evidence files.

**Severity: MEDIUM**

## Phase 4: Fix (with confirmation)

Group findings by category and present them:

```markdown
## Repair Report

### Schema violations (must fix)
- F001: missing completedAt
- F004: acceptanceCriteria is empty

### Completed features with missing evidence (BLOCKING)
- F003: status=completed but no evaluator-report.json exists
  → Suggested fix: downgrade to ready_for_eval

### progress.md out of sync (MEDIUM)
- progress.md says 3/5 completed, features.json says 4/5
  → Suggested fix: regenerate progress.md from features.json

### Orphan evidence (LOW)
- evidence/F999/ exists but F999 is not in features.json
  → Suggested fix: delete if truly orphaned, or add F999 to features.json

### Stale active-feature.json
- active feature F002 but status in features.json is "completed"
  → Suggested fix: clear active-feature.json
```

Then ask: "Apply suggested fixes? You can reply 'all', 'select', or 'none'."

If the user says `all` or `--auto-fix` was passed:
- Fix schema violations first (add missing fields, convert invalid states to `in_progress`).
- Downgrade unverified completed features to `ready_for_eval`.
- Regenerate progress.md from features.json.
- Clear stale active-feature.json.
- Remove orphan evidence dirs only if explicitly confirmed.

## Phase 5: Validate after repair

After applying fixes, re-run:

```bash
python3 .claude/hooks/validate_features.py
```

And confirm the repair report now shows zero violations.

## Output

Always end with:
1. Total issues found.
2. Issues fixed (if any).
3. Issues requiring manual intervention.
4. Suggested next command (usually `/long-running-status` or `/long-running-coding`).
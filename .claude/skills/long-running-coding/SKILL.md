---
name: long-running-coding
description: Continue a long-running project by selecting exactly one feature, delegating implementation to the installed feature-dev plugin, running evaluator, updating state, and committing.
argument-hint: "[optional feature-id]"
disable-model-invocation: true
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, Bash(git *), Bash(python3 *), Bash(npm *), Bash(pnpm *), Bash(yarn *), Bash(pytest *), Bash(make *), Bash(./scripts/*), Bash(mkdir *), Bash(date *), Skill(feature-dev *)
---

# Long-Running Coding Orchestrator

You are the orchestrator, not the builder. The installed external `/feature-dev` plugin is the builder.

Requested feature: `$ARGUMENTS`

## Hard rules

- Work on exactly one feature.
- Do not implement unrelated requirements.
- Do not mark a feature `completed` until evaluator returns `PASS` and evidence exists.
- Do not rewrite or vendor the external `feature-dev` plugin.
- Keep state in `.claude/long-running/`.
- If in doubt, preserve state and mark `blocked` or `needs_work` rather than pretending completion.

## Phase 1: Load state

Run/read:

```bash
pwd
git status --short
git log --oneline -10
```

Read:

- `CLAUDE.md`
- `.claude/long-running/progress.md`
- `.claude/long-running/features.json`
- `.claude/long-running/active-feature.json` if present
- `.claude/long-running/handoff.md` if present

If `features.json` is missing, stop and tell the user to run `/long-running-init`.

## Phase 2: Select exactly one feature

Selection rules:

1. If `$ARGUMENTS` contains a feature ID, select that feature.
2. Otherwise resume `active-feature.json` if it exists and status is not `completed`.
3. Otherwise choose the highest-priority feature with status `needs_work` or `pending` whose dependencies are completed.
4. If none are available, report status and blockers.

Before starting, ensure selected feature status is one of:

- `pending`
- `in_progress`
- `needs_work`
- `ready_for_eval` only if the next step is evaluation

## Phase 3: Create active feature lock

Update `.claude/long-running/active-feature.json`:

```json
{
  "featureId": "F00X",
  "phase": "delegated_to_feature_dev",
  "status": "in_progress",
  "startedAt": "ISO timestamp",
  "startedFromCommit": "git rev-parse --short HEAD",
  "dirty": false,
  "changedFiles": [],
  "evidenceDir": ".claude/long-running/evidence/F00X",
  "lastEvaluatorVerdict": null,
  "handoffPromptFile": ".claude/long-running/evidence/F00X/feature-dev-handoff.md"
}
```

Create the evidence directory.

Set the feature status in `features.json` to `in_progress` and append an `attempts` entry.

## Phase 4: Delegate to external feature-dev plugin

### 4a. Write handoff file

Write `.claude/long-running/evidence/F00X/feature-dev-handoff.md`:

```markdown
# Feature-dev handoff for F00X

Use the installed `/feature-dev` plugin workflow for this single selected feature.

## Feature
- ID: F00X
- Title: ...
- Description: ...
- Priority: high|medium|low
- Dependencies: [F00Y, ...]

## Acceptance criteria
1. [AC001] ...
2. [AC002] ...

## Test plan
- ...

## Constraints
- Implement only this feature. Do not touch unrelated code.
- Do NOT mark this feature `completed` in features.json — that is the orchestrator's job.
- Save verification artifacts under `.claude/long-running/evidence/F00X/`.
- After implementation, summarize: changed files, commands run, test results, risks, and any incomplete criteria.
```

### 4b. Invoke feature-dev skill

Invoke the installed `feature-dev` skill with the handoff context. Use the Skill tool:

```
Skill(feature-dev, <summary of handoff including feature ID, title, description, acceptance criteria, and evidence directory path>)
```

The `feature-dev` plugin will implement the feature. **Wait for it to complete** before proceeding.

### 4c. If Skill invocation fails

If the Skill tool cannot invoke `feature-dev` (e.g., plugin not installed or invocation rejected), fall back: present the handoff file path and ask the user to run:

```text
/feature-dev .claude/long-running/evidence/F00X/feature-dev-handoff.md
```

Then wait for the user to confirm feature-dev has completed before continuing.

## Phase 5: Collect evidence

Save evidence under `.claude/long-running/evidence/F00X/`:

- `commands.log`: commands run and outcomes
- `test-output.txt`: test/check output
- `git-diff.patch`: `git diff HEAD`
- screenshots/logs if relevant
- `feature-dev-summary.md`: builder summary

Run the most relevant verification commands from the feature `testPlan`. If the project has `scripts/check.sh` or `scripts/test.sh`, use them unless the feature needs a narrower command.

Set feature status to `ready_for_eval` after evidence exists.

## Phase 6: Run evaluator

Use the `evaluator` subagent to evaluate only the selected feature. Provide it:

- feature JSON
- acceptance criteria
- evidence directory
- `git diff HEAD`
- test output path
- relevant changed files

The evaluator must return JSON with:

```json
{
  "featureId": "F00X",
  "verdict": "PASS|NEEDS_WORK|BLOCKED",
  "criteria": [
    { "criterionId": "AC001", "status": "pass|fail|unknown", "evidence": [], "notes": "" }
  ],
  "requiredFixes": [],
  "riskNotes": []
}
```

Save it to `.claude/long-running/evidence/F00X/evaluator-report.json`.

## Phase 7: Decide

If evaluator verdict is `PASS`:

1. Update every satisfied acceptance criterion with status `pass` and evidence links.
2. Set feature status `completed`.
3. Set `completedAt` and `evidence.evaluatorReport`.
4. Update `progress.md` with implementation summary, tests, evidence, changed files.
5. Update `handoff.md` with next recommended feature.
6. Commit with:

```bash
git add .
git commit -m "feat(F00X): <short title>"
```

If verdict is `NEEDS_WORK`:

1. Count existing `attempts` entries for this feature in `features.json`.
2. Write `.claude/long-running/findings/F00X-attempt-N.md` with the evaluator's `requiredFixes`.
3. **If the feature has already had 3 or more `attempts` (including the current one):**
   - Auto-escalate to `blocked`. Set status `blocked` and record in `blockedReason`: "Exceeded 3 NEEDS_WORK attempts. Manual intervention required."
   - Update `progress.md` and `handoff.md` with the block reason and all findings.
   - Do NOT attempt another pass without user direction.
4. **If under 3 attempts:**
   - Set feature status `needs_work`.
   - Update `progress.md` with required fixes.
   - Either run another feature-dev pass for the same feature or stop with clear next steps.

If verdict is `BLOCKED`:

1. Set feature status `blocked`.
2. Record blocker in `progress.md`, `features.json`, and `handoff.md`.
3. Do not mark completed.

## Output

Return:

1. Selected feature ID/title.
2. Whether feature-dev ran or needs the user to invoke it.
3. Evidence files created.
4. Evaluator verdict.
5. State updates.
6. Commit hash if committed.
7. Next recommended feature.

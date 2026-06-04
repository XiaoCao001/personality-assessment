---
name: long-running-init
description: Initialize the Claude long-running harness for a complex project by clarifying requirements, decomposing features, and creating .claude/long-running state files.
argument-hint: "[optional project brief]"
disable-model-invocation: true
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, Bash(git *), Bash(pwd), Bash(ls *), Bash(mkdir *), Bash(chmod *), Bash(python3 *)
---

# Long-Running Init

You are the initializer for this project's long-running Claude Code harness.

Input brief: `$ARGUMENTS`

## Goal

Create a durable cross-session workspace for a multi-feature project:

- `.claude/long-running/progress.md`
- `.claude/long-running/features.json`
- `.claude/long-running/decisions.md`
- `.claude/long-running/handoff.md`
- `.claude/long-running/evidence/`
- optional project scripts: `scripts/bootstrap.sh`, `scripts/check.sh`, `scripts/test.sh`

Do not create or overwrite the external `feature-dev` plugin. It is assumed to be installed already.

## Phase 1: Discovery before writing

If the user has not already provided enough information, ask focused questions before writing files. Prefer one compact question set at a time.

Clarify:

1. Project goal and target users.
2. Existing repo or greenfield project.
3. Tech stack, package manager, test framework, launch command.
4. Non-goals and constraints.
5. First useful milestone.
6. How each feature can be verified.
7. Any dependencies between features.

Proceed without more questions only when you can produce small, testable features with concrete acceptance criteria.

## Phase 2: Decompose

Create features that can usually be completed in one Claude Code session.

Each feature must include:

- `id`: `F001`, `F002`, ...
- `title`
- `category`: `functional`, `technical`, `ui`, `api`, `test`, `infra`, or `docs`
- `description`
- `priority`: `high`, `medium`, or `low`
- `dependsOn`: array of feature IDs
- `steps`: short implementation outline
- `acceptanceCriteria`: array of objects `{ id, text, status, evidence }`; default status is `unknown`
- `testPlan`: commands or manual checks
- `status`: usually `pending`
- `evidence`: directory/test/report placeholders
- `attempts`: empty array
- `createdAt`, `updatedAt`, `completedAt`

## Phase 3: Materialize files

Create directories:

```bash
mkdir -p .claude/long-running/{evidence,findings,runtime,templates}
mkdir -p scripts
```

Create `.claude/long-running/features.json` that conforms to `.claude/long-running/features.schema.json`.

Create `.claude/long-running/progress.md` with:

```markdown
# Project Progress

## Project overview
[summary]

## Current status
- Phase: initialized
- Last updated: [ISO timestamp]
- Completed features: 0 / [total]
- Active feature: none

## Completed work
- [timestamp] Harness initialized

## Current risks / blockers
- None known

## Next recommended feature
- [F001] [title]

## Session log
### [timestamp] Initialization
- Created long-running state files.
- Generated feature list and verification plan.
```

Create `.claude/long-running/decisions.md` for long-term decisions.

Create `.claude/long-running/handoff.md` with the next action and files to read.

If no `scripts/check.sh` exists, create a conservative placeholder that explains how to customize it. Do not guess destructive commands.

## Phase 4: Git setup

Run:

```bash
git status --short
git log --oneline -5
```

If this is not a git repo, ask before running `git init`. If it is a repo, commit only after the user agrees or if they explicitly asked for a full initialization commit.

## Output

Return:

1. Files created or changed.
2. Number of features generated.
3. First recommended feature.
4. Any unanswered assumptions.
5. Suggested next command: `/long-running-coding` or `/long-running-coding F00X`.

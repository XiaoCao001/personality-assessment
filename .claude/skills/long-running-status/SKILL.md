---
name: long-running-status
description: Show the current project status for the Claude long-running harness by reading progress, features, active feature, evidence, and git state.
argument-hint: "[optional --json]"
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Bash(pwd), Bash(git *), Bash(python3 *), Bash(ls *)
---

# Long-Running Status

You are a status reporter. Do not modify files.

Arguments: `$ARGUMENTS`

## Steps

1. Run:

```bash
pwd
git status --short
git log --oneline -5
```

2. Read if present:

- `.claude/long-running/progress.md`
- `.claude/long-running/features.json`
- `.claude/long-running/active-feature.json`
- `.claude/long-running/handoff.md`

3. Count features by status:

- `pending`
- `in_progress`
- `ready_for_eval`
- `needs_work`
- `completed`
- `blocked`

4. Check evidence health:

- completed features missing evaluator report
- completed features missing evidence directory
- in-progress active feature without evidence directory
- dirty git state

## Output format

```markdown
## 项目状态报告

### 基本信息
- 项目名称: ...
- 工作目录: ...
- Git 状态: clean / dirty
- 最近提交: ...

### 进度概览
- 总功能数: X
- 已完成: X (X%)
- 进行中: X
- 待验收: X
- 需返工: X
- 待处理: X
- 阻塞: X

### 当前 active feature
- ID: ...
- Phase: ...
- Evidence: ...

### 最近完成
1. ...

### 需要注意
- ...

### 建议下一步
- ...
```

If `$ARGUMENTS` contains `--json`, also include a compact JSON summary at the end.

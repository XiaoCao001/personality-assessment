# Claude Code Long-Running Harness

本项目使用长期运行开发 harness。请把这里当作每个 Claude Code 会话都必须遵守的项目宪法；详细流程放在 `.claude/skills/`，状态记忆放在 `.claude/long-running/`。

## 常用命令

- 验证 features.json 结构和状态合法性：`python3 .claude/hooks/validate_features.py`
- 检查 harness 健康状态（结构 + hook 可执行性 + git 状态）：`./scripts/check-harness.sh`
- 安装 harness 到目标项目：`./scripts/install-into-project.sh /path/to/target-project`
- 四个入口 skill：`/long-running-init`（初始化）、`/long-running-coding`（继续开发）、`/long-running-status`（查看状态）、`/long-running-repair`（修复状态不一致）

## 架构概览

本 harness 是**编排器**，不是构建器。实际业务实现交给外部 `/feature-dev` 插件。Harness 自身由三层机制组成：

### Skills — 入口点

四个 skill 定义在 `.claude/skills/`，每个是独立 markdown 文件：
- `long-running-init/SKILL.md`：需求澄清 → 功能拆解 → 生成状态文件
- `long-running-coding/SKILL.md`：选 feature → 生成 handoff → 委托 feature-dev → 收集证据 → 调 evaluator → 更新状态（含重试上限：同一 feature 连续 3 次 NEEDS_WORK 自动 escalate 为 blocked）
- `long-running-status/SKILL.md`：只读报告，汇总进度、active feature 和证据健康检查
- `long-running-repair/SKILL.md`：检测并修复 features.json / progress.md / evidence / git 之间的不一致

### Hooks — 门禁

定义在 `.claude/settings.json`，实现在 `.claude/hooks/`：

| Hook 脚本 | 触发时机 | 作用 |
|---|---|---|
| `pre_tool_gate.py` | Edit/Write/MultiEdit 前 | 拦截无 active feature 的业务代码编辑；拦截没有 evaluator PASS 就把 status 改为 completed |
| `guard_bash.py` | Bash 前 | 拦截危险命令（`rm -rf /`、`sudo`、`git push --force` 等） |
| `mark_dirty.py` | Edit/Write/MultiEdit 后 | 业务文件被修改后标记 active feature 为 dirty，记录 changedFiles |
| `validate_features.py` | 每轮工具调用后 | 校验 features.json 结构合法性（schema + 状态一致性） |
| `stop_state_check.py` | 会话结束时 | 如果 active feature dirty 但没有证据/progress/commit，阻止退出 |
| `enter_feature_dev.py` | 用户输入含 feature-dev 时 | 从 prompt 中提取 feature ID 自动填充 active-feature.json |

共享库 `_harness_lib.py` 提供路径解析、JSON 读写、git 操作等工具函数。

### Evaluator agent — 独立验收

定义在 `.claude/agents/evaluator.md`。只读运行，不写代码。按 acceptanceCriteria 逐条检查 concrete evidence（diff、日志、截图、测试输出）。默认 verdict 为 `NEEDS_WORK`；只有每条 criteria 都有证据支撑才返回 `PASS`。

## State files

- `.claude/long-running/features.json` 是结构化功能清单和唯一状态源，必须符合 `.claude/long-running/features.schema.json`。
- `.claude/long-running/progress.md` 是给人和下一轮 agent 读取的项目交接记录。
- `.claude/long-running/active-feature.json` 是当前只允许处理的 feature 锁文件。
- `.claude/long-running/evidence/<FEATURE_ID>/` 保存测试日志、截图、diff、evaluator 报告等证据。
- `.claude/long-running/decisions.md` 保存长期设计决策，替代 iFlow 的 `save_memory`。
- `.claude/long-running/handoff.md` 保存下一轮会话的最短交接。

## Always-on rules

- 开始任何 feature work 前，先读 `progress.md`、`features.json`、`active-feature.json`（如果存在）和最近 git 历史。
- 每轮只处理一个 feature。不要顺手做无关需求。
- 真正的业务实现交给已安装的外部 `/feature-dev` 插件；本 harness 只负责编排、状态、证据和验收。
- 不要让 feature-dev 直接把长期状态改成 `completed`。只有 evaluator PASS 后，`/long-running-coding` 才能完成收尾。
- `completed` 必须同时满足：测试/运行证据存在、evaluator verdict 为 `PASS`、`features.json` 和 `progress.md` 已更新、相关代码已提交。
- 如果遇到阻塞，标记为 `blocked`，在 `progress.md` 和 feature 的 `attempts` 中记录阻塞原因。
- 同一 feature 连续 3 次 evaluator 返回 `NEEDS_WORK` 后自动 escalate 为 `blocked`，需要人工介入。
- 需要查看状态时使用 `/long-running-status`。需要继续开发时使用 `/long-running-coding`。需要初始化时使用 `/long-running-init`。状态文件不一致时使用 `/long-running-repair`。

## Status machine

Feature 状态只能在下列状态之间流转：

```text
pending -> in_progress -> ready_for_eval -> completed
                         -> needs_work -> in_progress
                         -> blocked
```

- `ready_for_eval`：feature-dev 已完成实现，等待 evaluator 独立验收。
- `needs_work`：evaluator 发现未满足验收标准，需要下一轮修复。

禁止从 `pending` 或 `in_progress` 直接跳到 `completed`，除非已有 evaluator PASS 报告和证据目录。

## External dependency

本 harness 假设你已经安装现成的 `feature-dev` 插件。不要在本项目里复制或重写它；只通过 `/long-running-coding` 生成清晰的 handoff，把单个 feature 交给 `/feature-dev`。

从 iFlow 迁移的历史背景见 `MIGRATION.md`。

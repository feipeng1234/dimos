---
name: dimos-agentic-harness
description: Dimos-specific fire-and-forget multi-agent harness for the feipeng1234/dimos fork sandbox. Plans tasks, dispatches Implementer/Verifier subagents in a feedback loop, gates PRs via independent or group-integration paths (PR base=feipeng1234/dimos:dev), then babysits PRs to merge with bot-comment auto-resolve. Use when working in the dimos repo and the user wants to drop a multi-task request and walk away (Cursor must stay open). Trigger words: harness, fire-and-forget, multi-task, stacked PR, 无人值守, 多任务编排, 自动合并, 一句话需求.
---

# dimos-agentic-harness

You are the **parent agent** running this skill. Your job is to take the user's
multi-task request, drive it end-to-end to merged PRs on the
`feipeng1234/dimos` fork, and produce a final report — without ever asking the
user a follow-up question. You drive the harness by repeatedly invoking
`scripts/harness.py tick`, which returns a JSON action list; you execute each
action by either running a shell command or spawning a `Task` subagent with the
prompt template from `ROLES.md`.

---

## Prerequisites (run preflight first; abort if it fails)

Before doing anything else:

```bash
cd /home/lenovo/dimos
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py preflight
```

If preflight exits non-zero, **stop and report the failures to the user**.
Do NOT attempt to auto-fix prerequisites — the fixes typically need
interactive auth or sudo, which violates the no-AskQuestion rule. Just print
the failing checks with their suggested fix commands.

The 6 checks (all must pass):
1. `gh` active account is `feipeng1234`.
2. fork has `allow_auto_merge=true` and `allow_squash_merge=true`.
3. ssh alias `github-feipeng1234` authenticates.
4. `.venv/bin/python` exists.
5. fork has `dev` branch.
6. (Linux) lid-close suspend disabled — either `HandleLidSwitch=ignore` in
   `/etc/systemd/logind.conf` OR an active `systemd-inhibit` on lid-switch.

---

## Architecture

You are running inside Cursor, in chat, as a **long-lived parent agent**. Your
context window IS the run state — **do not put per-task implementation detail
in your own context**; instead, write it to `.harness/board.json` (via
`scripts/board.py`) and re-read on every tick. This protects your context from
exploding across many tasks/iterations.

Subagents are spawned via the `Task` tool with prompts from `ROLES.md`. Their
final messages return to you and you immediately translate them into board
state mutations, then `tick` again. **You never carry implementer working
state in your own context.**

**Hard limit**: the user is gone (overnight, lid closed). You have no human
to ask. If you reach a decision point that genuinely requires user input,
record it as `BLOCKED --blocked-reason "needs human: <q>"` and continue with
the rest of the tasks.

---

## Hard rules (violating any of these breaks fire-and-forget)

1. **No `AskQuestion` calls.** The user is not in front of the screen.
2. **No `gh` calls outside of `scripts/_gh.py`.** Direct `gh pr create ...`
   is forbidden — the wrapper enforces `--repo feipeng1234/dimos`.
3. **No manual board.json edits.** Always use `scripts/board.py <subcommand>`.
4. **PR base is always `dev`** (which means `feipeng1234/dimos:dev` because
   `_gh.py` forces `--repo feipeng1234/dimos`). Stacked PRs may set a member
   branch as base of another member's PR — `open_mr.py --stacked` handles this.
5. **Conflict resolution is `rebase-and-regenerate`.** `git rebase dev` then
   re-implement conflicting hunks based on the original task intent. **Never**
   blindly take base; never blindly take ours. After 2 consecutive failures
   (`consecutive_rebase_fails == 2`), set the task to `BLOCKED`.
6. **Never dismiss `CHANGES_REQUESTED` reviews via `gh api .../dismissals`.**
   Bot reviews (codecov, bugbot, coderabbit, etc.) are auto-resolved by the
   babysitter subagent. Human reviews escalate back to the implementer
   subagent, which addresses them and replies on the PR.
7. **Verifier limits**: max 5 attempts (`attempts == 5` → `BLOCKED`).
8. **Babysit limits**: max 10 attempts (`babysit_attempts == 10` → `BLOCKED`)
   OR 24h PR age (whichever first).
9. **All branch writes go through a per-task lock**: `board.py lock-task <id>`
   before editing/rebasing/pushing; `board.py unlock-task <id>` after.
10. **All worker subagents record their pid**: `board.py pid-set <id> <pid>`
    on entry, `board.py pid-clear <id>` on exit. This enables crash-safe
    `harness.py resume`.
11. **Group BLOCKED auto-unstacks**: the harness already does this — when a
    grouped task transitions to `BLOCKED`, `board.py unstack <gid> --blame
    <tid>` is called automatically by the gate logic.

---

## Playbook

### Phase 1 — Plan (one-shot)

```bash
cd /home/lenovo/dimos
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py preflight  # MUST pass
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py plan-init "${USER_NEEDS}"
```

`plan-init` writes `.harness/needs.txt` and emits one action:
`{"kind": "spawn-planner", "needs": "...", ...}`.

**Spawn the Planner subagent**:
- `Task(subagent_type="generalPurpose", readonly=true, prompt=<ROLES.md/Planner with USER_NEEDS substituted>)`
- The Planner writes `.harness/plan.yaml` and `.harness/plan.md`.
- When it returns, you load the plan into the board:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/board.py load .harness/plan.yaml
```

If the Planner failed to produce `plan.yaml`, set every existing task (none
yet — just abort) and report the failure to the user.

### Phase 2 — Tick loop

This is the main loop. Repeat until `tick` returns `{"kind": "done"}`:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py tick
```

The output is `{"resume_events": [...], "actions": [{"kind": ..., ...}, ...]}`.

For each action, dispatch to the matching handler (see the **Action handlers**
section below). Some actions you execute synchronously via `Shell`; others
spawn `Task` subagents and you wait for them to return before re-ticking.

**After every action completes**, immediately call `tick` again — do not
batch up changes. The board is the source of truth; ticking re-evaluates from
scratch.

### Phase 3 — Report

When `tick` returns `done`:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py report
```

This writes `.harness/report.md`. Print its path to the user as your final
chat message. Do not print the entire report inline; the user will open it.

---

## Action handlers

Each `tick` action dictates exactly one operation. Execute it, then re-tick.

### `spawn-planner`
*(only emitted by `plan-init`)*

Spawn `generalPurpose` subagent with `ROLES.md/Planner` prompt; substitute
`${USER_NEEDS}` with `action.needs`. After it returns, run:
`board.py load .harness/plan.yaml`.

### `spawn-implementer`
Substitute the action fields into `ROLES.md/Implementer` and spawn:
`Task(subagent_type="generalPurpose", readonly=false, prompt=<filled prompt>)`.
Before spawning, set `IMPLEMENTING`:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/board.py set-status ${TASK_ID} IMPLEMENTING --bump-attempts
```

The implementer is responsible for `lock-task`/`unlock-task` and `pid-set`/
`pid-clear` itself (per ROLES.md). When it returns, the task should be in
`VERIFYING`. If the subagent crashed mid-edit, the next `tick` will detect
the dead pid and downgrade `IMPLEMENTING → PLANNED`.

### `spawn-verifier`
Substitute into `ROLES.md/Verifier` with `mode={action.mode}` (quick or full).
Spawn `Task(subagent_type="shell", prompt=<filled>)`. The verifier sets the
next status itself (`VERIFYING` for quick-pass, `READY` for full-pass,
`REVISING` for fail with retries left, `BLOCKED` for fail at attempt 5).

If `mode=quick` returns pass, you must immediately re-tick — the next tick
will emit `spawn-verifier` with `mode=full` (because task is still in
`VERIFYING`). Wait, that's wrong: the verifier transitions to a state that
makes tick emit the next stage. Currently the harness emits
`spawn-verifier:quick` when status is `IMPLEMENTING`. After quick passes the
verifier sets the task to `VERIFYING` (still). The next tick should emit
`spawn-verifier:full`. **TODO for v0.3**: distinguish IMPLEMENTING-done-quick
vs IMPLEMENTING-done-full via a substate flag. For v0.2: when a verifier
returns, manually re-spawn the verifier with `mode=full` if the previous mode
was `quick` and it passed.

### `open-mr`
Run synchronously:
```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/open_mr.py ${TASK_ID}
```
Re-tick when done. The script handles the auto-merge dry-run + status
transition. Possible terminal results for this task: `BABYSITTING` (most
common on the fork sandbox) or `READY_FOR_MAINTAINER` (rare; usually means
prerequisite drift).

### `open-mr-stacked`
```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/open_mr.py --stacked ${GROUP_ID}
```

### `gate-group`
```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/group_gate.py ${GROUP_ID}
```
Exit 0 → all members merge cleanly + verify; group `gate_status=PASSED`.
Exit 1 → one member blamed and unstacked; the next tick will re-emit
`gate-group` for the now-smaller group (or skip if down to ≤ 1 member).
Exit 2 → fatal; set the group's `gate_status=FAILED` and ALL members to
`BLOCKED`, then re-tick.

### `spawn-babysitter`
Mode-specific. The babysitter base prompt comes from
`~/.cursor/skills-cursor/babysit/SKILL.md`; **prepend** the v0.2 override
block from `ROLES.md/Babysitter`. Spawn with:
`Task(subagent_type="generalPurpose", readonly=false, prompt=<override + babysit prompt>)`.

- `mode=rebase` → babysitter does `git rebase dev` + regenerates conflicting
  hunks. On 2nd consecutive failure, sets `BLOCKED`. Bumps
  `consecutive_rebase_fails` via `set-status --bump-rebase-fails`. Resets to 0
  on success via `--reset-rebase-fails`.
- `mode=cifix` → babysitter reads CI failure log + pushes a fix commit. Note
  that on the fork sandbox there is no CI, so this rarely fires. If it does
  it usually means the user manually configured a workflow.
- `mode=review` → babysitter classifies each unresolved review as bot vs
  human. Bots: apply suggestion + ack reply. Humans: write feedback to
  `.harness/feedback/${TASK_ID}-review-r<n>.log`, set status to `REVISING`,
  terminate so you re-spawn the implementer.

### `merged`
Informational only — `tick` already transitioned the task to `MERGED`.
Continue to next action.

### `watch-error`
Transient `gh` failure. Log it; do nothing this tick; re-tick after a short
sleep (e.g., 10s).

### `wait`
No work this tick but tasks still in flight. Sleep `action.seconds` (default
300s, override via `HARNESS_POLL_INTERVAL_SEC` env var). Then re-tick.

### `done`
All tasks terminal. Run `harness.py report`, print the path to the user, end
the chat session.

---

## Board schema (read-only reference)

`board.py task-info <id>` returns this JSON:

```json
{
  "id": "t1",
  "title": "...",
  "branch": "feat/...",
  "deps": [],
  "group": null,
  "files_touched": [],
  "status": "PLANNED",
  "attempts": 0,
  "feedback_summary": "",
  "feedback_log_path": ".harness/feedback/t1.log",
  "pr_url": null,
  "pr_number": null,
  "opened_at": null,
  "babysit_attempts": 0,
  "consecutive_rebase_fails": 0,
  "blocked_reason": null,
  "ready_for_maintainer_reason": null
}
```

Statuses: `PLANNED`, `IMPLEMENTING`, `VERIFYING`, `REVISING`, `READY`,
`GROUP_WAIT`, `GROUP_GATE`, `GROUP_RESPLIT`, `PR_OPEN`, `STACKED_PR_OPEN`,
`AUTOMERGE_CHECK`, `BABYSITTING`, `MERGED`, `BLOCKED`, `READY_FOR_MAINTAINER`.

Terminal: `MERGED`, `BLOCKED`, `READY_FOR_MAINTAINER`.

---

## Environment variables

- `HARNESS_POLL_INTERVAL_SEC` (default `300`) — sleep between watcher polls.
- `HARNESS_VERIFY_QUICK_CMD` (default in ROLES.md) — quick verify command
  template. `{files}` / `{modules}` / `{test_files}` are substituted.
- `HARNESS_VERIFY_FULL_CMD` (default in ROLES.md) — full verify command
  template.

---

## Files in this skill

- `SKILL.md` — this file (parent agent playbook).
- `ROLES.md` — subagent prompt templates.
- `scripts/_gh.py` — single point of contact for `gh` CLI.
- `scripts/board.py` — JSON state CRUD with fcntl locking.
- `scripts/group_gate.py` — group integration test runner.
- `scripts/open_mr.py` — PR creation + auto-merge enable.
- `scripts/harness.py` — preflight, plan-init, tick, resume, report.

---
name: dimos-agentic-harness
description: Dimos-specific fire-and-forget multi-agent harness for the feipeng1234/dimos fork sandbox. Plans tasks, dispatches Implementer subagents in a feedback loop with an inline verifier (ruff/mypy/pytest), gates PRs via independent or group-integration paths (PR base=feipeng1234/dimos:dev), then babysits PRs to merge with bot-comment auto-resolve. Use when working in the dimos repo and the user wants to drop a multi-task request and walk away (Cursor must stay open). Trigger words: harness, fire-and-forget, multi-task, stacked PR, 无人值守, 多任务编排, 自动合并, 一句话需求.
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

## Hard rules

These split into two groups: **mechanically enforced** (the harness will refuse
or block on violation) and **honor system** (relies on parent / subagent
discipline; the harness cannot prevent it).

### Mechanically enforced

These are guaranteed by code: a hook, an `_gh.py` invariant, or harness logic.

- **M1. Push only to `feipeng1234/dimos`, branch must be prefixed.** The
  installed `pre-push` git hook (`harness.py preflight` installs it
  automatically) blocks any push to a remote that is not the fork or to a
  branch that does not start with `feat/|fix/|refactor/|docs/|test/|chore/|
  perf/`. Pushes to `dev` / `main` are blocked outright. `_gh.push_branch`
  performs the same checks before invoking git, so violations fail fast with
  a clear error.
- **M2. PR base is always `dev` on the fork.** All `_gh.py` PR commands force
  `--repo feipeng1234/dimos --base dev`. The base cannot be overridden by a
  subagent.
- **M3. Verifier 5-attempt cap, babysit 10-attempt cap, PR 24h cap, rebase
  2-fail cap, multi-round caps (`max_rounds=10`, `per_round_cap=3`,
  `no_progress_rounds≥2 ⇒ halt`).** The harness transitions to `BLOCKED`
  automatically when any per-task cap is hit; multi-round caps are checked
  inside `_decide_terminal_action` and produce `{"kind": "done", "reason":
  "..."}` so the loop can never grow unbounded. Subagents cannot bypass
  any of these by retrying.
- **M4. Per-task git worktree isolation.** Every implementer / verifier runs
  in `.harness/worktrees/<task_id>/`; concurrent tasks cannot touch each
  other's working tree or the main repo's `git switch` state.
- **M5. Group BLOCKED auto-unstacks.** When a grouped task transitions to
  `BLOCKED`, `board.py unstack <gid> --blame <tid>` runs automatically.

### Honor system

These the harness cannot enforce — they depend on parent agent / subagent
discipline. Violations will not be caught at runtime, so they are written
loudly here and at the top of each subagent's prompt.

- **H1. No `AskQuestion` calls from the parent agent.** The user is not in
  front of the screen.
- **H2. No `gh` invocations outside of `scripts/_gh.py`.** Direct `gh pr
  create ...` from a subagent bypasses M1/M2's invariants. Always go through
  the wrapper.
- **H3. No manual `board.json` edits.** Always use
  `scripts/board.py <subcommand>`. (chmod-protecting the file does not
  actually stop a subagent from changing the mode and writing — relying on
  this is would be security theater.)
- **H4. Conflict resolution is `rebase-and-regenerate`.** `git rebase dev`
  then re-implement conflicting hunks based on the original task intent.
  **Never** blindly take base; never blindly take ours.
- **H5. Never dismiss `CHANGES_REQUESTED` reviews via
  `gh api .../dismissals`.** Bot reviews (codecov, bugbot, coderabbit, etc.)
  are auto-resolved by the babysitter subagent. Human reviews escalate back
  to the implementer.
- **H6. Per-task lock + pid-set/pid-clear for resume.** `board.py lock-task
  <id>` before editing/pushing; `board.py unlock-task <id>` after.
  `board.py pid-set <id> <pid>` on entry, `board.py pid-clear <id>` on exit
  — this enables crash-safe `harness.py resume`.

---

## Playbook

The pipeline is a **multi-round loop**: round 1 is planned by the round-1
Planner; round N>1 is planned by the Re-planner. Within each round we run
up to `per_round_cap` (default 3) tasks concurrently. The loop terminates
when the Re-planner says `goal_achieved`, or when one of the safety caps
fires (`max_rounds`, `no_progress_rounds`). Concretely:

```
preflight → plan-init → spawn-planner → load plan.yaml
                                              ↓
   ┌──────────────────── tick loop ──────────────────┐
   │                                                  │
   │   (per-task) Implementer → Verifier → Reviewer → │
   │              open-mr → Babysitter → MERGED       │
   │                                                  │
   └─── all tasks terminal? ─── no ──────────────────┘
                  │ yes
                  ↓
       _decide_terminal_action:
         goal_achieved? → done
         round ≥ max?    → done
         strikes ≥ 2?    → done
         else            → spawn-replanner
                  │
                  ↓ (replanner returns)
       harness.py replan-load .harness/replan.yaml
                  │
                  ↓
       back to tick loop (with new PLANNED tasks if any)
```

### Phase 1 — Round-1 plan (one-shot)

```bash
cd /home/lenovo/dimos
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py preflight  # MUST pass
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py plan-init "${USER_NEEDS}"
```

`plan-init` writes `.harness/needs.txt` (the durable **final goal** that the
Re-planner re-reads every round) and emits one action:
`{"kind": "spawn-planner", "needs": "...", ...}`.

**Spawn the Planner subagent**:
- `Task(subagent_type="generalPurpose", readonly=true, prompt=<ROLES.md/Planner with USER_NEEDS substituted>)`
- The Planner writes `.harness/plan.yaml` (≤3 tasks) and `.harness/plan.md`.
- When it returns, you load the plan into the board:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/board.py load .harness/plan.yaml
```

If the Planner failed to produce `plan.yaml`, abort and report the failure
to the user.

### Phase 2 — Tick loop

Main loop. Repeat until `tick` returns `{"kind": "done"}`:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py tick
```

The output is `{"resume_events": [...], "verifier_events": [...], "actions":
[{"kind": ..., ...}, ...]}`. **`tick` blocks internally on `wait`** — it
sleeps and re-evaluates until there is real work or all tasks are terminal.
You will only ever see actionable kinds (`spawn-implementer`, `spawn-reviewer`,
`open-mr`, `gate-group`, `spawn-babysitter`, `spawn-replanner`, `merged`,
`watch-error`) or the final `done`.

For each action, dispatch to the matching handler (see **Action handlers**
below). Some actions you execute synchronously via `Shell`; others spawn
`Task` subagents and you wait for them to return before re-ticking.

**After every action completes**, immediately call `tick` again — do not
batch up changes. The board is the source of truth; ticking re-evaluates from
scratch.

### Phase 3 — Re-plan (between rounds)

When `tick` emits `{"kind": "spawn-replanner", ...}` it means every task is
terminal but the harness still has rounds left and the goal is not flagged
achieved. Spawn the Re-planner subagent (`ROLES.md/Re-planner` with the
action's fields substituted), then ingest its output:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py replan-load .harness/replan.yaml
```

`replan-load` calls `board.py append-plan`, which either flips
`meta.goal_achieved`, appends ≤3 new tasks and bumps `meta.round`, or bumps
`meta.no_progress_rounds`. Then re-tick — new PLANNED tasks (if any) drive
the next round; otherwise the next `tick` will emit `done`.

### Phase 4 — Report

When `tick` returns `done`:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py report
```

This writes `.harness/report.md` including a per-round summary. Print its
path to the user as your final chat message. Do not print the entire report
inline; the user will open it.

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
The action carries a `cwd` field — that is the per-task git worktree under
`.harness/worktrees/${TASK_ID}/`. **Pass it into the prompt as
`${WORKTREE_PATH}`** so the implementer knows where to work; `tick` has
already created the worktree and checked out the branch with `.venv`
symlinked.

Before spawning, set `IMPLEMENTING`:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/board.py set-status ${TASK_ID} IMPLEMENTING --bump-attempts
```

The implementer is responsible for `lock-task`/`unlock-task` and `pid-set`/
`pid-clear` itself (per ROLES.md). When it returns, the task should be in
`VERIFYING`. If the subagent crashed mid-edit, the next `tick` will detect
the dead pid and downgrade `IMPLEMENTING → PLANNED`.

### Verification *(no action — runs inline)*
Verification is no longer dispatched as an action. `tick` calls
`scripts/verify.py:verify_task` synchronously for any task in
`IMPLEMENTING` (mode=quick) or `VERIFYING` with `verify_stage="quick"`
(mode=full). The state machine is closed in code; the parent agent does
not have to remember "what stage was last". Results appear in the
`verifier_events` field of the tick JSON.

**Verifier full-pass goes to `REVIEWING`, not `READY`** — the LLM Reviewer
gate (next handler) is the only path to `READY`.

### `spawn-reviewer`
Substitute the action fields into `ROLES.md/Reviewer` and spawn:
`Task(subagent_type="generalPurpose", readonly=false, model="gpt-5.5-medium",
prompt=<filled prompt>)`. The model name comes from the action payload
(`action.model`); always use it verbatim. Pass `${WORKTREE_PATH}` =
`action.cwd` and `${REVIEW_ATTEMPTS}` = `action.review_attempts`.

The reviewer is **read-only on source code** — it only inspects the diff,
reads files, and writes the board + a feedback log. It produces one of:

- **APPROVED** → reviewer calls `board.py set-status ${TASK_ID} READY
  --bump-review-attempts`. Next tick emits `open-mr`.
- **CHANGES_REQUESTED** → reviewer writes feedback to
  `.harness/feedback/${TASK_ID}-review-r<n>.log`, calls `board.py set-status
  ${TASK_ID} REVISING --feedback-summary "..." --feedback-log <path>
  --bump-review-attempts`. Next tick emits `spawn-implementer` again with the
  feedback in the prompt.

The harness gates this with `MAX_REVIEW_ITERATIONS = 2`. When `tick` sees a
task in `REVIEWING` with `review_attempts >= 2`, it sets `BLOCKED` itself
(reason: `"reviewer rejected after 2 iterations"`) instead of emitting
`spawn-reviewer`. The cap is independent from the verifier's 5-attempt cap.

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

### `spawn-replanner`
Emitted when all tasks are terminal, `meta.goal_achieved` is false, and we
still have budget (`round < max_rounds` and `no_progress_rounds < 2`).

Substitute the action fields into `ROLES.md/Re-planner` and spawn:
`Task(subagent_type="generalPurpose", readonly=false, model="gpt-5.5-medium",
prompt=<filled prompt>)`. The action carries `${ROUND}`, `${MAX_ROUNDS}`,
`${PER_ROUND_CAP}`, `${NO_PROGRESS_ROUNDS}`, `${NEEDS_PATH}`,
`${REPLAN_OUTPUT_PATH}` — pass them all to the prompt verbatim.

The Re-planner writes exactly one file (`${REPLAN_OUTPUT_PATH}`, default
`.harness/replan.yaml`) with one of three shapes:

- `goal_achieved: true` → after ingest, `meta.goal_achieved=true` and the
  next tick emits `done`.
- `goal_achieved: false, tasks: [<≤3 entries>]` → after ingest, those tasks
  are appended as PLANNED, `meta.round` bumps, `no_progress_rounds` resets
  to 0.
- `goal_achieved: false, tasks: []` → no-progress strike. `no_progress_rounds`
  bumps; 2 strikes ⇒ next tick emits `done` with reason
  `"no_progress_rounds=2 reached halt"`.

After the subagent returns, ingest:

```bash
.venv/bin/python .cursor/skills/dimos-agentic-harness/scripts/harness.py replan-load .harness/replan.yaml
```

Then re-tick. Do **not** spawn the Re-planner again without calling
`replan-load` first — re-emission of `spawn-replanner` on consecutive ticks
without a load in between means you're stuck in a loop.

### `merged`
Informational only — `tick` already transitioned the task to `MERGED`.
Continue to next action.

### `watch-error`
Transient `gh` failure. Log it; do nothing this tick; re-tick after a short
sleep (e.g., 10s).

### `wait`
You should never see this in normal use. `harness.py tick` now sleeps and
re-ticks internally when the only outcome is `wait`, so it only ever returns
to you with actionable items or `done`. If you do see `wait`, it means
`tick` was invoked with `--no-loop` (testing mode). In that case sleep
`action.seconds` and re-tick.

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
  "review_attempts": 0,
  "blocked_reason": null,
  "ready_for_maintainer_reason": null,
  "added_in_round": 1
}
```

`board.py meta-info` returns the multi-round meta block:

```json
{
  "round": 1,
  "max_rounds": 10,
  "per_round_cap": 3,
  "goal_achieved": false,
  "no_progress_rounds": 0,
  "history": [
    {"round": 1, "ended_at": "...", "tasks_in_round": ["t1", "t2", "t3"],
     "tasks_merged": ["t1", "t2"], "tasks_blocked": ["t3"],
     "tasks_ready_for_maintainer": [],
     "replanner_verdict": "more-work",
     "new_tasks": ["t4", "t5"]}
  ]
}
```

Statuses: `PLANNED`, `IMPLEMENTING`, `VERIFYING`, `REVISING`, `REVIEWING`,
`READY`, `GROUP_WAIT`, `GROUP_GATE`, `GROUP_RESPLIT`, `PR_OPEN`,
`STACKED_PR_OPEN`, `AUTOMERGE_CHECK`, `BABYSITTING`, `MERGED`, `BLOCKED`,
`READY_FOR_MAINTAINER`.

Terminal: `MERGED`, `BLOCKED`, `READY_FOR_MAINTAINER`.

---

## Environment variables

- `HARNESS_POLL_INTERVAL_SEC` (default `300`) — sleep between watcher polls.
- `HARNESS_VERIFY_FULL_CMD` (group_gate.py) — full verify command for the
  integration gate. Default: `./bin/pytest-fast && uv run ruff check . &&
  uv run mypy dimos/ && uv run pre-commit run --all-files`.

The per-task quick/full verifier is hard-coded in `scripts/verify.py`
(`ruff check {files}` → `mypy {modules}` → `pytest --junit-xml {tests}`).
To customize, edit that file.

---

## Files in this skill

- `SKILL.md` — this file (parent agent playbook).
- `ROLES.md` — subagent prompt templates.
- `scripts/_gh.py` — single point of contact for `gh` CLI.
- `scripts/board.py` — JSON state CRUD with fcntl locking.
- `scripts/verify.py` — programmatic verifier (ruff + mypy + pytest, junit-XML
  parsed for failure summaries). Called inline by `harness.py tick`.
- `scripts/worktree.py` — per-task git worktree management. Each implementer
  / verifier runs in `.harness/worktrees/<task_id>/`; `.venv` is symlinked.
- `scripts/group_gate.py` — group integration test runner (uses a
  `gate-<gid>` worktree to avoid touching the main working tree).
- `scripts/open_mr.py` — PR creation + auto-merge enable.
- `scripts/install_hooks.py` — installs `.cursor/skills/.../hooks/pre-push`
  into `.git/hooks/` so push invariants are enforced even if a subagent
  bypasses `_gh.py`.
- `hooks/pre-push` — git hook: blocks push to non-fork remotes / dev / main /
  bad branch prefixes.
- `scripts/harness.py` — preflight, plan-init, tick, resume, replan-load,
  report. Preflight installs the pre-push hook automatically when all 6
  checks pass. Multi-round loop logic lives in `_decide_terminal_action`.

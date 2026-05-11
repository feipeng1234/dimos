# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User-facing entry point for the dimos-agentic-harness skill.

In v0.2 the harness is **driven by the Cursor parent agent** in chat. The
parent agent calls `harness.py tick` repeatedly; tick observes the board state
and emits a JSON list of actions for the parent agent to execute (mostly
spawning Task subagents using the prompts in ROLES.md). When tick has nothing
to do but tasks are still pending, it emits `{"kind": "wait"}` and the parent
agent should sleep before re-ticking.

Subcommands:
    preflight                      — verify all environment prerequisites; exit 0 ok / 1 fail
    plan-init "<needs>"            — bootstrap `.harness/` and stash the needs string
    tick                           — observe + advance state; emit JSON action list
    resume                         — same as tick but logs dead-worker downgrades
    report                         — write `.harness/report.md`
    status                         — pass-through to `board.py status`

This script must NOT call any LLM-using tool; the parent agent owns Task
spawning. This script only manipulates board state, reads `gh`, and runs
shell verifiers.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
BOARD = SCRIPTS / "board.py"
HARNESS_DIR = Path(".harness")  # relative to cwd; caller cd's to repo root
HEARTBEAT = HARNESS_DIR / "heartbeat"
REPORT_PATH = HARNESS_DIR / "report.md"
NEEDS_PATH = HARNESS_DIR / "needs.txt"

POLL_INTERVAL_SEC_DEFAULT = 300
MAX_BABYSIT_ATTEMPTS = 10
MAX_PR_AGE_HOURS = 24
MAX_VERIFIER_ATTEMPTS = 5
MAX_REBASE_FAILS = 2

TERMINAL = {"MERGED", "BLOCKED", "READY_FOR_MAINTAINER"}


def _load_gh():
    spec = importlib.util.spec_from_file_location("_gh", SCRIPTS / "_gh.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load _gh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_verify() -> Any:
    if "verify" in sys.modules:
        return sys.modules["verify"]
    spec = importlib.util.spec_from_file_location("verify", SCRIPTS / "verify.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = mod  # required for dataclasses to resolve __module__
    spec.loader.exec_module(mod)
    return mod


def _load_worktree() -> Any:
    if "worktree" in sys.modules:
        return sys.modules["worktree"]
    spec = importlib.util.spec_from_file_location("worktree", SCRIPTS / "worktree.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load worktree.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["worktree"] = mod
    spec.loader.exec_module(mod)
    return mod


def _board(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(BOARD), *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _read_board_json() -> dict:
    """Read board.json directly (read-only, no lock needed for snapshot)."""
    p = HARNESS_DIR / "board.json"
    if not p.exists():
        return {"tasks": [], "groups": []}
    return json.loads(p.read_text())


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _heartbeat(action: str, payload: Any = None) -> None:
    HARNESS_DIR.mkdir(exist_ok=True)
    HEARTBEAT.write_text(
        json.dumps(
            {
                "ts": _now_iso(),
                "action": action,
                "payload": payload,
            }
        )
        + "\n"
    )


# --- preflight ------------------------------------------------------------


def _preflight_check(name: str, ok: bool, hint: str = "") -> tuple[str, bool, str]:
    return (name, ok, hint)


def cmd_preflight() -> int:
    checks: list[tuple[str, bool, str]] = []

    proc = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True,
        text=True,
    )
    active = proc.stdout.strip()
    checks.append(
        _preflight_check(
            "gh active account = feipeng1234",
            proc.returncode == 0 and active == "feipeng1234",
            f"got '{active}'; run: gh auth switch -u feipeng1234",
        )
    )

    proc = subprocess.run(
        [
            "gh",
            "api",
            "repos/feipeng1234/dimos",
            "--jq",
            "{auto: .allow_auto_merge, squash: .allow_squash_merge}",
        ],
        capture_output=True,
        text=True,
    )
    auto_merge = squash = False
    if proc.returncode == 0:
        try:
            data = json.loads(proc.stdout)
            auto_merge = bool(data.get("auto"))
            squash = bool(data.get("squash"))
        except json.JSONDecodeError:
            pass
    checks.append(
        _preflight_check(
            "fork allow_auto_merge && allow_squash_merge",
            auto_merge and squash,
            "run: gh repo edit feipeng1234/dimos --enable-auto-merge --enable-squash-merge --delete-branch-on-merge",
        )
    )

    ssh_proc = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=no",
            "github-feipeng1234",
        ],
        capture_output=True,
        text=True,
    )
    auth_msg = (ssh_proc.stderr + ssh_proc.stdout).strip()
    checks.append(
        _preflight_check(
            "ssh github-feipeng1234 alias auth",
            "Hi feipeng1234" in auth_msg,
            "configure ~/.ssh/config Host github-feipeng1234 with the right IdentityFile",
        )
    )

    venv_python = Path(".venv/bin/python")
    checks.append(
        _preflight_check(
            f"venv exists at {venv_python}",
            venv_python.exists(),
            "run: cd /home/lenovo/dimos && uv sync --all-extras --no-extra dds",
        )
    )

    fetch = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", "dev"],
        capture_output=True,
        text=True,
    )
    has_dev = fetch.returncode == 0 and "refs/heads/dev" in fetch.stdout
    checks.append(
        _preflight_check(
            "fork has dev branch",
            has_dev,
            "run: git fetch upstream && git push origin upstream/main:dev --force-with-lease",
        )
    )

    if sys.platform.startswith("linux"):
        logind = Path("/etc/systemd/logind.conf")
        ignore_ok = False
        if logind.exists():
            try:
                content = logind.read_text()
                ignore_ok = "HandleLidSwitch=ignore" in content
            except OSError:
                pass
        # also accept if a systemd-inhibit on lid-switch is currently active
        try:
            lst = subprocess.run(
                ["systemd-inhibit", "--list", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "handle-lid-switch" in lst.stdout:
                ignore_ok = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        checks.append(
            _preflight_check(
                "lid-close suspend disabled (overnight prep)",
                ignore_ok,
                (
                    "either set HandleLidSwitch=ignore in /etc/systemd/logind.conf "
                    "and `systemctl restart systemd-logind`, OR run "
                    "`sudo systemd-inhibit --what=sleep:idle:handle-lid-switch "
                    "--who=dimos-harness --why='harness overnight' --mode=block "
                    "sleep infinity &` for this session"
                ),
            )
        )

    width = max(len(name) for name, _, _ in checks)
    all_ok = True
    for name, ok, hint in checks:
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {name.ljust(width)}  {'' if ok else hint}")
        if not ok:
            all_ok = False

    if all_ok:
        print("")
        print("installing git hooks (pre-push fork-only enforcement):")
        try:
            ih_spec = importlib.util.spec_from_file_location(
                "install_hooks", SCRIPTS / "install_hooks.py"
            )
            assert ih_spec and ih_spec.loader
            ih = importlib.util.module_from_spec(ih_spec)
            ih_spec.loader.exec_module(ih)
            for line in ih.install():
                print(line)
        except Exception as exc:
            print(f"  [WARN] hook install failed: {exc}")

    return 0 if all_ok else 1


# --- plan-init ------------------------------------------------------------


def cmd_plan_init(needs: str) -> int:
    HARNESS_DIR.mkdir(exist_ok=True)
    NEEDS_PATH.write_text(needs.strip() + "\n")
    rc, _, _ = _board("init")
    if rc != 0:
        return rc
    _heartbeat("plan-init", {"needs": needs})
    print(
        json.dumps(
            {
                "next_action": {
                    "kind": "spawn-planner",
                    "needs": needs,
                    "needs_path": str(NEEDS_PATH),
                    "subagent_type": "generalPurpose",
                    "readonly": False,
                }
            }
        )
    )
    return 0


# --- resume / tick action emission ----------------------------------------


def _resume_dead_workers(board: dict) -> list[dict]:
    """Walk non-terminal tasks; if a recorded pid is dead, downgrade.

    Returns a list of downgrade events for logging.
    """
    events: list[dict] = []
    pids_dir = HARNESS_DIR / "pids"
    for t in board["tasks"]:
        if t["status"] in TERMINAL:
            continue
        pid_file = pids_dir / f"{t['id']}.pid"
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        if alive:
            continue

        old = t["status"]
        if old in ("IMPLEMENTING", "REVISING"):
            new = "PLANNED"
            extra = []
        elif old == "VERIFYING":
            new = "VERIFYING"
            extra = []
        elif old in ("BABYSITTING",):
            new = "BABYSITTING"
            extra = []
        else:
            new = old
            extra = []
        _board("set-status", t["id"], new, *extra)
        _board("pid-clear", t["id"])
        events.append({"task_id": t["id"], "from": old, "to": new, "dead_pid": pid})
    return events


def _watch_one_pr(t: dict, gh) -> dict | None:
    """Inspect a BABYSITTING task's PR and decide an action.

    Returns the action dict (kind=...) or None if no action needed this tick
    (PR is healthy and waiting for auto-merge).
    """
    task_id = t["id"]
    pr_number = t.get("pr_number")
    if not pr_number:
        return None

    if t.get("babysit_attempts", 0) >= MAX_BABYSIT_ATTEMPTS:
        _board(
            "set-status",
            task_id,
            "BLOCKED",
            "--blocked-reason",
            f"babysit_attempts exceeded {MAX_BABYSIT_ATTEMPTS}",
        )
        return None

    if t.get("opened_at"):
        try:
            opened = dt.datetime.fromisoformat(t["opened_at"])
            age_hours = (dt.datetime.now(dt.timezone.utc) - opened).total_seconds() / 3600.0
            if age_hours > MAX_PR_AGE_HOURS:
                _board(
                    "set-status",
                    task_id,
                    "BLOCKED",
                    "--blocked-reason",
                    f"PR age {age_hours:.1f}h > {MAX_PR_AGE_HOURS}h",
                )
                return None
        except ValueError:
            pass

    try:
        view = gh.pr_view_json(pr_number)
    except Exception as exc:
        return {"kind": "watch-error", "task_id": task_id, "error": str(exc)}

    if view.get("mergedAt"):
        _board("set-status", task_id, "MERGED")
        try:
            _load_worktree().cleanup_worktree(task_id)
        except Exception as exc:
            print(f"[warn] cleanup_worktree({task_id}) failed: {exc}", file=sys.stderr)
        return {"kind": "merged", "task_id": task_id, "pr_number": pr_number}

    state = view.get("mergeStateStatus") or ""
    review_decision = view.get("reviewDecision") or ""

    rollup = view.get("statusCheckRollup") or []
    has_failure = any(
        (entry.get("conclusion") or entry.get("state") or "").upper()
        in ("FAILURE", "ERROR", "CANCELLED", "TIMED_OUT")
        for entry in rollup
    )

    if state == "CONFLICTING":
        if t.get("consecutive_rebase_fails", 0) >= MAX_REBASE_FAILS:
            _board(
                "set-status",
                task_id,
                "BLOCKED",
                "--blocked-reason",
                f"unresolvable conflict after {MAX_REBASE_FAILS} rebase attempts",
            )
            return None
        _board("set-status", task_id, "BABYSITTING", "--bump-babysit-attempts")
        return {
            "kind": "spawn-babysitter",
            "mode": "rebase",
            "task_id": task_id,
            "pr_number": pr_number,
            "branch": t["branch"],
        }

    if has_failure:
        _board("set-status", task_id, "BABYSITTING", "--bump-babysit-attempts")
        return {
            "kind": "spawn-babysitter",
            "mode": "cifix",
            "task_id": task_id,
            "pr_number": pr_number,
            "branch": t["branch"],
        }

    if review_decision == "CHANGES_REQUESTED":
        _board("set-status", task_id, "BABYSITTING", "--bump-babysit-attempts")
        return {
            "kind": "spawn-babysitter",
            "mode": "review",
            "task_id": task_id,
            "pr_number": pr_number,
            "branch": t["branch"],
        }

    return None


def _emit_dispatch_actions(board: dict) -> list[dict]:
    """Find the next implementer-eligible task.

    A task is eligible if status in {PLANNED, REVISING} AND all `deps` are
    MERGED. Grouped tasks are eligible (each member needs an implementer
    before the gate can run); their group only matters at PR time.

    Side effect: ensures the per-task git worktree exists before emitting
    the action so the implementer subagent can `cd` into it directly.
    """
    actions: list[dict] = []
    merged_ids = {t["id"] for t in board["tasks"] if t["status"] == "MERGED"}
    wt_mod = _load_worktree()
    for t in board["tasks"]:
        if t["status"] not in ("PLANNED", "REVISING"):
            continue
        if not all(dep in merged_ids for dep in t.get("deps", [])):
            continue
        wt_path = wt_mod.ensure_worktree(t["id"], t["branch"])
        actions.append(
            {
                "kind": "spawn-implementer",
                "task_id": t["id"],
                "title": t["title"],
                "branch": t["branch"],
                "cwd": str(wt_path),
                "files_touched": t.get("files_touched", []),
                "feedback_summary": t.get("feedback_summary", ""),
                "feedback_log_path": t.get("feedback_log_path", ""),
            }
        )
        break
    return actions


def _run_verifier_inline(board: dict) -> list[dict]:
    """Synchronously verify tasks that need quick or full verification.

    State machine: implementers transition `IMPLEMENTING → VERIFYING`
    (no verify_stage) on successful push. The dispatcher below picks
    these up for the quick pass, then on the next tick picks up
    `VERIFYING + stage="quick"` for the full pass. `IMPLEMENTING` is
    the in-progress state — the verifier never runs against it; if the
    impl dies, `_resume_dead_workers` reverts `IMPLEMENTING → PLANNED`.

    Returns a list of `{"kind": "verified", ...}` events for the heartbeat /
    parent-agent log.
    """
    events: list[dict] = []
    verify = _load_verify()
    for t in list(board["tasks"]):
        status = t["status"]
        stage = t.get("verify_stage")
        if status == "VERIFYING" and stage is None:
            mode = "quick"
        elif status == "VERIFYING" and stage == "quick":
            mode = "full"
        else:
            continue
        result = verify.verify_task(t["id"], mode)
        events.append(
            {
                "kind": "verified",
                "task_id": result.task_id,
                "mode": result.mode,
                "passed": result.passed,
                "next_status": result.next_status,
                "summary": result.summary,
            }
        )
    return events


def _emit_pr_open_actions(board: dict) -> list[dict]:
    actions: list[dict] = []
    tasks_by_id = {t["id"]: t for t in board["tasks"]}
    for t in board["tasks"]:
        if t["status"] != "READY":
            continue
        if t.get("group"):
            continue
        actions.append({"kind": "open-mr", "task_id": t["id"]})

    for g in board["groups"]:
        if g.get("gate_status") != "PASSED":
            continue
        members = [tasks_by_id[m] for m in g["members"] if m in tasks_by_id]
        if any(m["status"] == "READY" for m in members):
            actions.append({"kind": "open-mr-stacked", "group_id": g["id"]})
    return actions


def _emit_group_gate_actions(board: dict) -> list[dict]:
    actions: list[dict] = []
    tasks_by_id = {t["id"]: t for t in board["tasks"]}
    for g in board["groups"]:
        if g.get("gate_status") != "PENDING":
            continue
        members = [tasks_by_id[m] for m in g["members"] if m in tasks_by_id]
        if not members:
            continue
        if all(m["status"] == "READY" for m in members):
            actions.append({"kind": "gate-group", "group_id": g["id"]})
    return actions


def _all_terminal(board: dict) -> bool:
    if not board["tasks"]:
        return False
    return all(t["status"] in TERMINAL for t in board["tasks"])


def _tick_once(verbose_resume: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """Single non-blocking pass. Returns (resume_events, verifier_events, actions).

    `actions` may include `{"kind": "wait", ...}` or `{"kind": "done"}` when
    nothing else fires. Higher-level loop strips wait and re-calls.
    """
    HARNESS_DIR.mkdir(exist_ok=True)
    board = _read_board_json()

    resume_events = _resume_dead_workers(board)
    if verbose_resume and resume_events:
        for ev in resume_events:
            print(f"[resume] {ev}", file=sys.stderr)
    if resume_events:
        board = _read_board_json()

    gh = _load_gh()
    actions: list[dict] = []

    for t in list(board["tasks"]):
        if t["status"] != "BABYSITTING":
            continue
        action = _watch_one_pr(t, gh)
        if action:
            actions.append(action)
    board = _read_board_json()

    verifier_events = _run_verifier_inline(board)
    if verifier_events:
        board = _read_board_json()

    actions.extend(_emit_pr_open_actions(board))
    actions.extend(_emit_group_gate_actions(board))
    actions.extend(_emit_dispatch_actions(board))

    if not actions:
        if _all_terminal(board):
            actions = [{"kind": "done"}]
        else:
            actions = [
                {
                    "kind": "wait",
                    "seconds": int(
                        os.environ.get("HARNESS_POLL_INTERVAL_SEC", POLL_INTERVAL_SEC_DEFAULT)
                    ),
                }
            ]

    return resume_events, verifier_events, actions


def cmd_tick(verbose_resume: bool = False, loop: bool = True) -> int:
    """Observe + advance state. Default: loop until non-wait actions or done.

    With loop=True (default), if the only action is `wait`, sleep and re-tick
    so the parent agent never has to handle a wait action itself. Aggregated
    resume_events / verifier_events from all inner iterations are emitted.

    With loop=False, behaves like the v0.2 single-pass tick (used by tests).
    """
    all_resume: list[dict] = []
    all_verify: list[dict] = []
    while True:
        resume_events, verifier_events, actions = _tick_once(verbose_resume)
        all_resume.extend(resume_events)
        all_verify.extend(verifier_events)

        only_wait = len(actions) == 1 and actions[0].get("kind") == "wait"
        if loop and only_wait:
            wait_sec = int(actions[0].get("seconds") or POLL_INTERVAL_SEC_DEFAULT)
            time.sleep(wait_sec)
            continue
        break

    payload = {
        "resume_events": all_resume,
        "verifier_events": all_verify,
        "actions": actions,
    }
    _heartbeat("tick", payload)
    print(json.dumps(payload))
    return 0


def cmd_resume() -> int:
    return cmd_tick(verbose_resume=True)


# --- report ---------------------------------------------------------------


def cmd_report() -> int:
    board = _read_board_json()
    HARNESS_DIR.mkdir(exist_ok=True)

    by_status: dict[str, list[dict]] = {}
    for t in board["tasks"]:
        by_status.setdefault(t["status"], []).append(t)

    lines: list[str] = []
    lines.append("# dimos-agentic-harness run report")
    lines.append("")
    lines.append(f"Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    counts = {s: len(ts) for s, ts in by_status.items()}
    lines.append(f"- MERGED: {counts.get('MERGED', 0)}")
    lines.append(f"- READY_FOR_MAINTAINER: {counts.get('READY_FOR_MAINTAINER', 0)}")
    lines.append(f"- BLOCKED: {counts.get('BLOCKED', 0)}")
    lines.append(f"- in-progress: {sum(c for s, c in counts.items() if s not in TERMINAL)}")
    lines.append("")

    for status in ("MERGED", "READY_FOR_MAINTAINER", "BLOCKED"):
        lines.append(f"## {status}")
        lines.append("")
        items = by_status.get(status, [])
        if not items:
            lines.append("(none)")
            lines.append("")
            continue
        for t in items:
            url = t.get("pr_url") or "—"
            babysit = t.get("babysit_attempts", 0)
            attempts = t.get("attempts", 0)
            reason = t.get("blocked_reason") or t.get("ready_for_maintainer_reason") or ""
            lines.append(
                f"- **{t['id']}** {t['title']} — PR: {url} — verifier_attempts={attempts}"
                f" — babysit={babysit}" + (f" — reason: {reason}" if reason else "")
            )
        lines.append("")

    if board.get("groups"):
        lines.append("## Groups")
        lines.append("")
        for g in board["groups"]:
            members = ", ".join(g["members"])
            lines.append(f"- **{g['id']}** ({members}) — gate: {g.get('gate_status')}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(str(REPORT_PATH))
    _heartbeat("report")
    return 0


# --- main ------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, *rest = argv
    if cmd == "preflight":
        return cmd_preflight()
    if cmd == "plan-init":
        if len(rest) != 1:
            print('usage: harness.py plan-init "<needs>"', file=sys.stderr)
            return 2
        return cmd_plan_init(rest[0])
    if cmd == "tick":
        loop = "--no-loop" not in rest
        return cmd_tick(loop=loop)
    if cmd == "resume":
        loop = "--no-loop" not in rest
        return cmd_tick(verbose_resume=True, loop=loop)
    if cmd == "report":
        return cmd_report()
    if cmd == "status":
        rc, out, err = _board("status")
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return rc
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

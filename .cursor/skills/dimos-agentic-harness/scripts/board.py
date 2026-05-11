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

"""Persistent task board for the dimos-agentic-harness skill.

All harness state lives in `.harness/board.json`. Every read or write goes
through this CLI to guarantee fcntl-locked atomic updates. Subagents and
worker processes never touch the JSON file directly.

CLI: `python board.py <subcommand> [args]`. See `--help` for each command.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any

import typer
import yaml

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)


# --- Paths ----------------------------------------------------------------


HARNESS_DIR = Path(".harness")
BOARD_PATH = HARNESS_DIR / "board.json"
LOCKS_DIR = HARNESS_DIR / "locks"
PIDS_DIR = HARNESS_DIR / "pids"
LOGS_DIR = HARNESS_DIR / "logs"
FEEDBACK_DIR = HARNESS_DIR / "feedback"
BOARD_LOCK = HARNESS_DIR / "board.lock"


VALID_STATUSES = {
    "PLANNED",
    "IMPLEMENTING",
    "VERIFYING",
    "REVISING",
    "READY",
    "GROUP_WAIT",
    "GROUP_GATE",
    "GROUP_RESPLIT",
    "PR_OPEN",
    "STACKED_PR_OPEN",
    "AUTOMERGE_CHECK",
    "BABYSITTING",
    "MERGED",
    "BLOCKED",
    "READY_FOR_MAINTAINER",
}

TERMINAL_STATUSES = {"MERGED", "BLOCKED", "READY_FOR_MAINTAINER"}


# --- Locking primitives ---------------------------------------------------


@contextmanager
def _board_lock() -> Iterator[None]:
    """fcntl LOCK_EX on the board file for the duration of the block."""
    HARNESS_DIR.mkdir(exist_ok=True)
    fd = os.open(BOARD_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_board_locked() -> dict[str, Any]:
    if not BOARD_PATH.exists():
        raise typer.Exit(code=2)
    with BOARD_PATH.open("r") as f:
        return json.load(f)


def _write_board_locked(board: dict[str, Any]) -> None:
    tmp = BOARD_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(board, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(BOARD_PATH)


def _empty_task(task_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": task_id,
        "title": "",
        "branch": "",
        "deps": [],
        "group": None,
        "files_touched": [],
        "status": "PLANNED",
        "attempts": 0,
        "feedback_summary": "",
        "feedback_log_path": str(FEEDBACK_DIR / f"{task_id}.log"),
        "pr_url": None,
        "pr_number": None,
        "opened_at": None,
        "babysit_attempts": 0,
        "consecutive_rebase_fails": 0,
        "blocked_reason": None,
        "ready_for_maintainer_reason": None,
    }
    base.update(overrides)
    return base


def _empty_group(group_id: str, members: list[str]) -> dict[str, Any]:
    return {
        "id": group_id,
        "members": list(members),
        "integration_branch": f"integration/{group_id}",
        "gate_status": "PENDING",
    }


def _empty_board() -> dict[str, Any]:
    return {
        "version": 2,
        "tasks": [],
        "groups": [],
        "locks_dir": str(LOCKS_DIR),
        "pids_dir": str(PIDS_DIR),
    }


def _find_task(board: dict[str, Any], task_id: str) -> dict[str, Any]:
    for t in board["tasks"]:
        if t["id"] == task_id:
            return t
    raise typer.BadParameter(f"task {task_id!r} not found")


def _find_group(board: dict[str, Any], group_id: str) -> dict[str, Any]:
    for g in board["groups"]:
        if g["id"] == group_id:
            return g
    raise typer.BadParameter(f"group {group_id!r} not found")


def _is_lock_held(task_id: str) -> bool:
    """A lock is held iff the lockfile exists and the recorded pid is alive."""
    p = LOCKS_DIR / f"{task_id}.lock"
    if not p.exists():
        return False
    try:
        content = p.read_text().strip()
        pid = int(content)
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def _read_pid(task_id: str) -> int | None:
    p = PIDS_DIR / f"{task_id}.pid"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- Commands -------------------------------------------------------------


@app.command()
def init(
    gitignore_top: Path = typer.Option(
        Path(".gitignore"),
        "--gitignore",
        help="Path to repo .gitignore; appends `.harness/` if missing.",
    ),
) -> None:
    """Create .harness/{board.json,locks,pids,logs,feedback} and ensure gitignore."""
    HARNESS_DIR.mkdir(exist_ok=True)
    LOCKS_DIR.mkdir(exist_ok=True)
    PIDS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    FEEDBACK_DIR.mkdir(exist_ok=True)

    with _board_lock():
        if not BOARD_PATH.exists():
            _write_board_locked(_empty_board())

    if gitignore_top.exists():
        text = gitignore_top.read_text()
        if ".harness/" not in text:
            with gitignore_top.open("a") as f:
                if not text.endswith("\n"):
                    f.write("\n")
                f.write("\n# Agentic harness state\n.harness/\n")

    typer.echo(f"initialized {HARNESS_DIR}")


@app.command()
def load(plan_yaml: Path) -> None:
    """Atomically replace tasks + groups from a plan YAML file."""
    if not plan_yaml.exists():
        raise typer.BadParameter(f"plan file not found: {plan_yaml}")
    with plan_yaml.open("r") as f:
        plan = yaml.safe_load(f) or {}

    raw_tasks = plan.get("tasks") or []
    raw_groups = plan.get("groups") or []

    new_tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_tasks:
        if "id" not in raw:
            raise typer.BadParameter(f"task missing id: {raw}")
        if raw["id"] in seen_ids:
            raise typer.BadParameter(f"duplicate task id: {raw['id']}")
        seen_ids.add(raw["id"])
        new_tasks.append(_empty_task(raw["id"], **{k: v for k, v in raw.items() if k != "id"}))

    new_groups: list[dict[str, Any]] = []
    for raw in raw_groups:
        if "id" not in raw or "members" not in raw:
            raise typer.BadParameter(f"group missing id/members: {raw}")
        new_groups.append(_empty_group(raw["id"], raw["members"]))
        for m in raw["members"]:
            for t in new_tasks:
                if t["id"] == m:
                    t["group"] = raw["id"]
                    if t["status"] == "PLANNED":
                        t["status"] = "PLANNED"

    with _board_lock():
        board = _empty_board()
        board["tasks"] = new_tasks
        board["groups"] = new_groups
        _write_board_locked(board)

    typer.echo(f"loaded {len(new_tasks)} tasks, {len(new_groups)} groups")


@app.command("add-task")
def add_task(
    task_id: str,
    title: str = typer.Option("", "--title"),
    branch: str = typer.Option("", "--branch"),
    deps: list[str] = typer.Option([], "--dep"),
    group: str | None = typer.Option(None, "--group"),
    files_touched: list[str] = typer.Option([], "--file"),
) -> None:
    """Append a single task (debug/fallback; prefer `load`)."""
    with _board_lock():
        board = _read_board_locked()
        if any(t["id"] == task_id for t in board["tasks"]):
            raise typer.BadParameter(f"task {task_id} already exists")
        board["tasks"].append(
            _empty_task(
                task_id,
                title=title,
                branch=branch,
                deps=list(deps),
                group=group,
                files_touched=list(files_touched),
            )
        )
        _write_board_locked(board)
    typer.echo(f"added task {task_id}")


@app.command("add-group")
def add_group(group_id: str, members: list[str]) -> None:
    """Append a single group (debug/fallback)."""
    with _board_lock():
        board = _read_board_locked()
        if any(g["id"] == group_id for g in board["groups"]):
            raise typer.BadParameter(f"group {group_id} already exists")
        for m in members:
            t = _find_task(board, m)
            t["group"] = group_id
        board["groups"].append(_empty_group(group_id, members))
        _write_board_locked(board)
    typer.echo(f"added group {group_id} with {len(members)} members")


@app.command("set-status")
def set_status(
    task_id: str,
    status: str,
    feedback_summary: str = typer.Option("", "--feedback-summary"),
    feedback_log: str = typer.Option("", "--feedback-log"),
    blocked_reason: str = typer.Option("", "--blocked-reason"),
    ready_for_maintainer_reason: str = typer.Option("", "--ready-for-maintainer-reason"),
    pr_url: str = typer.Option("", "--pr-url"),
    pr_number: int = typer.Option(0, "--pr-number"),
    opened_at: str = typer.Option("", "--opened-at"),
    bump_attempts: bool = typer.Option(False, "--bump-attempts"),
    bump_babysit_attempts: bool = typer.Option(False, "--bump-babysit-attempts"),
    bump_rebase_fails: bool = typer.Option(False, "--bump-rebase-fails"),
    reset_rebase_fails: bool = typer.Option(False, "--reset-rebase-fails"),
) -> None:
    """Mutate a task's status and metadata. All fields optional."""
    if status not in VALID_STATUSES:
        raise typer.BadParameter(f"invalid status {status}. valid: {sorted(VALID_STATUSES)}")
    with _board_lock():
        board = _read_board_locked()
        t = _find_task(board, task_id)
        t["status"] = status
        if feedback_summary:
            t["feedback_summary"] = feedback_summary
        if feedback_log:
            t["feedback_log_path"] = feedback_log
        if blocked_reason:
            t["blocked_reason"] = blocked_reason
        if ready_for_maintainer_reason:
            t["ready_for_maintainer_reason"] = ready_for_maintainer_reason
        if pr_url:
            t["pr_url"] = pr_url
        if pr_number:
            t["pr_number"] = pr_number
        if opened_at:
            t["opened_at"] = opened_at
        if bump_attempts:
            t["attempts"] += 1
        if bump_babysit_attempts:
            t["babysit_attempts"] += 1
        if bump_rebase_fails:
            t["consecutive_rebase_fails"] += 1
        if reset_rebase_fails:
            t["consecutive_rebase_fails"] = 0
        _write_board_locked(board)
    typer.echo(f"{task_id} -> {status}")


@app.command("next-ready")
def next_ready() -> None:
    """Print the next task id whose deps are all MERGED and not group-waiting.

    Outputs the task id and exits 0 on success, exits 1 if no task ready.
    """
    with _board_lock():
        board = _read_board_locked()
    merged_ids = {t["id"] for t in board["tasks"] if t["status"] == "MERGED"}
    for t in board["tasks"]:
        if t["status"] != "PLANNED":
            continue
        if not all(dep in merged_ids for dep in t.get("deps", [])):
            continue
        typer.echo(t["id"])
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Print the full board as a table including lock + pid columns."""
    with _board_lock():
        board = _read_board_locked()

    headers = ["id", "status", "group", "attempts", "babysit", "lock", "pid", "title"]
    rows: list[list[str]] = []
    for t in board["tasks"]:
        pid_val = _read_pid(t["id"])
        if pid_val is None:
            pid_str = "-"
        else:
            alive = _pid_alive(pid_val)
            pid_str = f"{pid_val}{'' if alive else ' DEAD'}"
        rows.append(
            [
                t["id"],
                t["status"],
                t.get("group") or "-",
                str(t.get("attempts", 0)),
                str(t.get("babysit_attempts", 0)),
                "Y" if _is_lock_held(t["id"]) else "-",
                pid_str,
                (t.get("title") or "")[:50],
            ]
        )

    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]
    sep = "  ".join("-" * w for w in widths)
    typer.echo("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False)))
    typer.echo(sep)
    for r in rows:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=False)))

    if board["groups"]:
        typer.echo("")
        typer.echo("groups:")
        for g in board["groups"]:
            typer.echo(f"  {g['id']}: gate={g['gate_status']} members={','.join(g['members'])}")


@app.command("task-info")
def task_info(task_id: str) -> None:
    """Print a JSON object with the task's full record."""
    with _board_lock():
        board = _read_board_locked()
        t = _find_task(board, task_id)
    typer.echo(json.dumps(t))


@app.command("group-info")
def group_info(group_id: str) -> None:
    """Print a JSON object with the group's members + per-member branch/deps.

    Output: `{"id": "g1", "gate_status": "...", "members": [{"id": "t3",
    "branch": "feat/...", "deps": [...]}], "topo_order": ["t3", "t4"]}`.
    Topo order is computed from `deps`; cycles raise an error.
    """
    with _board_lock():
        board = _read_board_locked()
        g = _find_group(board, group_id)
        member_tasks = [_find_task(board, mid) for mid in g["members"]]

    member_set = set(g["members"])
    incoming: dict[str, set[str]] = {m["id"]: set() for m in member_tasks}
    for m in member_tasks:
        for dep in m.get("deps", []):
            if dep in member_set:
                incoming[m["id"]].add(dep)
    topo: list[str] = []
    pending = {mid: set(deps) for mid, deps in incoming.items()}
    while pending:
        ready = sorted([mid for mid, deps in pending.items() if not deps])
        if not ready:
            raise typer.BadParameter(f"cycle detected in group {group_id} deps")
        for mid in ready:
            topo.append(mid)
            del pending[mid]
            for remaining_deps in pending.values():
                remaining_deps.discard(mid)

    typer.echo(
        json.dumps(
            {
                "id": g["id"],
                "gate_status": g.get("gate_status", "PENDING"),
                "integration_branch": g.get("integration_branch", f"integration/{group_id}"),
                "members": [
                    {"id": m["id"], "branch": m["branch"], "deps": list(m.get("deps", []))}
                    for m in member_tasks
                ],
                "topo_order": topo,
            }
        )
    )


@app.command("set-gate-status")
def set_gate_status(group_id: str, status: str) -> None:
    """Set group.gate_status. Valid: PENDING, PASSED, FAILED."""
    if status not in {"PENDING", "PASSED", "FAILED"}:
        raise typer.BadParameter(f"invalid gate status: {status}")
    with _board_lock():
        board = _read_board_locked()
        g = _find_group(board, group_id)
        g["gate_status"] = status
        _write_board_locked(board)
    typer.echo(f"{group_id}.gate_status -> {status}")


@app.command()
def unstack(group_id: str, blame: str = typer.Option(..., "--blame")) -> None:
    """Remove the blamed task from its group; regroup or downgrade remainder.

    - Blamed task: `group=None`, status=`REVISING` (regenerate on top of dev).
    - If 2+ remain: keep the existing group with just the remainder.
    - If 1 remains: drop the group, set member's `group=None`, status=`READY`.
    """
    with _board_lock():
        board = _read_board_locked()
        g = _find_group(board, group_id)
        if blame not in g["members"]:
            raise typer.BadParameter(f"task {blame} is not a member of group {group_id}")
        blamed = _find_task(board, blame)
        blamed["group"] = None
        blamed["status"] = "REVISING"

        remaining = [m for m in g["members"] if m != blame]
        g["members"] = remaining
        g["gate_status"] = "PENDING"

        if len(remaining) <= 1:
            board["groups"] = [gg for gg in board["groups"] if gg["id"] != group_id]
            for m in remaining:
                t = _find_task(board, m)
                t["group"] = None
                if t["status"] in ("GROUP_WAIT", "GROUP_GATE", "GROUP_RESPLIT"):
                    t["status"] = "READY"

        _write_board_locked(board)

    typer.echo(f"unstacked {blame} from {group_id}; remaining={len(remaining)}")


@app.command("lock-task")
def lock_task(
    task_id: str,
    pid: int = typer.Option(0, "--pid", help="Owner pid to record (default: current process)."),
    timeout_sec: float = typer.Option(
        0.0, "--timeout-sec", help="0 = non-blocking; >0 = retry until timeout."
    ),
) -> None:
    """Claim a per-task lock.

    Implementation: fcntl flock on the lockfile is used briefly to
    serialize check-and-set; a pid is then persisted into the file
    and the flock is released. Subsequent readers consult `_is_lock_held`
    which combines file existence with pid liveness, so a crashed
    worker's stale lock is automatically considered released.

    Exits 0 if claim acquired, 1 if another live worker already holds it
    (after the optional retry window).
    """
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    owner_pid = pid or os.getpid()
    p = LOCKS_DIR / f"{task_id}.lock"
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while True:
        fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # serialize check-and-set
            os.lseek(fd, 0, 0)
            existing = os.read(fd, 64).decode("ascii", "ignore").strip()
            existing_pid: int | None = None
            try:
                existing_pid = int(existing) if existing else None
            except ValueError:
                existing_pid = None
            if existing_pid is not None and _pid_alive(existing_pid):
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                if time.monotonic() >= deadline:
                    typer.echo(
                        f"could not acquire lock for {task_id} within {timeout_sec}s "
                        f"(held by pid {existing_pid})",
                        err=True,
                    )
                    raise typer.Exit(code=1)
                time.sleep(0.2)
                continue
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, f"{owner_pid}\n".encode())
            os.fsync(fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        typer.echo(f"locked {task_id} pid={owner_pid}")
        return


@app.command("unlock-task")
def unlock_task(task_id: str) -> None:
    """Release a task lock by removing its lockfile."""
    p = LOCKS_DIR / f"{task_id}.lock"
    if p.exists():
        p.unlink()
        typer.echo(f"unlocked {task_id}")
    else:
        typer.echo(f"{task_id} was not locked")


@app.command("pid-set")
def pid_set(task_id: str, pid: int) -> None:
    """Record the worker pid currently driving a task (for resume)."""
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    (PIDS_DIR / f"{task_id}.pid").write_text(f"{pid}\n")
    typer.echo(f"{task_id} pid={pid}")


@app.command("pid-clear")
def pid_clear(task_id: str) -> None:
    """Clear the recorded pid (worker finished cleanly)."""
    p = PIDS_DIR / f"{task_id}.pid"
    if p.exists():
        p.unlink()
    typer.echo(f"{task_id} pid cleared")


if __name__ == "__main__":
    app()

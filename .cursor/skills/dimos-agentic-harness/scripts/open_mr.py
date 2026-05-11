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

"""Open PRs for ready tasks (independent or stacked) on the fork sandbox.

Usage:
    python open_mr.py <task-id>         # independent task → PR base = dev
    python open_mr.py --stacked <gid>   # group of tasks   → stacked PRs

Each PR is created via `_gh.py pr-create` (forced --repo feipeng1234/dimos),
then GitHub native auto-merge is enabled (squash). If auto-merge cannot be
enabled within the mergeable_state retry window, the task is marked
READY_FOR_MAINTAINER instead — never blocks waiting for a human to come back.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
BOARD = SCRIPTS / "board.py"
PLAN_MD = Path(".harness/plan.md")
NEEDS_TXT = Path(".harness/needs.txt")
WORKTREES_DIR = Path(".harness/worktrees")


def _load_gh():
    spec = importlib.util.spec_from_file_location("_gh", SCRIPTS / "_gh.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load _gh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gh = _load_gh()


def _board(*args: str, capture: bool = True) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(BOARD), *args],
        capture_output=capture,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _task_info(task_id: str) -> dict:
    rc, out, err = _board("task-info", task_id)
    if rc != 0:
        raise RuntimeError(f"task-info {task_id} failed: {err}")
    return json.loads(out)


def _group_info(group_id: str) -> dict:
    rc, out, err = _board("group-info", group_id)
    if rc != 0:
        raise RuntimeError(f"group-info {group_id} failed: {err}")
    return json.loads(out)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _extract_plan_section(task_id: str) -> str | None:
    """Return the prose paragraph for `task_id` from `.harness/plan.md`.

    Looks for a `### {task_id}` or `### {task_id} —` (or `### {task_id} -`) header
    and captures everything until the next `### ` / `## ` boundary. Returns None
    if the plan.md or the section is missing.
    """
    if not PLAN_MD.exists():
        return None
    try:
        text = PLAN_MD.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("### "):
            continue
        header = stripped[4:].strip()
        first_token = header.split()[0] if header else ""
        if first_token == task_id or header.startswith(f"{task_id} ") or header == task_id:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].lstrip()
        if s.startswith("### ") or s.startswith("## "):
            end = j
            break
    section = "\n".join(lines[start:end]).rstrip()
    return section or None


def _read_needs() -> str | None:
    if not NEEDS_TXT.exists():
        return None
    try:
        text = NEEDS_TXT.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _git_capture(cwd: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=str(cwd), check=False
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _collect_branch_context(branch: str) -> dict[str, str]:
    """Return commit summaries + diff stat for `branch` against `origin/dev`.

    Uses the per-task worktree at `.harness/worktrees/<task_id>/` when possible
    (the branch is already checked out there). Falls back to empty strings when
    the worktree is missing or git fails (e.g. after `prune` post-merge).
    """
    candidate_wts = sorted(WORKTREES_DIR.glob("*")) if WORKTREES_DIR.exists() else []
    chosen: Path | None = None
    for wt in candidate_wts:
        if not wt.is_dir():
            continue
        head = _git_capture(wt, ["rev-parse", "--abbrev-ref", "HEAD"])
        if head == branch:
            chosen = wt
            break
    if chosen is None:
        return {"commits": "", "diffstat": ""}
    _git_capture(chosen, ["fetch", "origin", "--quiet"])
    commits = _git_capture(chosen, ["log", "origin/dev..HEAD", "--format=- %s"])
    diffstat = _git_capture(chosen, ["diff", "--stat", "origin/dev...HEAD"])
    return {"commits": commits, "diffstat": diffstat}


def _build_body(
    task: dict,
    *,
    group_id: str | None = None,
    prev_pr_number: int | None = None,
) -> str:
    """Compose a rich PR body from plan.md + needs.txt + branch context.

    Sections (only emitted when their source exists):
        ## Why            — paragraph for this task from .harness/plan.md
        ## Changes        — commit summaries + diff --stat
        ## Test plan      — verbatim acceptance commands from .harness/needs.txt
        ## Verifier       — short summary from the board
        Stacked context   — group + depends-on, when applicable
    """
    title = task.get("title") or task.get("branch") or task.get("id") or "Change"
    parts: list[str] = [f"## {title}", ""]

    if group_id:
        parts.append(f"Part of stacked group `{group_id}`.")
        if prev_pr_number:
            parts.append(f"Depends on #{prev_pr_number}.")
        parts.append("")

    plan_section = _extract_plan_section(task["id"])
    if plan_section:
        parts.extend(["## Why", "", plan_section, ""])
    else:
        parts.extend(
            [
                "## Why",
                "",
                f"_No `.harness/plan.md` section found for `{task['id']}`. "
                "See the harness run for context._",
                "",
            ]
        )

    branch = task.get("branch") or ""
    ctx = _collect_branch_context(branch) if branch else {"commits": "", "diffstat": ""}
    if ctx["commits"] or ctx["diffstat"]:
        parts.append("## Changes")
        parts.append("")
        if ctx["commits"]:
            parts.extend(["**Commits:**", "", ctx["commits"], ""])
        if ctx["diffstat"]:
            parts.extend(
                ["**Diff stat (`origin/dev...HEAD`):**", "", "```", ctx["diffstat"], "```", ""]
            )

    needs = _read_needs()
    if needs:
        parts.extend(
            [
                "## Test plan",
                "",
                "Acceptance commands recorded for this harness run (`.harness/needs.txt`):",
                "",
                "```",
                needs,
                "```",
                "",
            ]
        )

    parts.extend(
        [
            "## Verifier",
            "",
            "```",
            task.get("feedback_summary") or "(verifier did not record a summary)",
            "```",
            "",
            "---",
            f"Task id: `{task['id']}` · branch: `{branch}`",
            "Generated by `dimos-agentic-harness`"
            + (" (stacked)" if group_id else "")
            + ". Conflict policy: rebase-and-regenerate.",
        ]
    )
    return "\n".join(parts).rstrip() + "\n"


def _body_independent(task: dict) -> str:
    return _build_body(task)


def _body_stacked(task: dict, group_id: str, prev_pr_number: int | None) -> str:
    return _build_body(task, group_id=group_id, prev_pr_number=prev_pr_number)


def _open_one_pr(task: dict, base: str, body: str) -> dict:
    """Push branch, open PR, attempt auto-merge, write board status. Returns final task record."""
    task_id = task["id"]
    branch = task["branch"]
    if not branch:
        raise RuntimeError(f"task {task_id} has no branch")

    print(f"[mr] pushing {branch}", file=sys.stderr)
    _gh.push_branch(branch, force=True)

    print(f"[mr] creating PR {branch} → {base}", file=sys.stderr)
    pr = _gh.pr_create(branch=branch, title=task["title"] or branch, body=body, base=base)
    pr_number = int(pr["number"])
    pr_url = pr.get("url") or ""

    _board(
        "set-status",
        task_id,
        "PR_OPEN",
        "--pr-url",
        pr_url,
        "--pr-number",
        str(pr_number),
        "--opened-at",
        _now_iso(),
    )

    retries = _gh.MERGEABLE_STATE_RETRIES
    interval = _gh.MERGEABLE_STATE_INTERVAL_SEC
    print(
        f"[mr] PR #{pr_number} created; checking mergeable state "
        f"(up to {int(retries * interval)}s)",
        file=sys.stderr,
    )
    stable = _gh.pr_view_with_mergeable_retry(pr_number)
    review_decision = stable.get("reviewDecision") or ""
    mergeable = stable.get("mergeable") or "UNKNOWN"
    rollup = stable.get("statusCheckRollup") or []

    review_ok = review_decision in ("", "APPROVED")
    if not review_ok or mergeable not in ("MERGEABLE", "CONFLICTING"):
        reason = f"reviewDecision={review_decision} mergeable={mergeable}"
        print(f"[mr] cannot enable auto-merge: {reason}", file=sys.stderr)
        _board(
            "set-status",
            task_id,
            "READY_FOR_MAINTAINER",
            "--ready-for-maintainer-reason",
            reason,
        )
        return _task_info(task_id)

    enabled = _gh.pr_enable_auto_merge(pr_number, method="squash")
    if enabled:
        _board("set-status", task_id, "BABYSITTING")
        print(f"[mr] {task_id} → BABYSITTING (auto-merge armed)", file=sys.stderr)
        return _task_info(task_id)

    # Auto-merge rejected. On a fork sandbox without branch protection / required
    # checks, `--auto` returns "Protected branch rules not configured for this
    # branch (enablePullRequestAutoMerge)". If the PR is otherwise mergeable
    # (review_ok, mergeable=MERGEABLE) AND has no pending checks to wait for,
    # auto-merge has nothing to do — fall back to a direct squash-merge.
    has_pending_checks = any(
        (entry.get("conclusion") or entry.get("state") or "").upper()
        in ("PENDING", "QUEUED", "IN_PROGRESS", "WAITING")
        for entry in rollup
    )
    if mergeable == "MERGEABLE" and not has_pending_checks:
        print(
            f"[mr] auto-merge rejected and no pending checks — direct merging #{pr_number}",
            file=sys.stderr,
        )
        if _gh.pr_merge_now(pr_number, method="squash"):
            _board("set-status", task_id, "BABYSITTING")
            print(f"[mr] {task_id} → BABYSITTING (direct-merged)", file=sys.stderr)
            return _task_info(task_id)
        reason = "direct merge fallback failed"
    else:
        reason = f"gh pr merge --auto rejected; mergeable={mergeable} pending={has_pending_checks}"

    print(f"[mr] {reason} for #{pr_number}", file=sys.stderr)
    _board(
        "set-status",
        task_id,
        "READY_FOR_MAINTAINER",
        "--ready-for-maintainer-reason",
        reason,
    )
    return _task_info(task_id)


def open_independent(task_id: str) -> int:
    task = _task_info(task_id)
    if task["status"] not in ("READY", "REVISING"):
        print(
            f"[mr] task {task_id} is in status {task['status']}; expected READY",
            file=sys.stderr,
        )
        return 2
    body = _body_independent(task)
    _open_one_pr(task, base="dev", body=body)
    return 0


def open_stacked(group_id: str) -> int:
    info = _group_info(group_id)
    if info.get("gate_status") != "PASSED":
        print(
            f"[mr] group {group_id} gate_status={info.get('gate_status')}; expected PASSED",
            file=sys.stderr,
        )
        return 2

    members_by_id = {m["id"]: m for m in info["members"]}
    topo = info["topo_order"]
    prev_branch = "dev"
    prev_pr_number: int | None = None

    for tid in topo:
        member_meta = members_by_id[tid]
        task = _task_info(tid)
        if task["status"] not in ("READY", "GROUP_GATE", "GROUP_WAIT"):
            print(
                f"[mr] member {tid} status {task['status']} — expected READY-ish",
                file=sys.stderr,
            )
            return 2
        body = _body_stacked(task, group_id, prev_pr_number)
        _board("set-status", tid, "STACKED_PR_OPEN")
        result = _open_one_pr(task, base=prev_branch, body=body)
        prev_pr_number = int(result["pr_number"]) if result.get("pr_number") else prev_pr_number
        prev_branch = member_meta["branch"]

    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "--stacked":
        if len(argv) != 2:
            print("usage: open_mr.py --stacked <group-id>", file=sys.stderr)
            return 2
        return open_stacked(argv[1])
    if len(argv) != 1:
        print("usage: open_mr.py <task-id>", file=sys.stderr)
        return 2
    return open_independent(argv[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

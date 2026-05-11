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

"""Single point of contact for all `gh` CLI invocations.

Every gh call from the harness MUST go through this module so that:
1. `--repo feipeng1234/dimos` is enforced (PRs always target the fork sandbox)
2. `mergeable_state == UNKNOWN` retry is centralized (GitHub takes 5-30s to compute)
3. Authentication is verified once (must be `feipeng1234` active)

Direct `gh ...` invocations from babysitters / open_mr.py / harness.py are
forbidden. CLI: `python _gh.py <subcommand> [args...]`.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from typing import Any

DIMOS_FORK = "feipeng1234/dimos"
DIMOS_BASE = "dev"
EXPECTED_GH_USER = "feipeng1234"

MERGEABLE_STATE_RETRIES = 6
MERGEABLE_STATE_INTERVAL_SEC = 5.0


class GhError(RuntimeError):
    """Wraps non-zero gh exits with stderr surfaced."""


def _run(
    args: list[str], capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
    cmd = ["gh", *args, "--repo", DIMOS_FORK]
    proc = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if check and proc.returncode != 0:
        raise GhError(
            f"gh failed (exit {proc.returncode}): {shlex.join(cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def assert_active_account() -> None:
    """Fail fast if the active gh account is not `feipeng1234`."""
    proc = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise GhError(f"`gh api user` failed: {proc.stderr}")
    login = proc.stdout.strip()
    if login != EXPECTED_GH_USER:
        raise GhError(
            f"Active gh account is '{login}', expected '{EXPECTED_GH_USER}'.\n"
            f"Run: gh auth switch -u {EXPECTED_GH_USER}"
        )


def push_branch(branch: str, force: bool = True) -> None:
    """git push the local branch to the fork's remote."""
    args = ["push", "origin", branch]
    if force:
        args.append("--force-with-lease")
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GhError(f"git push failed: {proc.stderr}")


def pr_create(branch: str, title: str, body: str, base: str = DIMOS_BASE) -> dict[str, Any]:
    """Create a PR; returns {'number': int, 'url': str}."""
    proc = _run(
        [
            "pr",
            "create",
            "--base",
            base,
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
        ]
    )
    url = proc.stdout.strip().splitlines()[-1]
    num_proc = _run(["pr", "view", url, "--json", "number,url"])
    return json.loads(num_proc.stdout)


def pr_view_json(num: int, fields: list[str] | None = None) -> dict[str, Any]:
    """gh pr view --json with sensible defaults for the harness."""
    default_fields = [
        "number",
        "url",
        "state",
        "mergedAt",
        "mergeable",
        "mergeStateStatus",
        "reviewDecision",
        "autoMergeRequest",
        "statusCheckRollup",
    ]
    field_list = ",".join(fields or default_fields)
    proc = _run(["pr", "view", str(num), "--json", field_list])
    return json.loads(proc.stdout)


def pr_view_with_mergeable_retry(num: int) -> dict[str, Any]:
    """Like pr_view_json but retries while mergeable is UNKNOWN.

    GitHub computes mergeable state asynchronously after PR creation; right
    after `pr create` we may see `mergeable=UNKNOWN`. Retries up to
    MERGEABLE_STATE_RETRIES times at MERGEABLE_STATE_INTERVAL_SEC intervals.
    """
    for _ in range(MERGEABLE_STATE_RETRIES):
        data = pr_view_json(num)
        mergeable = data.get("mergeable")
        if mergeable not in (None, "UNKNOWN"):
            return data
        time.sleep(MERGEABLE_STATE_INTERVAL_SEC)
    return pr_view_json(num)


def pr_enable_auto_merge(num: int, method: str = "squash") -> bool:
    """Enable GitHub native auto-merge. Returns True on success, False if unsupported."""
    method_flag = {"squash": "--squash", "merge": "--merge", "rebase": "--rebase"}[method]
    proc = _run(["pr", "merge", method_flag, "--auto", str(num)], check=False)
    return proc.returncode == 0


def pr_list_reviews(num: int) -> list[dict[str, Any]]:
    """Returns review list with author and state."""
    proc = _run(
        [
            "pr",
            "view",
            str(num),
            "--json",
            "reviews",
            "--jq",
            ".reviews",
        ]
    )
    return json.loads(proc.stdout or "[]")


def pr_list_comments(num: int) -> list[dict[str, Any]]:
    """Returns review-comments + issue-comments combined."""
    proc = _run(
        [
            "pr",
            "view",
            str(num),
            "--json",
            "comments",
            "--jq",
            ".comments",
        ]
    )
    return json.loads(proc.stdout or "[]")


# --- CLI dispatch ---------------------------------------------------------


def _cli(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, *rest = argv
    try:
        if cmd == "assert-active":
            assert_active_account()
            print(EXPECTED_GH_USER)
            return 0
        if cmd == "pr-create":
            branch, title, body = rest[0], rest[1], rest[2]
            base = rest[3] if len(rest) > 3 else DIMOS_BASE
            print(json.dumps(pr_create(branch, title, body, base)))
            return 0
        if cmd == "pr-view":
            num = int(rest[0])
            print(json.dumps(pr_view_json(num)))
            return 0
        if cmd == "pr-view-stable":
            num = int(rest[0])
            print(json.dumps(pr_view_with_mergeable_retry(num)))
            return 0
        if cmd == "pr-merge-auto":
            num = int(rest[0])
            method = rest[1] if len(rest) > 1 else "squash"
            ok = pr_enable_auto_merge(num, method)
            print("ok" if ok else "unsupported")
            return 0 if ok else 1
        if cmd == "push":
            branch = rest[0]
            force = "--no-force" not in rest
            push_branch(branch, force=force)
            return 0
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        return 2
    except GhError as exc:
        print(f"_gh error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))

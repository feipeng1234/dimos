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

"""Single point of contact for all `gh` CLI / `git push` calls in dimos-pr-flow.

Self-contained — does not depend on any other skill. Enforces:
1. Push only to feipeng1234/dimos.
2. Branch name must start with feat/|fix/|refactor/|docs/|test/|chore/|perf/.
3. PR base is always `dev` on the fork.
4. Squash auto-merge via `gh pr merge --auto`.

Raw `gh ...` / `git push ...` from SKILL.md are forbidden. Go through
this wrapper.

CLI: `python _gh.py <push|pr-create|pr-merge-auto> [args...]`
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from typing import Any

DIMOS_FORK = "feipeng1234/dimos"
DIMOS_BASE = "dev"

_VALID_BRANCH_PREFIXES = (
    "feat/",
    "fix/",
    "refactor/",
    "docs/",
    "test/",
    "chore/",
    "perf/",
)


class GhError(RuntimeError):
    """Wraps non-zero gh / git exits with stderr surfaced."""


def _assert_branch_name(branch: str) -> None:
    if not any(branch.startswith(p) for p in _VALID_BRANCH_PREFIXES):
        raise GhError(
            f"branch name {branch!r} does not match required prefix "
            f"({'|'.join(_VALID_BRANCH_PREFIXES)})"
        )


def _assert_origin_is_fork(cwd: str | None = None) -> None:
    proc = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    if proc.returncode != 0:
        raise GhError(f"`git remote get-url origin` failed: {proc.stderr.strip()}")
    url = proc.stdout.strip()
    if DIMOS_FORK not in url:
        raise GhError(
            f"refusing to push: origin remote is {url!r}, expected to contain {DIMOS_FORK!r}"
        )


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["gh", *args, "--repo", DIMOS_FORK]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise GhError(
            f"gh failed (exit {proc.returncode}): {shlex.join(cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def push_branch(branch: str, force: bool = True, cwd: str | None = None) -> None:
    """git push the local branch to the fork's remote.

    Invariants enforced before invoking git:
    - branch name must start with feat/|fix/|refactor/|docs/|test/|chore/|perf/
    - the cwd's `origin` remote must point to the feipeng1234/dimos fork
    """
    _assert_branch_name(branch)
    _assert_origin_is_fork(cwd)
    args = ["push", "origin", branch]
    if force:
        args.append("--force-with-lease")
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False, cwd=cwd)
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


def pr_enable_auto_merge(num: int, method: str = "squash") -> bool:
    """Enable GitHub native auto-merge. Returns True on success, False if rejected."""
    method_flag = {"squash": "--squash", "merge": "--merge", "rebase": "--rebase"}[method]
    proc = _run(["pr", "merge", method_flag, "--auto", str(num)], check=False)
    return proc.returncode == 0


# --- CLI dispatch ---------------------------------------------------------


def _cli(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, *rest = argv
    try:
        if cmd == "push":
            branch = rest[0]
            force = "--no-force" not in rest
            cwd: str | None = None
            for i, a in enumerate(rest):
                if a == "--cwd" and i + 1 < len(rest):
                    cwd = rest[i + 1]
            push_branch(branch, force=force, cwd=cwd)
            return 0
        if cmd == "pr-create":
            branch, title, body = rest[0], rest[1], rest[2]
            base = rest[3] if len(rest) > 3 else DIMOS_BASE
            print(json.dumps(pr_create(branch, title, body, base)))
            return 0
        if cmd == "pr-merge-auto":
            num = int(rest[0])
            method = rest[1] if len(rest) > 1 else "squash"
            ok = pr_enable_auto_merge(num, method)
            print("ok" if ok else "rejected")
            return 0 if ok else 1
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        return 2
    except GhError as exc:
        print(f"_gh error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))

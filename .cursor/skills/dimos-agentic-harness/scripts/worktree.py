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

"""Per-task git worktree isolation for the harness.

Each implementer / verifier runs in its own worktree under
`.harness/worktrees/<task_id>/` so concurrent tasks do not collide on the
main repo's working directory or `git switch` state.

Layout (relative to the main repo root):
    .harness/
        worktrees/
            t1/   ← independent working tree, branch=feat/...
            t2/   ← independent working tree, branch=feat/...
            gate-g1/  ← group integration worktree

The `.venv` directory is symlinked from the main repo so each worktree
shares the same installed Python packages without `uv sync`-ing N times.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys

WORKTREES_DIR = Path(".harness/worktrees")
WORKTREE_ADD_LOCK = Path(".harness/locks/worktree-add.lock")


@contextmanager
def _serialize_worktree_add() -> Iterator[None]:
    """Serialize `git worktree add` calls across concurrent harness invocations.

    `git worktree add` writes to the shared `.git/config` to register the new
    worktree. If two `git worktree add` calls run concurrently they race on the
    config lock and one fails with `不能锁定配置文件 .git/config / cannot lock
    config file .git/config`. We serialize via an `fcntl.flock` on a sentinel
    file under `.harness/locks/`. The lock is held only for the duration of
    the `git worktree add` (or the corresponding checkout in `reset_worktree`),
    not for the LFS smudge or any other work.
    """
    WORKTREE_ADD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with WORKTREE_ADD_LOCK.open("a") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _main_repo_root() -> Path:
    """Resolve the main repo root by asking git from the cwd of the script.

    We assume harness.py / verify.py are invoked with cwd=main repo root.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def _venv_symlink(worktree: Path) -> None:
    main_venv = _main_repo_root() / ".venv"
    if not main_venv.exists():
        return
    link = worktree / ".venv"
    if link.is_symlink() or link.exists():
        return
    os.symlink(main_venv, link)


def _no_lfs_env() -> dict[str, str]:
    """Env that skips LFS smudge so worktree creation doesn't pull GB of media.

    LFS files materialize as pointer text in the worktree. None of the
    optimization / refactor tasks in this harness touch LFS-tracked data
    (datasets, ROS bags, model weights). If a task ever needs a real LFS
    blob, the implementer can run `git lfs pull -I <path>` inside the
    worktree on demand.
    """
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return env


def ensure_worktree(task_id: str, branch: str, base_ref: str = "origin/dev") -> Path:
    """Create the worktree if missing; reuse if it already exists.

    Returns the absolute path to the worktree.
    """
    main_root = _main_repo_root()
    wt = main_root / WORKTREES_DIR / task_id
    if wt.exists():
        _venv_symlink(wt)
        return wt
    wt.parent.mkdir(parents=True, exist_ok=True)
    with _serialize_worktree_add():
        subprocess.run(
            ["git", "worktree", "add", "-B", branch, str(wt), base_ref],
            cwd=main_root,
            check=True,
            env=_no_lfs_env(),
        )
    _venv_symlink(wt)
    return wt


def reset_worktree(task_id: str, branch: str, base_ref: str = "origin/dev") -> Path:
    """Hard-reset an existing worktree's branch to base_ref.

    Used when a task transitions PLANNED → IMPLEMENTING after a previous
    failed attempt and we want to start clean. Drops uncommitted work.
    """
    main_root = _main_repo_root()
    wt = main_root / WORKTREES_DIR / task_id
    if not wt.exists():
        return ensure_worktree(task_id, branch, base_ref)
    no_lfs = _no_lfs_env()
    subprocess.run(["git", "fetch", "origin"], cwd=wt, check=False, env=no_lfs)
    subprocess.run(["git", "checkout", "-B", branch, base_ref], cwd=wt, check=True, env=no_lfs)
    subprocess.run(["git", "clean", "-fdx", "-e", ".venv"], cwd=wt, check=False)
    _venv_symlink(wt)
    return wt


def cleanup_worktree(task_id: str) -> None:
    """Remove the worktree (used after MERGED). Idempotent."""
    main_root = _main_repo_root()
    wt = main_root / WORKTREES_DIR / task_id
    if not wt.exists():
        return
    venv_link = wt / ".venv"
    if venv_link.is_symlink():
        venv_link.unlink()
    subprocess.run(
        ["git", "clean", "-fdx"],
        cwd=wt,
        check=False,
    )
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt)],
        cwd=main_root,
        check=False,
    )
    if wt.exists():
        # fall back: rm the directory if `worktree remove` failed
        import shutil

        shutil.rmtree(wt, ignore_errors=True)


def list_worktrees() -> list[Path]:
    main_root = _main_repo_root()
    base = main_root / WORKTREES_DIR
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


# --- CLI -------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: worktree.py {ensure|reset|cleanup|list} ...", file=sys.stderr)
        return 2
    cmd, *rest = argv
    if cmd == "ensure":
        if len(rest) < 2:
            print("usage: worktree.py ensure <task_id> <branch> [<base_ref>]", file=sys.stderr)
            return 2
        wt = ensure_worktree(rest[0], rest[1], rest[2] if len(rest) > 2 else "origin/dev")
        print(str(wt))
        return 0
    if cmd == "reset":
        if len(rest) < 2:
            print("usage: worktree.py reset <task_id> <branch> [<base_ref>]", file=sys.stderr)
            return 2
        wt = reset_worktree(rest[0], rest[1], rest[2] if len(rest) > 2 else "origin/dev")
        print(str(wt))
        return 0
    if cmd == "cleanup":
        if len(rest) != 1:
            print("usage: worktree.py cleanup <task_id>", file=sys.stderr)
            return 2
        cleanup_worktree(rest[0])
        return 0
    if cmd == "list":
        for p in list_worktrees():
            print(str(p))
        return 0
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

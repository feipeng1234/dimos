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

"""Tests for worktree.py against a throwaway git repo fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("worktree_mod", SCRIPTS_DIR / "worktree.py")
assert spec and spec.loader
worktree = importlib.util.module_from_spec(spec)
sys.modules["worktree_mod"] = worktree
spec.loader.exec_module(worktree)


@pytest.fixture
def fresh_repo(tmp_path: Path, monkeypatch) -> Path:
    """Bootstrap a tiny git repo with a `dev`-equivalent ref under origin/dev.

    Strategy: init a bare 'remote' alongside, push initial commit to it,
    so `origin/dev` resolves.
    """
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "dev"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=work, check=True, capture_output=True)
    (work / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=work,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "dev"], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "fetch", "origin"], cwd=work, check=True, capture_output=True)

    monkeypatch.chdir(work)
    return work


def test_ensure_creates_new(fresh_repo: Path) -> None:
    wt = worktree.ensure_worktree("t1", "feat/foo")
    assert wt.exists()
    assert wt.is_dir()
    head = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=wt,
        check=True,
        capture_output=True,
        text=True,
    )
    assert head.stdout.strip() == "feat/foo"


def test_ensure_reuses_existing(fresh_repo: Path) -> None:
    wt1 = worktree.ensure_worktree("t1", "feat/foo")
    (wt1 / "marker.txt").write_text("hi")
    wt2 = worktree.ensure_worktree("t1", "feat/foo")
    assert wt1 == wt2
    assert (wt2 / "marker.txt").exists()


def test_venv_symlink_when_main_has_venv(fresh_repo: Path) -> None:
    (fresh_repo / ".venv").mkdir()
    (fresh_repo / ".venv" / "marker").write_text("v")
    wt = worktree.ensure_worktree("t1", "feat/foo")
    link = wt / ".venv"
    assert link.is_symlink()
    assert (link / "marker").read_text() == "v"


def test_no_venv_symlink_when_no_main_venv(fresh_repo: Path) -> None:
    wt = worktree.ensure_worktree("t1", "feat/foo")
    assert not (wt / ".venv").exists()


def test_reset_drops_uncommitted(fresh_repo: Path) -> None:
    wt = worktree.ensure_worktree("t1", "feat/foo")
    (wt / "trash.txt").write_text("uncommitted")
    worktree.reset_worktree("t1", "feat/foo")
    assert not (wt / "trash.txt").exists()


def test_cleanup_removes(fresh_repo: Path) -> None:
    wt = worktree.ensure_worktree("t1", "feat/foo")
    assert wt.exists()
    worktree.cleanup_worktree("t1")
    assert not wt.exists()


def test_cleanup_idempotent(fresh_repo: Path) -> None:
    worktree.cleanup_worktree("nonexistent")  # should not raise


def test_list(fresh_repo: Path) -> None:
    worktree.ensure_worktree("t1", "feat/a")
    worktree.ensure_worktree("t2", "feat/b")
    paths = worktree.list_worktrees()
    names = {p.name for p in paths}
    assert names == {"t1", "t2"}

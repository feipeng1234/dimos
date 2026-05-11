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

"""Tests for install_hooks.py: install / reinstall / hook actually blocks bad pushes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("install_hooks_mod", SCRIPTS_DIR / "install_hooks.py")
assert spec and spec.loader
ih = importlib.util.module_from_spec(spec)
sys.modules["install_hooks_mod"] = ih
spec.loader.exec_module(ih)


@pytest.fixture
def tiny_repo(tmp_path: Path, monkeypatch) -> Path:
    """Empty git repo with a remote pointing at the fork URL.

    The hook checks the remote URL string, so a fake URL is enough — we never
    actually contact GitHub in tests.
    """
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=tmp_path, check=True, capture_output=True
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_install_creates_symlink(tiny_repo: Path) -> None:
    out = ih.install()
    hook = tiny_repo / ".git/hooks/pre-push"
    assert hook.is_symlink()
    assert "[link]" in " ".join(out) or "[ok]" in " ".join(out)


def test_install_idempotent(tiny_repo: Path) -> None:
    ih.install()
    out = ih.install()
    assert "[ok]" in " ".join(out)


def test_install_backs_up_existing(tiny_repo: Path) -> None:
    hook = tiny_repo / ".git/hooks"
    hook.mkdir(exist_ok=True)
    (hook / "pre-push").write_text("#!/bin/sh\necho pre-existing\n")
    out = ih.install()
    assert "[back]" in " ".join(out)
    backups = list(hook.glob("pre-push.harness-backup-*"))
    assert len(backups) == 1


def test_uninstall_removes_symlink(tiny_repo: Path) -> None:
    ih.install()
    out = ih.uninstall()
    hook = tiny_repo / ".git/hooks/pre-push"
    assert not hook.exists() and not hook.is_symlink()
    assert "[rm]" in " ".join(out)


def test_uninstall_keeps_non_harness_hook(tiny_repo: Path) -> None:
    hook = tiny_repo / ".git/hooks/pre-push"
    hook.parent.mkdir(exist_ok=True, parents=True)
    hook.write_text("#!/bin/sh\necho theirs\n")
    out = ih.uninstall()
    assert hook.exists()
    assert "[keep]" in " ".join(out)


# --- end-to-end: hook actually blocks a bad push --------------------------


def _make_remote_then_repo(tmp_path: Path, owner_repo: str) -> Path:
    """Create a bare remote whose path contains `owner/repo` and a wired worker.

    The pre-push hook does a substring match on the remote URL for
    `feipeng1234/dimos`, so the fixture path mirrors that shape.
    """
    bare = tmp_path / f"{owner_repo}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    work = tmp_path / "work"
    subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "dev"], cwd=work, check=True, capture_output=True)
    (work / "x.txt").write_text("a\n")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "i"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(bare)],
        cwd=work,
        check=True,
        capture_output=True,
    )
    return work


def test_hook_blocks_push_to_dev(tmp_path: Path, monkeypatch) -> None:
    work = _make_remote_then_repo(tmp_path, "feipeng1234/dimos")
    monkeypatch.chdir(work)
    ih.install()
    proc = subprocess.run(
        ["git", "push", "origin", "dev"], cwd=work, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "BLOCKED" in proc.stderr


def test_hook_blocks_bad_branch_name(tmp_path: Path, monkeypatch) -> None:
    work = _make_remote_then_repo(tmp_path, "feipeng1234/dimos")
    monkeypatch.chdir(work)
    ih.install()
    subprocess.run(["git", "checkout", "-b", "wip/foo"], cwd=work, check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "push", "origin", "wip/foo"], cwd=work, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "BLOCKED" in proc.stderr


def test_hook_blocks_push_to_non_fork(tmp_path: Path, monkeypatch) -> None:
    work = _make_remote_then_repo(tmp_path, "someoneelse/dimos")
    monkeypatch.chdir(work)
    ih.install()
    subprocess.run(["git", "checkout", "-b", "feat/foo"], cwd=work, check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "push", "origin", "feat/foo"], cwd=work, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "BLOCKED" in proc.stderr


def test_hook_allows_valid_push(tmp_path: Path, monkeypatch) -> None:
    work = _make_remote_then_repo(tmp_path, "feipeng1234/dimos")
    monkeypatch.chdir(work)
    ih.install()
    subprocess.run(["git", "checkout", "-b", "feat/foo"], cwd=work, check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "push", "origin", "feat/foo"], cwd=work, capture_output=True, text=True
    )
    assert proc.returncode == 0, f"unexpected push failure: {proc.stderr}"


def test_hook_chains_to_backup(tmp_path: Path, monkeypatch) -> None:
    """If a previous pre-push hook was backed up, the new hook chains into it."""
    work = _make_remote_then_repo(tmp_path, "feipeng1234/dimos")
    monkeypatch.chdir(work)
    hooks_dir = work / ".git/hooks"
    hooks_dir.mkdir(exist_ok=True, parents=True)
    marker = work / "chained_was_called.txt"
    chain = hooks_dir / "pre-push"
    chain.write_text(f"#!/bin/sh\necho ran > {marker}\nexit 0\n")
    chain.chmod(0o755)
    ih.install()
    subprocess.run(["git", "checkout", "-b", "feat/x"], cwd=work, check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "push", "origin", "feat/x"], cwd=work, capture_output=True, text=True
    )
    assert proc.returncode == 0, f"unexpected: {proc.stderr}"
    assert marker.exists(), "chained backup hook was not invoked"

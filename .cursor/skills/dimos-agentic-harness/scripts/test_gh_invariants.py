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

"""Tests for _gh.py invariant guards (branch name, origin remote)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gh_mod", SCRIPTS_DIR / "_gh.py")
assert spec and spec.loader
gh = importlib.util.module_from_spec(spec)
sys.modules["gh_mod"] = gh
spec.loader.exec_module(gh)


def test_assert_branch_name_accepts_valid() -> None:
    for ok in ("feat/x", "fix/y", "refactor/z", "docs/a", "test/b", "chore/c", "perf/d"):
        gh._assert_branch_name(ok)


def test_assert_branch_name_rejects_invalid() -> None:
    for bad in ("dev", "main", "feature/x", "wip/x", "x"):
        with pytest.raises(gh.GhError, match="does not match required prefix"):
            gh._assert_branch_name(bad)


def test_assert_origin_rejects_non_fork(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:someoneelse/dimos.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    with pytest.raises(gh.GhError, match="refusing to push"):
        gh._assert_origin_is_fork(cwd=str(tmp_path))


def test_assert_origin_accepts_fork(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:feipeng1234/dimos.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    gh._assert_origin_is_fork(cwd=str(tmp_path))


def test_push_branch_blocks_bad_name(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:feipeng1234/dimos.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    with pytest.raises(gh.GhError, match="does not match required prefix"):
        gh.push_branch("dev", force=False, cwd=str(tmp_path))


def test_push_branch_blocks_bad_remote(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:wrong/repo.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    with pytest.raises(gh.GhError, match="refusing to push"):
        gh.push_branch("feat/foo", force=False, cwd=str(tmp_path))

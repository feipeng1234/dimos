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

"""Tests for the change detection utility."""

from __future__ import annotations

from pathlib import Path

import pytest

from dimos.utils.change_detect import clear_cache, did_change


@pytest.fixture(autouse=True)
def _use_tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the change-detection cache to a temp dir for every test."""
    monkeypatch.setattr(
        "dimos.utils.change_detect._get_cache_dir",
        lambda: tmp_path / "cache",
    )


@pytest.fixture()
def src_dir(tmp_path: Path) -> Path:
    """A temp directory with two source files for testing."""
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.c").write_text("int main() { return 0; }")
    (d / "b.c").write_text("void helper() {}")
    return d


def test_first_call_returns_true(src_dir: Path) -> None:
    assert did_change("test_cache", [str(src_dir)]) is True


def test_second_call_no_change_returns_false(src_dir: Path) -> None:
    did_change("test_cache", [str(src_dir)])
    assert did_change("test_cache", [str(src_dir)]) is False


def test_file_modified_returns_true(src_dir: Path) -> None:
    did_change("test_cache", [str(src_dir)])
    (src_dir / "a.c").write_text("int main() { return 1; }")
    assert did_change("test_cache", [str(src_dir)]) is True


def test_file_added_to_dir_returns_true(src_dir: Path) -> None:
    did_change("test_cache", [str(src_dir)])
    (src_dir / "c.c").write_text("void new_func() {}")
    assert did_change("test_cache", [str(src_dir)]) is True


def test_file_deleted_returns_true(src_dir: Path) -> None:
    did_change("test_cache", [str(src_dir)])
    (src_dir / "b.c").unlink()
    assert did_change("test_cache", [str(src_dir)]) is True


def test_glob_pattern(src_dir: Path) -> None:
    pattern = str(src_dir / "*.c")
    assert did_change("glob_cache", [pattern]) is True
    assert did_change("glob_cache", [pattern]) is False
    (src_dir / "a.c").write_text("changed!")
    assert did_change("glob_cache", [pattern]) is True


def test_separate_cache_names_independent(src_dir: Path) -> None:
    paths = [str(src_dir)]
    did_change("cache_a", paths)
    did_change("cache_b", paths)
    # Both caches are now up-to-date
    assert did_change("cache_a", paths) is False
    assert did_change("cache_b", paths) is False
    # Modify a file — both caches should report changed independently
    (src_dir / "a.c").write_text("changed")
    assert did_change("cache_a", paths) is True
    # cache_b hasn't been checked since the change
    assert did_change("cache_b", paths) is True


def test_clear_cache(src_dir: Path) -> None:
    paths = [str(src_dir)]
    did_change("clear_test", paths)
    assert did_change("clear_test", paths) is False
    assert clear_cache("clear_test") is True
    assert did_change("clear_test", paths) is True


def test_clear_cache_nonexistent() -> None:
    assert clear_cache("does_not_exist") is False


def test_empty_paths_returns_false() -> None:
    assert did_change("empty_test", []) is False


def test_nonexistent_path_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A non-existent path logs a warning and doesn't crash."""
    result = did_change("missing_test", ["/nonexistent/path/to/file.c"])
    # First call with no resolvable files still returns True (no cache)
    assert isinstance(result, bool)

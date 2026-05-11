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

"""Tests for verify.py: junit parsing, decision table, file→module mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify_mod", SCRIPTS_DIR / "verify.py")
assert spec and spec.loader
verify = importlib.util.module_from_spec(spec)
sys.modules["verify_mod"] = verify
spec.loader.exec_module(verify)


# --- _parse_junit ---------------------------------------------------------


def test_parse_junit_no_file(tmp_path: Path) -> None:
    assert verify._parse_junit(tmp_path / "missing.xml") == []


def test_parse_junit_pass(tmp_path: Path) -> None:
    p = tmp_path / "j.xml"
    p.write_text(
        '<?xml version="1.0"?>'
        "<testsuites><testsuite>"
        '<testcase classname="m" name="t1"/>'
        "</testsuite></testsuites>"
    )
    assert verify._parse_junit(p) == []


def test_parse_junit_failure(tmp_path: Path) -> None:
    p = tmp_path / "j.xml"
    p.write_text(
        '<?xml version="1.0"?>'
        "<testsuites><testsuite>"
        '<testcase classname="m.Foo" name="t1">'
        '  <failure message="assert 1 == 2">trace</failure>'
        "</testcase>"
        '<testcase classname="m.Foo" name="t2">'
        '  <error message="boom">trace</error>'
        "</testcase>"
        "</testsuite></testsuites>"
    )
    out = verify._parse_junit(p)
    assert len(out) == 2
    assert "m.Foo::t1" in out[0]
    assert "assert 1 == 2" in out[0]
    assert "m.Foo::t2" in out[1]
    assert "boom" in out[1]


def test_parse_junit_caps_at_top_n(tmp_path: Path) -> None:
    cases = "".join(
        f'<testcase classname="m" name="t{i}"><failure message="f{i}">tr</failure></testcase>'
        for i in range(10)
    )
    p = tmp_path / "j.xml"
    p.write_text(f'<?xml version="1.0"?><testsuites><testsuite>{cases}</testsuite></testsuites>')
    assert len(verify._parse_junit(p, top_n=3)) == 3


def test_parse_junit_malformed(tmp_path: Path) -> None:
    p = tmp_path / "j.xml"
    p.write_text("not xml at all <<")
    out = verify._parse_junit(p)
    assert out == ["junit xml parse error"]


# --- _decide_next ---------------------------------------------------------


def test_decide_pass_quick() -> None:
    s, args = verify._decide_next(passed=True, mode="quick", attempts=0)
    assert s == "VERIFYING"
    assert "--verify-stage" in args and "quick" in args


def test_decide_pass_full() -> None:
    s, args = verify._decide_next(passed=True, mode="full", attempts=2)
    assert s == "REVIEWING"
    assert "--verify-stage" in args and "full" in args


def test_decide_fail_with_retries() -> None:
    s, args = verify._decide_next(passed=False, mode="quick", attempts=2)
    assert s == "REVISING"
    assert "--bump-attempts" in args


def test_decide_fail_at_max() -> None:
    s, args = verify._decide_next(passed=False, mode="full", attempts=4)
    assert s == "BLOCKED"
    assert "--blocked-reason" in args
    assert "--bump-attempts" in args


# --- _files_to_modules / _files_to_test_files -----------------------------


def test_files_to_modules_dedup() -> None:
    out = verify._files_to_modules(["a.py", "b.py", "a.py", "c.txt"])
    assert out == ["a.py", "b.py"]


def test_files_to_test_files_self(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test_foo.py").write_text("")
    out = verify._files_to_test_files(["test_foo.py"])
    assert out == ["test_foo.py"]


def test_files_to_test_files_sibling(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("")
    (tmp_path / "test_foo.py").write_text("")
    out = verify._files_to_test_files(["foo.py"])
    assert out == ["test_foo.py"]


def test_files_to_test_files_no_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("")
    assert verify._files_to_test_files(["foo.py"]) == []


# --- _format_summary ------------------------------------------------------


def test_format_summary_all_pass() -> None:
    s = verify._format_summary(
        verify.StepResult("ruff", 0, "", ""),
        verify.StepResult("mypy", 0, "", ""),
        verify.StepResult("pytest", 0, "", ""),
        [],
    )
    assert s == "pass"


def test_format_summary_pytest_with_failures() -> None:
    s = verify._format_summary(
        verify.StepResult("ruff", 0, "", ""),
        verify.StepResult("mypy", 0, "", ""),
        verify.StepResult("pytest", 1, "", ""),
        ["m::t1 - assert", "m::t2 - boom"],
    )
    assert "pytest:" in s
    assert "2 failure" in s
    assert "m::t1" in s


def test_format_summary_truncates() -> None:
    long = "x" * 500
    s = verify._format_summary(
        verify.StepResult("ruff", 1, long, ""),
        verify.StepResult("mypy", 0, "", ""),
        verify.StepResult("pytest", 0, "", ""),
        [],
    )
    assert len(s) <= verify.MAX_SUMMARY_CHARS
    assert s.endswith("...")


# --- end-to-end smoke (board CLI must work) -------------------------------


def test_verify_task_pass_no_files(tmp_path: Path, monkeypatch) -> None:
    """A task with files_touched=[] in quick mode skips all 3 steps and passes.

    Validates that verify_task can drive board.py CLI end-to-end:
    pid-set → set-status with --verify-stage quick → pid-clear.
    """
    monkeypatch.chdir(tmp_path)
    import subprocess as sp

    sp.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "init",
            "--gitignore",
            str(tmp_path / "x.gitignore"),
        ],
        check=True,
    )
    sp.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "add-task",
            "tsmoke",
            "--title",
            "smoke",
            "--branch",
            "feat/smoke",
        ],
        check=True,
    )
    sp.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "set-status",
            "tsmoke",
            "IMPLEMENTING",
        ],
        check=True,
    )

    # explicit cwd to bypass worktree management (no git repo in this fixture)
    result = verify.verify_task("tsmoke", "quick", cwd=tmp_path)
    assert result.passed is True
    assert result.next_status == "VERIFYING"

    info_proc = sp.run(
        [sys.executable, str(SCRIPTS_DIR / "board.py"), "task-info", "tsmoke"],
        check=True,
        capture_output=True,
        text=True,
    )
    import json as _json

    info = _json.loads(info_proc.stdout)
    assert info["status"] == "VERIFYING"
    assert info["verify_stage"] == "quick"

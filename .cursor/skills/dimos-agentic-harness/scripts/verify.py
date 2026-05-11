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

"""Programmatic verifier: runs ruff/mypy/pytest, parses junit XML, writes board.

Replaces the LLM-based Verifier subagent from v0.2. Called inline by
`harness.py` during `tick`. Synchronous; returns when verification is done
and the board has been updated.

Stages (all run unconditionally — last failure determines summary):
    1. ruff check {files_touched}
    2. mypy {derived modules}
    3. pytest --junit-xml=... {derived test files}

Exit codes are aggregated: pass iff all three return 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import IO, Any, Literal, cast
import xml.etree.ElementTree as ET

VENV = Path("/home/lenovo/dimos/.venv/bin")
RUFF = str(VENV / "ruff")
MYPY = str(VENV / "mypy")
PYTEST = str(VENV / "pytest")

HARNESS_DIR = Path(".harness")
FEEDBACK_DIR = HARNESS_DIR / "feedback"
JUNIT_DIR = HARNESS_DIR / "junit"
PIDS_DIR = HARNESS_DIR / "pids"

MAX_VERIFIER_ATTEMPTS = 5
MAX_SUMMARY_CHARS = 200

SCRIPTS = Path(__file__).resolve().parent
BOARD_PY = SCRIPTS / "board.py"


def _load_worktree() -> Any:
    if "worktree" in sys.modules:
        return sys.modules["worktree"]
    spec = importlib.util.spec_from_file_location("worktree", SCRIPTS / "worktree.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load worktree.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["worktree"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class StepResult:
    name: str
    rc: int
    stdout: str
    stderr: str


@dataclass
class VerifyResult:
    task_id: str
    mode: str
    passed: bool
    next_status: str
    summary: str
    log_path: Path


# --- helpers ---------------------------------------------------------------


def _board(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(BOARD_PY), *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _read_task(task_id: str) -> dict[str, Any]:
    rc, out, err = _board("task-info", task_id)
    if rc != 0:
        raise RuntimeError(f"board task-info {task_id} failed: {err}")
    return cast("dict[str, Any]", json.loads(out))


def _files_to_modules(files: list[str]) -> list[str]:
    """Map .py files under `dimos/` to their import paths; non-py paths kept as-is.

    Used as the mypy target. Returns deduplicated list preserving order.
    """
    seen: dict[str, None] = {}
    for f in files:
        if not f.endswith(".py"):
            continue
        seen.setdefault(f, None)
    return list(seen.keys())


def _files_to_test_files(files: list[str]) -> list[str]:
    """Derive test files: any test_*.py in files_touched, plus inferred siblings.

    For `dimos/foo/bar.py` we look for `dimos/foo/test_bar.py`. Returns existing
    paths only; if none found, returns empty list (caller decides fallback).
    """
    out: list[str] = []
    seen: set[str] = set()
    for f in files:
        p = Path(f)
        if p.name.startswith("test_") and p.suffix == ".py":
            if str(p) not in seen and p.exists():
                out.append(str(p))
                seen.add(str(p))
            continue
        if p.suffix != ".py":
            continue
        sibling = p.with_name(f"test_{p.name}")
        if sibling.exists() and str(sibling) not in seen:
            out.append(str(sibling))
            seen.add(str(sibling))
    return out


def _run_step(
    name: str,
    cmd: list[str],
    log_fp: IO[str],
    cwd: Path,
) -> StepResult:
    log_fp.write(f"\n$ ({name}) {' '.join(cmd)}\n")
    log_fp.flush()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    log_fp.write(proc.stdout)
    if proc.stderr:
        log_fp.write(f"\n--- stderr ---\n{proc.stderr}\n")
    log_fp.flush()
    return StepResult(name=name, rc=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _parse_junit(junit_path: Path, top_n: int = 3) -> list[str]:
    """Return up to N short failure descriptions from a pytest junit XML."""
    if not junit_path.exists():
        return []
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError:
        return ["junit xml parse error"]
    failures: list[str] = []
    for case in root.iter("testcase"):
        if len(failures) >= top_n:
            break
        for child in case:
            tag = child.tag.lower()
            if tag in ("failure", "error"):
                cls = case.get("classname") or ""
                name = case.get("name") or ""
                msg = (
                    (child.get("message") or "").splitlines()[0].strip()
                    if child.get("message")
                    else ""
                )
                ident = f"{cls}::{name}" if cls else name
                failures.append(f"{ident} - {msg}" if msg else ident)
                break
    return failures


def _format_summary(
    ruff: StepResult,
    mypy: StepResult,
    pytest: StepResult,
    pytest_failures: list[str],
) -> str:
    parts: list[str] = []
    if ruff.rc != 0:
        line = (ruff.stdout + ruff.stderr).strip().splitlines()[-1:] or ["ruff failed"]
        parts.append(f"ruff: {line[0]}")
    if mypy.rc != 0:
        line = (mypy.stdout + mypy.stderr).strip().splitlines()[-1:] or ["mypy failed"]
        parts.append(f"mypy: {line[0]}")
    if pytest.rc != 0:
        if pytest_failures:
            n = len(pytest_failures)
            head = " | ".join(pytest_failures)
            parts.append(f"pytest: {n} failure(s): {head}")
        else:
            tail = (pytest.stdout + pytest.stderr).strip().splitlines()[-1:] or ["pytest failed"]
            parts.append(f"pytest: {tail[0]}")
    summary = " || ".join(parts) if parts else "pass"
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 3] + "..."
    return summary


def _decide_next(
    passed: bool,
    mode: str,
    attempts: int,
) -> tuple[str, list[str]]:
    """Returns (next_status, extra_board_args).

    On full-mode pass we hand off to the LLM Reviewer (REVIEWING) instead of
    going straight to READY. The reviewer is the only path that can produce
    READY from this point on.
    """
    if passed:
        return "REVIEWING", ["--verify-stage", mode]
    if attempts + 1 >= MAX_VERIFIER_ATTEMPTS:
        return "BLOCKED", [
            "--blocked-reason",
            f"verifier exhausted at attempt {attempts + 1}",
            "--bump-attempts",
        ]
    return "REVISING", ["--bump-attempts"]


# --- main entry ------------------------------------------------------------


def verify_task(
    task_id: str,
    mode: Literal["quick", "full"],
    cwd: Path | None = None,
) -> VerifyResult:
    """Run the verification pipeline for one task. Updates board.json. Returns result.

    cwd: working directory to run commands in. If None, defaults to the per-task
    git worktree (created on demand if missing). Tests can pass an explicit cwd
    to bypass worktree management.
    """
    task = _read_task(task_id)
    attempts = int(task.get("attempts", 0))
    files = list(task.get("files_touched", []))
    modules = _files_to_modules(files)
    test_files = _files_to_test_files(files)
    if cwd is None:
        wt_mod = _load_worktree()
        cwd = wt_mod.ensure_worktree(task_id, task.get("branch") or f"feat/{task_id}")

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    JUNIT_DIR.mkdir(parents=True, exist_ok=True)
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = FEEDBACK_DIR / f"{task_id}-{mode}-r{attempts}.log"
    junit_path = JUNIT_DIR / f"{task_id}-{mode}.xml"

    _board("pid-set", task_id, str(os.getpid()))
    try:
        with log_path.open("w") as log_fp:
            log_fp.write(f"# verify {task_id} mode={mode} attempts={attempts}\n")
            log_fp.write(f"# files={files}\n")
            log_fp.write(f"# modules={modules}\n")
            log_fp.write(f"# test_files={test_files}\n")
            log_fp.write(f"# cwd={cwd}\n")

            py_files = [f for f in files if f.endswith(".py")]
            if py_files:
                ruff = _run_step("ruff", [RUFF, "check", *py_files], log_fp, cwd)
            else:
                ruff = StepResult(name="ruff", rc=0, stdout="(no python files)", stderr="")
                log_fp.write("\n$ (ruff) skipped: no python files in files_touched\n")

            if modules:
                mypy = _run_step("mypy", [MYPY, *modules], log_fp, cwd)
            else:
                mypy = StepResult(name="mypy", rc=0, stdout="(no modules)", stderr="")
                log_fp.write("\n$ (mypy) skipped: no modules\n")

            pytest_targets = test_files or (["dimos/"] if mode == "full" else [])
            if pytest_targets:
                pytest_cmd = [
                    PYTEST,
                    f"--junit-xml={junit_path}",
                    "-q",
                    "--no-header",
                    *pytest_targets,
                ]
                pytest = _run_step("pytest", pytest_cmd, log_fp, cwd)
            else:
                pytest = StepResult(name="pytest", rc=0, stdout="(no tests)", stderr="")
                log_fp.write("\n$ (pytest) skipped: no test files inferred (quick mode)\n")

        pytest_failures = _parse_junit(junit_path) if pytest.rc != 0 else []
        passed = ruff.rc == 0 and mypy.rc == 0 and pytest.rc == 0
        summary = _format_summary(ruff, mypy, pytest, pytest_failures)
        next_status, extra_args = _decide_next(passed, mode, attempts)

        board_args = [
            "set-status",
            task_id,
            next_status,
            "--feedback-summary",
            summary,
            "--feedback-log",
            str(log_path),
            *extra_args,
        ]
        rc, _, err = _board(*board_args)
        if rc != 0:
            raise RuntimeError(f"board set-status failed: {err}")

        return VerifyResult(
            task_id=task_id,
            mode=mode,
            passed=passed,
            next_status=next_status,
            summary=summary,
            log_path=log_path,
        )
    finally:
        _board("pid-clear", task_id)


# --- CLI -------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("quick", "full"):
        print("usage: verify.py <task_id> {quick|full}", file=sys.stderr)
        return 2
    task_id, mode = argv[0], argv[1]
    result = verify_task(task_id, mode)  # type: ignore[arg-type]
    print(
        json.dumps(
            {
                "task_id": result.task_id,
                "mode": result.mode,
                "passed": result.passed,
                "next_status": result.next_status,
                "summary": result.summary,
                "log_path": str(result.log_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

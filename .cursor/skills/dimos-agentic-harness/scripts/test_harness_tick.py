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

"""Tests for harness.py: tick action emission + sleep loop semantics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("harness_mod", SCRIPTS_DIR / "harness.py")
assert spec and spec.loader
harness = importlib.util.module_from_spec(spec)
sys.modules["harness_mod"] = harness
spec.loader.exec_module(harness)


def _init_git(tmp_path: Path) -> None:
    """Bootstrap a tiny repo so worktree.ensure_worktree works."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "dev"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "dev"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "fetch", "origin"], cwd=tmp_path, check=True, capture_output=True)


def _init_board(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "init",
            "--gitignore",
            str(tmp_path / "x.gitignore"),
        ],
        check=True,
        capture_output=True,
    )


def _add_task(task_id: str, status: str, branch: str = "feat/x") -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "add-task",
            task_id,
            "--branch",
            branch,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "set-status",
            task_id,
            status,
        ],
        check=True,
        capture_output=True,
    )


def test_tick_no_loop_returns_wait_when_idle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _init_board(tmp_path)
    _add_task("t1", "PR_OPEN")  # not eligible for any action

    rc = harness.cmd_tick(loop=False)
    assert rc == 0
    payload = json.loads((tmp_path / ".harness/heartbeat").read_text())["payload"]
    assert len(payload["actions"]) == 1
    assert payload["actions"][0]["kind"] == "wait"


def test_tick_no_loop_returns_done_when_all_terminal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _init_board(tmp_path)
    _add_task("t1", "MERGED")
    _add_task("t2", "BLOCKED")

    rc = harness.cmd_tick(loop=False)
    assert rc == 0
    payload = json.loads((tmp_path / ".harness/heartbeat").read_text())["payload"]
    assert payload["actions"] == [{"kind": "done"}]


def test_tick_no_loop_emits_implementer_for_planned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _init_git(tmp_path)  # ensure_worktree needs a real repo
    _init_board(tmp_path)
    _add_task("t1", "PLANNED", branch="feat/foo")

    rc = harness.cmd_tick(loop=False)
    assert rc == 0
    payload = json.loads((tmp_path / ".harness/heartbeat").read_text())["payload"]
    actions = payload["actions"]
    assert len(actions) == 1
    assert actions[0]["kind"] == "spawn-implementer"
    assert actions[0]["task_id"] == "t1"
    assert "cwd" in actions[0]
    assert ".harness/worktrees/t1" in actions[0]["cwd"]


def _set_verify_stage(task_id: str, stage: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "board.py"),
            "set-status",
            task_id,
            "VERIFYING",
            "--verify-stage",
            stage,
        ],
        check=True,
        capture_output=True,
    )


def _stub_verify_module(calls: list[tuple[str, str]]) -> SimpleNamespace:
    """Build a stub of verify.py that records calls but does no real work."""

    def stub_verify(task_id: str, mode: str) -> SimpleNamespace:
        calls.append((task_id, mode))
        return SimpleNamespace(
            task_id=task_id,
            mode=mode,
            passed=True,
            next_status="VERIFYING",
            summary="stub",
        )

    return SimpleNamespace(verify_task=stub_verify)


def test_run_verifier_inline_picks_up_verifying_no_stage(tmp_path: Path, monkeypatch) -> None:
    """Regression: VERIFYING with verify_stage=None must trigger quick mode.

    The implementer transitions IMPLEMENTING → VERIFYING (no stage) as its
    done-signal. The previous dispatcher only matched IMPLEMENTING (which is
    the in-progress state and is never verified), so all impl-finished tasks
    were silently skipped and the harness spin-waited.
    """
    monkeypatch.chdir(tmp_path)
    _init_board(tmp_path)
    _add_task("tv", "VERIFYING")

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(harness, "_load_verify", lambda: _stub_verify_module(calls))

    board = harness._read_board_json()
    events = harness._run_verifier_inline(board)
    assert len(events) == 1
    assert events[0]["task_id"] == "tv"
    assert events[0]["mode"] == "quick"
    assert calls == [("tv", "quick")]


def test_run_verifier_inline_picks_up_verifying_stage_quick(tmp_path: Path, monkeypatch) -> None:
    """VERIFYING with verify_stage=quick must trigger full mode."""
    monkeypatch.chdir(tmp_path)
    _init_board(tmp_path)
    _add_task("tv", "VERIFYING")
    _set_verify_stage("tv", "quick")

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(harness, "_load_verify", lambda: _stub_verify_module(calls))

    board = harness._read_board_json()
    events = harness._run_verifier_inline(board)
    assert calls == [("tv", "full")]
    assert events[0]["mode"] == "full"


def test_run_verifier_inline_skips_implementing(tmp_path: Path, monkeypatch) -> None:
    """Tasks in IMPLEMENTING (in-progress) must NOT be verified.

    If the implementer crashes mid-flight, `_resume_dead_workers` reverts the
    status to PLANNED for re-dispatch — the verifier never touches IMPLEMENTING.
    """
    monkeypatch.chdir(tmp_path)
    _init_board(tmp_path)
    _add_task("ti", "IMPLEMENTING")

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(harness, "_load_verify", lambda: _stub_verify_module(calls))

    board = harness._read_board_json()
    events = harness._run_verifier_inline(board)
    assert events == []
    assert calls == []


def test_emit_review_actions_emits_for_reviewing(tmp_path: Path, monkeypatch) -> None:
    """REVIEWING + review_attempts < MAX → one spawn-reviewer action."""
    monkeypatch.chdir(tmp_path)
    _init_git(tmp_path)  # ensure_worktree needs a real repo
    _init_board(tmp_path)
    _add_task("tr", "REVIEWING", branch="feat/review-me")

    board = harness._read_board_json()
    actions = harness._emit_review_actions(board)
    assert len(actions) == 1
    a = actions[0]
    assert a["kind"] == "spawn-reviewer"
    assert a["task_id"] == "tr"
    assert a["model"] == harness.REVIEWER_MODEL == "gpt-5.5-medium"
    assert a["review_attempts"] == 0
    assert ".harness/worktrees/tr" in a["cwd"]


def test_emit_review_actions_blocks_at_cap(tmp_path: Path, monkeypatch) -> None:
    """REVIEWING + review_attempts >= MAX → no action, status set to BLOCKED."""
    monkeypatch.chdir(tmp_path)
    _init_board(tmp_path)
    _add_task("tr", "REVIEWING", branch="feat/review-me")
    # bump review_attempts up to the cap
    for _ in range(harness.MAX_REVIEW_ITERATIONS):
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "board.py"),
                "set-status",
                "tr",
                "REVIEWING",
                "--bump-review-attempts",
            ],
            check=True,
            capture_output=True,
        )

    board = harness._read_board_json()
    actions = harness._emit_review_actions(board)
    assert actions == []

    info_proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "board.py"), "task-info", "tr"],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(info_proc.stdout)
    assert info["status"] == "BLOCKED"
    assert "reviewer rejected" in (info["blocked_reason"] or "")


def test_tick_loop_sleeps_then_returns_when_terminal(tmp_path: Path, monkeypatch) -> None:
    """loop=True with HARNESS_POLL_INTERVAL_SEC=0 should not sleep but also
    must NOT spin forever — it should converge to the same wait result on the
    next iteration. We force terminal mid-loop by mutating the board.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_POLL_INTERVAL_SEC", "0")
    _init_board(tmp_path)
    _add_task("t1", "PR_OPEN")  # idle → triggers wait

    real_sleep = time.sleep
    call_count = {"n": 0}

    def fake_sleep(s: float) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # mutate board: flip task to MERGED so next inner tick → done
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "board.py"),
                    "set-status",
                    "t1",
                    "MERGED",
                ],
                check=True,
                capture_output=True,
            )
        if call_count["n"] > 5:
            raise RuntimeError("sleep loop did not terminate")
        real_sleep(0)

    monkeypatch.setattr(harness.time, "sleep", fake_sleep)

    rc = harness.cmd_tick(loop=True)
    assert rc == 0
    payload = json.loads((tmp_path / ".harness/heartbeat").read_text())["payload"]
    assert payload["actions"] == [{"kind": "done"}]
    assert call_count["n"] == 1

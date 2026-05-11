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

SCRIPTS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("harness_mod", SCRIPTS_DIR / "harness.py")
assert spec and spec.loader
harness = importlib.util.module_from_spec(spec)
sys.modules["harness_mod"] = harness
spec.loader.exec_module(harness)


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
    _init_board(tmp_path)
    _add_task("t1", "PLANNED")

    rc = harness.cmd_tick(loop=False)
    assert rc == 0
    payload = json.loads((tmp_path / ".harness/heartbeat").read_text())["payload"]
    actions = payload["actions"]
    assert len(actions) == 1
    assert actions[0]["kind"] == "spawn-implementer"
    assert actions[0]["task_id"] == "t1"


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

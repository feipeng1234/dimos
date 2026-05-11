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

"""Tests for board.py multi-round meta block (round, history, append-plan)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
BOARD = SCRIPTS_DIR / "board.py"


def _board(tmp: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BOARD), *args],
        cwd=tmp,
        capture_output=True,
        text=True,
        check=check,
    )


def _init(tmp: Path) -> None:
    _board(tmp, "init", "--gitignore", str(tmp / "x.gitignore"))


def _write_plan(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body)
    return p


def test_init_creates_meta_block(tmp_path: Path) -> None:
    _init(tmp_path)
    meta = json.loads(_board(tmp_path, "meta-info").stdout)
    assert meta["round"] == 1
    assert meta["max_rounds"] == 10
    assert meta["per_round_cap"] == 3
    assert meta["goal_achieved"] is False
    assert meta["no_progress_rounds"] == 0
    assert meta["history"] == []


def test_v2_board_migrates_to_v3_on_read(tmp_path: Path) -> None:
    """A pre-existing v2 board.json (no meta block) gets default meta added
    transparently on the first read; the next write upgrades it on disk."""
    harness_dir = tmp_path / ".harness"
    harness_dir.mkdir()
    legacy = {
        "version": 2,
        "tasks": [
            {
                "id": "t1",
                "title": "legacy",
                "branch": "feat/legacy",
                "deps": [],
                "group": None,
                "files_touched": [],
                "status": "MERGED",
                "verify_stage": None,
                "attempts": 0,
                "feedback_summary": "",
                "feedback_log_path": ".harness/feedback/t1.log",
                "pr_url": None,
                "pr_number": None,
                "opened_at": None,
                "babysit_attempts": 0,
                "consecutive_rebase_fails": 0,
                "review_attempts": 0,
                "blocked_reason": None,
                "ready_for_maintainer_reason": None,
            }
        ],
        "groups": [],
        "locks_dir": ".harness/locks",
        "pids_dir": ".harness/pids",
    }
    (harness_dir / "board.json").write_text(json.dumps(legacy))

    meta = json.loads(_board(tmp_path, "meta-info").stdout)
    assert meta["round"] == 1
    assert meta["goal_achieved"] is False

    info = json.loads(_board(tmp_path, "task-info", "t1").stdout)
    assert info["title"] == "legacy"
    assert info["status"] == "MERGED"


def test_append_plan_more_work_adds_tasks_and_bumps_round(tmp_path: Path) -> None:
    _init(tmp_path)
    _board(tmp_path, "add-task", "t1", "--branch", "feat/a")
    _board(tmp_path, "set-status", "t1", "MERGED")

    _write_plan(
        tmp_path,
        "replan.yaml",
        "goal_achieved: false\n"
        "tasks:\n"
        "  - id: t2\n"
        "    title: 'next slice'\n"
        "    branch: feat/b\n"
        "  - id: t3\n"
        "    title: 'parallel slice'\n"
        "    branch: feat/c\n",
    )
    out = _board(tmp_path, "append-plan", "replan.yaml").stdout
    payload = json.loads(out)
    assert payload["action"] == "appended"
    assert payload["from_round"] == 1
    assert payload["to_round"] == 2
    assert payload["new_tasks"] == ["t2", "t3"]

    meta = json.loads(_board(tmp_path, "meta-info").stdout)
    assert meta["round"] == 2
    assert meta["no_progress_rounds"] == 0
    assert len(meta["history"]) == 1
    h = meta["history"][0]
    assert h["round"] == 1
    assert h["tasks_merged"] == ["t1"]
    assert h["new_tasks"] == ["t2", "t3"]
    assert h["replanner_verdict"] == "more-work"

    t2 = json.loads(_board(tmp_path, "task-info", "t2").stdout)
    assert t2["added_in_round"] == 2
    assert t2["status"] == "PLANNED"


def test_append_plan_goal_achieved_blocks_further_appends(tmp_path: Path) -> None:
    _init(tmp_path)
    _board(tmp_path, "add-task", "t1", "--branch", "feat/a")
    _board(tmp_path, "set-status", "t1", "MERGED")

    _write_plan(tmp_path, "replan.yaml", "goal_achieved: true\ntasks: []\n")
    out = _board(tmp_path, "append-plan", "replan.yaml").stdout
    assert json.loads(out)["action"] == "goal-achieved"

    meta = json.loads(_board(tmp_path, "meta-info").stdout)
    assert meta["goal_achieved"] is True
    assert meta["history"][-1]["replanner_verdict"] == "goal-achieved"

    second = _board(tmp_path, "append-plan", "replan.yaml", check=False)
    assert second.returncode != 0
    assert "goal_achieved already set" in (second.stderr + second.stdout)


def test_append_plan_no_progress_bumps_counter(tmp_path: Path) -> None:
    _init(tmp_path)
    _board(tmp_path, "add-task", "t1", "--branch", "feat/a")
    _board(tmp_path, "set-status", "t1", "MERGED")

    _write_plan(tmp_path, "replan.yaml", "goal_achieved: false\ntasks: []\n")
    out = _board(tmp_path, "append-plan", "replan.yaml").stdout
    p1 = json.loads(out)
    assert p1["action"] == "no-progress"
    assert p1["no_progress_rounds"] == 1

    out = _board(tmp_path, "append-plan", "replan.yaml").stdout
    p2 = json.loads(out)
    assert p2["no_progress_rounds"] == 2

    meta = json.loads(_board(tmp_path, "meta-info").stdout)
    assert meta["round"] == 1  # unchanged on no-progress
    assert meta["no_progress_rounds"] == 2


def test_append_plan_rejects_duplicate_id(tmp_path: Path) -> None:
    _init(tmp_path)
    _board(tmp_path, "add-task", "t1", "--branch", "feat/a")

    _write_plan(
        tmp_path,
        "replan.yaml",
        "goal_achieved: false\ntasks:\n  - id: t1\n    branch: feat/dup\n",
    )
    proc = _board(tmp_path, "append-plan", "replan.yaml", check=False)
    assert proc.returncode != 0
    assert "already exists" in (proc.stderr + proc.stdout)


def test_append_plan_rejects_over_cap(tmp_path: Path) -> None:
    _init(tmp_path)
    _board(tmp_path, "set-meta", "--per-round-cap", "2")

    _write_plan(
        tmp_path,
        "replan.yaml",
        "goal_achieved: false\n"
        "tasks:\n"
        "  - id: t1\n    branch: feat/a\n"
        "  - id: t2\n    branch: feat/b\n"
        "  - id: t3\n    branch: feat/c\n",
    )
    proc = _board(tmp_path, "append-plan", "replan.yaml", check=False)
    assert proc.returncode != 0
    assert "per_round_cap" in (proc.stderr + proc.stdout)


def test_append_plan_resets_no_progress_when_tasks_added(tmp_path: Path) -> None:
    _init(tmp_path)
    _board(tmp_path, "add-task", "t1", "--branch", "feat/a")
    _board(tmp_path, "set-status", "t1", "MERGED")

    _write_plan(tmp_path, "noop.yaml", "goal_achieved: false\ntasks: []\n")
    _board(tmp_path, "append-plan", "noop.yaml")
    assert json.loads(_board(tmp_path, "meta-info").stdout)["no_progress_rounds"] == 1

    _write_plan(
        tmp_path,
        "more.yaml",
        "goal_achieved: false\ntasks:\n  - id: t2\n    branch: feat/b\n",
    )
    _board(tmp_path, "append-plan", "more.yaml")
    meta = json.loads(_board(tmp_path, "meta-info").stdout)
    assert meta["no_progress_rounds"] == 0
    assert meta["round"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

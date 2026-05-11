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

"""Group integration gate runner.

For a group `<gid>`, integrates all member branches on top of fork:dev in
topological order, running the full verifier after each merge. On failure
(merge conflict OR verify failure), blames the offending task and asks
board.py to unstack it. On success, marks the group's gate_status PASSED.

Usage:
    python group_gate.py <group-id>

Exit codes:
    0  → all members merged + verified; gate PASSED.
    1  → blame done; caller should re-tick the harness.
    2  → fatal error (board missing, group missing, etc).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
BOARD = SCRIPTS / "board.py"
LOG_DIR = Path(".harness/logs")

VERIFY_FULL_DEFAULT = (
    "./bin/pytest-fast && uv run ruff check . "
    "&& uv run mypy dimos/ && uv run pre-commit run --all-files"
)


def _run(cmd: list[str], log_path: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.write(proc.stdout)
            if proc.stderr:
                f.write(f"\n--- stderr ---\n{proc.stderr}\n")
    return proc.returncode, proc.stdout, proc.stderr


def _board(*args: str) -> tuple[int, str, str]:
    return _run([sys.executable, str(BOARD), *args])


def _unstack(gid: str, blame_id: str, reason: str) -> None:
    print(f"[gate] BLAME {blame_id}: {reason}", file=sys.stderr)
    rc, _, err = _board("unstack", gid, "--blame", blame_id)
    if rc != 0:
        print(f"[gate] board.py unstack failed: {err}", file=sys.stderr)
    _board("set-gate-status", gid, "FAILED")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    gid = argv[0]

    rc, out, err = _board("group-info", gid)
    if rc != 0:
        print(f"[gate] group-info failed: {err}", file=sys.stderr)
        return 2
    info = json.loads(out)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{gid}-gate.log"
    log_path.write_text(f"# Group gate run for {gid}\n")

    integration_branch = info["integration_branch"]
    topo_ids = info["topo_order"]
    members_by_id = {m["id"]: m for m in info["members"]}

    rc, _, err = _run(["git", "fetch", "origin"], log_path)
    if rc != 0:
        print(f"[gate] git fetch failed: {err}", file=sys.stderr)
        _board("set-gate-status", gid, "FAILED")
        return 2

    rc, _, err = _run(
        ["git", "switch", "-C", integration_branch, "origin/dev"],
        log_path,
    )
    if rc != 0:
        print(f"[gate] git switch -C {integration_branch} failed: {err}", file=sys.stderr)
        _board("set-gate-status", gid, "FAILED")
        return 2

    verify_full = os.environ.get("HARNESS_VERIFY_FULL_CMD", VERIFY_FULL_DEFAULT)

    for tid in topo_ids:
        member = members_by_id[tid]
        branch = member["branch"]
        print(f"[gate] merging {tid} ({branch})", file=sys.stderr)

        rc, _, err = _run(["git", "merge", "--no-ff", "--no-edit", branch], log_path)
        if rc != 0:
            _run(["git", "merge", "--abort"], log_path)
            _unstack(gid, tid, f"merge conflict on top of {integration_branch}")
            return 1

        print(f"[gate] verify-full after {tid}", file=sys.stderr)
        rc, _, err = _run(["bash", "-lc", verify_full], log_path)
        if rc != 0:
            _unstack(gid, tid, "verify-full failed after merge")
            return 1

    _board("set-gate-status", gid, "PASSED")
    print(f"[gate] {gid} PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

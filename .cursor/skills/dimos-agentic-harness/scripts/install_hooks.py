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

"""Install harness git hooks into .git/hooks/.

Hooks live as source under `.cursor/skills/dimos-agentic-harness/hooks/`
and are symlinked into `.git/hooks/` so updates flow through automatically.
Worktrees share the same .git directory, so hooks apply to all of them.

Backups: if `.git/hooks/<name>` already exists and is NOT a symlink to our
source, it is moved to `.git/hooks/<name>.harness-backup-<ts>` before we
take over. Existing symlinks pointing into our hooks dir are reused.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
HOOKS_SRC = SCRIPTS.parent / "hooks"
HOOK_NAMES = ("pre-push",)


def _git_common_dir() -> Path:
    """Return the shared .git directory (shared across worktrees)."""
    proc = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip()).resolve()


def install(force: bool = False) -> list[str]:
    """Install all hooks. Returns a list of human-readable status lines."""
    git_dir = _git_common_dir()
    hooks_dst = git_dir / "hooks"
    hooks_dst.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for name in HOOK_NAMES:
        src = HOOKS_SRC / name
        if not src.exists():
            out.append(f"  [skip] {name}: source missing at {src}")
            continue
        dst = hooks_dst / name
        if dst.is_symlink():
            target = dst.resolve()
            if target == src.resolve():
                out.append(f"  [ok]   {name}: already linked")
                continue
            if not force:
                out.append(
                    f"  [warn] {name}: existing symlink → {target}; rerun with --force to replace"
                )
                continue
            dst.unlink()
        elif dst.exists():
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = hooks_dst / f"{name}.harness-backup-{ts}"
            shutil.move(str(dst), str(backup))
            out.append(f"  [back] {name}: existing file moved to {backup.name}")
        os.symlink(src.resolve(), dst)
        out.append(f"  [link] {name}: → {src}")
    return out


def uninstall() -> list[str]:
    git_dir = _git_common_dir()
    hooks_dst = git_dir / "hooks"
    out: list[str] = []
    for name in HOOK_NAMES:
        dst = hooks_dst / name
        if not dst.exists() and not dst.is_symlink():
            out.append(f"  [skip] {name}: not installed")
            continue
        if dst.is_symlink() and dst.resolve() == (HOOKS_SRC / name).resolve():
            dst.unlink()
            out.append(f"  [rm]   {name}: symlink removed")
        else:
            out.append(
                f"  [keep] {name}: not a harness symlink; refusing to remove (target: {dst.resolve() if dst.exists() else '(broken)'})"
            )
    return out


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "install"
    rest = argv[1:]
    if cmd == "install":
        force = "--force" in rest
        for line in install(force=force):
            print(line)
        return 0
    if cmd == "uninstall":
        for line in uninstall():
            print(line)
        return 0
    print("usage: install_hooks.py {install [--force] | uninstall}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

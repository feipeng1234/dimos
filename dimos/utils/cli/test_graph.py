# Copyright 2025-2026 Dimensional Inc.
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


import pathlib
import sys
import types

import pytest

from dimos.utils.cli.graph import main


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        main("/nonexistent/path.py")


def _ensure_blueprints_module() -> type:
    """Ensure ``dimos.core.blueprints`` is importable, creating a stub if needed.

    Returns the ``Blueprint`` class (real or stub).
    """
    try:
        from dimos.core.blueprints import Blueprint

        return Blueprint
    except (ImportError, ModuleNotFoundError):
        pass

    # Create a minimal stub so that _load_blueprints can import the module.
    bp_mod = types.ModuleType("dimos.core.blueprints")

    class _StubBlueprint:
        pass

    bp_mod.Blueprint = _StubBlueprint  # type: ignore[attr-defined]
    sys.modules.setdefault("dimos.core.blueprints", bp_mod)

    # Also ensure the parent packages exist in sys.modules.
    for parent in ("dimos.core",):
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)

    return _StubBlueprint


def test_no_blueprints(tmp_path: pathlib.Path) -> None:
    _ensure_blueprints_module()
    p = tmp_path / "empty.py"
    p.write_text("x = 42\n")
    with pytest.raises(RuntimeError, match="No Blueprint instances"):
        main(str(p))


def test_module_load_failure(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "bad.py"
    p.write_text("raise ImportError('boom')\n")
    with pytest.raises(ImportError, match="boom"):
        main(str(p))

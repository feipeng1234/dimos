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

"""ACT training entry point. Called directly by TrainerModule.

`train_bc` lazy-imports lerobot/torch, builds Hydra-style argv from
BCConfig, calls lerobot's trainer in-process, appends `dimos_meta.json`
to output_dir, returns the checkpoint path.

Stats live at `<dataset>/meta/stats.json` (written by DataPrep). Training
reads them via lerobot's loader; never recomputes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dimos.learning.dataprep import Episode
from dimos.learning.training.configs import BCConfig

DIMOS_META_FILENAME = "dimos_meta.json"


def train_bc(
    dataset_path: str | Path,
    cfg: BCConfig,
    output_dir: str | Path,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Train ACT on a prepared dataset. Returns checkpoint dir."""
    raise NotImplementedError


def _build_lerobot_argv(cfg: BCConfig, dataset_path: Path, output_dir: Path) -> list[str]:
    """Translate BCConfig → Hydra-style CLI args for lerobot's trainer."""
    raise NotImplementedError


def _write_dimos_meta(output_dir: Path, dataset_path: Path) -> None:
    """Read <dataset_path>/dimos_meta.json, add policy fields
    (joint_names, chunk_size, policy_type), write to <output_dir>/dimos_meta.json."""
    raise NotImplementedError


def train_val_split(
    episodes: list[Episode],
    val_episode_ids: list[int] | None = None,
    val_ratio: float | None = None,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Partition `episodes` indices into (train_ids, val_ids).

    Resolution order (first non-None wins):
      1. `val_episode_ids` — explicit whitelist
      2. `val_ratio`       — deterministic random split via `seed`
      3. both None         — empty val (everything is train)
    """
    raise NotImplementedError

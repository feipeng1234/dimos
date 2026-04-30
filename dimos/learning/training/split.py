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

"""Episode-level train/val split.

LeRobot v2 supports filtering by episode index at training time, so we don't
materialize two datasets. We compute the partition once and pass the index
lists to the trainer.

Resolution order (first non-None wins):
  1. `cfg.val_episode_ids` — explicit whitelist
  2. `cfg.val_ratio`       — deterministic random split via cfg.val_split_seed
  3. neither set            — empty val (everything is train)
"""

from __future__ import annotations

from dimos.learning.spec import Episode, FilterConfig


def train_val_split(
    episodes: list[Episode],
    cfg: FilterConfig | None,
) -> tuple[list[int], list[int]]:
    """Partition `episodes` (already filtered) into (train_ids, val_ids).

    Returns lists of episode *indices* into `episodes`, not Episode objects.
    LeRobot consumes index lists. Determinism is via `cfg.val_split_seed`.
    """
    raise NotImplementedError

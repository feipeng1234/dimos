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

"""Replay policy ActionChunks at coordinator tick rate (100 Hz).

Slow producer (ChunkPolicyModule @ ~30 Hz, jittery) → buffer →
deterministic 100 Hz JointCommandOutput. Stale-chunk and policy-stall
handling keep hardware safe when the policy falls behind or dies.
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.control.task import (
    BaseControlTask,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
)
from dimos.learning.policy.base import ActionChunk


@dataclass
class ActionReplayerConfig:
    joint_names:       list[str]
    priority:          int   = 10
    max_chunk_age_s:   float = 0.5    # drop chunks older than this at receive time
    hold_on_stall:     bool  = True   # hold last position if buffer drains
    temporal_ensemble: bool  = False  # ACT trick; off in v1


class ActionReplayer(BaseControlTask):
    """Buffer latest chunk; interpolate per tick; emit JointCommandOutput."""

    def __init__(self, name: str, config: ActionReplayerConfig) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def claim(self) -> ResourceClaim:
        """Claim `config.joint_names` at `config.priority`."""
        raise NotImplementedError

    def is_active(self) -> bool:
        """True iff buffer has a non-stale target for now (or hold_on_stall)."""
        raise NotImplementedError

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        """Pure lookup / interpolate over the buffer at `state.now`.
        Must complete in << 10 ms."""
        raise NotImplementedError

    def on_action_chunk(self, msg: ActionChunk) -> None:
        """Latest-wins push. Drop if msg too old; drop buffered entries
        at/after msg's first target_ts; append new (target_ts, positions)."""
        raise NotImplementedError

    def _interpolate(self, t: float) -> JointCommandOutput | None:
        raise NotImplementedError

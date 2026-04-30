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

"""Policy abstraction — what `ChunkPolicyModule` calls every inference tick.

The Policy protocol decouples model format (lerobot PreTrainedPolicy in v1,
ONNX/TorchScript in v2) from the inference module. Anything that satisfies
this protocol is droppable into a blueprint.

`ActionChunk` is the typed message published by `ChunkPolicyModule` and
consumed by `ActionReplayer`. v1 uses a pydantic model; v2 will replace it
with a generated LCM type so it can flow over the wire — the field layout
here matches what that LCM type will look like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict


class ActionChunk(BaseModel):
    """A predicted sequence of joint targets, plus the metadata to replay it.

    Fields:
        ts:           wall-clock time the chunk was produced (seconds).
        joint_names:  names matching the action key ordering used at training.
        positions:    shape (T, N) — T future steps, N = len(joint_names).
        dt:           expected interval between successive actions (seconds).
                      Replayer uses ts + i*dt as the target time for action i.
        chunk_id:     monotonic id for ordering / dedup at the replayer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float
    joint_names: list[str]
    positions: np.ndarray  # (T, N)
    dt: float
    chunk_id: int


@runtime_checkable
class Policy(Protocol):
    """What ChunkPolicyModule needs from any policy implementation."""

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> Policy:
        """Load a checkpoint directory. `path` is a lerobot checkpoint dir in v1.

        Implementations should also load the sidecar `dimos_meta.json` and
        `meta/stats.json` so `predict_chunk` can normalize/unnormalize without
        the caller doing it.
        """
        ...

    @property
    def chunk_size(self) -> int:
        """Number of actions emitted per `predict_chunk` call (T)."""
        ...

    @property
    def joint_names(self) -> list[str]:
        """Action joint names, matching the spec's action key ordering."""
        ...

    @property
    def expects_language(self) -> bool:
        """True if the policy reads `obs['language_text']` (VLAs); False otherwise."""
        ...

    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return shape (chunk_size, action_dim) — already unnormalized to joint space.

        `obs` keys must match `spec.observation`. The policy applies its own
        input normalization internally (using the stats it loaded with the
        checkpoint).
        """
        ...

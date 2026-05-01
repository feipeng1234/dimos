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

"""LeRobot ACT policy wrapper. Lazy-imports lerobot/torch in load()."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dimos.learning.policy.base import Policy


class LeRobotPolicy:
    _model: Any  # lerobot.policies.pretrained.PreTrainedPolicy
    _stats: dict[str, Any]
    _chunk_size: int
    _joint_names: list[str]
    _device: str

    def __init__(
        self,
        model: Any,
        stats: dict[str, Any],
        chunk_size: int,
        joint_names: list[str],
        device: str,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> LeRobotPolicy:
        """Load checkpoint dir: model.safetensors + meta/stats.json + dimos_meta.json."""
        raise NotImplementedError

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Normalize obs → forward pass → unnormalize → (chunk_size, action_dim)."""
        raise NotImplementedError


# Protocol conformance check at import time.
_: type[Policy] = LeRobotPolicy

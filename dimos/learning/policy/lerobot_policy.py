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

"""LeRobot policy wrapper.

Wraps any `lerobot.PreTrainedPolicy` (ACT, Diffusion, pi0, pi0.5) behind the
`Policy` protocol. This is the only Policy implementation in v1 — both
training entry points produce checkpoints loadable by this class.

Heavy deps (`lerobot`, `torch`) are imported lazily inside `load()` so simply
importing this module does not require a CUDA install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dimos.learning.policy.base import Policy


class LeRobotPolicy:
    """Adapter for lerobot's PreTrainedPolicy → DimOS Policy protocol."""

    # Type-erased to keep this file import-light. Concrete type:
    #   _model: lerobot.policies.pretrained.PreTrainedPolicy
    _model: Any
    _stats: dict[str, Any]
    _chunk_size: int
    _joint_names: list[str]
    _expects_language: bool
    _device: str

    def __init__(
        self,
        model: Any,
        stats: dict[str, Any],
        chunk_size: int,
        joint_names: list[str],
        expects_language: bool,
        device: str,
    ) -> None:
        """Direct constructor — prefer `LeRobotPolicy.load(path)` in user code."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> LeRobotPolicy:
        """Load a lerobot checkpoint directory.

        Expected layout under `path`:
            config.json / model.safetensors  - the lerobot checkpoint
            meta/stats.json                  - normalization stats
            dimos_meta.json                  - DimOS sidecar (spec + provenance)

        Auto-detects the policy class (act / diffusion / pi0 / pi0_5) from
        the lerobot config and sets `expects_language` accordingly.
        """
        raise NotImplementedError

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def expects_language(self) -> bool:
        return self._expects_language

    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Run one forward pass; return (chunk_size, action_dim).

        Steps:
          1. Normalize obs via `self._stats` (image: /255 + per-channel norm;
             vector: (x - mean) / std).
          2. Convert to torch tensors on `self._device`, add batch dim.
          3. Call `self._model.select_action_chunk(obs)` (or equivalent).
          4. Move back to numpy, drop batch dim.
          5. Unnormalize actions via `self._stats`.

        Matches the pipeline used inside lerobot's training loop, so live
        inference sees the same numerics as training-time evaluation.
        """
        raise NotImplementedError


# Sanity check: make the protocol relationship explicit at import time.
_: type[Policy] = LeRobotPolicy

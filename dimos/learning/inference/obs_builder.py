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

"""Live observation construction for inference.

`ObsBuilder` is the inference-time counterpart to `DataPrep.iter_episode_samples`.
At training time, samples are built by walking recorded streams; at inference
time we have the *latest* message on each live stream. The transformation
from per-stream messages to a model-ready obs dict must be identical between
the two paths or we get train/serve skew.

To guarantee that, `ObsBuilder` reuses `DataPrep.resolve_field` for field
projection + preprocess. The only thing it adds is a mapping from the spec's
`stream:` names to live message objects supplied by the caller (the
ChunkPolicyModule, which has the actual `In[Image]` / `In[JointState]` ports).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dimos.learning.spec import DatasetSpec


class ObsBuilder:
    """Builds the model-input dict from the latest live messages.

    Construction takes a `DatasetSpec`. `build()` takes a `{stream_name: msg}`
    dict where keys are the recorded stream names referenced by
    `spec.observation[*].stream`, and values are the live LCM messages.

    The caller (ChunkPolicyModule) is responsible for resolving its In ports
    to those stream names — that's a small static mapping it sets up once.
    """

    def __init__(self, spec: DatasetSpec) -> None:
        """Cache the observation StreamFields for fast lookup at tick rate."""
        raise NotImplementedError

    def build(self, live_messages: dict[str, Any]) -> dict[str, np.ndarray]:
        """Project + preprocess the latest message on each obs stream.

        Args:
            live_messages: stream_name -> latest message object (e.g.
                {"camera_color_image": <Image>, "coordinator_joint_state": <JointState>}).
                Every stream referenced by `spec.observation` must be present;
                missing streams raise.

        Returns:
            obs dict keyed by `spec.observation` keys (e.g. "cam_high",
            "joint_pos"). Values are np.ndarrays whose shapes/dtypes match
            what `iter_episode_samples` produced at training time.
        """
        raise NotImplementedError

    def required_streams(self) -> set[str]:
        """Stream names this builder reads from. Used by ChunkPolicyModule
        to wire its In ports + assert the live_messages dict is complete."""
        raise NotImplementedError

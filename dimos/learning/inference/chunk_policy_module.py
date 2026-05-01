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

"""Vision/VLA policy as a DimOS Module — produces action chunks at policy rate.

One module covers both v1 inference targets:
  - ACT      (10–30 Hz, vision + joint state)
  - pi0/pi0.5 (1–5 Hz, vision + joint state + language)

`ChunkPolicyModule` runs the policy in a background thread at `inference_rate_hz`,
publishes each output as an `ActionChunk` message, and is consumed by
`ActionReplayer` (in the coordinator's tick loop) which interpolates to 100 Hz.

Heavy ML deps (`lerobot`, `torch`) are imported lazily via `LeRobotPolicy.load`,
not at module import time — so just having this in a blueprint doesn't pull
CUDA into every install.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.learning.inference.obs_builder import ObsBuilder
from dimos.learning.policy.base import ActionChunk, Policy
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState


class ChunkPolicyModuleConfig(ModuleConfig):
    """Config for ChunkPolicyModule."""

    spec_path: str  # path to dataset.yaml — supplies obs construction
    policy_path: str  # path to lerobot checkpoint dir
    inference_rate_hz: float = 5.0  # 5 Hz default for VLA; 30 Hz for ACT
    device: str = "cuda"
    default_language: str = ""  # used when `language_text` port has no value yet


class ChunkPolicyModule(Module):
    """Runs a Policy at `inference_rate_hz`, publishes ActionChunks.

    Live message latching:
      - `color_image` and `joint_state` are cached on every receive; the
        policy thread reads the latest cached value at each tick.
      - `language_text` is optional; if the policy doesn't expect language
        (`policy.expects_language is False`) the port is ignored.

    The thread loop is best-effort wrt `inference_rate_hz`: if a forward pass
    takes longer than the period, the next tick fires immediately; we never
    queue stale work.
    """

    config: ChunkPolicyModuleConfig

    color_image: In[Image]
    joint_state: In[JointState]
    language_text: In[str]

    action_chunk: Out[ActionChunk]

    def __init__(self, **kwargs: Any) -> None:
        """Defer all heavy init to `start()`."""
        super().__init__(**kwargs)
        # Latched live messages — written by port callbacks, read by policy thread.
        self._latest_image: Image | None = None
        self._latest_joint_state: JointState | None = None
        self._latest_language: str | None = None
        self._latch_lock = threading.Lock()

        # Filled in start():
        self._policy: Policy | None = None
        self._obs_builder: ObsBuilder | None = None
        self._chunk_id: int = 0

        # Thread control
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    @rpc
    def start(self) -> None:
        """Load spec + policy, subscribe to ports, spawn the inference thread.

        Steps:
          1. `spec = DatasetSpec.from_file(config.spec_path)`
          2. `self._policy = LeRobotPolicy.load(config.policy_path, device=config.device)`
          3. `self._obs_builder = ObsBuilder(spec)`
          4. Subscribe color_image / joint_state / language_text -> latch handlers.
          5. Start the policy thread targeting `_run_loop`.
        """
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        """Stop the inference thread and call `super().stop()`."""
        raise NotImplementedError

    # ── agent surface ────────────────────────────────────────────────────────

    @rpc
    def set_language(self, text: str) -> None:
        """Override the language conditioning text without touching the
        upstream `language_text` port. Useful when an LLM agent skill drives
        VLA task switching."""
        raise NotImplementedError

    @rpc
    def reload_policy(self, policy_path: str, device: str | None = None) -> None:
        """Hot-swap the policy checkpoint without restarting the blueprint.
        Stops the inference thread, loads the new checkpoint, restarts."""
        raise NotImplementedError

    @rpc
    def get_status(self) -> dict[str, Any]:
        """Return {'running': bool, 'chunk_count': int, 'policy_path': str,
        'expects_language': bool, 'last_chunk_ts': float | None}."""
        raise NotImplementedError

    # ── inference loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Background thread. Sleep to next deadline, build obs, call policy,
        publish chunk. Logs and continues on any per-tick error so a single
        bad observation doesn't kill inference.
        """
        raise NotImplementedError

    def _build_live_obs(self) -> dict[str, Any] | None:
        """Snapshot the latched messages and assemble the dict the ObsBuilder wants.

        Returns None if any required stream hasn't received a message yet
        (the loop will skip this tick and try again).
        """
        raise NotImplementedError

    def _next_chunk_id(self) -> int:
        cid = self._chunk_id
        self._chunk_id += 1
        return cid

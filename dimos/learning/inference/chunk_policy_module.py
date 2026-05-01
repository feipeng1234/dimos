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

"""ACT inference Module @ ~30 Hz.

Reads `<policy_path>/dimos_meta.json` at start() to recover obs schema
(StreamField map + sync). Latches In ports; calls predict_chunk on the
freshest snapshot every tick; publishes ActionChunk over LCM.

Heavy ML deps (`lerobot`, `torch`) imported lazily inside `start()` —
this file is import-light.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.learning.dataprep import StreamField, SyncConfig
from dimos.learning.policy.base import ActionChunk, Policy
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState


class ChunkPolicyModuleConfig(ModuleConfig):
    policy_path: str
    inference_rate_hz: float = 30.0
    device: str = "cuda"


class ChunkPolicyModule(Module):
    config: ChunkPolicyModuleConfig

    color_image: In[Image]
    joint_state: In[JointState]
    action_chunk: Out[ActionChunk]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Latched live messages
        self._latest_image: Image | None = None
        self._latest_joint_state: JointState | None = None
        self._latch_lock = threading.Lock()

        # Filled in start():
        self._policy: Policy | None = None
        self._observation: dict[str, StreamField] = {}
        self._sync: SyncConfig | None = None
        self._chunk_id: int = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @rpc
    def start(self) -> None:
        """Lazy-import LeRobotPolicy; load checkpoint; read dimos_meta.json
        for observation/sync; subscribe to ports; spawn the loop thread."""
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        raise NotImplementedError

    @rpc
    def reload_policy(self, policy_path: str, device: str | None = None) -> None:
        """Hot-swap the checkpoint without restarting the blueprint."""
        raise NotImplementedError

    @rpc
    def get_status(self) -> dict[str, Any]:
        """{'running', 'chunk_count', 'policy_path', 'last_chunk_ts'}."""
        raise NotImplementedError

    def _run_loop(self) -> None:
        period = 1.0 / self.config.inference_rate_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            obs = self._build_live_obs()
            if obs is None:
                time.sleep(period)
                continue

            positions = self._policy.predict_chunk(obs)  # (T, action_dim)
            self.action_chunk.publish(
                ActionChunk(
                    ts=time.time(),
                    joint_names=self._policy.joint_names,
                    positions=positions,
                    dt=period,
                    chunk_id=self._next_chunk_id(),
                )
            )
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _build_live_obs(self) -> dict[str, np.ndarray] | None:
        """Snapshot latched messages under a lock, project each obs key
        through `resolve_field` using `self._observation`. Returns
        None if any required stream hasn't received a message yet."""
        raise NotImplementedError

    def _next_chunk_id(self) -> int:
        cid = self._chunk_id
        self._chunk_id += 1
        return cid

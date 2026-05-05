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

"""ACT inference Module — runs the policy in a background thread.

Reads ``<policy_path>/dimos_meta.json`` at start() to recover the obs
schema (StreamField map + sync). Latches In ports; calls
``policy.predict_chunk`` on the freshest snapshot every tick; publishes
each result as an ActionChunk over LCM.

When ``publish_joint_command=True`` (default), also emits the first
action of each chunk as a JointState on ``joint_command``. This lets a
coordinator's servo task consume the policy directly without an
``ActionReplayer`` in the 100 Hz tick loop. Once ActionReplayer is wired
in, callers can ignore that port.

Heavy ML deps (``lerobot``, ``torch``) are imported lazily inside
``start()`` so this file is import-light.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.learning.dataprep import StreamField, SyncConfig, resolve_field
from dimos.learning.policy.base import ActionChunk, Policy
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState

DIMOS_META_FILENAME = "dimos_meta.json"


class ChunkPolicyModuleConfig(ModuleConfig):
    policy_path: str
    inference_rate_hz: float = 30.0
    device: str = "cuda"
    # Emit chunk[0] as a JointState on `joint_command` each tick. Useful when
    # the downstream coordinator has a servo task but no ActionReplayer task.
    publish_joint_command: bool = True


class ChunkPolicyModule(Module):
    config: ChunkPolicyModuleConfig

    color_image: In[Image]
    joint_state: In[JointState]
    action_chunk: Out[ActionChunk]
    joint_command: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_image: Image | None = None
        self._latest_joint_state: JointState | None = None
        self._latch_lock = threading.Lock()

        # Filled in start():
        self._policy: Policy | None = None
        self._observation: dict[str, StreamField] = {}
        self._sync: SyncConfig | None = None
        self._chunk_id: int = 0
        self._last_chunk_ts: float | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @rpc
    def start(self) -> None:
        """Load checkpoint, read dimos_meta, subscribe ports, spawn loop thread."""
        super().start()

        meta_path = Path(self.config.policy_path) / DIMOS_META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {DIMOS_META_FILENAME} in {self.config.policy_path}")
        with open(meta_path) as f:
            meta = json.load(f)

        self._observation = {k: StreamField(**v) for k, v in (meta.get("observation") or {}).items()}
        sync_cfg = meta.get("sync") or {}
        if sync_cfg:
            self._sync = SyncConfig(**sync_cfg)

        from dimos.learning.policy.lerobot_policy import LeRobotPolicy
        self._policy = LeRobotPolicy.load(self.config.policy_path, device=self.config.device)

        self.color_image.subscribe(self._on_color_image)
        self.joint_state.subscribe(self._on_joint_state)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        super().stop()

    @rpc
    def reload_policy(self, policy_path: str, device: str | None = None) -> None:
        """Hot-swap the checkpoint without restarting the blueprint."""
        self.stop()
        self.config.policy_path = policy_path
        if device is not None:
            self.config.device = device
        self.start()

    @rpc
    def get_status(self) -> dict[str, Any]:
        return {
            "running":       self._thread is not None and self._thread.is_alive(),
            "chunk_count":   self._chunk_id,
            "policy_path":   self.config.policy_path,
            "last_chunk_ts": self._last_chunk_ts,
        }

    # ── port handlers ────────────────────────────────────────────────────────

    def _on_color_image(self, msg: Image) -> None:
        with self._latch_lock:
            self._latest_image = msg

    def _on_joint_state(self, msg: JointState) -> None:
        with self._latch_lock:
            self._latest_joint_state = msg

    # ── loop ─────────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        assert self._policy is not None
        period = 1.0 / self.config.inference_rate_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                obs = self._build_live_obs()
                if obs is None:
                    self._stop.wait(timeout=period)
                    continue

                positions = self._policy.predict_chunk(obs)  # (T, action_dim)
                chunk_ts = time.time()
                self.action_chunk.publish(ActionChunk(
                    ts=chunk_ts,
                    joint_names=self._policy.joint_names,
                    positions=positions,
                    dt=period,
                    chunk_id=self._next_chunk_id(),
                ))
                self._last_chunk_ts = chunk_ts

                if self.config.publish_joint_command:
                    js = JointState(
                        name=self._policy.joint_names,
                        position=[float(x) for x in positions[0]],
                        velocity=[],
                    )
                    js.ts = chunk_ts
                    self.joint_command.publish(js)
            except Exception:
                # Single bad tick must not kill the loop.
                traceback.print_exc()

            elapsed = time.monotonic() - t0
            if elapsed < period:
                self._stop.wait(timeout=period - elapsed)

    def _build_live_obs(self) -> dict[str, np.ndarray] | None:
        """Snapshot latched messages and project each obs key through `resolve_field`.
        Returns None if any required stream hasn't received a message yet.
        """
        with self._latch_lock:
            latest_image = self._latest_image
            latest_joints = self._latest_joint_state

        if not self._observation:
            # No spec — fall back to canonical port names.
            if latest_image is None or latest_joints is None:
                return None
            return {
                "image":       np.asarray(latest_image.data),
                "joint_state": np.asarray(latest_joints.position),
            }

        out: dict[str, np.ndarray] = {}
        for obs_key, sf in self._observation.items():
            port = self._guess_port(sf.stream)
            if port == "color_image":
                if latest_image is None:
                    return None
                out[obs_key] = resolve_field(latest_image, sf)
            elif port == "joint_state":
                if latest_joints is None:
                    return None
                out[obs_key] = resolve_field(latest_joints, sf)
            else:
                # Extend here when adding In ports for new sensor types.
                return None
        return out

    @staticmethod
    def _guess_port(stream_name: str) -> str:
        """Route a recorded stream name to one of this module's In ports."""
        n = stream_name.lower()
        if "image" in n or "camera" in n or "rgb" in n:
            return "color_image"
        if "joint_state" in n:
            return "joint_state"
        return n

    def _next_chunk_id(self) -> int:
        cid = self._chunk_id
        self._chunk_id += 1
        return cid

# Copyright 2025-2026 Dimensional Inc.
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

"""G1 trajectory playback -- replay a recorded JSON file as joint commands.

Reads a JSON recording (produced by ``g1_record.py``) and publishes the
joint positions through the ControlCoordinator at the original timing.

Usage (via blueprint):
    TRAJECTORY_FILE=macarena.json ROBOT_INTERFACE=enp60s0 dimos run unitree-g1-playback
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs import JointState
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

INTERP_DURATION = 1.0  # seconds to interpolate to first sample
CMD_HZ = 500


class G1Playback(Module):
    """Replay a recorded joint trajectory.

    Ports:
        joint_state (In[JointState]): current joint positions from coordinator
        joint_command (Out[JointState]): target joint positions
    """

    joint_state: In[JointState]
    joint_command: Out[JointState]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_state: JointState | None = None
        self._state_lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self.joint_state.subscribe(self._on_joint_state)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="g1-playback")
        self._thread.start()
        logger.info("G1Playback started")

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        super().stop()
        logger.info("G1Playback stopped")

    def _on_joint_state(self, msg: JointState) -> None:
        with self._state_lock:
            self._latest_state = msg

    def _get_positions(self) -> list[float] | None:
        with self._state_lock:
            if self._latest_state is None or not self._latest_state.position:
                return None
            return list(self._latest_state.position)

    def _get_joint_names(self) -> list[str]:
        with self._state_lock:
            if self._latest_state is None:
                return []
            return list(self._latest_state.name)

    def _run(self) -> None:
        traj_file = os.getenv("TRAJECTORY_FILE", "")
        if not traj_file:
            logger.error("TRAJECTORY_FILE env var not set, aborting")
            return

        with open(traj_file) as f:
            recording = json.load(f)

        names: list[str] = recording["joint_names"]
        samples: list[dict] = recording["samples"]
        if not samples:
            logger.error("Recording has no samples, aborting")
            return

        logger.info(f"Loaded {len(samples)} samples from {traj_file}")

        # Wait for initial joint state
        logger.info("Waiting for initial joint state...")
        while not self._stop_event.is_set():
            if self._get_positions() is not None:
                break
            time.sleep(0.1)
        if self._stop_event.is_set():
            return

        current = self._get_positions()
        if current is None:
            return

        first_pos = samples[0]["position"]
        dt = 1.0 / CMD_HZ
        interp_ticks = int(INTERP_DURATION * CMD_HZ)

        # Stage 1: interpolate from current position to first sample
        logger.info("Interpolating to first sample position...")
        for tick in range(interp_ticks):
            if self._stop_event.is_set():
                return
            ratio = (tick + 1) / interp_ticks
            pos = [(1.0 - ratio) * c + ratio * t for c, t in zip(current, first_pos, strict=False)]
            self._pub(names, pos)
            time.sleep(dt)

        # Stage 2: loop playback
        logger.info("Starting trajectory playback (loops until stopped)")
        while not self._stop_event.is_set():
            for i, sample in enumerate(samples):
                if self._stop_event.is_set():
                    return

                self._pub(names, sample["position"])

                # Sleep for delta-t to next sample
                if i + 1 < len(samples):
                    gap = samples[i + 1]["ts"] - sample["ts"]
                    if gap > 0:
                        time.sleep(gap)
                    # skip negative/zero gaps
                else:
                    # End of trajectory, small pause before looping
                    time.sleep(dt)

            logger.info("Trajectory loop complete, restarting...")

        logger.info("G1 playback stopped")

    def _pub(self, names: list[str], positions: list[float]) -> None:
        msg = JointState(name=names, position=positions)
        self.joint_command.publish(msg)


g1_playback = G1Playback.blueprint

__all__ = ["G1Playback", "g1_playback"]

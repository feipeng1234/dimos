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

"""G1 low-level example -- joint-level control via ControlCoordinator.

Mirrors the Unitree SDK g1_low_level_example.py behaviour exactly:

  Stage 1 (3s):  Interpolate from current position to zero posture
  Stage 2 (3s):  Swing ankles using PR mode (sinusoidal Pitch/Roll)
  Stage 3 (inf): Swing ankles using AB mode + wrist roll oscillation

The ControlCoordinator receives JointState commands over LCM, runs them
through per-joint arbitration, and writes to the UnitreeG1LowLevelAdapter.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs import JointState
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# G1 joint indices (matching G1JointIndex in the SDK example)
# ---------------------------------------------------------------------------
NUM_G1_JOINTS = 29

# Ankle joints (PR mode aliases)
LEFT_ANKLE_PITCH = 4
LEFT_ANKLE_ROLL = 5
RIGHT_ANKLE_PITCH = 10
RIGHT_ANKLE_ROLL = 11

# Wrist joints
LEFT_WRIST_ROLL = 19
RIGHT_WRIST_ROLL = 26

# Per-joint Kp/Kd from Unitree SDK (used by ConnectedWholeBody via coordinator)
# These are informational -- the coordinator's ConnectedWholeBody applies
# its own default gains. For the SDK-matching gains, use write_motor_commands
# directly (future enhancement).

# Timing
CMD_HZ = 500  # Match coordinator tick rate
DURATION = 3.0  # Each stage duration (seconds)
STAGE_1_TICKS = int(DURATION * CMD_HZ)
STAGE_2_TICKS = int(DURATION * CMD_HZ)

# Oscillation amplitudes (radians)
MAX_PITCH = math.pi * 30.0 / 180.0  # 30 degrees
MAX_ROLL = math.pi * 10.0 / 180.0  # 10 degrees
MAX_WRIST = math.pi * 30.0 / 180.0  # 30 degrees


class G1LowLevelControl(Module):
    """Example module for G1 low-level joint control.

    Replicates the Unitree SDK g1_low_level_example.py 3-stage sequence.
    The ControlCoordinator handles arbitration and hardware IO.

    Ports:
        joint_state (In[JointState]): receives current joint positions
        joint_command (Out[JointState]): publishes target joint positions
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
        self._thread = threading.Thread(target=self._run, daemon=True, name="g1-lowlevel")
        self._thread.start()
        logger.info("G1LowLevelControl started")

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        super().stop()
        logger.info("G1LowLevelControl stopped")

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

    # =========================================================================
    # 3-stage sequence (mirrors SDK example)
    # =========================================================================

    def _run(self) -> None:
        """Background thread: run the 3-stage SDK example sequence."""
        logger.info("Waiting for initial joint state...")
        while not self._stop_event.is_set():
            if self._get_positions() is not None:
                break
            time.sleep(0.1)
        if self._stop_event.is_set():
            return

        start_pos = self._get_positions()
        names = self._get_joint_names()
        if start_pos is None or not names:
            logger.error("No joint state received, aborting")
            return

        logger.info(f"Got initial state ({len(start_pos)} joints). Starting in 5s...")
        for i in range(5, 0, -1):
            if self._stop_event.is_set():
                return
            logger.info(f"  {i}...")
            time.sleep(1.0)

        dt = 1.0 / CMD_HZ
        zero_pos = [0.0] * NUM_G1_JOINTS

        # Stage 1: Interpolate current -> zero posture (3s)
        logger.info("Stage 1: interpolating to zero posture")
        for tick in range(STAGE_1_TICKS):
            if self._stop_event.is_set():
                return
            ratio = min((tick + 1) / STAGE_1_TICKS, 1.0)
            # Read latest state for smooth interpolation (like SDK does)
            current = self._get_positions() or start_pos
            pos = [(1.0 - ratio) * c + ratio * 0.0 for c in current]
            self._pub(names, pos)
            time.sleep(dt)

        # Stage 2: Swing ankles in PR mode (3s)
        logger.info("Stage 2: ankle oscillation (PR mode)")
        for tick in range(STAGE_2_TICKS):
            if self._stop_event.is_set():
                return
            t = (tick + 1) * dt
            pos = list(zero_pos)

            # Left ankle: pitch + roll sinusoid
            pos[LEFT_ANKLE_PITCH] = MAX_PITCH * math.sin(2.0 * math.pi * t)
            pos[LEFT_ANKLE_ROLL] = MAX_ROLL * math.sin(2.0 * math.pi * t)
            # Right ankle: pitch same phase, roll inverted
            pos[RIGHT_ANKLE_PITCH] = MAX_PITCH * math.sin(2.0 * math.pi * t)
            pos[RIGHT_ANKLE_ROLL] = -MAX_ROLL * math.sin(2.0 * math.pi * t)

            self._pub(names, pos)
            time.sleep(dt)

        # Stage 3: Swing ankles + wrist roll (runs indefinitely until stopped)
        logger.info("Stage 3: ankle + wrist oscillation (runs until stopped)")
        tick = 0
        while not self._stop_event.is_set():
            t = (tick + 1) * dt
            pos = list(zero_pos)

            # Ankle oscillation (same amplitudes as stage 2)
            pos[LEFT_ANKLE_PITCH] = MAX_PITCH * math.sin(2.0 * math.pi * t)
            pos[LEFT_ANKLE_ROLL] = MAX_ROLL * math.sin(2.0 * math.pi * t + math.pi)
            pos[RIGHT_ANKLE_PITCH] = -MAX_PITCH * math.sin(2.0 * math.pi * t)
            pos[RIGHT_ANKLE_ROLL] = -MAX_ROLL * math.sin(2.0 * math.pi * t + math.pi)

            # Wrist roll oscillation
            wrist_des = MAX_WRIST * math.sin(2.0 * math.pi * t)
            pos[LEFT_WRIST_ROLL] = wrist_des
            pos[RIGHT_WRIST_ROLL] = wrist_des

            self._pub(names, pos)
            tick += 1
            time.sleep(dt)

        logger.info("G1 low-level sequence stopped")

    def _pub(self, names: list[str], positions: list[float]) -> None:
        msg = JointState(name=names, position=positions)
        self.joint_command.publish(msg)


# Blueprint export
g1_low_level_control = G1LowLevelControl.blueprint

__all__ = ["G1LowLevelControl", "g1_low_level_control"]

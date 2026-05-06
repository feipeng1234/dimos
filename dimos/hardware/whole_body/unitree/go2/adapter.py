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

"""Unitree Go2 low-level adapter — direct 12-DOF joint control over DDS.

Uses ``rt/lowcmd`` / ``rt/lowstate`` DDS topics for 500 Hz motor-level
position/velocity/torque control, bypassing the high-level SportClient.

Important: The Go2 must first exit sport mode (via MotionSwitcherClient)
before low-level commands are accepted.

Motor ordering (12 leg joints):
  0-2: FR (hip, thigh, calf)
  3-5: FL
  6-8: RR
  9-11: RL
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from dimos.hardware.whole_body.spec import (
    POS_STOP,
    VEL_STOP,
    IMUState,
    MotorCommand,
    MotorState,
)
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.hardware.whole_body.registry import WholeBodyAdapterRegistry

logger = setup_logger()

_NUM_MOTORS = 12


class UnitreeGo2LowLevelAdapter:
    """WholeBodyAdapter implementation for Unitree Go2 — low-level DDS.

    The coordinator's tick loop drives the publish cadence.  Each call to
    ``write_motor_commands()`` updates the ``LowCmd_`` buffer, computes
    CRC, and publishes immediately — no background thread needed.

    Args:
        network_interface: DDS network interface ID (default: 0).
    """

    def __init__(self, network_interface: int = 0, **_: object) -> None:
        self._network_interface = network_interface

        self._connected = False
        self._lock = threading.Lock()

        # SDK objects (lazy-imported on connect)
        self._low_cmd = None
        self._publisher = None
        self._subscriber = None
        self._crc = None

        # Latest feedback
        self._low_state = None

    # =========================================================================
    # Connection
    # =========================================================================

    def connect(self) -> bool:
        """Connect to Go2 and release sport mode for low-level control."""
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC

            # 1. Initialise DDS transport
            logger.info(f"Initializing DDS (low-level) with interface {self._network_interface}...")
            ChannelFactoryInitialize(self._network_interface)

            # 2. Create publisher / subscriber
            self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._publisher.Init()

            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscriber.Init(self._on_low_state, 10)

            # 3. Initialise LowCmd with safe defaults
            self._low_cmd = unitree_go_msg_dds__LowCmd_()
            self._low_cmd.head[0] = 0xFE
            self._low_cmd.head[1] = 0xEF
            self._low_cmd.level_flag = 0xFF
            self._low_cmd.gpio = 0
            for i in range(20):
                self._low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
                self._low_cmd.motor_cmd[i].q = POS_STOP
                self._low_cmd.motor_cmd[i].kp = 0
                self._low_cmd.motor_cmd[i].dq = VEL_STOP
                self._low_cmd.motor_cmd[i].kd = 0
                self._low_cmd.motor_cmd[i].tau = 0

            self._crc = CRC()

            # 4. Release sport mode so low-level commands are accepted
            logger.info("Releasing sport mode...")
            self._release_sport_mode()

            self._connected = True
            logger.info("Go2 low-level adapter connected")
            return True

        except Exception as e:
            logger.error(f"Failed to connect Go2 low-level adapter: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        self._connected = False
        self._publisher = None
        self._subscriber = None
        self._low_cmd = None
        self._low_state = None
        logger.info("Go2 low-level adapter disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # =========================================================================
    # State Reading
    # =========================================================================

    def read_motor_states(self) -> list[MotorState]:
        """Read motor states for all 12 leg joints."""
        with self._lock:
            if self._low_state is None:
                return [MotorState()] * _NUM_MOTORS
            return [
                MotorState(
                    q=self._low_state.motor_state[i].q,
                    dq=self._low_state.motor_state[i].dq,
                    tau=self._low_state.motor_state[i].tau_est,
                )
                for i in range(_NUM_MOTORS)
            ]

    def read_imu(self) -> IMUState:
        """Read IMU state."""
        with self._lock:
            if self._low_state is None:
                return IMUState()
            imu = self._low_state.imu_state
            return IMUState(
                quaternion=tuple(imu.quaternion),
                gyroscope=tuple(imu.gyroscope),
                accelerometer=tuple(imu.accelerometer),
                rpy=tuple(imu.rpy),
            )

    def read_foot_forces(self) -> list[float]:
        """Read foot force sensors (4 feet)."""
        with self._lock:
            if self._low_state is None:
                return [0.0] * 4
            return [float(self._low_state.foot_force[i]) for i in range(4)]

    # =========================================================================
    # Control
    # =========================================================================

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        """Update command buffer, compute CRC, and publish immediately.

        Called by the coordinator tick loop on every tick — no background
        thread needed.
        """
        if len(commands) != _NUM_MOTORS:
            logger.error(f"Expected {_NUM_MOTORS} commands, got {len(commands)}")
            return False

        with self._lock:
            if self._low_cmd is None or self._crc is None or self._publisher is None:
                return False
            for i, cmd in enumerate(commands):
                self._low_cmd.motor_cmd[i].q = cmd.q
                self._low_cmd.motor_cmd[i].dq = cmd.dq
                self._low_cmd.motor_cmd[i].kp = cmd.kp
                self._low_cmd.motor_cmd[i].kd = cmd.kd
                self._low_cmd.motor_cmd[i].tau = cmd.tau
            self._low_cmd.crc = self._crc.Crc(self._low_cmd)
            self._publisher.Write(self._low_cmd)
        return True

    # =========================================================================
    # Internal
    # =========================================================================

    def _on_low_state(self, msg: object) -> None:
        """DDS callback for rt/lowstate."""
        with self._lock:
            self._low_state = msg

    def _release_sport_mode(self) -> None:
        """Exit sport mode so that low-level commands are accepted.

        Loops StandDown + ReleaseMode until CheckMode returns empty.
        """
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        sc = SportClient()
        sc.SetTimeout(5.0)
        sc.Init()

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()

        _status, result = msc.CheckMode()
        while result["name"]:
            sc.StandDown()
            msc.ReleaseMode()
            _status, result = msc.CheckMode()
            time.sleep(1)

        logger.info("Sport mode released — low-level control active")


def register(registry: WholeBodyAdapterRegistry) -> None:
    """Register this adapter with the whole-body registry."""
    registry.register("unitree_go2", UnitreeGo2LowLevelAdapter)


__all__ = ["UnitreeGo2LowLevelAdapter"]

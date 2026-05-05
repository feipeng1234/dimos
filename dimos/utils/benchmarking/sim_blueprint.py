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

"""Sim coordinator blueprint for the Go2 controller benchmark.

Mirrors the production hardware blueprint in shape (same TickLoop, same
TwistBase adapter protocol, same transports) - only the bottom edge is
swapped: instead of the real Go2 WebRTC adapter, we use
:class:`~dimos.hardware.drive_trains.go2_sim.adapter.Go2SimTwistBaseAdapter`
which runs an FOPDT plant model in-process.

Tick rate is fixed at 10 Hz to match the Go2 hardware control rate
(see :mod:`memory/project_go2_control_rate`).
"""

from __future__ import annotations

from dimos.control.components import (
    HardwareComponent,
    HardwareType,
    make_twist_base_joints,
)
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.JointState import JointState

GO2_TICK_RATE_HZ = 10.0

_base_joints = make_twist_base_joints("base")


def _go2_sim_base(hw_id: str = "base") -> HardwareComponent:
    return HardwareComponent(
        hardware_id=hw_id,
        hardware_type=HardwareType.BASE,
        joints=make_twist_base_joints(hw_id),
        adapter_type="go2_sim_twist_base",
    )


coordinator_go2_sim_base = ControlCoordinator.blueprint(
    tick_rate=GO2_TICK_RATE_HZ,
    hardware=[_go2_sim_base()],
    tasks=[
        TaskConfig(
            name="vel_base",
            type="velocity",
            joint_names=_base_joints,
            priority=10,
        ),
    ],
).transports(
    {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
        ("twist_command", Twist): LCMTransport("/cmd_vel", Twist),
    }
)


__all__ = ["GO2_TICK_RATE_HZ", "coordinator_go2_sim_base"]

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

"""Unitree G1 GR00T whole-body-control blueprint — real hardware.

Composes three Modules:

    G1WholeBodyConnection  ─ owns DDS rt/lowstate / rt/lowcmd in its own
                             worker; publishes motor_states + imu over
                             LCM, subscribes motor_command from LCM.
    ControlCoordinator     ─ runs at 500 Hz with two tasks:
        * groot_wbc  (priority 50)  legs+waist (15 DOF), GR00T balance/
                                    walk ONNX policies at 50 Hz.
        * servo_arms (priority 10)  14 arm joints, hold relaxed pose.
    WebsocketVisModule     ─ operator dashboard at :7779 (Arm + Dry-Run
                             toggles, WASD teleop).

The coordinator's ``transport_lcm`` whole-body adapter bridges to the
connection module via LCM, mirroring Mustafa's ``unitree-g1-coordinator``
blueprint — single architectural pattern for G1 low-level work.

Real-hardware safety profile: comes up unarmed and in dry-run.  Operator
verifies computed commands in the dashboard, then clicks Activate to
ramp from current pose to the bent-knee default over 10 s before handing
torque control to the policy.

Sim is a separate blueprint (``unitree-g1-groot-wbc-sim``).

Usage:
    ROBOT_INTERFACE=enp86s0 dimos run unitree-g1-groot-wbc

Environment:
    ROBOT_INTERFACE   DDS network interface (default ``"enp86s0"``).
    CYCLONEDDS_HOME   Required at runtime — must point at the cyclonedds
                      C install (e.g. ``~/cyclonedds/install``).  Add the
                      export to your shell rc.
    GROOT_MODEL_DIR   Directory containing ``balance.onnx`` +
                      ``walk.onnx`` (default: pulled via ``get_data("groot")``).
"""

from __future__ import annotations

import os

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.whole_body.spec import WholeBodyConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.std_msgs.Bool import Bool as DimosBool
from dimos.robot.unitree.g1.blueprints.basic._groot_wbc_common import (
    ARM_DEFAULT_POSE,
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_arms,
    g1_joints,
    g1_legs_waist,
)
from dimos.robot.unitree.g1.wholebody_connection import G1WholeBodyConnection
from dimos.utils.data import get_data
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

_g1_connection = G1WholeBodyConnection.blueprint(
    release_sport_mode=True,
    network_interface=os.getenv("ROBOT_INTERFACE", "enp86s0"),
)

_g1_coordinator = ControlCoordinator.blueprint(
    tick_rate=500.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[
        HardwareComponent(
            hardware_id="g1",
            hardware_type=HardwareType.WHOLE_BODY,
            joints=g1_joints,
            adapter_type="transport_lcm",
            wb_config=WholeBodyConfig(kp=tuple(G1_GROOT_KP), kd=tuple(G1_GROOT_KD)),
        ),
    ],
    tasks=[
        TaskConfig(
            name="groot_wbc",
            type="groot_wbc",
            joint_names=g1_legs_waist,
            priority=50,
            model_path=os.getenv("GROOT_MODEL_DIR", str(get_data("groot"))),
            hardware_id="g1",
            auto_start=True,
            # Real-hw safety: come up unarmed + dry-run.  Operator
            # arms via the dashboard Activate button after sanity
            # checks; activation ramps over 10 s.
            auto_arm=False,
            auto_dry_run=True,
            default_ramp_seconds=10.0,
        ),
        TaskConfig(
            name="servo_arms",
            type="servo",
            joint_names=g1_arms,
            priority=10,
            default_positions=ARM_DEFAULT_POSE,
            auto_start=True,
        ),
    ],
)

# Operator dashboard at http://localhost:7779/ — Arm + Dry-Run toggles,
# WASD teleop captured in the browser DOM (sidesteps the macOS Cocoa
# main-thread restriction that breaks pygame-based teleop).
_g1_ws_vis = WebsocketVisModule.blueprint()

unitree_g1_groot_wbc = autoconnect(_g1_connection, _g1_coordinator, _g1_ws_vis).transports(
    {
        # Bridge: G1WholeBodyConnection ↔ ControlCoordinator (transport_lcm).
        # Topic prefix is the HardwareComponent's hardware_id ("g1").
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("imu", Imu): LCMTransport("/g1/imu", Imu),
        ("motor_command", MotorCommandArray): LCMTransport("/g1/motor_command", MotorCommandArray),
        # Coordinator outputs.
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
        ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
        ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
        # Dashboard ↔ coordinator (cmd_vel + activate + dry_run).
        ("twist_command", Twist): LCMTransport("/g1/cmd_vel", Twist),
        ("activate", DimosBool): LCMTransport("/g1/activate", DimosBool),
        ("dry_run", DimosBool): LCMTransport("/g1/dry_run", DimosBool),
        ("cmd_vel", Twist): LCMTransport("/g1/cmd_vel", Twist),
    }
)

__all__ = ["unitree_g1_groot_wbc"]

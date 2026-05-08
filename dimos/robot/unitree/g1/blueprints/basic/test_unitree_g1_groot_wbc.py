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

"""Composition smoke tests for ``unitree_g1_groot_wbc`` (real-hw blueprint).

These tests do not exercise DDS or actually run anything — they just
verify that the blueprint composes correctly:

  * imports cleanly without the ``[unitree-dds]`` extra installed
  * deploys the three expected modules (connection + coordinator + dashboard)
  * picks the bridge adapter (``transport_lcm``) on the G1 hardware
    component, so the GR00T-WBC PR really did migrate off the deleted
    ``unitree_g1`` monolith
  * wires the LCM topics the coordinator's safety state machine needs
    (``/g1/activate``, ``/g1/dry_run``, ``/g1/cmd_vel``) and the
    bridge's own ports (``/g1/motor_states``, ``/g1/imu``, ``/g1/motor_command``)
  * keeps the real-hw safety profile on the GR00T task (unarmed +
    dry-run + 10-s ramp).
"""

from __future__ import annotations

from typing import Any

from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_groot_wbc import (
    unitree_g1_groot_wbc,
)


def _module_names() -> set[str]:
    return {atom.module.__name__ for atom in unitree_g1_groot_wbc.blueprints}


def _coordinator_kwargs() -> dict[str, Any]:
    for atom in unitree_g1_groot_wbc.blueprints:
        if atom.module.__name__ == "ControlCoordinator":
            return atom.kwargs
    raise AssertionError("ControlCoordinator not in blueprint composition")


def test_blueprint_composes_three_expected_modules() -> None:
    assert _module_names() == {
        "G1WholeBodyConnection",
        "ControlCoordinator",
        "WebsocketVisModule",
    }


def test_g1_hardware_uses_bridge_adapter() -> None:
    """We deleted the monolith ``unitree_g1`` adapter — verify the
    coordinator now binds via Mustafa's bridge (``transport_lcm``)."""
    hw = _coordinator_kwargs()["hardware"]
    assert len(hw) == 1, "expected exactly one hardware component"
    g1 = hw[0]
    assert g1.hardware_id == "g1"
    assert g1.adapter_type == "transport_lcm"
    assert len(g1.joints) == 29
    assert g1.wb_config is not None
    assert g1.wb_config.kp is not None and len(g1.wb_config.kp) == 29
    assert g1.wb_config.kd is not None and len(g1.wb_config.kd) == 29


def test_groot_task_is_safety_gated() -> None:
    """Real-hw blueprint must come up unarmed + dry-run + with a 10-s ramp.

    Sim auto-arms with no ramp via the sister ``unitree_g1_groot_wbc_sim``
    blueprint; that path is not under test here.
    """
    tasks = _coordinator_kwargs()["tasks"]
    groot = next(t for t in tasks if t.name == "groot_wbc")
    assert groot.type == "groot_wbc"
    assert groot.hardware_id == "g1"
    assert groot.priority == 50
    assert groot.auto_arm is False
    assert groot.auto_dry_run is True
    assert groot.default_ramp_seconds == 10.0


def test_servo_arms_holds_default_pose() -> None:
    tasks = _coordinator_kwargs()["tasks"]
    servo = next(t for t in tasks if t.name == "servo_arms")
    assert servo.type == "servo"
    assert servo.priority == 10
    assert servo.default_positions is not None
    assert len(servo.default_positions) == 14  # 7 left + 7 right arm joints


def test_bridge_topics_wired() -> None:
    """Bridge needs motor_states/imu/motor_command on /g1/* and the
    coordinator's safety inputs on /g1/activate, /g1/dry_run, /g1/cmd_vel."""
    topic_strings = {
        str(t.topic).split("#")[0] for t in unitree_g1_groot_wbc.transport_map.values()
    }
    expected = {
        "/g1/motor_states",
        "/g1/imu",
        "/g1/motor_command",
        "/g1/activate",
        "/g1/dry_run",
        "/g1/cmd_vel",
        "/g1/joint_command",
        "/coordinator/joint_state",
        "/odom",
    }
    missing = expected - topic_strings
    assert not missing, f"missing transports: {missing}"

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

"""Pre-configured blueprints for the ControlCoordinator.

This package provides ready-to-use coordinator blueprints for common setups.

Usage:
    # Run via CLI:
    dimos run coordinator-mock           # Mock 7-DOF arm
    dimos run coordinator-xarm7          # XArm7 real hardware
    dimos run coordinator-dual-mock      # Dual mock arms

    # Or programmatically:
    from dimos.control.blueprints import coordinator_mock
    coordinator = coordinator_mock.build()
    coordinator.loop()
"""

from __future__ import annotations

from dimos.control.blueprints.basic import (
    coordinator_basic,
    coordinator_mock,
    coordinator_piper,
    coordinator_xarm6,
    coordinator_xarm7,
)
from dimos.control.blueprints.dual import (
    coordinator_dual_mock,
    coordinator_dual_xarm,
    coordinator_piper_xarm,
)
from dimos.control.blueprints.mobile import (
    coordinator_mobile_manip_mock,
    coordinator_mock_twist_base,
)
from dimos.control.blueprints.teleop import (
    coordinator_cartesian_ik_mock,
    coordinator_cartesian_ik_piper,
    coordinator_combined_xarm6,
    coordinator_teleop_dual,
    coordinator_teleop_piper,
    coordinator_teleop_xarm6,
    coordinator_teleop_xarm7,
    coordinator_velocity_xarm6,
)

__all__ = [
    # Basic (single arm, trajectory)
    "coordinator_basic",
    "coordinator_cartesian_ik_mock",
    "coordinator_cartesian_ik_piper",
    "coordinator_combined_xarm6",
    # Dual arm (trajectory)
    "coordinator_dual_mock",
    "coordinator_dual_xarm",
    "coordinator_mobile_manip_mock",
    "coordinator_mock",
    # Mobile manipulation
    "coordinator_mock_twist_base",
    "coordinator_piper",
    "coordinator_piper_xarm",
    "coordinator_teleop_dual",
    "coordinator_teleop_piper",
    # Teleop (servo, velocity, cartesian IK, teleop IK)
    "coordinator_teleop_xarm6",
    "coordinator_teleop_xarm7",
    "coordinator_velocity_xarm6",
    "coordinator_xarm6",
    "coordinator_xarm7",
]

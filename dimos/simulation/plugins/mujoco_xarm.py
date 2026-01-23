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

"""MuJoCo engine registration with xArm robot specs."""

from __future__ import annotations

from dimos.simulation.engines.base import RobotSpec
from dimos.simulation.engines.mujoco_engine import MujocoEngine
from dimos.simulation.registry import registry


def register() -> None:
    registry.register_engine("mujoco", MujocoEngine)

    registry.register_robot(
        "xarm7",
        RobotSpec(
            name="xarm7",
            engine="mujoco",
            asset="xarm7_mj_description",
            dof=7,
            vendor="UFACTORY",
            model="xArm7",
        ),
    )
    registry.register_robot(
        "xarm6",
        RobotSpec(
            name="xarm6",
            engine="mujoco",
            asset="xarm6_mj_description",
            dof=6,
            vendor="UFACTORY",
            model="xArm6",
        ),
    )
    registry.register_robot(
        "xarm5",
        RobotSpec(
            name="xarm5",
            engine="mujoco",
            asset="xarm5_mj_description",
            dof=5,
            vendor="UFACTORY",
            model="xArm5",
        ),
    )


__all__ = [
    "register",
]

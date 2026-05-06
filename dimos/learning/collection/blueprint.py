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

"""Recording blueprints. RecordReplay is enabled via `--record-path`."""

from __future__ import annotations

from dimos.control.coordinator import ControlCoordinator
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport, pLCMTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.learning.collection.episode_monitor import (
    EpisodeMonitorModule,
    EpisodeStatus,
)
from dimos.msgs.sensor_msgs.Image import Image
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
from dimos.teleop.quest.blueprints import teleop_quest_xarm7, teleop_quest_xarm7_sim
from dimos.teleop.quest.quest_extensions import ArmTeleopModule
from dimos.teleop.quest.quest_types import Buttons

_DEFAULT_BUTTON_MAP = {"start": "A", "save": "B", "discard": "X"}
# EpisodeStatus is a Pydantic BaseModel (no lcm_encode), so it travels
# over a pickle transport.
_STATUS_TRANSPORT = pLCMTransport("/learning/episode_status")
_BUTTONS_TRANSPORT = LCMTransport("/teleop/buttons", Buttons)


learning_collect_quest_xarm7 = (
    autoconnect(
        teleop_quest_xarm7,
        RealSenseCamera.blueprint(enable_pointcloud=False),
        EpisodeMonitorModule.blueprint(button_map=_DEFAULT_BUTTON_MAP),
    )
    .transports(
        {
            ("buttons", Buttons): _BUTTONS_TRANSPORT,
            ("color_image", Image): LCMTransport("/camera/color_image", Image),
            ("status", EpisodeStatus): _STATUS_TRANSPORT,
        }
    )
    .default_record_modules(
        ArmTeleopModule,
        ControlCoordinator,
        RealSenseCamera,
        EpisodeMonitorModule,
    )
)


# Sim records the MuJoCo color_image stream in place of RealSenseCamera.
# teleop_quest_xarm7_sim already declares ArmTeleopModule, ControlCoordinator,
# and MujocoSimModule for recording, so we only extend with the EpisodeMonitor.
learning_collect_quest_xarm7_sim = (
    autoconnect(
        teleop_quest_xarm7_sim,
        EpisodeMonitorModule.blueprint(button_map=_DEFAULT_BUTTON_MAP),
    )
    .transports(
        {
            ("buttons", Buttons): _BUTTONS_TRANSPORT,
            ("status", EpisodeStatus): _STATUS_TRANSPORT,
        }
    )
    .default_record_modules(EpisodeMonitorModule)
)


__all__ = [
    "learning_collect_quest_xarm7",
    "learning_collect_quest_xarm7_sim",
]

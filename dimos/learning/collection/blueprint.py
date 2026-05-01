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

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.learning.collection.episode_monitor import (
    EpisodeMonitorModule,
    EpisodeStatus,
)
from dimos.msgs.sensor_msgs.Image import Image
from dimos.teleop.quest.blueprints import (
    teleop_quest_dual,
    teleop_quest_piper,
    teleop_quest_xarm6,
    teleop_quest_xarm7,
)
from dimos.teleop.quest.quest_types import Buttons

_DEFAULT_BUTTON_MAP = {"start": "A", "save": "B", "discard": "X"}
_TRANSPORTS = {
    ("buttons", Buttons): LCMTransport("/teleop/buttons", Buttons),
    ("color_image", Image): LCMTransport("/camera/color_image", Image),
    ("status", EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
}


learning_collect_quest_xarm7 = autoconnect(
    teleop_quest_xarm7,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(button_map=_DEFAULT_BUTTON_MAP),
).transports(_TRANSPORTS)


learning_collect_quest_piper = autoconnect(
    teleop_quest_piper,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(button_map=_DEFAULT_BUTTON_MAP),
).transports(_TRANSPORTS)


learning_collect_quest_xarm6 = autoconnect(
    teleop_quest_xarm6,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(button_map=_DEFAULT_BUTTON_MAP),
).transports(_TRANSPORTS)


learning_collect_quest_dual = autoconnect(
    teleop_quest_dual,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(button_map=_DEFAULT_BUTTON_MAP),
).transports(_TRANSPORTS)


__all__ = [
    "learning_collect_quest_dual",
    "learning_collect_quest_piper",
    "learning_collect_quest_xarm6",
    "learning_collect_quest_xarm7",
]

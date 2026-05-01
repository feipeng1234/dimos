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

"""Collection blueprints for the DimOS Learning Framework.

Each blueprint composes a teleop session + a camera + the
EpisodeMonitorModule (for live operator feedback). RecordReplay is NOT a
Module — it intercepts at the transport layer and is enabled via the CLI
flag `--record-path session.db`.

Usage:
    dimos run learning-collect-quest-xarm7 --record-path data/pick_red.db
"""

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

# ── XArm7 + Quest ────────────────────────────────────────────────────────────

learning_collect_quest_xarm7 = autoconnect(
    teleop_quest_xarm7,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(),
).transports(
    {
        ("buttons", Buttons): LCMTransport("/teleop/buttons", Buttons),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("status", EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
    }
)


# ── Piper + Quest ────────────────────────────────────────────────────────────

learning_collect_quest_piper = autoconnect(
    teleop_quest_piper,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(),
).transports(
    {
        ("buttons", Buttons): LCMTransport("/teleop/buttons", Buttons),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("status", EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
    }
)


# ── XArm6 + Quest ────────────────────────────────────────────────────────────

learning_collect_quest_xarm6 = autoconnect(
    teleop_quest_xarm6,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(),
).transports(
    {
        ("buttons", Buttons): LCMTransport("/teleop/buttons", Buttons),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("status", EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
    }
)


# ── Dual arm (XArm6 + Piper) + Quest ─────────────────────────────────────────

learning_collect_quest_dual = autoconnect(
    teleop_quest_dual,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(),
).transports(
    {
        ("buttons", Buttons): LCMTransport("/teleop/buttons", Buttons),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("status", EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
    }
)


__all__ = [
    "learning_collect_quest_dual",
    "learning_collect_quest_piper",
    "learning_collect_quest_xarm6",
    "learning_collect_quest_xarm7",
]

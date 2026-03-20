#!/usr/bin/env python3
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

"""Compatibility Unitree G1 blueprint exports.

This module restores the public blueprint names referenced by
``dimos.robot.all_blueprints``. The modular blueprint package referenced in
review comments is not present in this workspace, so these exports map the
legacy names to the currently available modules.
"""

from dimos.core.blueprints import autoconnect
from dimos.robot.unitree.connection.g1 import g1_connection
from dimos.robot.unitree.connection.g1sim import g1_sim_connection
from dimos.robot.unitree_webrtc.keyboard_pose_teleop import keyboard_pose_teleop
from dimos.robot.unitree_webrtc.keyboard_teleop import keyboard_teleop
from dimos.robot.unitree_webrtc.unitree_g1_skill_container import g1_skills


basic_ros = autoconnect(g1_connection())
standard = basic_ros
detection = standard
full_featured = standard
standard_with_shm = standard

basic_sim = autoconnect(g1_sim_connection())
standard_sim = basic_sim

agentic = autoconnect(standard, g1_skills())
agentic_sim = autoconnect(standard_sim, g1_skills())

with_joystick = autoconnect(standard, keyboard_teleop(), keyboard_pose_teleop())


__all__ = [
    "agentic",
    "agentic_sim",
    "basic_ros",
    "basic_sim",
    "detection",
    "full_featured",
    "standard",
    "standard_sim",
    "standard_with_shm",
    "with_joystick",
]

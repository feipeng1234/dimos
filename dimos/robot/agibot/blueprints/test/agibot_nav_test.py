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

"""
AGIbot Navigation Test Blueprint

Composes ROSNav (the standard ROS↔DimOS bridge) to validate the AGIbot's
navigation stack. ROSNav handles all ROS topic subscriptions, TF broadcasting,
and republishing to LCM/SHM so data is visible in lcmspy and rerun.

Usage:
    dimos run agibot-nav-test
"""

from dimos.core.blueprints import autoconnect
from dimos.navigation.rosnav import ros_nav

agibot_nav_test = autoconnect(
    ros_nav(),
)

__all__ = ["agibot_nav_test"]

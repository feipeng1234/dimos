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

"""DimOS Python API — top-level entry point.

Usage::

    from dimos.api import detect, topic_collect, topic_publish

    # Listen to a topic for 5 seconds
    images = topic_collect("/color_image", duration=5.0)

    # Detect an object — returns Detection2DBBox with chainable methods
    det = detect("person", images[-1])

    # Chain: detect → servo → publish
    twist = det.servo(cam_info)
    topic_publish("/cmd_vel", twist)

    # Chain: detect → project → to_pose → navigate
    det3d = det.project(cloud, cam_info, tf)
    goal = det3d.to_pose("map")

    # Chain: detect → track → follow loop
    tracker = det.track()
    best = tracker.best(new_image)
"""

from dimos.perception.detection.detect import detect
from dimos.robot.cli.topic import topic_collect, topic_publish

__all__ = [
    "detect",
    "topic_collect",
    "topic_publish",
]

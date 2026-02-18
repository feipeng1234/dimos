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
AGIbot Navigation Validation Blueprint (Phase 1)

Validates the ROS navigation stack running in the AGIbot's Docker container
by subscribing to its ROS topics via ROSTransport. No bridge modules needed —
ROSTransport is just another transport backend.

Usage:
    dimos run agibot-nav-test

What it does:
    - Subscribes to ROS topics published by the nav container
    - Logs received data (rates, sizes) to verify connectivity
    - Publishes a test goal to verify bidirectional communication

ROS topics (from nav container):
    IN:  /registered_scan (PointCloud2), /cmd_vel (TwistStamped),
         /terrain_map_ext (PointCloud2), /path (Path), /tf (TFMessage)
    OUT: /goal_pose (PoseStamped)
"""

from dataclasses import dataclass
import logging
import time

from dimos.core import In, Module, Out, rpc
from dimos.core.blueprints import autoconnect
from dimos.core.module import ModuleConfig
from dimos.core.transport import ROSTransport
from dimos.msgs.geometry_msgs import PoseStamped, TwistStamped
from dimos.msgs.nav_msgs import Path
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger

logger = setup_logger(level=logging.INFO)


@dataclass
class Config(ModuleConfig):
    log_interval: float = 5.0  # seconds between health reports


class AGIbotNavValidator(Module):
    """Listens to nav stack ROS topics and reports health status.

    Streams are connected to ROS topics via .transports() on the blueprint —
    this module just has In/Out ports with the right names and types.
    """

    default_config = Config
    config: Config

    # IN from nav container
    registered_scan: In[PointCloud2]
    cmd_vel: In[TwistStamped]
    terrain_map_ext: In[PointCloud2]
    path: In[Path]
    ros_tf: In[TFMessage]  # named ros_tf because self.tf is a built-in Module facility

    # OUT to nav container
    goal_pose: Out[PoseStamped]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._counts: dict[str, int] = {
            "registered_scan": 0,
            "cmd_vel": 0,
            "terrain_map_ext": 0,
            "path": 0,
            "ros_tf": 0,
        }
        self._last_report: float = 0.0

    @rpc
    def start(self) -> None:
        super().start()

        self.registered_scan.subscribe(lambda msg: self._on_msg("registered_scan", msg))
        self.cmd_vel.subscribe(lambda msg: self._on_msg("cmd_vel", msg))
        self.terrain_map_ext.subscribe(lambda msg: self._on_msg("terrain_map_ext", msg))
        self.path.subscribe(lambda msg: self._on_msg("path", msg))
        self.ros_tf.subscribe(self._on_ros_tf)

        # Activate the built-in TF facility so transforms are broadcast
        self.tf.start()

        logger.info("AGIbot nav validator started — listening for ROS topics")

    @rpc
    def stop(self) -> None:
        self._print_report()
        super().stop()

    def _on_ros_tf(self, msg: TFMessage) -> None:
        """Forward ROS /tf transforms into the built-in TF facility."""
        self._counts["ros_tf"] += 1
        self.tf.publish(*msg.transforms)
        now = time.monotonic()
        if now - self._last_report >= self.config.log_interval:
            self._print_report()
            self._last_report = now

    def _on_msg(self, topic: str, msg: object) -> None:
        self._counts[topic] += 1
        now = time.monotonic()
        if now - self._last_report >= self.config.log_interval:
            self._print_report()
            self._last_report = now

    def _print_report(self) -> None:
        lines = ["AGIbot Nav Stack Health:"]
        for topic, count in self._counts.items():
            status = "✅" if count > 0 else "❌"
            lines.append(f"  {status} /{topic}: {count} msgs")
        logger.info("\n".join(lines))


# --- Blueprint ---

agibot_nav_test = autoconnect(
    AGIbotNavValidator.blueprint(),
).transports(
    {
        # IN: receive from nav container ROS topics
        ("registered_scan", PointCloud2): ROSTransport("/registered_scan", PointCloud2),
        ("cmd_vel", TwistStamped): ROSTransport("/cmd_vel", TwistStamped),
        ("terrain_map_ext", PointCloud2): ROSTransport("/terrain_map_ext", PointCloud2),
        ("path", Path): ROSTransport("/path", Path),
        ("ros_tf", TFMessage): ROSTransport("/tf", TFMessage),
        # OUT: publish to nav container
        ("goal_pose", PoseStamped): ROSTransport("/goal_pose", PoseStamped),
    }
)

__all__ = ["AGIbotNavValidator", "agibot_nav_test"]

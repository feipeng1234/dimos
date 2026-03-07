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

"""Go2 Fleet Connection - manage multiple Go2 robots as a fleet"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.transport import LCMTransport, pSHMTransport
from dimos.msgs.geometry_msgs import Twist
from dimos.msgs.sensor_msgs import CameraInfo
from dimos.robot.unitree.go2.connection import (
    GO2Connection,
    Go2ConnectionProtocol,
    make_connection,
)

if TYPE_CHECKING:
    from dimos.core.module_coordinator import ModuleCoordinator
    from dimos.core.rpc_client import ModuleProxy

logger = logging.getLogger(__name__)


class Go2FleetConnection(GO2Connection):
    """Inherits all single-robot behaviour from GO2Connection for the primary
    (first) robot. Additional robots only receive broadcast commands
    (move, standup, liedown, publish_request).
    """

    def __init__(
        self,
        ips: list[str] | None = None,
        cfg: GlobalConfig = global_config,
        *args: object,
        **kwargs: object,
    ) -> None:
        if not ips:
            raw = cfg.robot_ips
            if not raw:
                raise ValueError(
                    "No IPs provided. Pass ips= or set ROBOT_IPS (e.g. ROBOT_IPS=10.0.0.102,10.0.0.209)"
                )
            ips = [ip.strip() for ip in raw.split(",") if ip.strip()]
        self._extra_ips = ips[1:]
        self._extra_connections: list[Go2ConnectionProtocol] = []
        super().__init__(*args, ip=ips[0], cfg=cfg, **kwargs)

    @rpc
    def start(self) -> None:
        for ip in self._extra_ips:
            logger.info(f"Connecting to fleet Go2 at {ip}...")
            conn = make_connection(ip, self._global_config)
            conn.start()
            self._extra_connections.append(conn)

        # Parent starts primary robot, subscribes sensors, calls standup() on all
        super().start()
        for conn in self._extra_connections:
            conn.balance_stand()
            if self._global_config.disable_obstacle_avoidance:
                conn.disable_obstacle_avoidance()

    @rpc
    def stop(self) -> None:
        # one robot's error should not prevent others from stopping
        for conn in self._extra_connections:
            try:
                conn.liedown()
                conn.stop()
            except Exception as e:
                logger.error(f"Error stopping fleet Go2: {e}")
        self._extra_connections.clear()
        super().stop()

    @property
    def _all_connections(self) -> list[Go2ConnectionProtocol]:
        return [self.connection, *self._extra_connections]

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        results = [conn.move(twist, duration) for conn in self._all_connections]
        return all(results)

    @rpc
    def standup(self) -> bool:
        results = [conn.standup() for conn in self._all_connections]
        return all(results)

    @rpc
    def liedown(self) -> bool:
        results = [conn.liedown() for conn in self._all_connections]
        return all(results)

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        """Publish a request to all robots, return primary's response."""
        for conn in self._extra_connections:
            conn.publish_request(topic, data)
        return self.connection.publish_request(topic, data)


go2_fleet_connection = Go2FleetConnection.blueprint


def deploy(dimos: ModuleCoordinator, ips: list[str], prefix: str = "") -> ModuleProxy:
    from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE

    connection = dimos.deploy(Go2FleetConnection, ips)

    connection.pointcloud.transport = pSHMTransport(
        f"{prefix}/lidar", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )
    connection.color_image.transport = pSHMTransport(
        f"{prefix}/image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )
    connection.cmd_vel.transport = LCMTransport(f"{prefix}/cmd_vel", Twist)
    connection.camera_info.transport = LCMTransport(f"{prefix}/camera_info", CameraInfo)
    connection.start()

    return connection


__all__ = ["Go2FleetConnection", "deploy", "go2_fleet_connection"]

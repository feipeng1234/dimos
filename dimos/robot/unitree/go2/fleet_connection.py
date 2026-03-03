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

"""Go2 Fleet Connection — broadcast cmd_vel to multiple Go2 robots over WebRTC.

All robots receive the same velocity commands. Video, odometry, and lidar
are published from the primary (first) robot only.
"""

from __future__ import annotations

import logging
import time
from threading import Thread

from reactivex.disposable import Disposable

from dimos import spec
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import (
    PoseStamped,
    Quaternion,
    Transform,
    Twist,
    Vector3,
)
from dimos.msgs.sensor_msgs import CameraInfo, Image, PointCloud2
from dimos.robot.unitree.connection import UnitreeWebRTCConnection

logger = logging.getLogger(__name__)


def _camera_info_static() -> CameraInfo:
    fx, fy, cx, cy = (819.553492, 820.646595, 625.284099, 336.808987)
    width, height = (1280, 720)
    return CameraInfo(
        frame_id="camera_optical",
        height=height,
        width=width,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x=0,
        binning_y=0,
    )


class Go2FleetConnection(Module, spec.Camera, spec.Pointcloud):
    """Manages multiple Go2 WebRTC connections with broadcast cmd_vel.

    All robots receive the same velocity commands. Video, odometry, and lidar
    are published from the primary (first) robot only.

    Args:
        ips: List of robot IP addresses.
    """

    cmd_vel: In[Twist]
    pointcloud: Out[PointCloud2]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    def __init__(self, ips: list[str], *args: object, **kwargs: object) -> None:
        if not ips:
            raise ValueError("At least one IP address is required")
        self._ips = ips
        self._connections: list[UnitreeWebRTCConnection] = []
        self._camera_info_thread: Thread | None = None
        self._latest_video_frame: Image | None = None
        Module.__init__(self, *args, **kwargs)

    @rpc
    def start(self) -> None:
        super().start()

        for ip in self._ips:
            logger.info(f"Connecting to Go2 at {ip}...")
            conn = UnitreeWebRTCConnection(ip)
            conn.start()
            self._connections.append(conn)

        # Publish streams from primary (first) robot
        primary = self._connections[0]

        def onimage(image: Image) -> None:
            self.color_image.publish(image)
            self._latest_video_frame = image

        self._disposables.add(primary.lidar_stream().subscribe(self.lidar.publish))
        self._disposables.add(primary.odom_stream().subscribe(self._publish_tf))
        self._disposables.add(primary.video_stream().subscribe(onimage))
        self._disposables.add(Disposable(self.cmd_vel.subscribe(self._broadcast_move)))

        self._camera_info_thread = Thread(
            target=self._publish_camera_info, daemon=True
        )
        self._camera_info_thread.start()

        for conn in self._connections:
            conn.standup()
        time.sleep(3)
        for conn in self._connections:
            conn.balance_stand()

    @rpc
    def stop(self) -> None:
        for conn in self._connections:
            try:
                conn.liedown()
                conn.stop()
            except Exception as e:
                logger.error(f"Error stopping Go2 at {conn}: {e}")

        if self._camera_info_thread and self._camera_info_thread.is_alive():
            self._camera_info_thread.join(timeout=1.0)

        self._connections.clear()
        super().stop()

    def _broadcast_move(self, twist: Twist) -> None:
        for conn in self._connections:
            try:
                conn.move(twist)
            except Exception as e:
                logger.error(f"Error sending move: {e}")

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        return all(
            conn.move(twist, duration) for conn in self._connections
        )

    @rpc
    def standup(self) -> bool:
        return all(conn.standup() for conn in self._connections)

    @rpc
    def liedown(self) -> bool:
        return all(conn.liedown() for conn in self._connections)

    @classmethod
    def _odom_to_tf(cls, odom: PoseStamped) -> list[Transform]:
        return [
            Transform.from_pose("base_link", odom),
            Transform(
                translation=Vector3(0.3, 0.0, 0.0),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
                child_frame_id="camera_link",
                ts=odom.ts,
            ),
            Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
                frame_id="camera_link",
                child_frame_id="camera_optical",
                ts=odom.ts,
            ),
        ]

    def _publish_tf(self, msg: PoseStamped) -> None:
        self.tf.publish(*self._odom_to_tf(msg))
        if self.odom.transport:
            self.odom.publish(msg)

    def _publish_camera_info(self) -> None:
        while True:
            self.camera_info.publish(_camera_info_static())
            time.sleep(1.0)


go2_fleet_connection = Go2FleetConnection.blueprint

__all__ = ["Go2FleetConnection", "go2_fleet_connection"]

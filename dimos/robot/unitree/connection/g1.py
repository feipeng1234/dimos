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

from threading import Thread
import time
from typing import Any, Protocol

from reactivex.disposable import Disposable
import rerun as rr
import rerun.blueprint as rrb

from dimos import spec
from dimos.core import DimosCluster, In, Module, Out, rpc
from dimos.core.global_config import GlobalConfig
from dimos.dashboard.rerun_init import connect_rerun
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Transform, Twist, Vector3
from dimos.msgs.sensor_msgs import CameraInfo, Image, PointCloud2
from dimos.robot.unitree.connection.connection import UnitreeWebRTCConnection
from dimos.robot.unitree_webrtc.type.lidar import LidarMessage
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

class G1ConnectionProtocol(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def lidar_stream(self): ...  # Observable-like
    def odom_stream(self): ...  # Observable-like
    def video_stream(self): ...  # Observable-like
    def move(self, twist: Twist, duration: float = 0.0) -> Any: ...
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]: ...


def _camera_info_static() -> CameraInfo:
    # Shared default intrinsics used by GO2 (good enough for sim visualization).
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


class G1Connection(Module, spec.Camera, spec.Pointcloud):
    cmd_vel: In[Twist]
    # Match GO2 naming so the mapping stack wires up automatically:
    # - VoxelGridMapper expects `lidar: In[LidarMessage]`
    # - Some consumers expect `pointcloud: Out[PointCloud2]`
    lidar: Out[LidarMessage]
    pointcloud: Out[PointCloud2]
    odom: Out[PoseStamped]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    ip: str | None
    connection_type: str | None = None
    _global_config: GlobalConfig

    connection: G1ConnectionProtocol | None
    camera_info_static: CameraInfo = _camera_info_static()
    _camera_info_thread: Thread | None = None

    def __init__(
        self,
        ip: str | None = None,
        connection_type: str | None = None,
        global_config: GlobalConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._global_config = global_config or GlobalConfig()
        self.ip = ip if ip is not None else self._global_config.robot_ip
        self.connection_type = connection_type or self._global_config.unitree_connection_type
        self.connection = None
        super().__init__(*args, **kwargs)

    @rpc
    def start(self) -> None:
        super().start()

        match self.connection_type:
            case "webrtc":
                assert self.ip is not None, "IP address must be provided"
                self.connection = UnitreeWebRTCConnection(self.ip)
            case "mujoco":
                from dimos.robot.unitree_webrtc.mujoco_connection import MujocoConnection

                self.connection = MujocoConnection(self._global_config)
            case "replay":
                raise ValueError("Replay connection not implemented for G1 robot")
            case _:
                raise ValueError(f"Unknown connection type: {self.connection_type}")

        assert self.connection is not None
        self.connection.start()

        # Initialize static Rerun assets. Dynamic transforms are handled by tf_rerun polling.
        if self._global_config.viewer_backend.startswith("rerun"):
            self._init_rerun_world()

        def onimage(image: Image) -> None:
            self.color_image.publish(image)
            if self._global_config.viewer_backend.startswith("rerun"):
                rr.log("world/robot/camera/rgb", image.to_rerun())

        def on_lidar(msg: LidarMessage) -> None:
            # Publish both views; LidarMessage is a PointCloud2 subclass.
            self.lidar.publish(msg)
            self.pointcloud.publish(msg)

        self._disposables.add(self.connection.lidar_stream().subscribe(on_lidar))
        self._disposables.add(self.connection.odom_stream().subscribe(self._publish_tf))
        self._disposables.add(self.connection.video_stream().subscribe(onimage))
        self._disposables.add(Disposable(self.cmd_vel.subscribe(self.move)))

        self._camera_info_thread = Thread(
            target=self._publish_camera_info,
            daemon=True,
            name="G1CameraInfoThread",
        )
        self._camera_info_thread.start()

    @rpc
    def stop(self) -> None:
        self.connection.stop()
        if self._camera_info_thread and self._camera_info_thread.is_alive():
            self._camera_info_thread.join(timeout=1.0)
        super().stop()

    @classmethod
    def rerun_views(cls):  # type: ignore[no-untyped-def]
        return [
            rrb.Spatial2DView(
                name="G1 Camera",
                origin="world/robot/camera/rgb",
            )
        ]

    def _init_rerun_world(self) -> None:
        """Log static Rerun entities and attach semantic paths to TF frames (GO2 pattern)."""
        connect_rerun(global_config=self._global_config)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Attach semantic paths to named TF frames.
        rr.log(
            "world",
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
                parent_frame="world",  # type: ignore[call-arg]
            ),
            static=True,
        )
        rr.log(
            "world/robot",
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
                parent_frame="base_link",  # type: ignore[call-arg]
            ),
            static=True,
        )
        rr.log("world/robot/axes", rr.TransformAxes3D(0.5), static=True)  # type: ignore[attr-defined]

        rr.log(
            "world/robot/camera",
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
                parent_frame="camera_optical",  # type: ignore[call-arg]
            ),
            static=True,
        )
        rr.log("world/robot/camera", _camera_info_static().to_rerun(), static=True)

    def _publish_camera_info(self) -> None:
        while True:
            self.camera_info.publish(_camera_info_static())
            time.sleep(1.0)

    @classmethod
    def _odom_to_tf(cls, odom: PoseStamped) -> list[Transform]:
        camera_link = Transform(
            translation=Vector3(0.3, 0.0, 0.0),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
            child_frame_id="camera_link",
            ts=odom.ts,
        )

        camera_optical = Transform(
            translation=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
            frame_id="camera_link",
            child_frame_id="camera_optical",
            ts=odom.ts,
        )

        return [
            Transform.from_pose("base_link", odom),
            camera_link,
            camera_optical,
        ]

    def _publish_tf(self, msg: PoseStamped) -> None:
        transforms = self._odom_to_tf(msg)
        self.tf.publish(*transforms)
        if self.odom.transport:
            self.odom.publish(msg)

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> None:
        self.connection.move(twist, duration)

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        logger.info("Publishing request", topic=topic)
        return self.connection.publish_request(topic, data)  # type: ignore[no-any-return]


g1_connection = G1Connection.blueprint


def deploy(dimos: DimosCluster, ip: str, local_planner: spec.LocalPlanner) -> G1Connection:
    connection = dimos.deploy(G1Connection, ip)  # type: ignore[attr-defined]
    connection.cmd_vel.connect(local_planner.cmd_vel)
    connection.start()
    return connection  # type: ignore[no-any-return]


__all__ = ["G1Connection", "deploy", "g1_connection"]

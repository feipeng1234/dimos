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

"""LiDAR module wrappers that convert LidarHardware observables into module streams."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any

import reactivex as rx
from reactivex import operators as ops

from dimos.core import Module, ModuleConfig, Out, rpc
from dimos.msgs.geometry_msgs import Quaternion, Transform, Vector3
from dimos.spec import perception

if TYPE_CHECKING:
    from collections.abc import Callable

    from dimos.hardware.sensors.lidar.spec import LidarHardware
    from dimos.msgs.sensor_msgs.Imu import Imu
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def default_lidar_transform() -> Transform:
    return Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="lidar_link",
    )


@dataclass
class LidarModuleConfig(ModuleConfig):
    frame_id: str = "lidar_link"
    transform: Transform | None = field(default_factory=default_lidar_transform)
    hardware: Callable[[], LidarHardware[Any]] | LidarHardware[Any] | None = None
    frequency: float = 0.0  # Hz, 0 means no limit


class LidarModule(Module[LidarModuleConfig], perception.Lidar):
    """Generic LiDAR module — pointcloud only.

    Publishes PointCloud2 messages and TF transforms.
    """

    pointcloud: Out[PointCloud2]

    hardware: LidarHardware[Any]

    config: LidarModuleConfig
    default_config = LidarModuleConfig

    @rpc
    def start(self) -> None:
        super().start()
        self._init_hardware()

        stream = self.hardware.pointcloud_stream()

        if self.config.frequency > 0:
            stream = stream.pipe(ops.sample(1.0 / self.config.frequency))

        self._disposables.add(
            stream.subscribe(lambda pc: self.pointcloud.publish(pc)),
        )

        self._disposables.add(
            rx.interval(1.0).subscribe(lambda _: self._publish_tf()),
        )

    def _init_hardware(self) -> None:
        if callable(self.config.hardware):
            self.hardware = self.config.hardware()
        else:
            self.hardware = self.config.hardware  # type: ignore[assignment]

    def _publish_tf(self) -> None:
        if not self.config.transform:
            return
        ts = time.time()
        lidar_link = self.config.transform
        lidar_link.ts = ts
        self.tf.publish(lidar_link)

    def stop(self) -> None:
        if self.hardware and hasattr(self.hardware, "stop"):
            self.hardware.stop()
        super().stop()


# ---------------------------------------------------------------------------
# Livox-specific: LiDAR + IMU
# ---------------------------------------------------------------------------


@dataclass
class LivoxLidarModuleConfig(LidarModuleConfig):
    enable_imu: bool = True


class LivoxLidarModule(LidarModule, perception.IMU):
    """Livox LiDAR module — pointcloud + IMU.

    Extends LidarModule with IMU stream support for sensors like the Mid-360.
    """

    imu: Out[Imu]

    config: LivoxLidarModuleConfig  # type: ignore[assignment]
    default_config = LivoxLidarModuleConfig  # type: ignore[assignment]

    @rpc
    def start(self) -> None:
        super().start()

        if self.config.enable_imu:
            imu_stream = self.hardware.imu_stream()
            if imu_stream is not None:
                self._disposables.add(
                    imu_stream.subscribe(lambda imu_msg: self.imu.publish(imu_msg)),
                )


lidar_module = LidarModule.blueprint
livox_lidar_module = LivoxLidarModule.blueprint

__all__ = [
    "LidarModule",
    "LivoxLidarModule",
    "lidar_module",
    "livox_lidar_module",
]

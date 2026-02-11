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

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from reactivex.observable import Observable

from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.service import Configurable  # type: ignore[attr-defined]


class LidarConfig(Protocol):
    frame_id_prefix: str | None
    frequency: float  # Hz, point cloud output rate


LidarConfigT = TypeVar("LidarConfigT", bound=LidarConfig)


class LidarHardware(ABC, Configurable[LidarConfigT], Generic[LidarConfigT]):
    """Abstract base class for LiDAR hardware drivers."""

    @abstractmethod
    def pointcloud_stream(self) -> Observable[PointCloud2]:
        """Observable stream of point clouds from the LiDAR."""
        pass

    def imu_stream(self) -> Observable[Imu] | None:
        """Optional observable stream of IMU data. Returns None if unsupported."""
        return None

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

"""Livox Mid-360 LiDAR hardware driver using Livox SDK2 ctypes bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import json
import logging
from queue import Empty, Queue
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
import open3d.core as o3c  # type: ignore[import-untyped]
from reactivex import create

from dimos.hardware.sensors.lidar.livox.sdk import (
    AsyncControlCallbackType,
    ImuDataCallbackType,
    InfoChangeCallbackType,
    LivoxLidarDeviceType,
    LivoxLidarEthernetPacket,
    LivoxLidarInfo,
    LivoxLidarPointDataType,
    LivoxLidarWorkMode,
    LivoxSDK,
    PointCloudCallbackType,
    get_packet_timestamp_ns,
    parse_cartesian_high_points,
    parse_cartesian_low_points,
    parse_imu_data,
)
from dimos.hardware.sensors.lidar.spec import LidarConfig, LidarHardware
from dimos.msgs.geometry_msgs import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.reactive import backpressure

if TYPE_CHECKING:
    from reactivex.observable import Observable

logger = logging.getLogger(__name__)

# Gravity constant for converting accelerometer data from g to m/s^2
GRAVITY_MS2 = 9.80665


@dataclass
class LivoxMid360Config(LidarConfig):
    """Configuration for the Livox Mid-360 LiDAR."""

    host_ip: str = "192.168.1.5"
    lidar_ips: list[str] = field(default_factory=lambda: ["192.168.1.155"])
    frequency: float = 10.0  # Hz, point cloud output rate
    frame_id: str = "lidar_link"
    frame_id_prefix: str | None = None
    enable_imu: bool = True
    sdk_lib_path: str | None = None

    # SDK port configuration
    cmd_data_port: int = 56100
    push_msg_port: int = 56200
    point_data_port: int = 56300
    imu_data_port: int = 56400
    log_data_port: int = 56500
    host_cmd_data_port: int = 56101
    host_push_msg_port: int = 56201
    host_point_data_port: int = 56301
    host_imu_data_port: int = 56401
    host_log_data_port: int = 56501

    # Socket buffer size (bytes) to avoid packet loss at high data rates
    recv_buf_size: int = 4 * 1024 * 1024  # 4 MB


class LivoxMid360(LidarHardware["LivoxMid360Config"]):
    """Livox Mid-360 LiDAR driver using SDK2 ctypes bindings.

    Produces Observable[PointCloud2] at the configured frame rate (~10 Hz),
    and optionally Observable[Imu] at ~200 Hz.

    Uses a thread-safe queue to pass data from SDK C-thread callbacks
    to Python consumer threads that build messages.
    """

    default_config = LivoxMid360Config

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._sdk: LivoxSDK | None = None
        self._config_path: str | None = None
        self._stop_event = threading.Event()

        # Point cloud frame accumulator state
        self._pc_queue: Queue[tuple[int, int, np.ndarray, np.ndarray]] = Queue(maxsize=4096)
        self._pc_consumer_thread: threading.Thread | None = None
        self._pc_observer: Any[PointCloud2] | None = None

        # IMU state
        self._imu_queue: Queue[tuple[int, np.ndarray]] = Queue(maxsize=4096)
        self._imu_consumer_thread: threading.Thread | None = None
        self._imu_observer: Any[Imu] | None = None

        # Device tracking
        self._connected_handles: dict[int, LivoxLidarDeviceType] = {}

    def _frame_id(self, suffix: str) -> str:
        prefix = self.config.frame_id_prefix
        if prefix:
            return f"{prefix}/{suffix}"
        return suffix

    # ------------------------------------------------------------------
    # SDK config generation
    # ------------------------------------------------------------------

    def _write_sdk_config(self) -> str:
        """Generate a temporary JSON config file for the SDK."""
        config = {
            "MID360": {
                "lidar_net_info": {
                    "cmd_data_port": self.config.cmd_data_port,
                    "push_msg_port": self.config.push_msg_port,
                    "point_data_port": self.config.point_data_port,
                    "imu_data_port": self.config.imu_data_port,
                    "log_data_port": self.config.log_data_port,
                },
                "host_net_info": [
                    {
                        "host_ip": self.config.host_ip,
                        "multicast_ip": "224.1.1.5",
                        "cmd_data_port": self.config.host_cmd_data_port,
                        "push_msg_port": self.config.host_push_msg_port,
                        "point_data_port": self.config.host_point_data_port,
                        "imu_data_port": self.config.host_imu_data_port,
                        "log_data_port": self.config.host_log_data_port,
                    }
                ],
            }
        }
        fd, path = tempfile.mkstemp(suffix=".json", prefix="livox_mid360_")
        with open(fd, "w") as f:
            json.dump(config, f)
        return path

    # ------------------------------------------------------------------
    # SDK callbacks (called from C threads - keep minimal, just enqueue)
    # ------------------------------------------------------------------

    def _on_point_cloud(
        self,
        handle: int,
        dev_type: int,
        packet_ptr: LivoxLidarEthernetPacket,  # type: ignore[override]
        client_data: object,
    ) -> None:
        """SDK point cloud callback. Copies data and enqueues for processing."""
        if self._stop_event.is_set():
            return
        try:
            packet = packet_ptr.contents if hasattr(packet_ptr, "contents") else packet_ptr
            data_type = packet.data_type

            if data_type == LivoxLidarPointDataType.CARTESIAN_HIGH:
                xyz, reflectivities, _tags = parse_cartesian_high_points(packet)
            elif data_type == LivoxLidarPointDataType.CARTESIAN_LOW:
                xyz, reflectivities, _tags = parse_cartesian_low_points(packet)
            else:
                return  # skip spherical for now

            ts_ns = get_packet_timestamp_ns(packet)
            frame_cnt = packet.frame_cnt

            self._pc_queue.put_nowait((frame_cnt, ts_ns, xyz, reflectivities))
        except Exception:
            logger.debug("Error in point cloud callback", exc_info=True)

    def _on_imu_data(
        self,
        handle: int,
        dev_type: int,
        packet_ptr: LivoxLidarEthernetPacket,  # type: ignore[override]
        client_data: object,
    ) -> None:
        """SDK IMU callback. Copies data and enqueues."""
        if self._stop_event.is_set():
            return
        try:
            packet = packet_ptr.contents if hasattr(packet_ptr, "contents") else packet_ptr
            ts_ns = get_packet_timestamp_ns(packet)
            imu_points = parse_imu_data(packet)
            self._imu_queue.put_nowait((ts_ns, imu_points))
        except Exception:
            logger.debug("Error in IMU callback", exc_info=True)

    def _on_info_change(
        self,
        handle: int,
        info_ptr: LivoxLidarInfo,  # type: ignore[override]
        client_data: object,
    ) -> None:
        """SDK device info change callback. Tracks connected devices."""
        try:
            info = info_ptr.contents if hasattr(info_ptr, "contents") else info_ptr
            dev_type = LivoxLidarDeviceType(info.dev_type)
            sn = info.sn.decode("utf-8", errors="replace").rstrip("\x00")
            ip = info.lidar_ip.decode("utf-8", errors="replace").rstrip("\x00")
            self._connected_handles[handle] = dev_type
            logger.info(
                "Livox device connected: handle=%d type=%s sn=%s ip=%s",
                handle,
                dev_type.name,
                sn,
                ip,
            )

            # Set to normal work mode
            self._sdk_set_work_mode(handle)

            # Enable/disable IMU based on config
            if self.config.enable_imu:
                self._sdk_enable_imu(handle)
        except Exception:
            logger.debug("Error in info change callback", exc_info=True)

    def _sdk_set_work_mode(self, handle: int) -> None:
        if self._sdk is None:
            return

        def _on_work_mode(status: int, handle: int, response: object, client_data: object) -> None:
            if status == 0:
                logger.info("Work mode set to NORMAL for handle %d", handle)
            else:
                logger.warning("Failed to set work mode for handle %d: status=%d", handle, status)

        _cb = AsyncControlCallbackType(_on_work_mode)
        self._sdk._callbacks[f"work_mode_cb_{handle}"] = _cb
        self._sdk.lib.SetLivoxLidarWorkMode(handle, int(LivoxLidarWorkMode.NORMAL), _cb, None)

    def _sdk_enable_imu(self, handle: int) -> None:
        if self._sdk is None:
            return

        def _on_imu_enable(status: int, handle: int, response: object, client_data: object) -> None:
            if status == 0:
                logger.info("IMU enabled for handle %d", handle)
            else:
                logger.warning("Failed to enable IMU for handle %d: status=%d", handle, status)

        _cb = AsyncControlCallbackType(_on_imu_enable)
        self._sdk._callbacks[f"imu_enable_cb_{handle}"] = _cb
        self._sdk.lib.EnableLivoxLidarImuData(handle, _cb, None)

    # ------------------------------------------------------------------
    # Consumer threads (Python-side, build messages from queued data)
    # ------------------------------------------------------------------

    def _pointcloud_consumer_loop(self) -> None:
        """Drain the point cloud queue, accumulate by time window, emit PointCloud2."""
        frame_points: list[np.ndarray] = []
        frame_reflectivities: list[np.ndarray] = []
        frame_timestamp: float | None = None
        frame_interval = 1.0 / self.config.frequency if self.config.frequency > 0 else 0.1
        last_emit = time.monotonic()

        while not self._stop_event.is_set():
            try:
                _frame_cnt, ts_ns, xyz, reflectivities = self._pc_queue.get(timeout=0.05)
            except Empty:
                # Check if we should emit on timeout
                if frame_points and (time.monotonic() - last_emit) >= frame_interval:
                    self._emit_pointcloud(frame_points, frame_reflectivities, frame_timestamp)
                    frame_points = []
                    frame_reflectivities = []
                    frame_timestamp = None
                    last_emit = time.monotonic()
                continue

            frame_points.append(xyz)
            frame_reflectivities.append(reflectivities)
            if frame_timestamp is None:
                frame_timestamp = ts_ns / 1e9

            # Emit when time window is full
            if (time.monotonic() - last_emit) >= frame_interval:
                self._emit_pointcloud(frame_points, frame_reflectivities, frame_timestamp)
                frame_points = []
                frame_reflectivities = []
                frame_timestamp = None
                last_emit = time.monotonic()

        # Emit any remaining data
        if frame_points:
            self._emit_pointcloud(frame_points, frame_reflectivities, frame_timestamp)

    def _emit_pointcloud(
        self,
        frame_points: list[np.ndarray],
        frame_reflectivities: list[np.ndarray],
        frame_timestamp: float | None,
    ) -> None:
        if not self._pc_observer or self._stop_event.is_set():
            return

        all_points = np.concatenate(frame_points, axis=0)
        all_reflectivities = np.concatenate(frame_reflectivities, axis=0)

        if len(all_points) == 0:
            return

        # Build Open3D tensor point cloud
        pcd_t = o3d.t.geometry.PointCloud()
        pcd_t.point["positions"] = o3c.Tensor(all_points, dtype=o3c.float32)
        # Store reflectivity as intensity (normalized to 0-1 range)
        pcd_t.point["intensities"] = o3c.Tensor(
            all_reflectivities.astype(np.float32).reshape(-1, 1) / 255.0,
            dtype=o3c.float32,
        )

        pc2 = PointCloud2(
            pointcloud=pcd_t,
            frame_id=self._frame_id(self.config.frame_id),
            ts=frame_timestamp if frame_timestamp else time.time(),
        )

        try:
            self._pc_observer.on_next(pc2)
        except Exception:
            logger.debug("Error emitting point cloud", exc_info=True)

    def _imu_consumer_loop(self) -> None:
        """Drain the IMU queue and emit Imu messages."""
        while not self._stop_event.is_set():
            try:
                ts_ns, imu_points = self._imu_queue.get(timeout=0.5)
            except Empty:
                continue

            if not self._imu_observer or self._stop_event.is_set():
                continue

            ts = ts_ns / 1e9
            for i in range(len(imu_points)):
                pt = imu_points[i]
                imu_msg = Imu(
                    angular_velocity=Vector3(
                        float(pt["gyro_x"]),
                        float(pt["gyro_y"]),
                        float(pt["gyro_z"]),
                    ),
                    linear_acceleration=Vector3(
                        float(pt["acc_x"]) * GRAVITY_MS2,
                        float(pt["acc_y"]) * GRAVITY_MS2,
                        float(pt["acc_z"]) * GRAVITY_MS2,
                    ),
                    frame_id=self._frame_id("imu_link"),
                    ts=ts,
                )
                try:
                    self._imu_observer.on_next(imu_msg)
                except Exception:
                    logger.debug("Error emitting IMU data", exc_info=True)

    # ------------------------------------------------------------------
    # Public API: Observable streams
    # ------------------------------------------------------------------

    @cache
    def pointcloud_stream(self) -> Observable[PointCloud2]:
        """Observable stream of PointCloud2 messages (~10 Hz full-frame scans)."""

        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._pc_observer = observer
            try:
                self._start_sdk()
            except Exception as e:
                observer.on_error(e)
                return

            def dispose() -> None:
                self._pc_observer = None
                self.stop()

            return dispose

        return backpressure(create(subscribe))

    @cache
    def imu_stream(self) -> Observable[Imu]:
        """Observable stream of Imu messages (~200 Hz)."""

        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._imu_observer = observer

            def dispose() -> None:
                self._imu_observer = None

            return dispose

        return backpressure(create(subscribe))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start_sdk(self) -> None:
        """Initialize and start the Livox SDK."""
        if self._sdk is not None:
            return  # already running

        self._stop_event.clear()

        # Write temp config
        self._config_path = self._write_sdk_config()

        # Load and init SDK
        self._sdk = LivoxSDK(self.config.sdk_lib_path)
        self._sdk.init(self._config_path, self.config.host_ip)

        # Register callbacks (must keep references to prevent GC)
        pc_cb = PointCloudCallbackType(self._on_point_cloud)
        imu_cb = ImuDataCallbackType(self._on_imu_data)
        info_cb = InfoChangeCallbackType(self._on_info_change)

        self._sdk.set_point_cloud_callback(pc_cb)
        if self.config.enable_imu:
            self._sdk.set_imu_callback(imu_cb)
        self._sdk.set_info_change_callback(info_cb)

        # Start SDK background threads
        self._sdk.start()
        logger.info(
            "Livox SDK started (host_ip=%s, lidar_ips=%s)",
            self.config.host_ip,
            self.config.lidar_ips,
        )

        # Start consumer threads
        self._pc_consumer_thread = threading.Thread(
            target=self._pointcloud_consumer_loop, daemon=True, name="livox-pc-consumer"
        )
        self._pc_consumer_thread.start()

        if self.config.enable_imu:
            self._imu_consumer_thread = threading.Thread(
                target=self._imu_consumer_loop, daemon=True, name="livox-imu-consumer"
            )
            self._imu_consumer_thread.start()

    def stop(self) -> None:
        """Stop the SDK and all consumer threads."""
        self._stop_event.set()

        if self._sdk is not None:
            self._sdk.uninit()
            self._sdk = None

        for thread in [self._pc_consumer_thread, self._imu_consumer_thread]:
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)

        self._pc_consumer_thread = None
        self._imu_consumer_thread = None
        self._connected_handles.clear()

        # Cleanup temp config
        if self._config_path:
            import os

            try:
                os.unlink(self._config_path)
            except OSError:
                pass
            self._config_path = None

        logger.info("Livox Mid-360 stopped")

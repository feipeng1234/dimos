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

"""ctypes bindings for Livox SDK2 (liblivox_lidar_sdk_shared.so).

Provides Python access to the Livox LiDAR SDK2 C API for device
communication, point cloud streaming, and IMU data.

The SDK .so must be pre-built from https://github.com/Livox-SDK/Livox-SDK2.
Set LIVOX_SDK2_LIB_PATH env var or install to /usr/local/lib.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from enum import IntEnum
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LivoxLidarDeviceType(IntEnum):
    HUB = 0
    MID40 = 1
    TELE = 2
    HORIZON = 3
    MID70 = 6
    AVIA = 7
    MID360 = 9
    INDUSTRIAL_HAP = 10
    HAP = 15
    PA = 16


class LivoxLidarPointDataType(IntEnum):
    IMU = 0x00
    CARTESIAN_HIGH = 0x01
    CARTESIAN_LOW = 0x02
    SPHERICAL = 0x03


class LivoxLidarWorkMode(IntEnum):
    NORMAL = 0x01
    WAKE_UP = 0x02
    SLEEP = 0x03
    ERROR = 0x04
    POWER_ON_SELF_TEST = 0x05
    MOTOR_STARTING = 0x06
    MOTOR_STOPPING = 0x07
    UPGRADE = 0x08


class LivoxLidarScanPattern(IntEnum):
    NON_REPETITIVE = 0x00
    REPETITIVE = 0x01
    REPETITIVE_LOW_FRAME_RATE = 0x02


class LivoxLidarStatus(IntEnum):
    SEND_FAILED = -9
    HANDLER_IMPL_NOT_EXIST = -8
    INVALID_HANDLE = -7
    CHANNEL_NOT_EXIST = -6
    NOT_ENOUGH_MEMORY = -5
    TIMEOUT = -4
    NOT_SUPPORTED = -3
    NOT_CONNECTED = -2
    FAILURE = -1
    SUCCESS = 0


# ---------------------------------------------------------------------------
# Packed C structures (all #pragma pack(1) in the SDK header)
# ---------------------------------------------------------------------------


class LivoxLidarCartesianHighRawPoint(ctypes.LittleEndianStructure):
    """High-resolution cartesian point (14 bytes). Coordinates in mm."""

    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("z", ctypes.c_int32),
        ("reflectivity", ctypes.c_uint8),
        ("tag", ctypes.c_uint8),
    ]


class LivoxLidarCartesianLowRawPoint(ctypes.LittleEndianStructure):
    """Low-resolution cartesian point (8 bytes). Coordinates in cm."""

    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_int16),
        ("y", ctypes.c_int16),
        ("z", ctypes.c_int16),
        ("reflectivity", ctypes.c_uint8),
        ("tag", ctypes.c_uint8),
    ]


class LivoxLidarSpherPoint(ctypes.LittleEndianStructure):
    """Spherical coordinate point (10 bytes)."""

    _pack_ = 1
    _fields_ = [
        ("depth", ctypes.c_uint32),
        ("theta", ctypes.c_uint16),
        ("phi", ctypes.c_uint16),
        ("reflectivity", ctypes.c_uint8),
        ("tag", ctypes.c_uint8),
    ]


class LivoxLidarImuRawPoint(ctypes.LittleEndianStructure):
    """IMU data point (24 bytes). Gyro in rad/s, accel in g."""

    _pack_ = 1
    _fields_ = [
        ("gyro_x", ctypes.c_float),
        ("gyro_y", ctypes.c_float),
        ("gyro_z", ctypes.c_float),
        ("acc_x", ctypes.c_float),
        ("acc_y", ctypes.c_float),
        ("acc_z", ctypes.c_float),
    ]


class LivoxLidarEthernetPacket(ctypes.LittleEndianStructure):
    """Point cloud / IMU ethernet packet header (32 bytes + variable data)."""

    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint8),
        ("length", ctypes.c_uint16),
        ("time_interval", ctypes.c_uint16),  # unit: 0.1 us
        ("dot_num", ctypes.c_uint16),
        ("udp_cnt", ctypes.c_uint16),
        ("frame_cnt", ctypes.c_uint8),
        ("data_type", ctypes.c_uint8),
        ("time_type", ctypes.c_uint8),
        ("rsvd", ctypes.c_uint8 * 12),
        ("crc32", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint8 * 8),
        ("data", ctypes.c_uint8 * 1),  # variable-length payload
    ]


class LivoxLidarInfo(ctypes.LittleEndianStructure):
    """Device info received on connection."""

    _pack_ = 1
    _fields_ = [
        ("dev_type", ctypes.c_uint8),
        ("sn", ctypes.c_char * 16),
        ("lidar_ip", ctypes.c_char * 16),
    ]


class LivoxLidarAsyncControlResponse(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("ret_code", ctypes.c_uint8),
        ("error_key", ctypes.c_uint16),
    ]


# ---------------------------------------------------------------------------
# numpy dtypes for efficient batch parsing of point data
# ---------------------------------------------------------------------------

CART_HIGH_DTYPE = np.dtype(
    [("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("reflectivity", "u1"), ("tag", "u1")]
)
assert CART_HIGH_DTYPE.itemsize == 14

CART_LOW_DTYPE = np.dtype(
    [("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("reflectivity", "u1"), ("tag", "u1")]
)
assert CART_LOW_DTYPE.itemsize == 8

IMU_DTYPE = np.dtype(
    [
        ("gyro_x", "<f4"),
        ("gyro_y", "<f4"),
        ("gyro_z", "<f4"),
        ("acc_x", "<f4"),
        ("acc_y", "<f4"),
        ("acc_z", "<f4"),
    ]
)
assert IMU_DTYPE.itemsize == 24


# ---------------------------------------------------------------------------
# Callback function pointer types
# ---------------------------------------------------------------------------

# void cb(uint32_t handle, uint8_t dev_type, LivoxLidarEthernetPacket* data, void* client_data)
PointCloudCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint32,
    ctypes.c_uint8,
    ctypes.POINTER(LivoxLidarEthernetPacket),
    ctypes.c_void_p,
)

# Same signature for IMU
ImuDataCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint32,
    ctypes.c_uint8,
    ctypes.POINTER(LivoxLidarEthernetPacket),
    ctypes.c_void_p,
)

# void cb(uint32_t handle, const LivoxLidarInfo* info, void* client_data)
InfoChangeCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint32,
    ctypes.POINTER(LivoxLidarInfo),
    ctypes.c_void_p,
)

# void cb(uint32_t handle, uint8_t dev_type, const char* info, void* client_data)
InfoCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint32,
    ctypes.c_uint8,
    ctypes.c_char_p,
    ctypes.c_void_p,
)

# void cb(livox_status status, uint32_t handle, LivoxLidarAsyncControlResponse* response, void* client_data)
AsyncControlCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # livox_status
    ctypes.c_uint32,
    ctypes.POINTER(LivoxLidarAsyncControlResponse),
    ctypes.c_void_p,
)


# ---------------------------------------------------------------------------
# Packet parsing helpers
# ---------------------------------------------------------------------------


def parse_cartesian_high_points(
    packet: LivoxLidarEthernetPacket,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse high-res cartesian points from a packet.

    Returns:
        (xyz_meters, reflectivities, tags) where xyz is (N,3) float32 in meters.
    """
    dot_num = packet.dot_num
    if dot_num == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)

    data_ptr = ctypes.addressof(packet) + LivoxLidarEthernetPacket.data.offset
    buf = (ctypes.c_uint8 * (dot_num * CART_HIGH_DTYPE.itemsize)).from_address(data_ptr)
    points = np.frombuffer(buf, dtype=CART_HIGH_DTYPE, count=dot_num)

    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32) / 1000.0
    return xyz, points["reflectivity"].copy(), points["tag"].copy()


def parse_cartesian_low_points(
    packet: LivoxLidarEthernetPacket,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse low-res cartesian points. Returns xyz in meters."""
    dot_num = packet.dot_num
    if dot_num == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)

    data_ptr = ctypes.addressof(packet) + LivoxLidarEthernetPacket.data.offset
    buf = (ctypes.c_uint8 * (dot_num * CART_LOW_DTYPE.itemsize)).from_address(data_ptr)
    points = np.frombuffer(buf, dtype=CART_LOW_DTYPE, count=dot_num)

    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32) / 100.0
    return xyz, points["reflectivity"].copy(), points["tag"].copy()


def parse_imu_data(packet: LivoxLidarEthernetPacket) -> np.ndarray:
    """Parse IMU data from a packet. Returns structured array with gyro/accel fields."""
    dot_num = packet.dot_num
    if dot_num == 0:
        return np.empty(0, dtype=IMU_DTYPE)

    data_ptr = ctypes.addressof(packet) + LivoxLidarEthernetPacket.data.offset
    buf = (ctypes.c_uint8 * (dot_num * IMU_DTYPE.itemsize)).from_address(data_ptr)
    return np.frombuffer(buf, dtype=IMU_DTYPE, count=dot_num).copy()


def get_packet_timestamp_ns(packet: LivoxLidarEthernetPacket) -> int:
    """Extract the 64-bit nanosecond timestamp from a packet."""
    return int.from_bytes(bytes(packet.timestamp), byteorder="little")


# ---------------------------------------------------------------------------
# SDK library loader
# ---------------------------------------------------------------------------

_SDK_LIB_NAMES = [
    "livox_lidar_sdk_shared",
    "liblivox_lidar_sdk_shared.so",
    "liblivox_lidar_sdk_shared.so.2",
]

_SDK_SEARCH_PATHS = [
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
]


def _find_sdk_library(lib_path: str | None = None) -> str:
    """Find the Livox SDK2 shared library.

    Search order:
    1. Explicit lib_path argument
    2. LIVOX_SDK2_LIB_PATH environment variable
    3. ctypes.util.find_library
    4. Well-known system paths
    """
    if lib_path and Path(lib_path).exists():
        return lib_path

    env_path = os.environ.get("LIVOX_SDK2_LIB_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    for name in _SDK_LIB_NAMES:
        found = ctypes.util.find_library(name)
        if found:
            return found

    for search_dir in _SDK_SEARCH_PATHS:
        for name in _SDK_LIB_NAMES:
            candidate = Path(search_dir) / name
            if candidate.exists():
                return str(candidate)

    raise RuntimeError(
        "Livox SDK2 shared library not found. "
        "Build from https://github.com/Livox-SDK/Livox-SDK2 and install, "
        "or set LIVOX_SDK2_LIB_PATH to the .so file path."
    )


def load_sdk(lib_path: str | None = None) -> ctypes.CDLL:
    """Load the Livox SDK2 shared library and set up function signatures."""
    path = _find_sdk_library(lib_path)
    lib = ctypes.CDLL(path)
    logger.info("Loaded Livox SDK2 from %s", path)

    # --- SDK lifecycle ---
    lib.LivoxLidarSdkInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    lib.LivoxLidarSdkInit.restype = ctypes.c_bool

    lib.LivoxLidarSdkStart.argtypes = []
    lib.LivoxLidarSdkStart.restype = ctypes.c_bool

    lib.LivoxLidarSdkUninit.argtypes = []
    lib.LivoxLidarSdkUninit.restype = None

    # --- Callbacks ---
    lib.SetLivoxLidarPointCloudCallBack.argtypes = [PointCloudCallbackType, ctypes.c_void_p]
    lib.SetLivoxLidarPointCloudCallBack.restype = None

    lib.SetLivoxLidarImuDataCallback.argtypes = [ImuDataCallbackType, ctypes.c_void_p]
    lib.SetLivoxLidarImuDataCallback.restype = None

    lib.SetLivoxLidarInfoChangeCallback.argtypes = [InfoChangeCallbackType, ctypes.c_void_p]
    lib.SetLivoxLidarInfoChangeCallback.restype = None

    lib.SetLivoxLidarInfoCallback.argtypes = [InfoCallbackType, ctypes.c_void_p]
    lib.SetLivoxLidarInfoCallback.restype = None

    # --- Device configuration ---
    lib.SetLivoxLidarWorkMode.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        AsyncControlCallbackType,
        ctypes.c_void_p,
    ]
    lib.SetLivoxLidarWorkMode.restype = ctypes.c_int32

    lib.EnableLivoxLidarImuData.argtypes = [
        ctypes.c_uint32,
        AsyncControlCallbackType,
        ctypes.c_void_p,
    ]
    lib.EnableLivoxLidarImuData.restype = ctypes.c_int32

    lib.DisableLivoxLidarImuData.argtypes = [
        ctypes.c_uint32,
        AsyncControlCallbackType,
        ctypes.c_void_p,
    ]
    lib.DisableLivoxLidarImuData.restype = ctypes.c_int32

    lib.SetLivoxLidarPclDataType.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        AsyncControlCallbackType,
        ctypes.c_void_p,
    ]
    lib.SetLivoxLidarPclDataType.restype = ctypes.c_int32

    lib.SetLivoxLidarScanPattern.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        AsyncControlCallbackType,
        ctypes.c_void_p,
    ]
    lib.SetLivoxLidarScanPattern.restype = ctypes.c_int32

    # --- Console logging ---
    lib.DisableLivoxSdkConsoleLogger.argtypes = []
    lib.DisableLivoxSdkConsoleLogger.restype = None

    return lib


class LivoxSDK:
    """Convenience wrapper managing SDK lifecycle and callback registration.

    Prevents garbage collection of ctypes callback references (which would
    cause segfaults when the SDK tries to call them from C threads).
    """

    def __init__(self, lib_path: str | None = None) -> None:
        self._lib = load_sdk(lib_path)
        self._initialized = False
        # prevent GC of callback C function pointers
        self._callbacks: dict[str, Any] = {}

    @property
    def lib(self) -> ctypes.CDLL:
        return self._lib

    def init(self, config_path: str, host_ip: str, quiet: bool = True) -> None:
        """Initialize the SDK with a JSON config file and host IP."""
        if self._initialized:
            raise RuntimeError("SDK already initialized. Call uninit() first.")
        if quiet:
            self._lib.DisableLivoxSdkConsoleLogger()
        ok = self._lib.LivoxLidarSdkInit(
            config_path.encode(),
            host_ip.encode(),
            None,
        )
        if not ok:
            raise RuntimeError(
                f"LivoxLidarSdkInit failed (config={config_path}, host_ip={host_ip})"
            )
        self._initialized = True

    def start(self) -> None:
        """Start the SDK (begins device communication on background threads)."""
        if not self._initialized:
            raise RuntimeError("SDK not initialized. Call init() first.")
        ok = self._lib.LivoxLidarSdkStart()
        if not ok:
            raise RuntimeError("LivoxLidarSdkStart failed")

    def uninit(self) -> None:
        """Shut down the SDK and release resources."""
        if self._initialized:
            self._lib.LivoxLidarSdkUninit()
            self._initialized = False
            self._callbacks.clear()

    def set_point_cloud_callback(self, callback: Any) -> None:
        self._callbacks["pointcloud"] = callback  # prevent GC
        self._lib.SetLivoxLidarPointCloudCallBack(callback, None)

    def set_imu_callback(self, callback: Any) -> None:
        self._callbacks["imu"] = callback
        self._lib.SetLivoxLidarImuDataCallback(callback, None)

    def set_info_change_callback(self, callback: Any) -> None:
        self._callbacks["info_change"] = callback
        self._lib.SetLivoxLidarInfoChangeCallback(callback, None)

    def set_info_callback(self, callback: Any) -> None:
        self._callbacks["info"] = callback
        self._lib.SetLivoxLidarInfoCallback(callback, None)

    def set_work_mode(self, handle: int, mode: LivoxLidarWorkMode, callback: Any = None) -> int:
        cb = callback or AsyncControlCallbackType(0)  # type: ignore[arg-type]
        if callback:
            self._callbacks[f"work_mode_{handle}"] = callback
        return self._lib.SetLivoxLidarWorkMode(handle, int(mode), cb, None)  # type: ignore[no-any-return]

    def enable_imu_data(self, handle: int, callback: Any = None) -> int:
        cb = callback or AsyncControlCallbackType(0)  # type: ignore[arg-type]
        if callback:
            self._callbacks[f"imu_enable_{handle}"] = callback
        return self._lib.EnableLivoxLidarImuData(handle, cb, None)  # type: ignore[no-any-return]

    def disable_imu_data(self, handle: int, callback: Any = None) -> int:
        cb = callback or AsyncControlCallbackType(0)  # type: ignore[arg-type]
        if callback:
            self._callbacks[f"imu_disable_{handle}"] = callback
        return self._lib.DisableLivoxLidarImuData(handle, cb, None)  # type: ignore[no-any-return]

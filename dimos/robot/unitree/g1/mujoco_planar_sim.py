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

"""Single-file kinematic planar mujoco sim for G1 navigation testing.

The robot is a box on a 3-DOF kinematic-feeling base (slide_x + slide_y +
hinge_yaw) driven by velocity actuators, so it respects scene collisions
without needing a biped controller. ``cmd_vel`` is interpreted in the
robot's body frame.

The lidar approximates a Livox Mid-360 mounted upside-down on the G1:
  * 360 deg horizontal x 59 deg vertical FOV (-7..+52 deg about the
    sensor "up" axis; flipped to -52..+7 deg in world by mounting the
    site rotated 180 deg about X).
  * 0.1..70 m range, ~200,000 pts/s.
  * Non-repetitive scan: each scan draws the next chunk of a Halton
    quasi-random sequence over the FOV, so over many scans the FOV
    fills in evenly. This is a convincing visual approximation, not a
    bit-exact reproduction of the real Risley-prism rosette.

Outputs:
  * ``odom``  -- ``nav_msgs.Odometry`` at ``odom_hz``
  * ``lidar`` -- ``sensor_msgs.PointCloud2`` (sensor frame) at ``lidar_hz``

Inputs:
  * ``cmd_vel`` -- ``geometry_msgs.Twist`` (body frame: ``linear.x`` forward,
    ``linear.y`` strafe, ``angular.z`` yaw)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# Mid-360 spec (vertical FOV is asymmetric: -7 deg below to +52 deg above the
# sensor "up" axis). The 180-deg site rotation in the wrapper MJCF physically
# flips the sensor, so these sensor-frame bounds end up as -52..+7 in world.
_MID360_V_MIN_DEG = -7.0
_MID360_V_MAX_DEG = 52.0
_MID360_FOV_H_DEG = 360.0
_MID360_RANGE_MIN_M = 0.1
_MID360_RANGE_MAX_M = 70.0
_MID360_POINTS_PER_SECOND = 200_000
# G1 base sits ~0.7 m off the ground when standing; box half-height matches that.
_DEFAULT_BOX_HALF_EXTENTS = (0.18, 0.18, 0.65)
_DEFAULT_SENSOR_Z_OFFSET = 0.6  # mid360 sits ~0.6 m above box centre (head height)
_PI = math.pi


def _halton(index: int, base: int) -> float:
    """One element of the Halton low-discrepancy sequence in [0, 1)."""
    fraction = 1.0
    result = 0.0
    n = index
    while n > 0:
        fraction /= base
        result += fraction * (n % base)
        n //= base
    return result


def _default_scene_xml() -> str:
    """Path to the in-repo HSSD-house scene used by the rosnav cross-wall test."""
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "data" / "hssd_house" / "scene_hssd_house.xml")


class G1MujocoPlanarSimConfig(ModuleConfig):
    scene_xml: str = Field(default_factory=lambda _: _default_scene_xml())
    # Default spawn pulls from GlobalConfig.mujoco_start_pos so the existing
    # nav-sim CLI override (--mujoco-start-pos "x, y") keeps working.
    spawn_x: float = Field(default_factory=lambda m: m["g"].mujoco_start_pos_float[0])
    spawn_y: float = Field(default_factory=lambda m: m["g"].mujoco_start_pos_float[1])
    spawn_yaw: float = 0.0
    box_half_extents: tuple[float, float, float] = _DEFAULT_BOX_HALF_EXTENTS
    sensor_z_offset: float = _DEFAULT_SENSOR_Z_OFFSET
    physics_dt: float = 0.002
    lidar_hz: float = 10.0
    odom_hz: float = 50.0
    lidar_points_per_scan: int = _MID360_POINTS_PER_SECOND // 10
    lidar_range_min: float = _MID360_RANGE_MIN_M
    lidar_range_max: float = _MID360_RANGE_MAX_M
    fov_h_deg: float = _MID360_FOV_H_DEG
    fov_v_min_deg: float = _MID360_V_MIN_DEG
    fov_v_max_deg: float = _MID360_V_MAX_DEG
    velocity_kv: float = 200.0
    box_mass: float = 35.0
    # Frames default to the nav_stack convention: PGO/SimplePlanner consume
    # odometry tagged ``world`` and a registered scan in the same frame.
    odom_frame: str = "world"
    base_frame: str = "base_link"
    # When true, lidar points are transformed to ``odom_frame`` before
    # publishing so the output is a "registered_scan" -- the form the
    # nav_stack's TerrainAnalysis / PGO consume directly. Set false to
    # publish in the lidar's own sensor frame instead.
    lidar_in_world_frame: bool = True
    sensor_frame: str = "lidar"


class G1MujocoPlanarSim(Module):
    """Kinematic planar G1 sim with a Mid-360-flavoured lidar."""

    config: G1MujocoPlanarSimConfig
    cmd_vel: In[Twist]
    odom: Out[Odometry]
    lidar: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._cmd_lock = threading.Lock()
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._sim_thread: threading.Thread | None = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._slide_x_qpos: int = -1
        self._slide_y_qpos: int = -1
        self._yaw_qpos: int = -1
        self._slide_x_act: int = -1
        self._slide_y_act: int = -1
        self._yaw_act: int = -1
        self._robot_body_id: int = -1
        self._lidar_site_id: int = -1
        self._ray_dirs_sensor: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._ray_cursor: int = 0

    @rpc
    def start(self) -> None:
        super().start()
        self._build_model()
        self._build_ray_pattern()
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        self._sim_thread = threading.Thread(
            target=self._sim_loop,
            name="G1MujocoPlanarSim",
            daemon=True,
        )
        self._sim_thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    def _build_model(self) -> None:
        cfg = self.config
        scene_path = Path(cfg.scene_xml).expanduser().resolve()
        if not scene_path.is_file():
            raise FileNotFoundError(f"Mujoco scene XML not found: {scene_path}")

        bx, by, bz = cfg.box_half_extents
        spawn_z = bz + 0.01  # avoid initial floor penetration
        # Wrapper MJCF lives next to the scene so include + relative asset paths
        # resolve against the same directory as the original scene file.
        wrapper_xml = (
            f'<?xml version="1.0"?>\n'
            f'<mujoco model="g1_planar_sim_wrapper">\n'
            f'  <include file="{scene_path.name}"/>\n'
            f'  <option timestep="{cfg.physics_dt}"/>\n'
            f"  <worldbody>\n"
            f'    <body name="planar_robot" pos="{cfg.spawn_x} {cfg.spawn_y} {spawn_z}">\n'
            f'      <joint name="slide_x" type="slide" axis="1 0 0" limited="false"/>\n'
            f'      <joint name="slide_y" type="slide" axis="0 1 0" limited="false"/>\n'
            f'      <joint name="hinge_yaw" type="hinge" axis="0 0 1" limited="false"/>\n'
            f'      <geom name="planar_robot_box" type="box"'
            f' size="{bx} {by} {bz}" rgba="0.2 0.4 0.9 1" mass="{cfg.box_mass}"/>\n'
            f'      <site name="lidar" pos="0 0 {cfg.sensor_z_offset}"'
            f' euler="{_PI} 0 0"/>\n'
            f"    </body>\n"
            f"  </worldbody>\n"
            f"  <actuator>\n"
            f'    <velocity name="vx" joint="slide_x" kv="{cfg.velocity_kv}"/>\n'
            f'    <velocity name="vy" joint="slide_y" kv="{cfg.velocity_kv}"/>\n'
            f'    <velocity name="wz" joint="hinge_yaw" kv="{cfg.velocity_kv}"/>\n'
            f"  </actuator>\n"
            f"</mujoco>\n"
        )
        wrapper_path = scene_path.parent / f"_g1_planar_sim_wrapper_{os.getpid()}.xml"
        wrapper_path.write_text(wrapper_xml)
        try:
            model = mujoco.MjModel.from_xml_path(str(wrapper_path))
        finally:
            wrapper_path.unlink(missing_ok=True)
        data = mujoco.MjData(model)
        self._model = model
        self._data = data

        self._slide_x_qpos = self._joint_qpos_addr("slide_x")
        self._slide_y_qpos = self._joint_qpos_addr("slide_y")
        self._yaw_qpos = self._joint_qpos_addr("hinge_yaw")
        self._slide_x_act = self._actuator_id("vx")
        self._slide_y_act = self._actuator_id("vy")
        self._yaw_act = self._actuator_id("wz")
        self._robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "planar_robot")
        self._lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        if self._robot_body_id < 0 or self._lidar_site_id < 0:
            raise RuntimeError("planar_robot body or lidar site missing in compiled model")

        data.qpos[self._yaw_qpos] = cfg.spawn_yaw
        mujoco.mj_forward(model, data)

    def _joint_qpos_addr(self, name: str) -> int:
        assert self._model is not None
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Joint {name!r} missing in compiled model")
        return int(self._model.jnt_qposadr[joint_id])

    def _actuator_id(self, name: str) -> int:
        assert self._model is not None
        actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise RuntimeError(f"Actuator {name!r} missing in compiled model")
        return int(actuator_id)

    def _build_ray_pattern(self) -> None:
        """Pre-compute a long Halton sequence of unit ray directions in the
        sensor frame. ``_publish_lidar`` reads a sliding window into this
        sequence each scan, so consecutive scans expose different angles --
        non-repetitive in the short term, uniform in the long term.
        """
        cfg = self.config
        # 200 scans of unique angles before the cursor wraps -- 20 s at 10 Hz.
        n_total = max(cfg.lidar_points_per_scan * 200, 200_000)
        v_min = math.radians(cfg.fov_v_min_deg)
        v_max = math.radians(cfg.fov_v_max_deg)
        h_full = math.radians(cfg.fov_h_deg)
        idxs = np.arange(1, n_total + 1, dtype=np.int64)
        u = np.fromiter((_halton(int(i), 2) for i in idxs), dtype=np.float64, count=n_total)
        v = np.fromiter((_halton(int(i), 3) for i in idxs), dtype=np.float64, count=n_total)
        # Azimuth wraps around [-pi, pi] (full 360 by default).
        azimuth = (u * h_full) - (h_full / 2.0)
        # Sample elevation uniformly in sin(elev) so polar areas don't get
        # over-sampled relative to equatorial ones.
        sin_e = math.sin(v_min) + v * (math.sin(v_max) - math.sin(v_min))
        elev = np.arcsin(sin_e)
        cos_e = np.cos(elev)
        x = cos_e * np.cos(azimuth)
        y = cos_e * np.sin(azimuth)
        z = sin_e
        self._ray_dirs_sensor = np.stack([x, y, z], axis=1).astype(np.float64)
        self._ray_cursor = 0

    def _next_ray_chunk(self, n: int) -> np.ndarray:
        total = self._ray_dirs_sensor.shape[0]
        end = self._ray_cursor + n
        if end <= total:
            chunk = self._ray_dirs_sensor[self._ray_cursor : end]
            self._ray_cursor = end
            return chunk
        head = self._ray_dirs_sensor[self._ray_cursor :]
        wrap_n = n - head.shape[0]
        tail = self._ray_dirs_sensor[:wrap_n]
        self._ray_cursor = wrap_n
        return np.concatenate([head, tail], axis=0)

    def _on_cmd_vel(self, twist: Twist) -> None:
        with self._cmd_lock:
            self._cmd_vx = float(twist.linear.x)
            self._cmd_vy = float(twist.linear.y)
            self._cmd_wz = float(twist.angular.z)

    def _sim_loop(self) -> None:
        assert self._model is not None and self._data is not None
        cfg = self.config
        physics_dt = cfg.physics_dt
        steps_per_lidar = max(round((1.0 / cfg.lidar_hz) / physics_dt), 1)
        steps_per_odom = max(round((1.0 / cfg.odom_hz) / physics_dt), 1)
        wall_start = time.time()
        sim_step = 0
        next_lidar_step = steps_per_lidar
        next_odom_step = steps_per_odom
        while not self._stop_event.is_set():
            with self._cmd_lock:
                vx_body = self._cmd_vx
                vy_body = self._cmd_vy
                wz = self._cmd_wz
            yaw = float(self._data.qpos[self._yaw_qpos])
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            self._data.ctrl[self._slide_x_act] = vx_body * cy - vy_body * sy
            self._data.ctrl[self._slide_y_act] = vx_body * sy + vy_body * cy
            self._data.ctrl[self._yaw_act] = wz

            mujoco.mj_step(self._model, self._data)
            sim_step += 1

            if sim_step >= next_odom_step:
                self._publish_odom()
                next_odom_step += steps_per_odom
            if sim_step >= next_lidar_step:
                self._publish_lidar()
                next_lidar_step += steps_per_lidar

            target_wall = wall_start + sim_step * physics_dt
            slack = target_wall - time.time()
            if slack > 0:
                self._stop_event.wait(slack)

    def _publish_odom(self) -> None:
        assert self._data is not None
        cfg = self.config
        x = float(self._data.qpos[self._slide_x_qpos])
        y = float(self._data.qpos[self._slide_y_qpos])
        z = float(self._data.xpos[self._robot_body_id][2])
        yaw = float(self._data.qpos[self._yaw_qpos])
        vx_world = float(self._data.qvel[self._slide_x_qpos])
        vy_world = float(self._data.qvel[self._slide_y_qpos])
        wz = float(self._data.qvel[self._yaw_qpos])
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        vx_body = vx_world * cy + vy_world * sy
        vy_body = -vx_world * sy + vy_world * cy
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        self.odom.publish(
            Odometry(
                ts=time.time(),
                frame_id=cfg.odom_frame,
                child_frame_id=cfg.base_frame,
                pose=Pose(
                    position=Vector3(x, y, z),
                    orientation=Quaternion(0.0, 0.0, qz, qw),
                ),
                twist=Twist(
                    linear=Vector3(vx_body, vy_body, 0.0),
                    angular=Vector3(0.0, 0.0, wz),
                ),
            )
        )

    def _publish_lidar(self) -> None:
        assert self._model is not None and self._data is not None
        cfg = self.config
        n = int(cfg.lidar_points_per_scan)
        dirs_sensor = self._next_ray_chunk(n)
        site_pos = self._data.site_xpos[self._lidar_site_id].copy()
        site_mat = self._data.site_xmat[self._lidar_site_id].reshape(3, 3).copy()
        # Sensor-frame unit vectors -> world-frame unit vectors.
        dirs_world = (site_mat @ dirs_sensor.T).T

        dist = np.zeros(n, dtype=np.float64)
        geomid = np.full(n, -1, dtype=np.int32)
        mujoco.mj_multiRay(
            self._model,
            self._data,
            site_pos.astype(np.float64),
            np.ascontiguousarray(dirs_world).ravel(),
            None,
            1,
            self._robot_body_id,
            geomid,
            dist,
            None,
            n,
            cfg.lidar_range_max,
        )

        valid = (geomid >= 0) & (dist >= cfg.lidar_range_min) & (dist <= cfg.lidar_range_max)
        out_frame = cfg.odom_frame if cfg.lidar_in_world_frame else cfg.sensor_frame
        if not np.any(valid):
            self.lidar.publish(
                PointCloud2.from_numpy(
                    np.zeros((0, 3), dtype=np.float32),
                    frame_id=out_frame,
                    timestamp=time.time(),
                )
            )
            return
        points_sensor = dirs_sensor[valid] * dist[valid][:, None]
        if cfg.lidar_in_world_frame:
            points = (site_mat @ points_sensor.T).T + site_pos
        else:
            points = points_sensor
        self.lidar.publish(
            PointCloud2.from_numpy(
                points.astype(np.float32),
                frame_id=out_frame,
                timestamp=time.time(),
            )
        )


__all__ = ["G1MujocoPlanarSim", "G1MujocoPlanarSimConfig"]

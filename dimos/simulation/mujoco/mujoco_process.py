#!/usr/bin/env python3

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

import base64
import json
import os
import pickle
import signal
import sys
import time
from typing import Any

import mujoco
from mujoco import viewer
import numpy as np
from numpy.typing import NDArray

# NOTE: do NOT eagerly import anything that pulls in open3d from
# top-level — open3d's bundled GLFW (libglfw.3.dylib inside the wheel)
# registers a competing set of Cocoa GLFW* classes the moment its
# pybind module loads.  If those classes are registered before mujoco's
# viewer initialises its own GLFW (which happens inside
# viewer.launch_passive), Cocoa monitor enumeration goes sideways and
# launch_passive segfaults in _glfwGetVideoModeCocoa with EXC_BAD_ACCESS
# at offset 0x100 of a NULL _GLFWmonitor pointer.  PointCloud2 and
# depth_camera both import open3d at module scope, so they're imported
# lazily inside _step_once below — by then launch_passive has already
# loaded mujoco's GLFW first.
from dimos.core.global_config import GlobalConfig
from dimos.simulation.mujoco.constants import (
    DEPTH_CAMERA_FOV,
    LIDAR_FPS,
    LIDAR_RESOLUTION,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from dimos.simulation.mujoco.model import load_model, load_scene_xml
from dimos.simulation.mujoco.person_on_track import PersonPositionController
from dimos.simulation.mujoco.shared_memory import ShmReader
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _auto_detect_headless() -> bool:
    """Best-effort guess at whether a GUI viewer can run.

    Linux without ``$DISPLAY`` (or ``$WAYLAND_DISPLAY``) → headless.
    macOS we assume Cocoa is available; users in genuinely headless
    macOS contexts should set ``mujoco_headless=True`` explicitly.
    """
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


class MockController:
    """Controller that reads commands from shared memory.

    Includes a watchdog: if no new command arrives for ``stale_timeout``
    seconds, the cached command zeroes out.  Without this, a single
    transient cmd_vel publish (e.g. one stray twist from the dimos-viewer
    websocket on connect) would persist forever — the SHM cmd buffer
    only changes when somebody calls ``MujocoConnection.move()`` again,
    and most twist senders send the moving command once and never
    follow up with an explicit zero.  In kinematic-robot mode that
    surfaces as the robot spinning indefinitely.
    """

    _STALE_TIMEOUT = 0.5  # seconds — half a cmd_vel publish period @ 10 Hz

    def __init__(self, shm_interface: ShmReader) -> None:
        self.shm = shm_interface
        self._command = np.zeros(3, dtype=np.float32)
        self._last_update = 0.0

    def get_command(self) -> NDArray[Any]:
        """Get the current movement command."""
        now = time.time()
        cmd_data = self.shm.read_command()
        if cmd_data is not None:
            linear, angular = cmd_data
            # MuJoCo expects [forward, lateral, rotational]
            self._command[0] = linear[0]  # forward/backward
            self._command[1] = linear[1]  # left/right
            self._command[2] = angular[2]  # rotation
            self._last_update = now
        elif self._last_update and now - self._last_update > self._STALE_TIMEOUT:
            # No fresh command for a while — fall back to "stop" so the
            # kinematic robot doesn't spin forever on a stale wz.
            self._command[:] = 0.0
        result: NDArray[Any] = self._command.copy()
        return result

    def stop(self) -> None:
        """Stop method to satisfy InputController protocol."""
        pass


def _run_simulation(config: GlobalConfig, shm: ShmReader) -> None:
    robot_name = config.robot_model or "unitree_go1"
    if robot_name == "unitree_go2":
        robot_name = "unitree_go1"

    controller = MockController(shm)
    model, data = load_model(
        controller,
        robot=robot_name,
        scene_xml=load_scene_xml(config),
        mujoco_room=config.mujoco_room,
        kinematic_robot=config.mujoco_kinematic_robot,
    )

    if model is None or data is None:
        raise ValueError("Failed to load MuJoCo model: model or data is None")

    match robot_name:
        case "unitree_go1":
            z = 0.3
        case "unitree_g1":
            z = 0.8
        case _:
            z = 0

    pos = config.mujoco_start_pos_float

    data.qpos[0:3] = [pos[0], pos[1], z]

    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    lidar_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_front_camera")

    person_position_controller = PersonPositionController(model)

    lidar_left_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_left_camera")
    lidar_right_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_right_camera"
    )

    shm.signal_ready()

    camera_size = (VIDEO_WIDTH, VIDEO_HEIGHT)

    # Create renderers (offscreen — these don't need a viewer window)
    rgb_renderer = mujoco.Renderer(model, height=camera_size[1], width=camera_size[0])
    depth_renderer = mujoco.Renderer(model, height=camera_size[1], width=camera_size[0])
    depth_renderer.enable_depth_rendering()

    depth_left_renderer = mujoco.Renderer(model, height=camera_size[1], width=camera_size[0])
    depth_left_renderer.enable_depth_rendering()

    depth_right_renderer = mujoco.Renderer(model, height=camera_size[1], width=camera_size[0])
    depth_right_renderer.enable_depth_rendering()

    scene_option = mujoco.MjvOption()

    # Timing control
    last_video_time = 0.0
    last_lidar_time = 0.0
    video_interval = 1.0 / VIDEO_FPS
    lidar_interval = 1.0 / LIDAR_FPS

    # In kinematic mode we run mj_forward only (no physics on the
    # robot), so we cache the home joint pose + spawn z and write them
    # back every tick to keep the robot upright and at floor height.
    home_qpos_robot = np.array(data.qpos[7 : 7 + int(model.nu)]).copy()
    spawn_base_z = float(data.qpos[2])
    _kin_tick_counter = [0]
    _kin_last_log = [0.0]
    # Forward margin on the wall ray: just enough that the body's
    # centre stops before puncturing the wall surface.
    KIN_WALL_MARGIN = 0.2
    # Ray height: cast at floor level rather than the body centre
    # (z≈0.8).  At chest height the ray starts INSIDE the robot's own
    # waist/chest collision geom and exits at ~0.01 m, blocking every
    # tick.  At floor level the ray is below the chest geom; the legs/
    # feet are at the body's lateral offsets (±0.089 m), so a forward
    # ray cast from the body centre's xy doesn't intersect them either.
    KIN_RAY_Z = 0.05
    _kin_geomid_out = np.zeros(1, dtype=np.int32)

    def _step_once(m_viewer: Any) -> None:
        nonlocal last_video_time, last_lidar_time
        step_start = time.time()

        if config.mujoco_kinematic_robot:
            # No physics on the robot — we move qpos directly, then
            # mj_forward updates kinematics so cameras + sensors track
            # the new pose.  Wall collision is handled by a single
            # mj_ray cast in the proposed travel direction; if it hits
            # a static geom (walls, furniture) within step + margin,
            # we cancel the translation for this tick.
            cmd = controller.get_command()  # (vx, vy, wz) in robot frame
            vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
            dt = float(model.opt.timestep) * float(config.mujoco_steps_per_frame)

            qw, qz = float(data.qpos[3]), float(data.qpos[6])
            yaw = np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
            cyaw, syaw = float(np.cos(yaw)), float(np.sin(yaw))

            dx = dt * (vx * cyaw - vy * syaw)
            dy = dt * (vx * syaw + vy * cyaw)

            move_mag = float(np.hypot(dx, dy))
            blocked = False
            hit_dist = -1.0
            if move_mag > 0.0:
                pnt = np.array(
                    [float(data.qpos[0]), float(data.qpos[1]), KIN_RAY_Z],
                    dtype=np.float64,
                )
                vec = np.array([dx / move_mag, dy / move_mag, 0.0], dtype=np.float64)
                # mj_ray with flg_static=1 includes all geoms (static +
                # dynamic).  We can't simply switch to flg_static=0 — that
                # would skip wall geoms attached to room bodies.  Instead,
                # casting at floor level (KIN_RAY_Z) keeps the ray below
                # the robot's chest/waist geoms, so the ray starts in
                # free space and properly hits walls.
                hit_dist = float(mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, _kin_geomid_out))
                if 0.0 < hit_dist < move_mag + KIN_WALL_MARGIN:
                    blocked = True

            if not blocked:
                data.qpos[0] += dx
                data.qpos[1] += dy
            data.qpos[2] = spawn_base_z
            yaw_after = yaw + wz * dt
            data.qpos[3] = float(np.cos(yaw_after * 0.5))
            data.qpos[4] = 0.0
            data.qpos[5] = 0.0
            data.qpos[6] = float(np.sin(yaw_after * 0.5))
            data.qpos[7 : 7 + int(model.nu)] = home_qpos_robot
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)

            _kin_tick_counter[0] += 1
            now = time.time()
            if now - _kin_last_log[0] > 1.0:
                print(
                    f"kinematic tick={_kin_tick_counter[0]} cmd=({vx:.2f},{vy:.2f},{wz:.2f}) "
                    f"pos=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f}) "
                    f"yaw={yaw_after:.2f} blocked={blocked} hit_dist={hit_dist:.2f} "
                    f"geom={int(_kin_geomid_out[0])}",
                    flush=True,
                )
                _kin_last_log[0] = now
        else:
            # Step simulation
            for _ in range(config.mujoco_steps_per_frame):
                mujoco.mj_step(model, data)

        person_position_controller.tick(data)

        if m_viewer is not None:
            m_viewer.sync()

        # Always update odometry
        pos = data.qpos[0:3].copy()
        quat = data.qpos[3:7].copy()  # (w, x, y, z)
        shm.write_odom(pos, quat, time.time())

        current_time = time.time()

        # Video rendering
        if current_time - last_video_time >= video_interval:
            rgb_renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
            pixels = rgb_renderer.render()
            shm.write_video(pixels)
            last_video_time = current_time

        # Lidar/depth rendering
        if current_time - last_lidar_time >= lidar_interval:
            # Render all depth cameras
            depth_renderer.update_scene(data, camera=lidar_camera_id, scene_option=scene_option)
            depth_front = depth_renderer.render()

            depth_left_renderer.update_scene(
                data, camera=lidar_left_camera_id, scene_option=scene_option
            )
            depth_left = depth_left_renderer.render()

            depth_right_renderer.update_scene(
                data, camera=lidar_right_camera_id, scene_option=scene_option
            )
            depth_right = depth_right_renderer.render()

            shm.write_depth(depth_front, depth_left, depth_right)

            # Process depth images into lidar message
            all_points = []
            cameras_data = [
                (
                    depth_front,
                    data.cam_xpos[lidar_camera_id],
                    data.cam_xmat[lidar_camera_id].reshape(3, 3),
                ),
                (
                    depth_left,
                    data.cam_xpos[lidar_left_camera_id],
                    data.cam_xmat[lidar_left_camera_id].reshape(3, 3),
                ),
                (
                    depth_right,
                    data.cam_xpos[lidar_right_camera_id],
                    data.cam_xmat[lidar_right_camera_id].reshape(3, 3),
                ),
            ]

            # Lazy import — pulls in open3d.  See top-of-file note.
            from dimos.simulation.mujoco.depth_camera import depth_image_to_point_cloud

            for depth_image, camera_pos, camera_mat in cameras_data:
                points = depth_image_to_point_cloud(
                    depth_image, camera_pos, camera_mat, fov_degrees=DEPTH_CAMERA_FOV
                )
                if points.size > 0:
                    all_points.append(points)

            if all_points:
                # Lazy imports — both pull in open3d.  See top-of-file note.
                import open3d as o3d  # type: ignore[import-untyped]

                from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

                combined_points = np.vstack(all_points)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(combined_points)
                pcd = pcd.voxel_down_sample(voxel_size=LIDAR_RESOLUTION)

                lidar_msg = PointCloud2(
                    pointcloud=pcd,
                    ts=time.time(),
                    frame_id="world",
                )
                shm.write_lidar(lidar_msg)

            last_lidar_time = current_time

        # Control simulation speed.  Per-iter physics advance is
        # `timestep × steps_per_frame` (kinematic integrates that span
        # directly, policy mode runs that many mj_step calls), so the
        # wall-clock target is the same span — gives 1× real time and
        # stops the kinematic loop from spinning at 500 Hz.
        target_period = model.opt.timestep * float(config.mujoco_steps_per_frame)
        time_until_next_step = target_period - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    headless = (
        config.mujoco_headless if config.mujoco_headless is not None else _auto_detect_headless()
    )
    try:
        if headless:
            # Offscreen mode: no GUI viewer.  Required for CI, headless
            # macOS, and any Linux box without a display.
            while not shm.should_stop():
                _step_once(m_viewer=None)
        else:
            with viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            ) as m_viewer:
                m_viewer.cam.lookat = config.mujoco_camera_position_float[0:3]
                m_viewer.cam.distance = config.mujoco_camera_position_float[3]
                m_viewer.cam.azimuth = config.mujoco_camera_position_float[4]
                m_viewer.cam.elevation = config.mujoco_camera_position_float[5]

                while m_viewer.is_running() and not shm.should_stop():
                    _step_once(m_viewer=m_viewer)
    finally:
        person_position_controller.stop()


if __name__ == "__main__":
    # Print early so the parent's [mujoco]-log pump immediately shows
    # this subprocess is alive — useful when debugging "sim crashes
    # silently" reports.  flush=True because we don't yet know if
    # PYTHONUNBUFFERED is honored on the launcher (mjpython on macOS).
    print(f"mujoco_process: starting (pid={os.getpid()}, sys={sys.platform})", flush=True)

    global_config = pickle.loads(base64.b64decode(sys.argv[1]))
    shm_names = json.loads(sys.argv[2])

    shm = ShmReader(shm_names)

    def signal_handler(_signum: int, _frame: Any) -> None:
        # Signal the main loop to exit gracefully so the viewer context
        # manager can close the window and clean up resources.
        print(f"mujoco_process: signal {_signum} received, requesting stop", flush=True)
        shm.signal_stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("mujoco_process: entering _run_simulation", flush=True)
        _run_simulation(global_config, shm)
    except BaseException as e:
        # Surface any uncaught exception (incl. KeyboardInterrupt and
        # SystemExit which BaseException catches) so the parent sees
        # *why* the subprocess died instead of just an exit code.
        import traceback

        print(
            f"mujoco_process: ERROR in _run_simulation: {type(e).__name__}: {e}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        print("mujoco_process: cleanup", flush=True)
        shm.cleanup()

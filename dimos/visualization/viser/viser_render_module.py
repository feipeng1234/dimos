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

"""Viser-based 3D viewer module for dimos.

Streams a Gaussian splat scene + the robot (MJCF meshes, FK from
``/coordinator/joint_state`` + ``/odom``) into a browser at
http://localhost:<port>/.

This is render-only — the viewer subscribes to existing LCM topics and
does not feed back into the control path.  Teleop continues to come
from the existing command-center dashboard.
"""

from __future__ import annotations

from pathlib import Path as FilePath
import threading
import time
from typing import Any

import mujoco
import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Path import Path as PathMsg
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger
from dimos.visualization.viser.camera import CameraSpec, g1_d435_default, world_pose
from dimos.visualization.viser.robot_meshes import (
    RobotMeshes,
    apply_state,
    dimos_joint_to_mjcf,
    load_robot_meshes,
)
from dimos.visualization.viser.scene_editor import SceneEditor
from dimos.visualization.viser.splat import SplatAlignment, load_splat

logger = setup_logger()


class ViserRenderModule(Module):
    """Viser viewer that overlays the live robot on a Gaussian splat.

    Inputs:
        joint_state: per-joint q values from the coordinator.
        odom: base pose from the sim (or future real-hw) adapter.
    """

    joint_state: In[JointState]
    odom: In[PoseStamped]
    path: In[PathMsg]
    # Optional lidar pointcloud overlay.  Subscribers that don't have a
    # /lidar publisher in their stack (e.g. real-Go2 builds going through
    # WebRTC) will simply never receive a message and the overlay stays empty.
    lidar: In[PointCloud2]
    clicked_point: Out[PointStamped]

    def __init__(
        self,
        splat_path: str | FilePath | None,
        mjcf_path: str | FilePath,
        *,
        port: int = 8082,
        alignment_yaml: str | FilePath | None = None,
        render_hz: float = 30.0,
        camera_spec: CameraSpec | None = None,
        scene_mesh_path: str | FilePath | None = None,
        scene_mesh_scale: float = 1.0,
        scene_mesh_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scene_mesh_rotation_zyx_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scene_mesh_y_up: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Empty / None splat_path means "no splat in the viewer" — useful when
        # the world is provided as a mesh instead (DIMOS_SCENE_MESH_PATH).
        self._splat_path = FilePath(splat_path) if splat_path else None
        self._mjcf_path = FilePath(mjcf_path)
        self._alignment_yaml = FilePath(alignment_yaml) if alignment_yaml else None
        self._port = port
        self._render_dt = 1.0 / float(render_hz)
        self._camera_spec = camera_spec if camera_spec is not None else g1_d435_default()
        self._scene_mesh_path = FilePath(scene_mesh_path) if scene_mesh_path else None
        self._scene_mesh_scale = scene_mesh_scale
        self._scene_mesh_translation = scene_mesh_translation
        self._scene_mesh_rotation_zyx_deg = scene_mesh_rotation_zyx_deg
        self._scene_mesh_y_up = scene_mesh_y_up

        # viser handles for view-mode toggle
        self._splat_handle: Any = None
        self._scene_mesh_handle: Any = None

        # Mutable shared state — written from In subscribers, read from
        # the render loop.  Plain dict + lock; values are lightweight.
        self._state_lock = threading.Lock()
        self._latest_joints: dict[str, float] = {}
        self._latest_base_pos: np.ndarray | None = None
        self._latest_base_wxyz: np.ndarray | None = None

        self._server: Any = None  # viser.ViserServer
        self._body_frames: dict[int, Any] = {}  # body_id -> viser frame handle
        self._camera_body_id: int | None = None
        self._camera_frustum: Any = None  # viser frustum handle
        self._robot: RobotMeshes | None = None
        self._render_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._path_handle: Any = None
        # Lidar overlay state.  GUI checkbox toggles visibility; the
        # handle holds the most recently uploaded scene element so we
        # can replace it on the next /lidar message instead of
        # accumulating cloud-on-cloud.  `_lidar_visible` is read by
        # the In subscriber so toggling off skips the upload entirely.
        self._lidar_handle: Any = None
        self._lidar_visible: bool = True
        self._lidar_checkbox: Any = None

    @rpc
    def start(self) -> None:
        super().start()

        import viser

        alignment = (
            SplatAlignment.from_yaml(self._alignment_yaml)
            if self._alignment_yaml and self._alignment_yaml.exists()
            else SplatAlignment()
        )

        if self._splat_path is not None:
            logger.info(f"Viser: loading splat from {self._splat_path}")
            splat = load_splat(self._splat_path, alignment=alignment)
            logger.info(f"Viser: loaded {len(splat.centers)} Gaussians")
        else:
            splat = None
            logger.info("Viser: splat disabled (no splat_path provided)")

        logger.info(f"Viser: loading robot meshes from {self._mjcf_path}")
        from dimos.simulation.mujoco.model import get_assets

        self._robot = load_robot_meshes(self._mjcf_path, assets=get_assets())
        logger.info(
            f"Viser: {len(self._robot.geoms)} visual meshes across "
            f"{len(self._robot.body_names)} bodies"
        )

        self._server = viser.ViserServer(host="0.0.0.0", port=self._port)
        # Strip the floating control panel down to just a collapse button —
        # the viewer is render-only, no GUI controls live in the panel, and
        # viser exposes no API to hide the panel entirely.
        self._server.gui.set_panel_label(None)
        self._server.gui.configure_theme(
            control_layout="collapsible",
            show_logo=False,
            show_share_button=False,
            dark_mode=True,
        )
        logger.info(f"Viser viewer: http://localhost:{self._port}/")

        if splat is not None:
            self._splat_handle = self._server.scene.add_gaussian_splats(
                "/splat",
                centers=splat.centers,
                covariances=splat.covariances,
                rgbs=splat.rgbs,
                opacities=splat.opacities,
            )

        # Optional scene mesh (.usdz / .glb / etc.) — drawn in the same
        # world frame as the robot.  ``MeshCameraModule`` ray-casts the
        # same mesh to feed the head-camera RGB topic.
        if self._scene_mesh_path is not None and self._scene_mesh_path.exists():
            from dimos.mapping.mesh_scene import (
                SceneMeshAlignment,
                load_scene_mesh,
            )

            try:
                mesh_alignment = SceneMeshAlignment(
                    scale=self._scene_mesh_scale,
                    rotation_zyx_deg=self._scene_mesh_rotation_zyx_deg,
                    translation=self._scene_mesh_translation,
                    y_up=self._scene_mesh_y_up,
                )
                logger.info(f"Viser: loading scene mesh {self._scene_mesh_path}")
                scene_mesh = load_scene_mesh(self._scene_mesh_path, alignment=mesh_alignment)
                vertices = np.asarray(scene_mesh.vertices, dtype=np.float32)
                faces = np.asarray(scene_mesh.triangles, dtype=np.int32)
                # Forward per-vertex colors when the loader extracted them
                # (USD ``displayColor`` primvar or material ``diffuseColor``).
                # ``add_mesh_simple`` only accepts a single color in this
                # viser build, so for the colored path we go through
                # ``add_mesh_trimesh`` which preserves per-vertex visual data.
                vertex_colors_raw = (
                    np.asarray(scene_mesh.vertex_colors) if scene_mesh.has_vertex_colors() else None
                )
                if vertex_colors_raw is not None and len(vertex_colors_raw) == len(vertices):
                    import trimesh

                    rgba = np.empty((len(vertices), 4), dtype=np.uint8)
                    rgba[:, :3] = (np.clip(vertex_colors_raw, 0.0, 1.0) * 255.0).astype(np.uint8)
                    rgba[:, 3] = 255
                    tm = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        vertex_colors=rgba,
                        process=False,
                    )
                    self._scene_mesh_handle = self._server.scene.add_mesh_trimesh(
                        "/scene_mesh", mesh=tm
                    )
                    color_msg = "with per-vertex colors"
                else:
                    self._scene_mesh_handle = self._server.scene.add_mesh_simple(
                        "/scene_mesh",
                        vertices=vertices,
                        faces=faces,
                        color=(180, 180, 180),
                        opacity=1.0,
                    )
                    color_msg = "no vertex colors found, falling back to grey"
                logger.info(
                    f"Viser: scene mesh added "
                    f"({len(vertices)} verts, {len(faces)} tris, {color_msg})"
                )
                # Frame each connecting client on the mesh's bounding box so
                # the user lands looking at the scene rather than at viser's
                # default camera (which sits at the origin and ends up
                # *inside* a 6m × 12m × 3m room).
                bbox_min = vertices.min(axis=0)
                bbox_max = vertices.max(axis=0)
                center = (bbox_min + bbox_max) * 0.5
                extent = float(np.linalg.norm(bbox_max - bbox_min))
                cam_pos = center + np.array(
                    [extent * 0.6, -extent * 0.6, extent * 0.4],
                    dtype=np.float32,
                )

                @self._server.on_client_connect
                def _frame_camera_on_mesh(client: Any) -> None:
                    client.camera.position = tuple(float(x) for x in cam_pos)
                    client.camera.look_at = tuple(float(x) for x in center)
            except Exception as e:
                logger.warning(f"Viser: scene mesh load failed: {e}")

        # View-mode toggle in the GUI — only shown when *both* a splat
        # and a scene mesh are loaded (so the user has something to
        # switch between).  In every other case we just show whatever
        # backdrop got loaded; no dropdown clutter.
        if self._splat_handle is not None and self._scene_mesh_handle is not None:
            view_mode_dropdown = self._server.gui.add_dropdown(
                "View mode", ["Splat", "Mesh"], initial_value="Mesh"
            )

            def _apply_view_mode(mode: str) -> None:
                if self._splat_handle is not None:
                    self._splat_handle.visible = mode == "Splat"
                if self._scene_mesh_handle is not None:
                    self._scene_mesh_handle.visible = mode == "Mesh"

            _apply_view_mode("Mesh")

        # Lidar overlay toggle.  Shown unconditionally — when no /lidar
        # publisher is connected the cloud just stays empty, but the
        # checkbox makes it discoverable that the overlay exists.
        self._lidar_checkbox = self._server.gui.add_checkbox(
            "Show lidar pointcloud", initial_value=self._lidar_visible
        )

        @self._lidar_checkbox.on_update
        def _on_lidar_toggle(_: Any) -> None:
            self._lidar_visible = bool(self._lidar_checkbox.value)
            if self._lidar_handle is not None:
                self._lidar_handle.visible = self._lidar_visible

            @view_mode_dropdown.on_update
            def _on_view_mode(_event: Any) -> None:
                _apply_view_mode(view_mode_dropdown.value)

        # One frame per body; meshes are added as children so they
        # follow when the body frame moves.
        for body_id, body_name in enumerate(self._robot.body_names):
            self._body_frames[body_id] = self._server.scene.add_frame(
                f"/robot/{body_name}",
                show_axes=False,
            )
        for i, geom in enumerate(self._robot.geoms):
            color_rgb = (
                int(geom.rgba[0] * 255),
                int(geom.rgba[1] * 255),
                int(geom.rgba[2] * 255),
            )
            self._server.scene.add_mesh_simple(
                f"/robot/{geom.body_name}/geom_{i}",
                vertices=geom.vertices,
                faces=geom.faces,
                color=color_rgb,
                opacity=float(geom.rgba[3]) if geom.rgba[3] > 0 else 1.0,
                position=tuple(geom.local_pos),
                wxyz=tuple(geom.local_wxyz),
            )

        # Camera frustum overlay — shows where a robot-mounted RGB sensor
        # would look from.  Stays None if the configured mount body
        # isn't in this MJCF (e.g. swap to a robot without head_link).
        cam_body_id = mujoco.mj_name2id(
            self._robot.model, mujoco.mjtObj.mjOBJ_BODY, self._camera_spec.body_name
        )
        if cam_body_id < 0:
            logger.warning(
                f"Viser: camera mount body '{self._camera_spec.body_name}' not in MJCF; "
                "frustum overlay disabled"
            )
        else:
            self._camera_body_id = cam_body_id
            self._camera_frustum = self._server.scene.add_camera_frustum(
                "/robot/_camera_frustum",
                fov=float(np.radians(self._camera_spec.vfov_deg)),
                aspect=float(self._camera_spec.aspect),
                scale=float(self._camera_spec.frustum_scale),
                color=self._camera_spec.frustum_color,
            )

        # In-viewer scene editor.  Spawns boxes / planes the user can
        # drag with transform-control gizmos; "Export OBJ" writes them
        # to data/mujoco_sim/dimos_office_edited.obj for hand-off into
        # the MJCF.
        self._scene_editor = SceneEditor(server=self._server)
        self._scene_editor.attach()

        # Click-to-navigate. We arm a one-shot scene click callback when
        # the user presses "Set nav goal", because viser disables camera
        # orbit while the click callback is registered (App.tsx:514) — so
        # leaving it always-on would break LMB orbit globally.
        nav_goal_button = self._server.gui.add_button("Set nav goal")

        @nav_goal_button.on_click
        def _arm_nav_goal_click(_event: Any) -> None:
            nav_goal_button.disabled = True
            nav_goal_button.label = "Click on floor..."

            @self._server.scene.on_pointer_event(event_type="click")
            def _on_floor_click(event: Any) -> None:
                try:
                    self._handle_floor_click(event)
                finally:
                    self._server.scene.remove_pointer_callback()

            @self._server.scene.on_pointer_callback_removed
            def _rearm_button() -> None:
                nav_goal_button.disabled = False
                nav_goal_button.label = "Set nav goal"

        try:
            unsub = self.path.subscribe(self._on_path)
            self.register_disposable(Disposable(unsub))
        except Exception as e:
            logger.warning(f"Viser: path subscribe failed: {e}")

        try:
            unsub = self.lidar.subscribe(self._on_lidar)
            self.register_disposable(Disposable(unsub))
        except Exception as e:
            logger.warning(f"Viser: lidar subscribe failed: {e}")

        try:
            unsub = self.joint_state.subscribe(self._on_joint_state)
            self.register_disposable(Disposable(unsub))
        except Exception as e:
            logger.warning(f"Viser: joint_state subscribe failed: {e}")

        try:
            unsub = self.odom.subscribe(self._on_odom)
            self.register_disposable(Disposable(unsub))
        except Exception as e:
            logger.warning(f"Viser: odom subscribe failed: {e}")

        self._render_thread = threading.Thread(
            target=self._render_loop, name="viser-render", daemon=True
        )
        self._render_thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
        super().stop()

    def _handle_floor_click(self, event: Any) -> None:
        """Project the click ray onto the z=0 floor and publish a goal."""
        ray_origin = event.ray_origin
        ray_direction = event.ray_direction
        if ray_origin is None or ray_direction is None:
            return

        ox, oy, oz = ray_origin
        dx, dy, dz = ray_direction
        if abs(dz) < 1e-6:
            logger.info("Viser nav-goal: click ray is parallel to floor, ignoring")
            return
        t = -oz / dz
        if t <= 0:
            logger.info("Viser nav-goal: click is above the horizon, ignoring")
            return
        x = ox + t * dx
        y = oy + t * dy

        marker_color = (0, 200, 255)
        try:
            self._server.scene.add_icosphere(
                "/nav_goal_marker",
                radius=0.08,
                position=(float(x), float(y), 0.05),
                color=marker_color,
            )
        except Exception as e:
            logger.debug(f"Viser nav-goal marker failed: {e}")

        point = PointStamped(x=float(x), y=float(y), z=0.0, ts=time.time(), frame_id="map")
        self.clicked_point.publish(point)
        logger.info(f"Viser nav-goal: published clicked_point=({x:.3f}, {y:.3f})")

    def _on_path(self, msg: PathMsg) -> None:
        """Draw the planner's path as a polyline floating above the floor."""
        poses = msg.poses
        if len(poses) < 2:
            handle = self._path_handle
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
                self._path_handle = None
            return

        path_height = 0.10  # lift above floor so it doesn't z-fight with the splat
        pts = np.array(
            [[p.position.x, p.position.y, path_height] for p in poses],
            dtype=np.float32,
        )
        # add_line_segments wants (N, 2, 3): start/end of each segment.
        segments = np.stack([pts[:-1], pts[1:]], axis=1)

        try:
            self._path_handle = self._server.scene.add_line_segments(
                "/nav_path",
                points=segments,
                colors=(255, 30, 30),
                line_width=4.0,
            )
        except Exception as e:
            logger.debug(f"Viser nav-path render failed: {e}")

    def _on_lidar(self, msg: PointCloud2) -> None:
        """Replace the lidar overlay in viser with the latest pointcloud.

        The publisher hands us an ``open3d`` PointCloud whose points are
        already in the world frame (this is what ``VoxelGridMapper``
        consumes too — see its docstring).  We pass the (N, 3) array to
        viser's ``add_point_cloud``; the previous handle is overwritten
        in-place so we don't accumulate cloud-on-cloud across frames.
        """
        if not self._lidar_visible or self._server is None:
            return
        try:
            pcd = msg.pointcloud
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.size == 0:
                return
            # Color choice: subtle teal so the cloud is visible against
            # both the dark mesh background and the light splat.  Per-point
            # colors would be nicer but the lidar source doesn't carry
            # them and viser's add_point_cloud accepts a single uniform
            # color that broadcasts across all points.
            colors = np.broadcast_to(np.array([0, 200, 255], dtype=np.uint8), pts.shape).copy()
            self._lidar_handle = self._server.scene.add_point_cloud(
                "/lidar_overlay",
                points=pts,
                colors=colors,
                point_size=0.02,
            )
            self._lidar_handle.visible = self._lidar_visible
        except Exception as e:
            logger.debug(f"Viser lidar overlay update failed: {e}")

    def _on_joint_state(self, msg: JointState) -> None:
        names = list(msg.name)
        positions = list(msg.position)
        if not names or len(names) != len(positions):
            return
        with self._state_lock:
            for n, q in zip(names, positions, strict=False):
                self._latest_joints[dimos_joint_to_mjcf(n)] = float(q)

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._state_lock:
            self._latest_base_pos = np.array(
                [msg.position.x, msg.position.y, msg.position.z],
                dtype=np.float64,
            )
            # PoseStamped quaternion is (x, y, z, w); MuJoCo / Viser want (w, x, y, z).
            self._latest_base_wxyz = np.array(
                [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z],
                dtype=np.float64,
            )

    def _render_loop(self) -> None:
        assert self._robot is not None
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            with self._state_lock:
                joints = dict(self._latest_joints)
                base_pos = None if self._latest_base_pos is None else self._latest_base_pos.copy()
                base_wxyz = (
                    None if self._latest_base_wxyz is None else self._latest_base_wxyz.copy()
                )

            try:
                apply_state(
                    self._robot,
                    base_pos=base_pos,
                    base_wxyz=base_wxyz,
                    joint_positions=joints,
                )
                self._update_camera_frustum()
                xpos = self._robot.data.xpos
                xquat = self._robot.data.xquat
                for body_id, frame in self._body_frames.items():
                    frame.position = tuple(float(x) for x in xpos[body_id])
                    frame.wxyz = tuple(float(x) for x in xquat[body_id])
            except Exception as e:
                logger.debug(f"Viser render tick failed: {e}")

            next_tick += self._render_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def _update_camera_frustum(self) -> None:
        """Place the camera frustum at the current pose of its mount body."""
        if self._camera_frustum is None or self._camera_body_id is None:
            return
        assert self._robot is not None
        body_pos = self._robot.data.xpos[self._camera_body_id]
        body_wxyz = self._robot.data.xquat[self._camera_body_id]
        cam_pos, cam_wxyz = world_pose(body_pos, body_wxyz, self._camera_spec)
        self._camera_frustum.position = tuple(float(x) for x in cam_pos)
        self._camera_frustum.wxyz = tuple(float(x) for x in cam_wxyz)


viser_render = ViserRenderModule.blueprint

__all__ = ["ViserRenderModule", "viser_render"]

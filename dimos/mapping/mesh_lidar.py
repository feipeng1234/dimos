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

"""Mesh-backed simulated lidar publisher.

Generates ``/lidar`` ``PointCloud2`` messages by ray-casting a 360 deg
horizontal x configurable vertical fan against a static scene mesh
loaded from ``.usdz`` / ``.glb`` / etc.  Replaces ``MujocoSimModule``'s
depth-render-based lidar on platforms where that path doesn't work
(e.g. macOS — see ``StaticCostmapModule`` for context).

Subscribes to ``/odom`` to know where the robot is each tick.  The
ray-cast itself runs on Open3D's CPU ``RaycastingScene`` — vectorised
C++ over a BVH, no GPU.  At 1024 rays it's well under 10 ms per scan
on M-series.

Designed to feed the existing ``VoxelGridMapper → CostMapper`` path,
so click-to-nav planning gets a real costmap derived from real
geometry, not the all-free placeholder.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.mesh_scene import (
    SceneMeshAlignment,
    load_scene_mesh,
    make_raycasting_scene,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MeshLidarConfig(ModuleConfig):
    """Configuration for ``MeshLidarModule``."""

    scene_path: str = ""
    """Path to a ``.usdz`` / ``.glb`` / ``.obj`` / ``.ply`` / ``.stl``
    scene mesh.  Empty disables the publisher."""

    scene_scale: float = 1.0
    """Multiplicative scale applied at load time.  Use 0.01 for sources
    in centimeters (Sketchfab default), 0.0254 for inches."""

    scene_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame offset applied after scaling + rotation."""

    scene_rotation_zyx_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Yaw / pitch / roll in degrees, applied after the y_up swap."""

    scene_y_up: bool = True
    """Flip Y-up to Z-up to match dimos world convention."""

    sensor_height_m: float = 1.0
    """Lidar sensor height above ``/odom`` z, in meters."""

    n_rays_horizontal: int = 1024
    """Azimuthal samples per scan."""

    vfov_deg: float = 30.0
    """Vertical field-of-view in degrees (centred at horizon)."""

    n_rays_vertical: int = 16
    """Elevation samples per scan."""

    max_range_m: float = 30.0
    """Hits beyond this range are dropped from the published cloud."""

    publish_hz: float = 5.0
    """How often to ray-cast and publish."""

    frame_id: str = "world"


class MeshLidarModule(Module):
    """Ray-cast a static scene mesh to produce a simulated ``/lidar`` cloud."""

    config: MeshLidarConfig = Field(default_factory=MeshLidarConfig)
    odom: In[PoseStamped]
    pointcloud: Out[PointCloud2]

    @rpc
    def start(self) -> None:
        super().start()

        cfg = self.config
        if not cfg.scene_path:
            logger.info("MeshLidarModule: scene_path empty, publisher disabled")
            return

        path = Path(cfg.scene_path).expanduser()
        alignment = SceneMeshAlignment(
            scale=cfg.scene_scale,
            rotation_zyx_deg=cfg.scene_rotation_zyx_deg,
            translation=cfg.scene_translation,
            y_up=cfg.scene_y_up,
        )
        logger.info(f"MeshLidarModule: loading scene mesh {path}")
        self._mesh = load_scene_mesh(path, alignment=alignment)
        bbox = self._mesh.get_axis_aligned_bounding_box()
        logger.info(
            f"MeshLidarModule: loaded mesh, "
            f"{len(self._mesh.vertices)} verts, "
            f"{len(self._mesh.triangles)} tris, "
            f"bbox min={np.asarray(bbox.min_bound).round(2)}, "
            f"max={np.asarray(bbox.max_bound).round(2)}"
        )

        self._scene = make_raycasting_scene(self._mesh)
        self._ray_dirs = self._build_ray_directions()
        self._latest_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._pose_lock = threading.Lock()
        self._stop_event = threading.Event()

        try:
            unsub = self.odom.subscribe(self._on_odom)
            from reactivex.disposable import Disposable

            self.register_disposable(Disposable(unsub))
        except Exception as e:
            logger.warning(f"MeshLidarModule: odom subscribe failed: {e}")

        self._thread = threading.Thread(
            target=self._publish_loop,
            name="mesh-lidar-publisher",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            f"MeshLidarModule publishing /lidar at {cfg.publish_hz} Hz "
            f"({cfg.n_rays_horizontal}x{cfg.n_rays_vertical} rays, "
            f"max_range={cfg.max_range_m}m)"
        )

    @rpc
    def stop(self) -> None:
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
            if hasattr(self, "_thread") and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        super().stop()

    def _build_ray_directions(self) -> np.ndarray:
        """Construct a (n_h, n_v, 3) array of unit ray directions in world frame.

        Body-X-forward convention: rays sweep azimuth around body Z (yaw),
        elevation tips up/down within ``vfov_deg``.  We rotate the rays
        per-tick by the robot's yaw to face wherever the robot points.
        """
        cfg = self.config
        az = np.linspace(0.0, 2.0 * np.pi, cfg.n_rays_horizontal, endpoint=False)
        if cfg.n_rays_vertical == 1:
            el = np.array([0.0])
        else:
            half = np.radians(cfg.vfov_deg / 2.0)
            el = np.linspace(-half, half, cfg.n_rays_vertical)
        # (n_h, n_v): meshgrid in the natural lidar fan order.
        az_g, el_g = np.meshgrid(az, el, indexing="xy")
        cx = np.cos(el_g) * np.cos(az_g)
        cy = np.cos(el_g) * np.sin(az_g)
        cz = np.sin(el_g)
        return np.stack([cx, cy, cz], axis=-1).astype(np.float32)

    def _on_odom(self, msg: PoseStamped) -> None:
        # Robot odom is usually (x, y, theta) for ground robots; we
        # only really care about the planar position + yaw for the
        # lidar fan.  Extract them here so the publish loop is fast.
        with self._pose_lock:
            pos = np.array(
                [msg.position.x, msg.position.y, msg.position.z],
                dtype=np.float32,
            )
            wxyz = np.array(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ],
                dtype=np.float64,
            )
            self._latest_pose = (pos, wxyz)

    def _wxyz_to_yaw(self, wxyz: np.ndarray) -> float:
        # Standard wxyz quaternion -> yaw (rotation about Z).
        w, x, y, z = wxyz
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _publish_loop(self) -> None:
        import open3d.core as o3c

        cfg = self.config
        period = 1.0 / cfg.publish_hz if cfg.publish_hz > 0 else 1.0
        while not self._stop_event.is_set():
            with self._pose_lock:
                pose = self._latest_pose
            if pose is None:
                # No odom yet; emit a single scan from origin so the
                # downstream costmap pipeline gets primed even before
                # the robot's first odom message.
                pos = np.zeros(3, dtype=np.float32)
                yaw = 0.0
            else:
                pos = pose[0].copy()
                yaw = self._wxyz_to_yaw(pose[1])

            # Rotate the precomputed body-frame fan into world frame.
            cy, sy = np.cos(yaw), np.sin(yaw)
            rotmat = np.array(
                [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            dirs_world = (self._ray_dirs.reshape(-1, 3) @ rotmat.T).astype(np.float32)
            origin = pos + np.array([0.0, 0.0, cfg.sensor_height_m], dtype=np.float32)
            origins = np.tile(origin, (len(dirs_world), 1))
            rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)

            try:
                hit = self._scene.cast_rays(o3c.Tensor(rays, dtype=o3c.Dtype.Float32))
                t_hit = hit["t_hit"].numpy()
                mask = np.isfinite(t_hit) & (t_hit < cfg.max_range_m)
                if mask.any():
                    pts = origins[mask] + dirs_world[mask] * t_hit[mask, None]
                    msg = PointCloud2.from_numpy(
                        pts.astype(np.float32),
                        frame_id=cfg.frame_id,
                        timestamp=time.time(),
                    )
                    self.pointcloud.publish(msg)
            except Exception as e:
                logger.debug(f"MeshLidarModule cast/publish failed: {e}")

            self._stop_event.wait(period)


mesh_lidar = MeshLidarModule.blueprint


def cli_main() -> None:
    """``python -m dimos.mapping.mesh_lidar <path>`` — quick load + ray-cast smoke test.

    Useful to verify a downloaded ``.usdz`` / ``.glb`` loads cleanly and
    has reasonable extents before wiring it into the sim.
    """
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m dimos.mapping.mesh_lidar <scene_path> [scale]")
        sys.exit(2)
    path = Path(sys.argv[1])
    scale = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
    align = SceneMeshAlignment(scale=scale)
    mesh = load_scene_mesh(path, alignment=align)
    bbox = mesh.get_axis_aligned_bounding_box()
    print(f"verts={len(mesh.vertices)}, tris={len(mesh.triangles)}")
    print(f"bbox min={np.asarray(bbox.min_bound)}, max={np.asarray(bbox.max_bound)}")
    scene = make_raycasting_scene(mesh)
    import open3d.core as o3c

    n = 1024
    az = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    origins = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
    dirs = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=-1)
    rays = o3c.Tensor(np.concatenate([origins, dirs], axis=-1), dtype=o3c.Dtype.Float32)
    t0 = time.perf_counter()
    hit = scene.cast_rays(rays)
    dt = (time.perf_counter() - t0) * 1000
    rng = hit["t_hit"].numpy()
    finite = rng[np.isfinite(rng)]
    print(
        f"ray-cast {n} rays in {dt:.1f}ms, "
        f"{len(finite)}/{n} hits, range "
        f"min={finite.min() if len(finite) else 'nan':.2f}, "
        f"median={np.median(finite) if len(finite) else 'nan':.2f}, "
        f"max={finite.max() if len(finite) else 'nan':.2f}"
    )


if __name__ == "__main__":
    cli_main()


__all__ = ["MeshLidarConfig", "MeshLidarModule", "mesh_lidar"]

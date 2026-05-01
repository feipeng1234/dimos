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

"""Load a 3D scene mesh from disk for ray-casting + visualization.

Supports:
  * ``.glb`` / ``.gltf`` / ``.obj`` / ``.ply`` / ``.stl``  — via Open3D's
    ``read_triangle_mesh``.
  * ``.usdz`` / ``.usd`` / ``.usdc``  — via ``pxr.Usd`` (install ``usd-core``).

Returned form is a single concatenated ``open3d.geometry.TriangleMesh``
in world frame, with optional scale + Y-up→Z-up + translation applied.

The same mesh feeds:
  1. ``MeshLidarModule`` — ``Open3D RaycastingScene`` ray-cast for /lidar.
  2. ``ViserRenderModule`` — drawn in the browser as collidable geometry
     overlaid on the splat (toggleable in the viewer's view-mode dropdown).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class SceneMeshAlignment:
    """How to transform a raw scene mesh into dimos world frame.

    Apply order: scale → rotation (y_up swap then zyx euler) → translation.
    """

    scale: float = 1.0
    """Multiplicative scale.  Use 0.01 if the source is centimeters."""

    rotation_zyx_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Yaw / pitch / roll in degrees, applied after the y_up swap."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame offset applied last."""

    y_up: bool = True
    """Most exporters (Blender, glTF, Apple USDZ) are Y-up.  When ``True``
    rotate the mesh -90 deg about world X to match dimos's Z-up convention."""


def _world_rotation(alignment: SceneMeshAlignment) -> np.ndarray:
    """Compose the y-up swap + ZYX Euler into one 3x3."""
    rad = np.radians(alignment.rotation_zyx_deg)
    cz, sz = np.cos(rad[0]), np.sin(rad[0])
    cy, sy = np.cos(rad[1]), np.sin(rad[1])
    cx, sx = np.cos(rad[2]), np.sin(rad[2])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rzyx = rz @ ry @ rx
    if alignment.y_up:
        y_to_z = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            dtype=np.float64,
        )
        return rzyx @ y_to_z
    return rzyx


def _load_usd_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Walk every Mesh prim in a USD stage and concatenate to one o3d mesh."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise ImportError("loading .usdz/.usd requires usd-core: `uv pip install usd-core`") from e

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {path}")

    all_pts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    vtx_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        if pts_attr is None or len(pts_attr) == 0:
            continue
        pts = np.asarray(pts_attr, dtype=np.float32)
        face_verts = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

        # Bake the prim's local-to-world transform into the points so the
        # composite scene comes out in stage-root coordinates.
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        m = np.asarray(xform, dtype=np.float64).T  # USD matrices are row-major
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts_world = (m @ pts_h.T).T[:, :3].astype(np.float32)

        # USD allows quads / n-gons; fan-triangulate so o3d gets pure tris.
        tris: list[tuple[int, int, int]] = []
        cursor = 0
        for n in face_counts:
            for k in range(1, n - 1):
                tris.append(
                    (
                        int(face_verts[cursor]) + vtx_offset,
                        int(face_verts[cursor + k]) + vtx_offset,
                        int(face_verts[cursor + k + 1]) + vtx_offset,
                    )
                )
            cursor += n

        if not tris:
            continue
        all_pts.append(pts_world)
        all_tris.append(np.asarray(tris, dtype=np.int32))
        vtx_offset += len(pts_world)

    if not all_pts:
        raise RuntimeError(f"no Mesh prims with triangles found in {path}")

    pts = np.concatenate(all_pts, axis=0).astype(np.float64)
    tris = np.concatenate(all_tris, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    return mesh


def load_scene_mesh(
    path: str | Path,
    alignment: SceneMeshAlignment | None = None,
) -> o3d.geometry.TriangleMesh:
    """Load a scene mesh from disk and apply alignment to put it in dimos world frame.

    Args:
        path: file path.  Supported extensions: ``.usdz``, ``.usd``, ``.usdc``,
            ``.glb``, ``.gltf``, ``.obj``, ``.ply``, ``.stl``.
        alignment: scale / rotation / translation to apply.

    Returns:
        an ``open3d.geometry.TriangleMesh`` in dimos world frame with vertex
        normals computed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"scene mesh not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".usdz", ".usd", ".usdc", ".usda"}:
        mesh = _load_usd_mesh(path)
    else:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            raise RuntimeError(f"o3d.io.read_triangle_mesh returned an empty mesh for {path}")

    align = alignment or SceneMeshAlignment()
    if align.scale != 1.0:
        mesh.scale(align.scale, center=np.zeros(3))
    rot = _world_rotation(align)
    if not np.allclose(rot, np.eye(3)):
        mesh.rotate(rot, center=np.zeros(3))
    if any(align.translation):
        mesh.translate(np.asarray(align.translation, dtype=np.float64))

    mesh.compute_vertex_normals()
    return mesh


def make_raycasting_scene(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.t.geometry.RaycastingScene:
    """Wrap a TriangleMesh into Open3D's BVH-backed ray-casting scene."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


__all__ = [
    "SceneMeshAlignment",
    "load_scene_mesh",
    "make_raycasting_scene",
]

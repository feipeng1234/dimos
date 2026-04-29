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

"""In-viewer scene editor — spawn boxes/planes, drag with gizmos, save to OBJ.

For the case where the user has a gsplat but no Blender skills.  Drop
boxes for obstacles, thin boxes for walls/floors, position with the
viser transform-control gizmos, resize per-axis from the GUI panel,
then export the lot as a single OBJ that
``data/mujoco_sim/g1_gear_wbc.xml`` already knows how to consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Defaults for newly-spawned primitives. The size sliders adopt these
# on first use; the user is expected to tweak from the panel.
_DEFAULT_BOX_SIZE: tuple[float, float, float] = (0.5, 0.5, 0.5)
_DEFAULT_PLANE_SIZE: tuple[float, float, float] = (1.0, 1.0, 0.05)
_SIZE_MIN = 0.05
_SIZE_MAX = 20.0
_SIZE_STEP = 0.05
_BOX_COLOR: tuple[int, int, int] = (255, 140, 60)
_PLANE_COLOR: tuple[int, int, int] = (90, 180, 255)
_ACTIVE_OPACITY = 1.0
_INACTIVE_OPACITY = 0.55


@dataclass
class _Primitive:
    """One spawned scene primitive plus its transform-control gizmo."""

    name: str
    kind: str  # "box" | "plane"
    dimensions: list[float]  # mutable so resize sliders can edit in place
    controls: Any  # TransformControlsHandle
    box: Any  # BoxHandle — viser supports re-assigning .dimensions live


@dataclass
class SceneEditor:
    """Wires GUI controls + click placement onto an existing viser server."""

    server: Any  # viser.ViserServer
    output_dir: Path = field(
        default_factory=lambda: Path("data/mujoco_sim/dimos_office_edited.obj").parent
    )
    output_filename: str = "dimos_office_edited.obj"

    _primitives: list[_Primitive] = field(default_factory=list)
    _active: _Primitive | None = None
    _next_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pending_kind: str | None = None
    _suspend_size_callback: bool = False

    # GUI handles
    _size_x: Any = None
    _size_y: Any = None
    _size_z: Any = None
    _status_label: Any = None
    _active_label: Any = None

    def attach(self) -> None:
        """Add GUI controls to the viser panel."""
        with self.server.gui.add_folder("Scene editor"):
            self._size_x = self.server.gui.add_number(
                "Size X (m)",
                _DEFAULT_BOX_SIZE[0],
                min=_SIZE_MIN,
                max=_SIZE_MAX,
                step=_SIZE_STEP,
            )
            self._size_y = self.server.gui.add_number(
                "Size Y (m)",
                _DEFAULT_BOX_SIZE[1],
                min=_SIZE_MIN,
                max=_SIZE_MAX,
                step=_SIZE_STEP,
            )
            self._size_z = self.server.gui.add_number(
                "Size Z (m)",
                _DEFAULT_BOX_SIZE[2],
                min=_SIZE_MIN,
                max=_SIZE_MAX,
                step=_SIZE_STEP,
            )
            self._active_label = self.server.gui.add_markdown("_Active: none._")

            box_button = self.server.gui.add_button("Add box (click floor)")
            plane_button = self.server.gui.add_button("Add plane (click floor)")
            delete_button = self.server.gui.add_button("Delete active")
            export_button = self.server.gui.add_button("Export OBJ")
            self._status_label = self.server.gui.add_markdown("_Idle._")

        for size_input in (self._size_x, self._size_y, self._size_z):
            size_input.on_update(self._on_size_changed)

        @box_button.on_click
        def _arm_box(_event: Any) -> None:
            self._arm("box")

        @plane_button.on_click
        def _arm_plane(_event: Any) -> None:
            self._arm("plane")

        @delete_button.on_click
        def _delete_active(_event: Any) -> None:
            self._delete_active()

        @export_button.on_click
        def _export(_event: Any) -> None:
            self._export_obj()

    # ------------------------------------------------------------------
    # Spawn / arm
    # ------------------------------------------------------------------
    def _arm(self, kind: str) -> None:
        # Pull defaults appropriate for the kind. User-edited sliders
        # take precedence — only refresh when going from one kind to
        # another fresh start.
        if kind == "plane" and self._size_z.value == _DEFAULT_BOX_SIZE[2]:
            self._suspend_size_callback = True
            self._size_z.value = _DEFAULT_PLANE_SIZE[2]
            self._suspend_size_callback = False

        self._pending_kind = kind
        self._status_label.content = f"_Click on the floor to drop a **{kind}**…_"

        @self.server.scene.on_pointer_event(event_type="click")
        def _on_floor_click(event: Any) -> None:
            try:
                self._handle_floor_click(event)
            finally:
                self.server.scene.remove_pointer_callback()

        @self.server.scene.on_pointer_callback_removed
        def _disarm() -> None:
            self._pending_kind = None
            self._status_label.content = f"_Idle. {len(self._primitives)} primitive(s)._"

    def _handle_floor_click(self, event: Any) -> None:
        ray_origin = event.ray_origin
        ray_direction = event.ray_direction
        if ray_origin is None or ray_direction is None:
            return
        ox, oy, oz = ray_origin
        dx, dy, dz = ray_direction
        if abs(dz) < 1e-6:
            return
        t = -oz / dz
        if t <= 0:
            return
        x = float(ox + t * dx)
        y = float(oy + t * dy)
        kind = self._pending_kind or "box"
        with self._lock:
            self._spawn(kind, (x, y, 0.0))

    def _spawn(self, kind: str, position: tuple[float, float, float]) -> None:
        self._next_id += 1
        name = f"/scene_editor/{kind}_{self._next_id}"
        dimensions = [
            float(self._size_x.value),
            float(self._size_y.value),
            float(self._size_z.value),
        ]
        color = _BOX_COLOR if kind == "box" else _PLANE_COLOR

        # Sit base on z=0 by lifting center half-height.
        cx, cy, _cz = position
        center = (cx, cy, dimensions[2] / 2.0)

        controls = self.server.scene.add_transform_controls(
            name,
            scale=0.6,
            line_width=2.0,
            position=center,
        )
        box = self.server.scene.add_box(
            f"{name}/geom",
            color=color,
            dimensions=tuple(dimensions),
            opacity=_INACTIVE_OPACITY,
            side="double",
        )
        primitive = _Primitive(
            name=name,
            kind=kind,
            dimensions=dimensions,
            controls=controls,
            box=box,
        )
        self._primitives.append(primitive)

        # Click on the box itself to make it the active primitive (so
        # the size sliders bind to it). Box is a child of the gizmo so
        # the viser scene hierarchy keeps them together.
        @box.on_click
        def _select(_event: Any) -> None:
            self._set_active(primitive)

        self._set_active(primitive)
        self._status_label.content = f"_Added {kind}. {len(self._primitives)} total._"

    # ------------------------------------------------------------------
    # Selection + resize
    # ------------------------------------------------------------------
    def _set_active(self, primitive: _Primitive | None) -> None:
        with self._lock:
            # Dim the previously active primitive.
            if self._active is not None and self._active is not primitive:
                try:
                    self._active.box.opacity = _INACTIVE_OPACITY
                except Exception:
                    pass

            self._active = primitive

            if primitive is None:
                self._active_label.content = "_Active: none._"
                return

            try:
                primitive.box.opacity = _ACTIVE_OPACITY
            except Exception:
                pass

            # Reflect the active primitive's dimensions in the sliders
            # without re-firing the on_update callback.
            self._suspend_size_callback = True
            try:
                self._size_x.value = float(primitive.dimensions[0])
                self._size_y.value = float(primitive.dimensions[1])
                self._size_z.value = float(primitive.dimensions[2])
            finally:
                self._suspend_size_callback = False
            self._active_label.content = (
                f"_Active: **{primitive.name.rsplit('/', 1)[-1]}** "
                f"({primitive.dimensions[0]:.2f} x {primitive.dimensions[1]:.2f} "
                f"x {primitive.dimensions[2]:.2f} m)._"
            )

    def _on_size_changed(self, _event: Any) -> None:
        if self._suspend_size_callback:
            return
        if self._active is None:
            # No active primitive — sliders just affect the next spawn.
            return
        new_dims = (
            float(self._size_x.value),
            float(self._size_y.value),
            float(self._size_z.value),
        )
        with self._lock:
            self._active.dimensions = list(new_dims)
            try:
                self._active.box.dimensions = new_dims
            except Exception as e:
                logger.debug(f"Scene editor: live resize failed: {e}")
            self._active_label.content = (
                f"_Active: **{self._active.name.rsplit('/', 1)[-1]}** "
                f"({new_dims[0]:.2f} x {new_dims[1]:.2f} x {new_dims[2]:.2f} m)._"
            )

    def _delete_active(self) -> None:
        with self._lock:
            target = self._active or (self._primitives[-1] if self._primitives else None)
            if target is None:
                self._status_label.content = "_Nothing to delete._"
                return
            if target in self._primitives:
                self._primitives.remove(target)
            self._active = None
        try:
            target.box.remove()
            target.controls.remove()
        except Exception as e:
            logger.debug(f"Scene editor: removal failed: {e}")
        self._active_label.content = "_Active: none._"
        self._status_label.content = f"_Deleted. {len(self._primitives)} remaining._"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_obj(self) -> None:
        with self._lock:
            primitives = list(self._primitives)
        if not primitives:
            self._status_label.content = "_Nothing to export._"
            return

        verts: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for prim in primitives:
            position = np.asarray(prim.controls.position, dtype=np.float64)
            wxyz = np.asarray(prim.controls.wxyz, dtype=np.float64)
            R = _quat_to_matrix(wxyz)
            half = np.array(prim.dimensions, dtype=np.float64) * 0.5
            corners_local = np.array(
                [
                    [-half[0], -half[1], -half[2]],
                    [half[0], -half[1], -half[2]],
                    [half[0], half[1], -half[2]],
                    [-half[0], half[1], -half[2]],
                    [-half[0], -half[1], half[2]],
                    [half[0], -half[1], half[2]],
                    [half[0], half[1], half[2]],
                    [-half[0], half[1], half[2]],
                ]
            )
            corners_world = (corners_local @ R.T) + position
            base = len(verts) + 1  # OBJ is 1-indexed
            verts.extend((float(x), float(y), float(z)) for x, y, z in corners_world)
            face_offsets = [
                (0, 1, 2),
                (0, 2, 3),  # -Z
                (4, 6, 5),
                (4, 7, 6),  # +Z
                (0, 4, 5),
                (0, 5, 1),  # -Y
                (1, 5, 6),
                (1, 6, 2),  # +X
                (2, 6, 7),
                (2, 7, 3),  # +Y
                (3, 7, 4),
                (3, 4, 0),  # -X
            ]
            for a, b, c in face_offsets:
                faces.append((base + a, base + b, base + c))

        out_path = self.output_dir / self.output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write(f"# {len(primitives)} primitive(s) from in-viewer scene editor\n")
            f.write("# Exported at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("o scene_editor_export\n")
            for x, y, z in verts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a} {b} {c}\n")
        logger.info(f"Scene editor: wrote {out_path} ({len(verts)} verts, {len(faces)} faces)")
        self._status_label.content = f"_Wrote `{out_path}` ({len(primitives)} primitives)._"


def _quat_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

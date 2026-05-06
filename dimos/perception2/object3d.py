# Copyright 2026 Dimensional Inc.
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

"""Object3D — a single detected object's persistent record.

Pydantic-typed for clean serialization through memory2's codec layer.
Stores world-frame center + oriented bbox + class label + observation
counter.  Pointcloud bytes are kept as base64-encoded numpy arrays so
the whole record stays JSON-friendly for the codec.
"""

from __future__ import annotations

import base64

import numpy as np
from pydantic import BaseModel


class Object3D(BaseModel):
    """One persistent object in the world-frame map.

    Created from the first detection that doesn't merge with an
    existing entry; updated in place on every re-observation (which
    increments ``n_obs``, refreshes ``ts``, and recomputes the
    aggregated pointcloud + bbox).
    """

    name: str
    """Human-readable label from the detector (e.g. ``"chair"``,
    ``"american flag on the wall"``).  Becomes the LCM tag and the
    text on the viser overlay."""

    center: tuple[float, float, float]
    """World-frame xyz of the oriented bounding box center, meters."""

    extent: tuple[float, float, float]
    """Oriented bounding box dimensions ``(dx, dy, dz)``, meters,
    expressed in the object's local frame."""

    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion ``(w, x, y, z)`` mapping object-local frame to
    world frame.  Identity for AABB; non-identity for OBB.  Viser's
    ``add_box`` consumes wxyz directly."""

    confidence: float = 1.0
    """Detector confidence in [0, 1].  Qwen-VL doesn't return one, so
    we default to 1.0 there; later detectors can populate it."""

    ts: float
    """Wall-clock timestamp of the most recent observation, seconds."""

    n_obs: int = 1
    """Number of times this object has been re-observed since first
    detection.  Used to gate publishing (skip flicker after 1 hit)
    and to break ties in the agent-facing list."""

    pointcloud_b64: str | None = None
    """Aggregated world-frame pointcloud, ``np.float32 (N, 3)``,
    base64-encoded.  Optional — kept for viser overlay refinement
    and goto-object navigation; may be None when storage budget
    requires dropping it."""

    @classmethod
    def from_pointcloud(
        cls,
        name: str,
        pointcloud_xyz: np.ndarray,
        ts: float,
        confidence: float = 1.0,
    ) -> Object3D:
        """Construct from an Nx3 world-frame pointcloud.  Computes AABB."""
        if pointcloud_xyz.size == 0:
            raise ValueError("pointcloud_xyz must be non-empty")
        pts = np.asarray(pointcloud_xyz, dtype=np.float32)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        center = (mn + mx) / 2.0
        extent = mx - mn
        return cls(
            name=name,
            center=(float(center[0]), float(center[1]), float(center[2])),
            extent=(float(extent[0]), float(extent[1]), float(extent[2])),
            confidence=confidence,
            ts=ts,
            n_obs=1,
            pointcloud_b64=_pack_points(pts),
        )

    def points(self) -> np.ndarray | None:
        """Decode the stored pointcloud back to ``np.float32 (N, 3)``."""
        if self.pointcloud_b64 is None:
            return None
        return _unpack_points(self.pointcloud_b64)

    def merged(self, other_pts: np.ndarray, ts: float, max_points: int = 4096) -> Object3D:
        """Return a new Object3D with ``other_pts`` accumulated.

        Re-fits the AABB on the union and downsamples to ``max_points``
        via uniform random sampling so storage stays bounded.
        Identity (name) is preserved; ``n_obs`` increments.
        """
        existing = self.points()
        if existing is None or existing.size == 0:
            combined = np.asarray(other_pts, dtype=np.float32)
        else:
            combined = np.vstack([existing, np.asarray(other_pts, dtype=np.float32)])
        if combined.shape[0] > max_points:
            idx = np.random.choice(combined.shape[0], size=max_points, replace=False)
            combined = combined[idx]
        mn = combined.min(axis=0)
        mx = combined.max(axis=0)
        center = (mn + mx) / 2.0
        extent = mx - mn
        return self.model_copy(
            update=dict(
                center=(float(center[0]), float(center[1]), float(center[2])),
                extent=(float(extent[0]), float(extent[1]), float(extent[2])),
                ts=ts,
                n_obs=self.n_obs + 1,
                pointcloud_b64=_pack_points(combined),
            )
        )


def _pack_points(pts: np.ndarray) -> str:
    """numpy float32 (N, 3) → base64 ASCII for JSON-safe codec."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _unpack_points(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)


__all__ = ["Object3D"]

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

"""LineSegments3D: collection of 3D line segments for graph edge visualization.

On the wire uses ``nav_msgs/Path`` — consecutive pose pairs form segments.
Renders as ``rr.LineStrips3D`` with each segment as a separate strip.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, BinaryIO

from dimos_lcm.nav_msgs import Path as LCMPath

from dimos.types.timestamped import Timestamped

if TYPE_CHECKING:
    from rerun._baseclasses import Archetype


class LineSegments3D(Timestamped):
    """Line segments for graph edge visualization.

    Wire format: ``nav_msgs/Path`` — consecutive pose pairs are segments.
    """

    msg_name = "nav_msgs.LineSegments3D"
    ts: float
    frame_id: str
    _segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]]

    def __init__(
        self,
        ts: float = 0.0,
        frame_id: str = "map",
        segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.ts = ts if ts != 0 else time.time()
        self._segments = segments or []

    def lcm_encode(self) -> bytes:
        raise NotImplementedError("Encoded on C++ side")

    @classmethod
    def lcm_decode(cls, data: bytes | BinaryIO) -> LineSegments3D:
        lcm_msg = LCMPath.lcm_decode(data)
        header_ts = lcm_msg.header.stamp.sec + lcm_msg.header.stamp.nsec / 1e9
        frame_id = lcm_msg.header.frame_id

        segments = []
        poses = lcm_msg.poses
        for i in range(0, len(poses) - 1, 2):
            p1, p2 = poses[i], poses[i + 1]
            segments.append((
                (p1.pose.position.x, p1.pose.position.y, p1.pose.position.z),
                (p2.pose.position.x, p2.pose.position.y, p2.pose.position.z),
            ))
        return cls(ts=header_ts, frame_id=frame_id, segments=segments)

    def to_rerun(
        self,
        z_offset: float = 1.7,
        color: tuple[int, int, int, int] = (0, 255, 150, 255),
        radii: float = 0.04,
    ) -> Archetype:
        """Render as ``rr.LineStrips3D`` — each segment is a separate strip."""
        import rerun as rr

        if not self._segments:
            return rr.LineStrips3D([])

        strips = [
            [[p1[0], p1[1], p1[2] + z_offset], [p2[0], p2[1], p2[2] + z_offset]]
            for p1, p2 in self._segments
        ]

        return rr.LineStrips3D(
            strips,
            colors=[color] * len(strips),
            radii=[radii] * len(strips),
        )

    def __len__(self) -> int:
        return len(self._segments)

    def __str__(self) -> str:
        return f"LineSegments3D(frame_id='{self.frame_id}', segments={len(self._segments)})"

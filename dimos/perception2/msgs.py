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

"""ObjectBoundingBoxes — wire-format msg for the viser overlay.

Published by ``ObjectPerception`` at ~2 Hz; consumed by
``ViserRenderModule`` which renders one ``add_box`` + ``add_label`` per
entry under ``/perception2/objects/<name>``.

Wire format mirrors ``EntityMarkers``: JSON-encoded payload over an
LCM string channel, with ``lcm_encode`` / ``lcm_decode`` shims so
``autoconnect`` assigns ``LCMTransport`` instead of pickling.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import struct
import time

from dimos.types.timestamped import Timestamped


@dataclass
class ObjectBoundingBox:
    """One per-object 3D box for the viser overlay."""

    name: str
    """Detector label, e.g. ``"chair"``.  Used both as the floating
    text and as the suffix on the viser scene-tree node so the box is
    addressable / removable per object."""

    center: tuple[float, float, float]
    """World-frame xyz of the box center, meters."""

    extent: tuple[float, float, float]
    """Box dimensions ``(dx, dy, dz)`` in the object-local frame, meters."""

    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion ``(w, x, y, z)``: object-local frame → world frame."""

    confidence: float = 1.0
    """Detector confidence in [0, 1].  Drives label suffix opacity in
    the overlay so unreliable detections render dimly."""

    n_obs: int = 1
    """Re-observation count.  Lets the overlay throttle "flicker"
    boxes (those with ``n_obs == 1`` are typically suppressed)."""


class ObjectBoundingBoxes(Timestamped):
    """A batch of object 3D boxes published per perception tick.

    Encoded as a single JSON blob — the box list is small (~tens of
    objects max in a typical office) so JSON beats msgpack/protobuf
    for debuggability + zero schema-registry churn.
    """

    msg_name = "perception2_msgs.ObjectBoundingBoxes"

    def __init__(
        self,
        boxes: list[ObjectBoundingBox] | None = None,
        ts: float | None = None,
    ) -> None:
        self.boxes: list[ObjectBoundingBox] = boxes or []
        self.ts: float = ts if ts is not None else time.time()

    def _encode_one(self, buf: BytesIO) -> None:
        payload = json.dumps(
            [
                {
                    "name": b.name,
                    "center": list(b.center),
                    "extent": list(b.extent),
                    "orientation_wxyz": list(b.orientation_wxyz),
                    "confidence": b.confidence,
                    "n_obs": b.n_obs,
                }
                for b in self.boxes
            ]
        ).encode()
        buf.write(struct.pack(">I", len(payload)))
        buf.write(payload)

    def encode(self) -> bytes:
        buf = BytesIO()
        self._encode_one(buf)
        return buf.getvalue()

    @classmethod
    def _decode_one(cls, buf: BytesIO) -> ObjectBoundingBoxes:
        (length,) = struct.unpack(">I", buf.read(4))
        payload = json.loads(buf.read(length).decode())
        boxes = [
            ObjectBoundingBox(
                name=item["name"],
                center=tuple(item["center"]),  # type: ignore[arg-type]
                extent=tuple(item["extent"]),  # type: ignore[arg-type]
                orientation_wxyz=tuple(item["orientation_wxyz"]),  # type: ignore[arg-type]
                confidence=float(item.get("confidence", 1.0)),
                n_obs=int(item.get("n_obs", 1)),
            )
            for item in payload
        ]
        return cls(boxes=boxes)

    @classmethod
    def decode(cls, data: bytes) -> ObjectBoundingBoxes:
        return cls._decode_one(BytesIO(data))

    def lcm_encode(self) -> bytes:
        return self.encode()

    @classmethod
    def lcm_decode(cls, data: bytes, **kwargs: object) -> ObjectBoundingBoxes:
        return cls.decode(data)


__all__ = ["ObjectBoundingBox", "ObjectBoundingBoxes"]

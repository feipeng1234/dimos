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

"""perception2 — clean replacement for dimos.perception.detection.

Replaces the old YoloE/COCO-class detector + unbounded in-process ObjectDB
with a Qwen-VL detector + memory2-backed object store.  Headline wins:

* Real names (``"american flag"``, ``"wooden coffee table"``) instead of
  YoloE-PF's "centerpiece"/"brickwork"/"waffle" hallucinations.
* Bounded persistent storage via ``memory2.SqliteStore`` — survives
  restarts, evicts oldest entries past a configurable cap, supports
  semantic search (``EmbedText`` + cosine) and spatial filters
  (``.near(pose, radius)``).
* Per-object 3D bounding boxes published as a ``ObjectBoundingBoxes``
  LCM message, rendered in viser as oriented boxes + floating labels.
* Three agent-callable skills: ``list_objects``, ``find_object``,
  ``goto_object``.
"""

from dimos.perception2.module import ObjectPerception, ObjectPerceptionConfig
from dimos.perception2.msgs import ObjectBoundingBox, ObjectBoundingBoxes
from dimos.perception2.object3d import Object3D

__all__ = [
    "Object3D",
    "ObjectBoundingBox",
    "ObjectBoundingBoxes",
    "ObjectPerception",
    "ObjectPerceptionConfig",
]

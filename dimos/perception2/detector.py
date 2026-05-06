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

"""Detector layer — image in, list of 2D bboxes + labels out.

Defines a minimal ``Detector`` protocol and one concrete implementation
``VlDetector`` backed by any ``VlModel`` (Qwen-VL by default).  The VLM
returns proper noun-phrase labels ("american flag", "wooden coffee
table") instead of YoloE-PF's "centerpiece"/"brickwork"/"waffle"
hallucinations on small office images.

The 2D detection type comes from the existing
``perception.detection`` package — we don't redefine it because the
data carrier is fine; only the *detector* and *storage* needed
replacing.
"""

from __future__ import annotations

from typing import Protocol

from dimos.models.vl.base import VlModel
from dimos.models.vl.create import create as create_vl_model
from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D


class Detector(Protocol):
    """Anything that turns one image into a list of 2D bboxes + labels."""

    def detect(self, image: Image) -> list[Detection2DBBox]:
        """Run detection on a single image.  Returns possibly-empty list."""
        ...


class VlDetector:
    """VL-model-backed open-vocabulary detector.

    Calls ``vl_model.query_detections(image, query)`` per frame; the
    underlying prompt asks for "objects in this scene" — the VLM
    returns whatever it considers nameable, with proper noun phrases.

    Throttling lives in the calling Module, not here — keeping this
    class stateless makes it easy to swap in a different VL backend
    or a non-VL detector later without touching the perception loop.
    """

    def __init__(
        self,
        vl_model: VlModel | None = None,
        query: str = "list every distinct salient object in this scene",
        model_name: str = "qwen",
    ) -> None:
        """Construct the detector.

        Parameters
        ----------
        vl_model:
            Pre-built VL model.  If None, ``create_vl_model(model_name)``
            is used — same factory the rest of dimos goes through, so
            the global ``detection_model`` config picks the backend.
        query:
            The detection prompt.  Default asks for everything; pass
            something narrower (e.g. ``"only people"``) when you have
            a specific use case.
        model_name:
            Used only if ``vl_model`` is None.  Forwarded to
            ``dimos.models.vl.create``.
        """
        self._vl_model = vl_model if vl_model is not None else create_vl_model(model_name)
        self._query = query

    def detect(self, image: Image) -> list[Detection2DBBox]:
        result: ImageDetections2D[Detection2DBBox] = self._vl_model.query_detections(
            image, self._query
        )
        return list(result.detections)


__all__ = ["Detector", "VlDetector"]

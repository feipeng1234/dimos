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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dimos.msgs.sensor_msgs.Image import Image

# 1-byte prefix tags so a single codec can store both JPEG-compressed RGB
# images and raw LCM-encoded non-RGB images (depth, gray16) in the same stream.
_TAG_JPEG = b"J"
_TAG_RAW = b"R"


class JpegCodec:
    """Codec for Image types — JPEG for RGB-compatible, raw LCM for the rest.

    RGB/RGBA/BGR/BGRA images are JPEG-compressed inside an LCM Image envelope
    (lossy). Depth and grayscale images can't be JPEG-encoded, so they fall
    back to raw LCM encoding (lossless, but larger). The encoded payload is
    prefixed with a 1-byte tag so decode can dispatch correctly.
    """

    def __init__(self, quality: int = 50) -> None:
        self._quality = quality

    def encode(self, value: Image) -> bytes:
        from dimos.msgs.sensor_msgs.Image import ImageFormat

        non_rgb = {
            ImageFormat.DEPTH,
            ImageFormat.DEPTH16,
            ImageFormat.GRAY,
            ImageFormat.GRAY16,
        }
        if value.format in non_rgb:
            return _TAG_RAW + value.lcm_encode()
        return _TAG_JPEG + value.lcm_jpeg_encode(quality=self._quality)

    def decode(self, data: bytes) -> Image:
        from dimos.msgs.sensor_msgs.Image import Image

        tag = data[:1]
        if tag == _TAG_RAW:
            return Image.lcm_decode(data[1:])
        if tag == _TAG_JPEG:
            return Image.lcm_jpeg_decode(data[1:])
        # Pre-tag recordings: assume JPEG (the old behavior).
        return Image.lcm_jpeg_decode(data)

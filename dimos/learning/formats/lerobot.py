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

"""LeRobot v2 dataset writer.

Layout:
    <output.path>/
        meta/info.json          schema, fps, total episodes/frames
        meta/episodes.jsonl     per-episode metadata
        meta/stats.json         per-feature stats (from DataPrep.compute_stats)
        data/chunk-000/episode_*.parquet
        videos/chunk-000/<image_key>/episode_*.mp4
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from dimos.learning.dataprep import OutputConfig, Sample


def write(samples: Iterator[Sample], output: OutputConfig) -> Path:
    """Drain samples, write parquet+MP4, call DataPrep.compute_stats,
    serialize stats to meta/stats.json. Return the dataset root path."""
    raise NotImplementedError

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

"""HDF5 dataset writer. Single .hdf5 with one group per episode + stats group."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from dimos.learning.dataprep import OutputConfig, Sample


def write(samples: Iterator[Sample], output: OutputConfig) -> Path:
    """Write samples to a single HDF5 file (stats as group attrs).
    Returns the file path."""
    raise NotImplementedError

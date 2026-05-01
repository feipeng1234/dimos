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

"""Dataset-shape types + pure helpers.

Sub-configs (StreamField, SyncConfig, OutputConfig, EpisodeExtractor) and
data records (Episode, Sample) live here. So do the stateless functions
that walk samples — `resolve_field`, `compute_stats`, `extract_episodes`,
`iter_episode_samples`. Importable without booting a Module.

`DataPrepModule` (in `dataprep_module.py`) is a thin wrapper that runs
these helpers on a thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dimos.protocol.service.spec import BaseConfig

Writer = Callable[[Iterator["Sample"], "OutputConfig"], Path]

if TYPE_CHECKING:
    from dimos.memory2.store.sqlite import SqliteStore


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────


class EpisodeExtractor(BaseConfig):
    extractor: Literal["episode_status", "ranges", "whole_session"] = "episode_status"
    status_stream: str = "episode_status"
    ranges: list[tuple[float, float]] | None = None


class StreamField(BaseConfig):
    stream: str
    field: str | None = None


class SyncConfig(BaseConfig):
    anchor: str
    rate_hz: float
    tolerance_ms: float
    strategy: Literal["nearest", "interp"] = "nearest"


class OutputConfig(BaseConfig):
    format: Literal["lerobot", "hdf5", "rlds"] = "lerobot"
    path: Path
    metadata: dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Data records
# ─────────────────────────────────────────────────────────────────────────────


class Episode(BaseModel):
    id: str
    start_ts: float
    end_ts: float
    task_label: str | None = None
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


class Sample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float
    episode_id: str
    observation: dict[str, np.ndarray]
    action: dict[str, np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — used by ChunkPolicyModule, format writers, DataPrepModule
# ─────────────────────────────────────────────────────────────────────────────


def resolve_field(msg: Any, ref: StreamField) -> np.ndarray:
    """Project `msg` through `ref` (attribute access). Single source of
    truth for obs/action construction across train and live."""
    raise NotImplementedError


def extract_episodes(store: SqliteStore, cfg: EpisodeExtractor) -> list[Episode]:
    """Walk recorded EpisodeStatus events (or ranges/whole_session) into Episodes."""
    raise NotImplementedError


def iter_episode_samples(
    store: SqliteStore,
    episode: Episode,
    streams: dict[str, StreamField],  # observation ∪ action
    sync: SyncConfig,
) -> Iterator[Sample]:
    """Yield synced (obs, action) Samples for one episode."""
    raise NotImplementedError


def compute_stats(
    samples: Iterator[Sample],
    image_subsample: int = 10,
    quantile_reservoir: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-feature mean/std/min/max/q01/q99 in one pass.

    Welford for mean/std; reservoir sample for quantiles. Image features
    subsampled (every Nth frame) to bound cost.
    """
    raise NotImplementedError


def get_writer(format_name: str) -> Writer:
    """Lazy-import the format writer's `write` function."""
    if format_name == "lerobot":
        from dimos.learning.formats.lerobot import write
    elif format_name == "hdf5":
        from dimos.learning.formats.hdf5 import write
    elif format_name == "rlds":
        from dimos.learning.formats.rlds import write
    else:
        raise ValueError(f"Unknown format: {format_name!r}")
    return write

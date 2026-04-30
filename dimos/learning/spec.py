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

"""Data definitions for the DimOS Learning Framework.

Contains the YAML/JSON-backed `DatasetSpec` schema and the runtime data
classes (`Episode`, `Sample`) shared between collection, training, and
inference. No logic — just typed records and constants. Safe to import
from anywhere (no circular dependencies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# DatasetSpec — the YAML/JSON schema
# ─────────────────────────────────────────────────────────────────────────────


class DatasetSpec(BaseModel):
    """Top-level spec. Same instance used at build, load, and inference time.

    A `DatasetSpec` (loaded from YAML/JSON) is the contract between data
    collection (raw RecordReplay session -> on-disk dataset) and training
    (loading the same spec to feed a model). The same spec also drives
    inference observation construction.
    """

    source: Path  # path to session.db produced by RecordReplay
    episodes: EpisodeConfig
    observation: dict[str, StreamField]  # obs key -> stream field
    action: dict[str, StreamField]  # action key -> stream field
    sync: SyncConfig
    filters: FilterConfig | None = None
    output: OutputConfig | None = None  # only required by DataPrep.build()

    @classmethod
    def from_file(cls, path: str | Path) -> DatasetSpec:
        """Load from .yaml/.yml/.json (dispatch by extension)."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write to .yaml/.yml/.json (round-trip safe)."""
        raise NotImplementedError


class EpisodeConfig(BaseModel):
    """How to slice the continuous recording into episodes."""

    extractor: Literal["buttons", "ranges", "whole_session"] = "buttons"

    # BUTTONS extractor: friendly names map to Quest Buttons attrs via BUTTON_ALIASES.
    # The state machine always discards an in-progress episode if the recording ends
    # without an explicit save/discard press.
    button_stream: str = "buttons"
    start: str = "A"  # rising edge -> begin episode
    save: str = "B"  # rising edge -> end + save
    discard: str = "X"  # rising edge -> end + drop

    # RANGES extractor: explicit absolute timestamps
    ranges: list[tuple[float, float]] | None = None

    # Default label/description applied to every extracted episode unless overridden.
    # task_description is the free-form natural-language string used as language
    # conditioning for VLA policies (e.g. "pick up the red cube and place it on the blue plate").
    default_task_label: str | None = None
    default_task_description: str | None = None


class StreamField(BaseModel):
    """Pointer to a field in a recorded stream — one (obs|action) key's data source."""

    stream: str  # LCM stream / topic name as recorded in session.db
    type: str | None = None  # optional dotted type (e.g. "sensor_msgs.Image"); for codec dispatch
    field: str | None = None  # attribute on the message; None = whole message
    preprocess: str | None = None  # named preprocess hook (e.g. "jpeg_decode", "normalize_image")


class SyncConfig(BaseModel):
    """How to build per-timestep samples by aligning multiple streams."""

    anchor: str  # key in `observation` that drives the timeline
    rate_hz: float = 30.0  # downsample anchor to this rate; 0 = use anchor's native rate
    tolerance_ms: float = 50.0  # max allowed time delta when picking nearest sample
    strategy: Literal["nearest", "interp"] = "nearest"


class FilterConfig(BaseModel):
    """Per-episode filters applied after extraction."""

    success_only: bool = True
    min_duration_s: float = 0.0
    max_duration_s: float | None = None
    task_labels: list[str] | None = None  # whitelist; None = all

    # Train/val split. Episodes whose index lands in val become the validation set
    # at training time; everything else is train. `val_episode_ids` takes precedence
    # over `val_ratio`. Both None = no split (everything is train).
    val_episode_ids: list[int] | None = None
    val_ratio: float | None = None
    val_split_seed: int = 0


class OutputConfig(BaseModel):
    """Where and how to write the built dataset."""

    format: Literal["lerobot", "hdf5", "rlds"]
    path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime data
# ─────────────────────────────────────────────────────────────────────────────


# Friendly Quest controller names -> Buttons attribute names.
# Override by supplying an attribute name directly in the spec.
BUTTON_ALIASES: dict[str, str] = {
    "A": "right_primary",
    "B": "right_secondary",
    "X": "left_primary",
    "Y": "left_secondary",
    "LT": "left_trigger",
    "RT": "right_trigger",
    "LG": "left_grip",
    "RG": "right_grip",
    "MENU_L": "left_menu",
    "MENU_R": "right_menu",
}


class Episode(BaseModel):
    """A single demonstration carved from a session."""

    id: str
    start_ts: float
    end_ts: float
    task_label: str | None = None  # short categorical tag (e.g. "pick_red_cube")
    task_description: str | None = None  # free-form natural-language string for VLA conditioning
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


class Sample(BaseModel):
    """One synchronized timestep: aligned obs + action at ts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float
    episode_id: str
    observation: dict[str, np.ndarray]
    action: dict[str, np.ndarray]


# DatasetSpec is defined before its referenced subclasses so it reads as the
# top-of-file entry point. Resolve those forward references now that every
# referenced class exists in the module namespace.
DatasetSpec.model_rebuild()

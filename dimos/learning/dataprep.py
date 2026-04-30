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

"""Dataset builder/loader for the DimOS Learning Framework.

`DataPrep` is the single user-facing entry point. It reads a `DatasetSpec`
(see `dimos.learning.spec`) and either:
  - builds a training-ready dataset on disk in HDF5/RLDS/LeRobot, or
  - returns a PyTorch Dataset for training.

Stateless helpers (episode extraction, sample iteration, field resolution)
live as `@staticmethod`s on `DataPrep` so they share one namespace and are
callable without an instance — the live `ObsBuilder` at inference time
reuses `DataPrep.resolve_field` for that reason.

Workflow:
    # 1. Record a teleop session (Sam's PR #1708)
    dimos --blueprint quest_teleop_xarm7 --record-path session.db

    # 2. Build a training-ready dataset
    python -m dimos.learning.dataprep build dataset.yaml

    # 3. Train using the same spec
    from dimos.learning.dataprep import DataPrep
    dp = DataPrep.from_file("dataset.yaml")
    ds = dp.load()
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.learning.spec import (
    DatasetSpec,
    Episode,
    EpisodeConfig,
    FilterConfig,
    OutputConfig,
    Sample,
    StreamField,
)

Writer = Callable[[Iterator[Sample], OutputConfig], Path]

if TYPE_CHECKING:
    import torch

    from dimos.memory2.store.sqlite import SqliteStore


# ─────────────────────────────────────────────────────────────────────────────
# DataPrep — the only thing this module exports besides `main()`
# ─────────────────────────────────────────────────────────────────────────────


class DataPrep:
    """Build / load / inspect a dataset from a `DatasetSpec`.

    Holds the open `SqliteStore` and the cached, filtered episode list so
    repeated operations on the same spec (e.g. `inspect()` then `build()`)
    don't redo work. Construction is cheap — the store and episodes are
    computed lazily on first access.

    Not a DimOS Module: no ports, no runtime lifecycle. It's a stateful
    façade over the static helpers below.
    """

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, spec: DatasetSpec) -> None:
        """Bind to a spec. Does not open the store or extract episodes yet."""
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str | Path) -> DataPrep:
        """Convenience: `DataPrep.from_file("dataset.yaml")`."""
        raise NotImplementedError

    # ── lazy-cached state ────────────────────────────────────────────────────

    @property
    def store(self) -> SqliteStore:
        """Open the recording's SqliteStore on first access; cached thereafter."""
        raise NotImplementedError

    @property
    def episodes(self) -> list[Episode]:
        """Extract + filter episodes on first access; cached thereafter.

        Equivalent to:
            DataPrep.filter_episodes(
                DataPrep.extract_episodes(store, spec.episodes),
                spec.filters,
            )
        """
        raise NotImplementedError

    # ── operations ───────────────────────────────────────────────────────────

    def iter_samples(self) -> Iterator[Sample]:
        """Yield synced Samples across every episode, in episode order."""
        raise NotImplementedError

    def build(self) -> Path:
        """End-to-end: source session.db -> on-disk dataset in spec.output.format.

        Returns the path written. Requires `spec.output` to be set. Dispatches
        to the appropriate writer in `dimos.learning.formats` via `_get_writer`.
        """
        raise NotImplementedError

    def load(self) -> torch.utils.data.Dataset[Sample]:
        """Training-time loader: returns a PyTorch Dataset over the source recording.

        Materializes Samples on-the-fly (lazy). Does not require `spec.output`.
        Pre-extracts episodes once and indexes anchor timestamps for O(1)
        `__getitem__`.
        """
        raise NotImplementedError

    def inspect(self) -> dict[str, Any]:
        """Stats for a session: episode count, duration distribution,
        per-stream sample counts. Used by `python -m dimos.learning.dataprep inspect`.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying SqliteStore. Safe to call multiple times."""
        raise NotImplementedError

    def __enter__(self) -> DataPrep:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── stateless helpers ────────────────────────────────────────────────────
    #
    # Static so they're callable without an instance. `resolve_field` in
    # particular is reused by `dimos.learning.inference.obs_builder` to build
    # live observations, so train and infer share exactly one code path for
    # field projection + preprocess.

    @staticmethod
    def extract_episodes(store: SqliteStore, cfg: EpisodeConfig) -> list[Episode]:
        """Extract episode boundaries per the configured strategy.

        BUTTONS: scan cfg.button_stream for rising edges on cfg.start/save/discard.
            State machine:
                IDLE   --start press-->          RECORDING  (begin episode)
                RECORDING --save press-->        IDLE       (commit, success=True)
                RECORDING --discard press-->     IDLE       (drop)
                RECORDING --start press-->       RECORDING  (auto-commit, begin new)
                session ends mid-episode:        always discard

        RANGES: emit one Episode per (start_ts, end_ts) tuple in cfg.ranges.

        WHOLE:  emit a single Episode covering the entire recording's time range.
        """
        raise NotImplementedError

    @staticmethod
    def filter_episodes(eps: list[Episode], cfg: FilterConfig | None) -> list[Episode]:
        """Apply success / duration / label whitelist filters. `None` = pass-through.

        Note: train/val split fields on FilterConfig (`val_episode_ids`,
        `val_ratio`) are *not* applied here — they're consumed by the trainer,
        which needs the full episode list to materialize both splits.
        """
        raise NotImplementedError

    @staticmethod
    def iter_episode_samples(
        store: SqliteStore,
        episode: Episode,
        spec: DatasetSpec,
    ) -> Iterator[Sample]:
        """Yield synced (obs, action) Samples for a single episode.

        Walks the anchor stream at sync.rate_hz between episode.start_ts and
        episode.end_ts. For each anchor timestamp, pulls the nearest observation/
        action from each configured stream within sync.tolerance_ms. Applies any
        declared preprocess (e.g. jpeg_decode for Image, field projection for
        JointState). Skips frames where any required stream lacks a sample
        within tolerance.
        """
        raise NotImplementedError

    @staticmethod
    def resolve_field(msg: Any, ref: StreamField) -> np.ndarray:
        """Pull a single field from a stream message and convert to np.ndarray.

        Applies ref.field projection (attribute access) and ref.preprocess hook
        (named transform like jpeg_decode). Returns a numpy array suitable for
        inclusion in a Sample.

        Reused by the live ObsBuilder at inference time — single source of
        truth for observation construction across train and infer.
        """
        raise NotImplementedError

    @staticmethod
    def _get_writer(format_name: str) -> Writer:
        """Lazy-import the `write` function for a given format. Avoids loading
        heavy deps (h5py, tfds, lerobot) for unused formats.
        """
        if format_name == "lerobot":
            from dimos.learning.formats.lerobot import write
        elif format_name == "hdf5":
            from dimos.learning.formats.hdf5 import write
        elif format_name == "rlds":
            from dimos.learning.formats.rlds import write
        else:
            raise ValueError(
                f"Unknown dataset format: {format_name!r}. Supported: lerobot, hdf5, rlds."
            )
        return write


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entrypoint: `build` / `inspect` / `review` a dataset spec."""
    raise NotImplementedError


if __name__ == "__main__":
    main()

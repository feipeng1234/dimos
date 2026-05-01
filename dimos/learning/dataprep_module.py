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

"""DataPrepModule — wraps the dataprep pipeline as a Module with RPC surface.

All dataset-shape types and pure helpers live in `dataprep.py`. This file
just adds the Module lifecycle + thread + status tracking.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.learning.dataprep import (
    EpisodeExtractor,
    OutputConfig,
    StreamField,
    SyncConfig,
)


class DataPrepModuleConfig(ModuleConfig):
    source:      str
    episodes:    EpisodeExtractor
    observation: dict[str, StreamField]
    action:      dict[str, StreamField]
    sync:        SyncConfig
    output:      OutputConfig
    auto_run:    bool = False


class DataPrepModule(Module):
    """Wraps a long-running dataset build job."""

    config: DataPrepModuleConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "state":         "idle",   # idle | running | succeeded | failed
            "current_phase": None,     # scan_episodes | write | done
            "progress_pct":  0.0,
            "dataset_path":  None,
            "error":         None,
        }

    @rpc
    def start(self) -> None:
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        raise NotImplementedError

    @rpc
    def build(self) -> None:
        """Spawn a daemon thread running the build pipeline. Returns immediately."""
        raise NotImplementedError

    @rpc
    def get_status(self) -> dict[str, Any]:
        raise NotImplementedError

    @rpc
    def inspect(self) -> dict[str, Any]:
        """Read-only summary: episode count, drop rates, joint names, stats presence."""
        raise NotImplementedError

    def _run_build(self) -> None:
        """Thread target. Opens session.db, calls extract_episodes /
        iter_episode_samples / format writer, snapshots config to
        <output.path>/dimos_meta.json. Updates _status under _lock."""
        raise NotImplementedError


__all__ = ["DataPrepModule", "DataPrepModuleConfig"]

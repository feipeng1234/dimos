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

"""ACT training Module (v1, inline).

Runs `train_bc` on a daemon thread inside its own worker. No subprocess.
Lazy imports keep `lerobot` / `torch` / CUDA out of the worker until
`train()` is called. Metrics → TensorBoard. No cancel() in v1.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig


class TrainerModuleConfig(ModuleConfig):
    dataset_path: str = ""
    output_dir: str = ""
    config_path: str | None = None  # optional BCConfig YAML override
    auto_run: bool = False
    tensorboard_port: int = 6006


class TrainerModule(Module):
    config: TrainerModuleConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "state": "idle",  # idle | running | succeeded | failed
            "checkpoint_dir": None,
            "error": None,
        }

    @rpc
    def start(self) -> None:
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        raise NotImplementedError

    @rpc
    def train(
        self,
        dataset_path: str | None = None,
        output_dir: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Spawn a daemon thread running train_bc; returns immediately.
        Raises if a run is already in progress."""
        raise NotImplementedError

    @rpc
    def get_status(self) -> dict[str, Any]:
        raise NotImplementedError

    def _run_training(
        self,
        dataset_path: str,
        output_dir: str,
        config_overrides: dict[str, Any] | None,
    ) -> None:
        """Thread target. Lazy-imports train_bc + BCConfig; updates _status."""
        raise NotImplementedError


__all__ = ["TrainerModule", "TrainerModuleConfig"]

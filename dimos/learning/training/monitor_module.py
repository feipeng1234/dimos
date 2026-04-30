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

"""Visualize / log training progress.

Subscribes to the unified `TrainProgress` + `TrainDone` streams from
`TrainerModule` (which covers both build and train phases via `phase` field);
logs to:
  - rerun (if the rerun bridge is available — already a DimOS dep)
  - JSONL file (structured, post-hoc analysis)
  - stdout (always, terse summary line per event)

Optional in any blueprint. Sits passively on the bus so the same training
session can have multiple monitors (one in dev, one writing to a server).
"""

from __future__ import annotations

from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.learning.training.trainer_module import TrainDone, TrainProgress


class LearningMonitorModuleConfig(ModuleConfig):
    """Where to send progress events.

    Attributes:
        log_to_rerun: forward every event to the rerun bridge if importable.
        log_to_stdout: print one terse summary line per event.
        jsonl_path: if set, append JSON-per-line to this file.
        train_loss_smoothing: EMA smoothing factor for the rerun loss curve.
    """

    log_to_rerun: bool = True
    log_to_stdout: bool = True
    jsonl_path: str | None = None
    train_loss_smoothing: float = 0.9


class LearningMonitorModule(Module):
    """Pure subscriber. Owns no work, just visualization fan-out."""

    config: LearningMonitorModuleConfig

    train_progress: In[TrainProgress]
    train_done: In[TrainDone]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._jsonl_handle: Any = None  # opened in start()
        self._train_loss_ema: float | None = None

    @rpc
    def start(self) -> None:
        """Open the JSONL file (if configured), subscribe to both ports."""
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        """Flush + close the JSONL file; super().stop()."""
        raise NotImplementedError

    # ── handlers (called from port subscriptions) ────────────────────────────

    def _on_train_progress(self, msg: TrainProgress) -> None:
        """Forward to enabled sinks. Routes by `msg.phase`:
          - phase == "build": log dataset progress (episodes, samples)
          - phase in {"train","eval"}: log loss curves (with EMA smoothing for rerun)
          - other phases: log message line only.
        """
        raise NotImplementedError

    def _on_train_done(self, msg: TrainDone) -> None:
        """Final summary line; close JSONL section."""
        raise NotImplementedError

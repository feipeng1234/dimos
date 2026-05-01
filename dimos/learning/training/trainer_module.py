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

"""DimOS Module wrapper around the v1 training pipeline.

A single training job is two subprocesses run in sequence:
  1. `python -m dimos.learning.dataprep build`     (skipped if output exists)
  2. `python -m dimos.learning.training.train ...`

`TrainerModule` runs both, parses their progress lines, and republishes them
under one unified `TrainProgress` stream with a `phase` field. There is no
separate builder Module — building is always a precursor to training in v1,
so the wiring tax of two Modules + a chain port wasn't worth it.

Why subprocess: keeps `lerobot`, `torch`, CUDA out of the runtime's import
graph. Process isolation also means a CUDA OOM doesn't poison the runtime.

Wiring patterns:
  - Default blueprint:     `auto_run=True` -> module fires on start()
  - Agent skill:           agent calls `@rpc train(...)` directly
  - Build-only (rare):     agent calls `@rpc build_only(spec_path)`
  - External trigger:      publish `TrainTrigger` on the trigger port
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import threading
from typing import Any, Literal

from pydantic import BaseModel

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out

# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────


class TrainTrigger(BaseModel):
    """Start a training job. Empty trigger uses module config defaults."""

    spec_path: str | None = None
    output_dir: str | None = None
    config_kind: Literal["bc", "vla"] | None = None
    config_overrides: dict[str, Any] = {}  # merged onto BCConfig/VLAConfig
    skip_build: bool = False  # set when caller knows the dataset is already built
    job_id: str | None = None


class TrainProgress(BaseModel):
    """Unified progress event covering both build and train phases.

    `phase` indicates which subprocess the event came from. For phase=="build"
    the train-specific fields (loss, val_loss, step counts) are zero/None.
    For phase=="train" the build-specific fields are zero.
    """

    job_id: str
    phase: Literal["build", "load", "train", "eval", "save", "done", "failed"]
    message: str = ""

    # Build-phase fields (meaningful only when phase == "build")
    samples_written: int = 0
    current_episode: int = 0
    total_episodes: int = 0

    # Train-phase fields (meaningful only when phase in {"load","train","eval","save"})
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    val_loss: float | None = None
    eta_s: float | None = None


class TrainDone(BaseModel):
    """Terminal event with the final checkpoint dir or an error."""

    job_id: str
    success: bool
    dataset_path: Path | None = None  # the (possibly newly-built) dataset
    checkpoint_dir: Path | None = None  # None on failure
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────


class TrainerModuleConfig(ModuleConfig):
    """Trainer module config.

    Attributes:
        spec_path: default spec path (used for both build and train).
        output_dir: default checkpoint output directory.
        config_kind: "bc" (ACT/Diffusion) or "vla" (pi0/pi0.5).
        config_path: optional BCConfig/VLAConfig YAML override.
        python_executable: subprocess python; "" = current sys.executable.
        skip_build_if_exists: if the dataset path already exists on disk,
            skip the build phase. Default True (idempotent).
        auto_run: if True, start a job on `start()`.
        max_concurrent: cap on simultaneous jobs. v1 uses 1.
    """

    spec_path: str = ""
    output_dir: str = ""
    config_kind: Literal["bc", "vla"] = "bc"
    config_path: str | None = None
    python_executable: str = ""
    skip_build_if_exists: bool = True
    auto_run: bool = False
    max_concurrent: int = 1


class TrainerModule(Module):
    """Spawns dataprep build (if needed) then train; reports unified progress."""

    config: TrainerModuleConfig

    trigger: In[TrainTrigger]

    progress: Out[TrainProgress]
    done: Out[TrainDone]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # job_id -> (build_proc, train_proc, watcher_thread). Each proc may be None.
        self._jobs: dict[str, dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()
        self._next_job_id = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    @rpc
    def start(self) -> None:
        """Subscribe to `trigger`. If `auto_run`, kick off one training job."""
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        """Cancel all in-flight jobs, then super().stop()."""
        raise NotImplementedError

    # ── agent / external surface ─────────────────────────────────────────────

    @rpc
    def train(
        self,
        spec_path: str | None = None,
        output_dir: str | None = None,
        config_kind: Literal["bc", "vla"] | None = None,
        config_overrides: dict[str, Any] | None = None,
        skip_build: bool = False,
    ) -> str:
        """Start a build-then-train job. Returns job_id.

        All arguments override `config` for this job only. If `skip_build` or
        `config.skip_build_if_exists` and the dataset is on disk, the build
        phase is skipped.
        """
        raise NotImplementedError

    @rpc
    def build_only(self, spec_path: str | None = None) -> str:
        """Run only the dataset-build subprocess; do not train.

        Convenience for the rare standalone case (CI dataset bake, debugging
        a new spec). Returns job_id; emits TrainProgress with phase=="build"
        events and a TrainDone with checkpoint_dir=None on completion.
        """
        raise NotImplementedError

    @rpc
    def cancel(self, job_id: str) -> bool:
        """SIGTERM the active subprocess (build or train); True if cancelled."""
        raise NotImplementedError

    @rpc
    def list_jobs(self) -> list[str]:
        """Return active job ids."""
        raise NotImplementedError

    @rpc
    def list_checkpoints(self, output_dir: str | None = None) -> list[str]:
        """Scan `output_dir` (defaults to config.output_dir) and return paths
        to checkpoint subdirectories. Useful for agent flows like 'train then
        deploy the latest checkpoint'."""
        raise NotImplementedError

    # ── internals ────────────────────────────────────────────────────────────

    def _on_trigger(self, msg: TrainTrigger) -> None:
        """Port handler — calls `self.train(...)`."""
        raise NotImplementedError

    def _run_job(
        self,
        job_id: str,
        spec_path: str,
        output_dir: str,
        config_kind: Literal["bc", "vla"],
        config_overrides: dict[str, Any],
        skip_build: bool,
        train: bool,
    ) -> None:
        """Background thread driving one job through its phases.

        Sequence:
          1. Resolve dataset path from spec.
          2. If `skip_build` is False and dataset doesn't exist (or
             `skip_build_if_exists` is False), spawn `dataprep build` and
             stream its progress as phase=="build".
          3. If `train` is True, spawn `train` and stream progress as
             phase in {"load","train","eval","save"}.
          4. Emit terminal TrainDone.
        """
        raise NotImplementedError

    def _spawn_build(self, spec_path: str, job_id: str) -> subprocess.Popen[str]:
        """Build argv for `python -m dimos.learning.dataprep build --progress-json`."""
        raise NotImplementedError

    def _spawn_train(
        self,
        spec_path: str,
        output_dir: str,
        config_kind: Literal["bc", "vla"],
        config_overrides: dict[str, Any],
        job_id: str,
    ) -> subprocess.Popen[str]:
        """Build argv for `python -m dimos.learning.training.train <kind> --progress-json`."""
        raise NotImplementedError

    def _stream_build_progress(self, job_id: str, proc: subprocess.Popen[str]) -> int:
        """Read stdout JSON-per-line from the build subprocess; publish each
        as TrainProgress(phase="build"). Returns subprocess exit code."""
        raise NotImplementedError

    def _stream_train_progress(self, job_id: str, proc: subprocess.Popen[str]) -> int:
        """Read stdout JSON-per-line from the train subprocess; publish each
        as TrainProgress(phase in {"load","train","eval","save"}). Returns exit code."""
        raise NotImplementedError

    def _allocate_job_id(self) -> str:
        jid = f"train-{self._next_job_id}"
        self._next_job_id += 1
        return jid

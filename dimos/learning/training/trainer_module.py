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

"""ACT training Module — thin wrapper around `train_bc`.

Spawns `train_bc` on a daemon thread (which subprocesses
``python -m lerobot.scripts.train``). Exposes:

    @rpc start()              lifecycle (auto-fires train if auto_run)
    @rpc train(...)            kick off a training job
    @rpc get_status()          current state + checkpoint dir
    @rpc stop()                best-effort shutdown

There is no cancel(): the lerobot subprocess is sent SIGTERM only on
process exit. Heavy deps (torch, lerobot) stay in the subprocess.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.learning.training.configs import BCConfig
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class TrainerModuleConfig(ModuleConfig):
    dataset_path: str = ""
    output_dir: str = ""
    # ACT hyperparams. CLI override pattern:
    #   -o trainermodule.bc.steps=2000  -o trainermodule.bc.batch_size=4
    bc: BCConfig = BCConfig()
    # Optional JSON file with BCConfig overrides; merged on top of `bc`.
    config_path: str | None = None
    auto_run: bool = False
    overwrite: bool = True   # wipe output_dir before training (lerobot refuses to overwrite)
    resume: bool = False     # pass --resume=true to lerobot
    # Lerobot 0.3.x does not write tensorboard event files; the launch is a
    # no-op there and shows an empty UI. Disabled until we wire a stdout
    # parser → SummaryWriter on our side.
    tensorboard: bool = False
    tensorboard_port: int = 6006
    tensorboard_host: str = "0.0.0.0"


class TrainerModule(Module):
    config: TrainerModuleConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "state":           "idle",   # idle | running | succeeded | failed
            "checkpoint_dir":  None,
            "tensorboard_url": None,
            "error":           None,
        }
        self._tb_proc: subprocess.Popen[bytes] | None = None

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.auto_run:
            self.train()

    @rpc
    def stop(self) -> None:
        # Train thread is daemon: dies with the process. No mid-run interrupt.
        if self._tb_proc is not None and self._tb_proc.poll() is None:
            logger.info("[trainer] stopping tensorboard pid=%s", self._tb_proc.pid)
            self._tb_proc.terminate()
            try:
                self._tb_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._tb_proc.kill()
        super().stop()

    @rpc
    def train(
        self,
        dataset_path: str | None = None,
        output_dir: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Spawn a daemon thread running ``train_bc``; returns immediately.
        Raises if a run is already in progress."""
        with self._lock:
            if self._status["state"] == "running":
                raise RuntimeError("training already in progress")
            self._status.update(state="running", checkpoint_dir=None, error=None)

        ds = dataset_path or self.config.dataset_path
        od = output_dir or self.config.output_dir
        if not ds or not od:
            with self._lock:
                self._status.update(state="failed",
                                    error="dataset_path and output_dir are required")
            raise ValueError("dataset_path and output_dir are required")

        self._maybe_start_tensorboard(Path(od))

        self._thread = threading.Thread(
            target=self._run_training,
            args=(ds, od, config_overrides),
            daemon=True,
        )
        self._thread.start()

    @rpc
    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _run_training(
        self,
        dataset_path: str,
        output_dir: str,
        config_overrides: dict[str, Any] | None,
    ) -> None:
        try:
            from dimos.learning.training.train import train_bc

            cfg_kwargs = self.config.bc.model_dump()
            if self.config.config_path:
                with open(self.config.config_path) as f:
                    cfg_kwargs.update(json.load(f))
            cfg = BCConfig(**cfg_kwargs)

            ckpt = train_bc(
                dataset_path, cfg, output_dir,
                config_overrides=config_overrides,
                overwrite=self.config.overwrite,
                resume=self.config.resume,
            )

            with self._lock:
                self._status.update(state="succeeded", checkpoint_dir=str(ckpt))
        except Exception as e:
            with self._lock:
                self._status.update(
                    state="failed",
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                )

    # ── tensorboard ──────────────────────────────────────────────────────────

    def _maybe_start_tensorboard(self, logdir: Path) -> None:
        """Spawn ``tensorboard --logdir <logdir>`` if enabled and available."""
        if not self.config.tensorboard or self.config.tensorboard_port == 0:
            return
        if self._tb_proc is not None and self._tb_proc.poll() is None:
            return

        tb_bin = shutil.which("tensorboard")
        if tb_bin is None:
            logger.warning(
                "[trainer] tensorboard binary not found on PATH — skipping. "
                "Install with: pip install tensorboard"
            )
            return

        # Do NOT pre-create logdir — lerobot's cfg.validate() refuses if the
        # output dir exists. Tensorboard polls happily on a missing dir.
        port = self.config.tensorboard_port
        host = self.config.tensorboard_host
        try:
            self._tb_proc = subprocess.Popen(
                [tb_bin, "--logdir", str(logdir),
                 "--port", str(port), "--host", host,
                 "--reload_interval", "5"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("[trainer] failed to launch tensorboard: %s", e)
            return

        view_host = "localhost" if host in ("0.0.0.0", "") else host
        url = f"http://{view_host}:{port}/"
        with self._lock:
            self._status["tensorboard_url"] = url
        logger.info("[trainer] tensorboard launched pid=%s — view at %s",
                    self._tb_proc.pid, url)


__all__ = ["TrainerModule", "TrainerModuleConfig"]

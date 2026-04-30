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

"""Training blueprints for the DimOS Learning Framework.

Each blueprint composes TrainerModule + LearningMonitorModule. TrainerModule
handles both the dataset-build and the training subprocesses internally
(see its docstring) — there is no separate builder Module in v1.

Variants:
    learning_train_act    - auto build (if needed) then train ACT (BC)
    learning_train_vla    - auto build (if needed) then finetune pi0/pi0.5
    learning_train_idle   - module idle, agent drives via @rpc

Defaults (spec_path, output_dir, ...) are placeholders; override at run
time via CLI flags or @rpc calls. Per-job overrides on the trigger payload
take precedence over module config.

Usage:
    dimos run learning-train-act \\
        --TrainerModule.config.spec_path dataset.yaml \\
        --TrainerModule.config.output_dir runs/act_pick_red
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.learning.training.monitor_module import LearningMonitorModule
from dimos.learning.training.trainer_module import (
    TrainDone,
    TrainerModule,
    TrainProgress,
)

# Topic names — shared across all variants so monitors / agents subscribe once.
_T_TRAIN_PROGRESS = "/learning/train/progress"
_T_TRAIN_DONE = "/learning/train/done"


# ── ACT (BC) — auto build (if needed) then train ─────────────────────────────

learning_train_act = autoconnect(
    TrainerModule.blueprint(config_kind="bc", auto_run=True),
    LearningMonitorModule.blueprint(log_to_rerun=True),
).transports(
    {
        ("progress", TrainProgress): LCMTransport(_T_TRAIN_PROGRESS, TrainProgress),
        ("done", TrainDone): LCMTransport(_T_TRAIN_DONE, TrainDone),
        ("train_progress", TrainProgress): LCMTransport(_T_TRAIN_PROGRESS, TrainProgress),
        ("train_done", TrainDone): LCMTransport(_T_TRAIN_DONE, TrainDone),
    }
)


# ── pi0 / pi0.5 (VLA finetune) — auto build (if needed) then train ───────────

learning_train_vla = autoconnect(
    TrainerModule.blueprint(config_kind="vla", auto_run=True),
    LearningMonitorModule.blueprint(log_to_rerun=True),
).transports(
    {
        ("progress", TrainProgress): LCMTransport(_T_TRAIN_PROGRESS, TrainProgress),
        ("done", TrainDone): LCMTransport(_T_TRAIN_DONE, TrainDone),
        ("train_progress", TrainProgress): LCMTransport(_T_TRAIN_PROGRESS, TrainProgress),
        ("train_done", TrainDone): LCMTransport(_T_TRAIN_DONE, TrainDone),
    }
)


# ── Idle — TrainerModule waits for explicit @rpc / external trigger ──────────
# Agent-driven: agent skill calls TrainerModule.train(...) (or .build_only(...))
# over RPC. Same module, no auto behavior.

learning_train_idle = autoconnect(
    TrainerModule.blueprint(auto_run=False),
    LearningMonitorModule.blueprint(log_to_rerun=True),
).transports(
    {
        ("progress", TrainProgress): LCMTransport(_T_TRAIN_PROGRESS, TrainProgress),
        ("done", TrainDone): LCMTransport(_T_TRAIN_DONE, TrainDone),
        ("train_progress", TrainProgress): LCMTransport(_T_TRAIN_PROGRESS, TrainProgress),
        ("train_done", TrainDone): LCMTransport(_T_TRAIN_DONE, TrainDone),
    }
)


__all__ = [
    "learning_train_act",
    "learning_train_idle",
    "learning_train_vla",
]

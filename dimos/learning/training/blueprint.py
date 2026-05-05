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

"""ACT training blueprint. RPC-only surface (no streams).

Defaults are tuned for the local pickplace_001 demo: 2k steps, batch=4, CPU.
For real training, override via:
    dimos run learning-train -o trainermodule.bc.steps=100000 -o trainermodule.bc.device=cuda
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.learning.training.configs import BCConfig
from dimos.learning.training.trainer_module import TrainerModule

learning_train = autoconnect(
    TrainerModule.blueprint(
        dataset_path="data/datasets/pickplace_001",
        output_dir="data/runs/act_pickplace_001",
        bc=BCConfig(
            steps=2000,
            batch_size=4,
            device="cpu",
        ),
        auto_run=True,
        overwrite=True,
    ),
).transports({})


__all__ = ["learning_train"]

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

"""ACT training config (v1). Fields are the opinionated subset DimOS exposes;
unset = lerobot default. Translated to Hydra-style argv inside `train_bc`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BCConfig(BaseModel):
    policy_type: Literal["act"] = "act"

    # Action chunking
    chunk_size: int = 50  # future actions per inference call
    n_obs_steps: int = 1  # obs history length

    # ACT model arch
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    use_vae: bool = True
    kl_weight: float = 10.0

    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained: bool = True

    # Optim
    steps: int = 100_000
    batch_size: int = 8
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    # Eval / checkpointing
    save_every: int = 10_000
    eval_every: int = 5_000
    seed: int = 0
    device: str = "cuda"

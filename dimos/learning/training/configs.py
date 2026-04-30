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

"""Trainer configs for v1.

Two pydantic configs, one per training entry point:
  - BCConfig    -> consumed by train_bc      (ACT, optionally Diffusion)
  - VLAConfig   -> consumed by finetune_vla  (pi0, pi0.5)

Both are translated into a `lerobot` training config inside the trainer;
fields here are the small, opinionated subset DimOS users actually need to
tune. Anything not exposed falls back to the lerobot default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BCConfig(BaseModel):
    """Behavior-cloning trainer config (v1: ACT, with Diffusion as a flag)."""

    policy_type: Literal["act", "diffusion"] = "act"

    # Action chunking
    chunk_size: int = 50  # number of future actions predicted per inference call
    n_obs_steps: int = 1  # observation history length passed to the policy

    # ACT model arch (ignored for Diffusion)
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


class VLAConfig(BaseModel):
    """VLA finetune config (v1: pi0, pi0.5)."""

    policy_type: Literal["pi0", "pi0_5"] = "pi0_5"

    # Pretrained checkpoint — HF hub id or local path
    pretrained_path: str

    # Finetune mode
    finetune_mode: Literal["full", "lora"] = "lora"
    lora_rank: int = 16
    freeze_vision: bool = True
    freeze_language: bool = True

    # Action chunking — pi0/pi0.5 default
    chunk_size: int = 50

    # Optim
    steps: int = 30_000
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-4

    # Eval / checkpointing
    save_every: int = 5_000
    eval_every: int = 2_500
    seed: int = 0
    device: str = "cuda"

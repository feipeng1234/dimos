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

"""Training entry points for v1.

Two functions, both thin wrappers around `lerobot`:
  - train_bc(spec, cfg, output_dir)      -> ACT (or Diffusion) BC training
  - finetune_vla(spec, cfg, output_dir)  -> pi0 / pi0.5 finetune

Both:
  1. Materialize the dataset via `DataPrep.build()` if `spec.output.path`
     doesn't already exist (idempotent).
  2. Open the materialized dataset as a `lerobot.LeRobotDataset`.
  3. Translate the DimOS config to a lerobot config, build the policy.
  4. Compute / load `meta/stats.json`.
  5. Compute the train/val split.
  6. Call lerobot's training loop.
  7. Write the checkpoint + a sidecar `dimos_meta.json` so inference can
     reconstruct everything from `output_dir` alone.

We do NOT reimplement the training loop, optimizer schedule, normalization,
action chunking, language tokenization, or checkpoint format. Riding on
lerobot keeps this file small and means a `pi0.5` upgrade is a config bump.
"""

from __future__ import annotations

from pathlib import Path

from dimos.learning.spec import DatasetSpec
from dimos.learning.training.configs import BCConfig, VLAConfig

# Sidecar written next to the lerobot checkpoint so inference can recover
# the spec + dataset path that produced this policy.
DIMOS_META_FILENAME = "dimos_meta.json"


def train_bc(spec: DatasetSpec, cfg: BCConfig, output_dir: str | Path) -> Path:
    """Train an ACT (or Diffusion) BC policy on `spec`.

    Returns the path to the final checkpoint directory. The returned dir
    contains the lerobot checkpoint + `dimos_meta.json` linking back to the
    spec and dataset used.
    """
    raise NotImplementedError


def finetune_vla(spec: DatasetSpec, cfg: VLAConfig, output_dir: str | Path) -> Path:
    """Finetune a pretrained pi0 / pi0.5 on `spec`.

    Loads `cfg.pretrained_path` (HF hub id or local), wraps it for the
    requested `finetune_mode` (full or LoRA), runs lerobot's training loop,
    and writes the resulting checkpoint to `output_dir`.

    Returns the checkpoint directory path.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Internals — translate DimOS configs into the lerobot training entry point.
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_dataset(spec: DatasetSpec) -> Path:
    """If `spec.output.path` doesn't exist on disk yet, run `DataPrep.build()`.

    Returns the resolved dataset path. Raises if `spec.output` is None
    (training requires a materialized dataset).
    """
    raise NotImplementedError


def _build_lerobot_config_bc(spec: DatasetSpec, cfg: BCConfig, dataset_path: Path) -> object:
    """Translate a DimOS BCConfig + spec into a lerobot training config.

    Returns the lerobot config object opaque to the rest of this file —
    everything lerobot-specific stays inside the implementation.
    """
    raise NotImplementedError


def _build_lerobot_config_vla(spec: DatasetSpec, cfg: VLAConfig, dataset_path: Path) -> object:
    """Translate a DimOS VLAConfig + spec into a lerobot training config."""
    raise NotImplementedError


def _write_dimos_meta(output_dir: Path, spec: DatasetSpec, dataset_path: Path) -> None:
    """Write `dimos_meta.json` next to the checkpoint.

    Schema:
        {
            "dimos_version": "...",
            "spec": <round-tripped DatasetSpec JSON>,
            "dataset_path": "<absolute path>",
            "lerobot_version": "..."
        }
    Used by inference to rehydrate the spec without a separate yaml.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entrypoint:

        python -m dimos.learning.training.train bc  <spec.yaml> --output <dir> [...]
        python -m dimos.learning.training.train vla <spec.yaml> --output <dir> --pretrained <id> [...]
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()

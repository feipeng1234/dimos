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

"""LeRobot ACT policy wrapper. Lazy-imports lerobot/torch in load()."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dimos.learning.policy.base import Policy

DIMOS_META_FILENAME = "dimos_meta.json"


class LeRobotPolicy:
    _model: Any  # lerobot ACTPolicy / PreTrainedPolicy
    _stats: dict[str, Any]
    _dimos_meta: dict[str, Any]
    _chunk_size: int
    _joint_names: list[str]
    _device: str

    def __init__(
        self,
        model: Any,
        stats: dict[str, Any],
        dimos_meta: dict[str, Any],
        chunk_size: int,
        joint_names: list[str],
        device: str,
    ) -> None:
        self._model = model
        self._stats = stats
        self._dimos_meta = dimos_meta
        self._chunk_size = chunk_size
        self._joint_names = joint_names
        self._device = device

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> LeRobotPolicy:
        """Load checkpoint + dataset stats + dimos_meta sidecar.

        ``path`` may be a TrainerModule output dir (we walk into
        ``checkpoints/last/pretrained_model/``) or an exact
        ``pretrained_model`` dir.
        """
        path = Path(path)
        pretrained_dir, run_dir = _resolve_checkpoint_dirs(path)

        meta_path = run_dir / DIMOS_META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {DIMOS_META_FILENAME} in {run_dir}")
        with open(meta_path) as f:
            dimos_meta = json.load(f)

        stats_path = _find_stats(run_dir, dimos_meta)
        with open(stats_path) as f:
            stats = json.load(f)

        # Lazy import — keeps torch/CUDA out of the dimos runtime at module load.
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError:
            try:
                from lerobot.common.policies.act.modeling_act import ACTPolicy
            except ImportError as e:
                raise RuntimeError(
                    "lerobot is required to load a checkpoint; install with "
                    "`pip install lerobot` (>=0.3)"
                ) from e

        model = ACTPolicy.from_pretrained(str(pretrained_dir))
        model.eval()
        model.to(device)

        chunk_size = int(dimos_meta.get("chunk_size", 50))
        joint_names = dimos_meta.get("joint_names") or _infer_joint_names(model)

        return cls(
            model=model,
            stats=stats,
            dimos_meta=dimos_meta,
            chunk_size=chunk_size,
            joint_names=joint_names,
            device=device,
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass on ``obs``. Returns shape ``(chunk_size, action_dim)``.

        ``obs`` keys are the dataset spec's observation keys (e.g. "image",
        "joint_state"). They are translated to lerobot's canonical names
        (``observation.images.*`` / ``observation.state``) using the
        dimos_meta sidecar so train and infer agree.
        """
        import torch  # lazy

        batch = self._build_batch(obs)
        with torch.inference_mode():
            chunk = self._forward_chunk(batch)
        if chunk.ndim == 3:
            chunk = chunk[0]  # (B, T, A) → (T, A)
        return chunk.detach().cpu().numpy()

    # ── internals ────────────────────────────────────────────────────────────

    def _build_batch(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        import torch

        batch: dict[str, Any] = {}
        observation_map = self._dimos_meta.get("observation", {})
        for user_key, value in obs.items():
            arr = np.asarray(value)
            if arr.ndim >= 3:
                # HWC uint8 → 1xCxHxW float32 / 255 (lerobot's expected layout).
                chw = np.transpose(arr, (2, 0, 1)) if arr.shape[-1] in (1, 3, 4) else arr
                t = torch.from_numpy(chw.astype(np.float32) / 255.0).unsqueeze(0)
                feat = f"observation.images.{user_key}"
            else:
                t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
                # Single low-dim observation is canonical "observation.state".
                low_dim_other = any(
                    k != user_key and k in obs and np.asarray(obs[k]).ndim < 3
                    for k in observation_map
                )
                feat = f"observation.{user_key}" if low_dim_other else "observation.state"
            batch[feat] = t.to(self._device)
        return batch

    def _forward_chunk(self, batch: dict[str, Any]) -> Any:
        """Prefer ``predict_action_chunk`` (newer API); fall back to repeated
        ``select_action`` after ``reset()`` to assemble a chunk."""
        if hasattr(self._model, "predict_action_chunk"):
            return self._model.predict_action_chunk(batch)
        if hasattr(self._model, "select_action"):
            import torch

            self._model.reset()
            actions = [self._model.select_action(batch) for _ in range(self._chunk_size)]
            return torch.stack(actions, dim=1)  # (B, T, A)
        raise RuntimeError("lerobot policy has neither predict_action_chunk nor select_action")


def _resolve_checkpoint_dirs(path: Path) -> tuple[Path, Path]:
    """Return ``(pretrained_model_dir, run_dir)`` for any supported input path.

    Run dir layout (lerobot 0.3+)::

        <run_dir>/
          dimos_meta.json
          checkpoints/<step>/pretrained_model/   # lerobot safetensors
          checkpoints/last -> symlink to latest
    """
    if (path / "model.safetensors").exists():
        # `…/checkpoints/<step>/pretrained_model` → run_dir is 3 parents up.
        return path, path.parent.parent.parent

    last = path / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last, path

    ckpts = path / "checkpoints"
    if ckpts.is_dir():
        numeric = sorted(
            (p for p in ckpts.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        if numeric and (numeric[-1] / "pretrained_model").exists():
            return numeric[-1] / "pretrained_model", path

    raise FileNotFoundError(
        f"No lerobot checkpoint found under {path}. "
        f"Expected {path}/checkpoints/last/pretrained_model/ "
        f"or a numeric checkpoint subdir."
    )


def _find_stats(run_dir: Path, dimos_meta: dict[str, Any]) -> Path:
    """Locate ``stats.json`` near a checkpoint.

    Lookup order:
      1. ``<run_dir>/meta/stats.json``
      2. dimos_meta's recorded ``dataset_path`` / ``source``
      3. ``<run_dir>/../datasets/<name>/meta/stats.json`` (sibling convention)
    """
    candidates: list[Path] = [
        run_dir / "meta" / "stats.json",
        run_dir / "stats.json",
    ]
    metadata = dimos_meta.get("metadata") or {}
    for key in ("dataset_path", "source"):
        v = metadata.get(key) or dimos_meta.get(key)
        if v and Path(v).suffix not in (".db", ".sqlite"):
            candidates.append(Path(v) / "meta" / "stats.json")

    for parent in (run_dir.parent, run_dir.parent.parent):
        if parent and (parent / "datasets").is_dir():
            for d in (parent / "datasets").iterdir():
                if (d / "meta" / "stats.json").is_file():
                    candidates.append(d / "meta" / "stats.json")

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"stats.json not found near {run_dir}; tried: {candidates}")


def _infer_joint_names(model: Any) -> list[str]:
    """Synthetic joint-name fallback when dimos_meta didn't record any."""
    cfg = getattr(model, "config", None)
    action_dim: int | None = None
    if cfg is not None:
        out_shapes = getattr(cfg, "output_shapes", None) or {}
        if "action" in out_shapes:
            action_dim = out_shapes["action"][-1]
        if action_dim is None:
            af = getattr(cfg, "action_feature", None)
            action_dim = getattr(af, "shape", [None])[-1] if af is not None else None
    if action_dim is None:
        action_dim = 7
    return [f"joint{i}" for i in range(action_dim)]


# Protocol conformance assertion at import time.
_: type[Policy] = LeRobotPolicy

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

"""ACT training entry point.

`train_bc` subprocesses ``python -m lerobot.scripts.train`` with argv
translated from `BCConfig`. Lerobot is never imported in-process so the
dimos runtime stays free of torch/CUDA. After a successful run we write
``dimos_meta.json`` next to the checkpoint so `LeRobotPolicy.load` can
recover the obs/action schema.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dimos.learning.dataprep import Episode
from dimos.learning.training.configs import BCConfig

DIMOS_META_FILENAME = "dimos_meta.json"


def train_bc(
    dataset_path: str | Path,
    cfg: BCConfig,
    output_dir: str | Path,
    config_overrides: dict[str, Any] | None = None,
    overwrite: bool = True,
    resume: bool = False,
) -> Path:
    """Train ACT on a prepared LeRobot v2 dataset. Returns the checkpoint dir.

    Args:
        overwrite: if True (default) wipes ``output_dir`` before launching.
            Lerobot's ``cfg.validate()`` refuses to run if the dir exists.
        resume: pass ``--resume=true`` to lerobot. Takes precedence over
            ``overwrite``.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    if resume:
        overwrite = False
    elif overwrite and output_dir.exists():
        print(f"[train_bc] removing existing {output_dir}", flush=True)
        shutil.rmtree(output_dir)

    argv = _build_lerobot_argv(cfg, dataset_path, output_dir)
    if resume:
        argv.append("--resume=true")
    if config_overrides:
        for k, v in config_overrides.items():
            argv.append(f"--{k}={v}")

    print(f"[train_bc] launching lerobot ({len(argv)} args, output → {output_dir})", flush=True)
    # Log alongside output_dir, not inside — lerobot creates output_dir itself
    # and its validate() refuses if it already exists.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.lerobot.log"

    # Stream lerobot stdout to a full log file + filtered terminal output.
    interesting = (
        "step:", "loss:", "Logs will be", "Creating dataset", "Output dir",
        "Saved checkpoint", "epoch", "loss", "lr=", "ETA", "ERROR", "Error",
        "WARNING", "Traceback",
    )
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log_path, "w") as logf:
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if any(kw in line for kw in interesting):
                sys.stdout.write(line)
                sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        print(f"[train_bc] lerobot exited {proc.returncode} — full log: {log_path}", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, argv)

    _write_dimos_meta(output_dir, dataset_path, cfg)
    return output_dir


def _build_lerobot_argv(cfg: BCConfig, dataset_path: Path, output_dir: Path) -> list[str]:
    """Translate BCConfig → argv for ``lerobot.scripts.train`` (lerobot 0.3.x).

    LeRobot 0.4.x renamed the entry point to ``lerobot.scripts.lerobot_train``
    and adjusted some draccus flag names — adjust this function if you pin
    a different version.
    """
    return [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--policy.type={cfg.policy_type}",
        f"--policy.chunk_size={cfg.chunk_size}",
        f"--policy.n_action_steps={cfg.chunk_size}",
        f"--policy.n_obs_steps={cfg.n_obs_steps}",
        f"--policy.dim_model={cfg.hidden_dim}",
        f"--policy.n_encoder_layers={cfg.n_layers}",
        f"--policy.n_decoder_layers={cfg.n_layers}",
        f"--policy.n_heads={cfg.n_heads}",
        f"--policy.use_vae={str(cfg.use_vae).lower()}",
        f"--policy.kl_weight={cfg.kl_weight}",
        f"--policy.vision_backbone={cfg.vision_backbone}",
        f"--policy.pretrained_backbone_weights={'ResNet18_Weights.IMAGENET1K_V1' if cfg.pretrained else 'null'}",
        f"--policy.device={cfg.device}",
        # push_to_hub defaults True in lerobot and triggers a repo_id requirement.
        "--policy.push_to_hub=false",
        "--dataset.repo_id=local",
        f"--dataset.root={dataset_path}",
        f"--steps={cfg.steps}",
        f"--batch_size={cfg.batch_size}",
        f"--optimizer.lr={cfg.lr}",
        f"--optimizer.weight_decay={cfg.weight_decay}",
        f"--save_freq={cfg.save_every}",
        f"--eval_freq={cfg.eval_every}",
        "--wandb.enable=false",
        f"--seed={cfg.seed}",
        f"--output_dir={output_dir}",
        # Note: do NOT pass --env — its choice-class decoder rejects "none";
        # leaving it unset disables eval cleanly.
    ]


def _write_dimos_meta(output_dir: Path, dataset_path: Path, cfg: BCConfig) -> None:
    """Write the inference sidecar at ``<output_dir>/dimos_meta.json``.

    Combines the dataset's dimos_meta (obs/action streams, sync) with policy
    fields (type, chunk_size, n_obs_steps) and the dataset_path so
    `LeRobotPolicy.load` can resolve `meta/stats.json`.
    """
    src = dataset_path / DIMOS_META_FILENAME
    base: dict[str, Any] = json.load(open(src)) if src.exists() else {}
    base.update({
        "dataset_path": str(dataset_path),
        "policy_type":  cfg.policy_type,
        "chunk_size":   cfg.chunk_size,
        "n_obs_steps":  cfg.n_obs_steps,
        "joint_names":  base.get("joint_names"),  # often None; inference falls back
    })
    with open(output_dir / DIMOS_META_FILENAME, "w") as f:
        json.dump(base, f, indent=2, default=str)


def train_val_split(
    episodes: list[Episode],
    val_episode_ids: list[int] | None = None,
    val_ratio: float | None = None,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Partition episode indices into (train_ids, val_ids).

    Resolution order: ``val_episode_ids`` (whitelist) > ``val_ratio``
    (deterministic via ``seed``) > both None (everything in train).
    """
    n = len(episodes)
    all_ids = list(range(n))

    if val_episode_ids is not None:
        val_set = set(val_episode_ids)
        return ([i for i in all_ids if i not in val_set],
                [i for i in all_ids if i in val_set])

    if val_ratio is not None:
        rng = random.Random(seed)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        n_val = int(round(n * val_ratio))
        return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])

    return all_ids, []


# ─────────────────────────────────────────────────────────────────────────────
# CLI: `python -m dimos.learning.training.train bc <dataset> --output ...`
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(prog="dimos.learning.training.train")
    sub = parser.add_subparsers(dest="kind", required=True)

    p_bc = sub.add_parser("bc", help="Train an ACT (BC) policy")
    p_bc.add_argument("dataset", help="path to LeRobot v2 dataset directory")
    p_bc.add_argument("--output", required=True, help="checkpoint output directory")
    p_bc.add_argument("--config", help="path to BCConfig JSON override")
    p_bc.add_argument("--steps", type=int)
    p_bc.add_argument("--batch-size", type=int)
    p_bc.add_argument("--chunk-size", type=int)
    p_bc.add_argument("--device", type=str)
    p_bc.add_argument("-o", "--override", action="append", default=[],
                      help="extra lerobot CLI override, e.g. -o optimizer.lr=5e-5")

    args = parser.parse_args()

    if args.kind == "bc":
        cfg_kwargs: dict[str, Any] = json.load(open(args.config)) if args.config else {}
        for k, v in (("steps", args.steps), ("batch_size", args.batch_size),
                     ("chunk_size", args.chunk_size), ("device", args.device)):
            if v is not None:
                cfg_kwargs[k] = v
        cfg = BCConfig(**cfg_kwargs)

        overrides: dict[str, Any] = {}
        for o in args.override:
            if "=" not in o:
                parser.error(f"--override must be key=value, got {o!r}")
            k, v = o.split("=", 1)
            overrides[k] = v

        out = train_bc(args.dataset, cfg, args.output, config_overrides=overrides)
        print(f"[train_bc] checkpoint at: {out}")


if __name__ == "__main__":
    main()

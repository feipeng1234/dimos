# Stage 2 — Training

ACT only (BC). Reads `data/datasets/<name>/`, writes `data/runs/<name>/`.

`TrainerModule` runs `train_bc(...)` on a daemon thread inside its own
worker. Lazy imports keep `lerobot` / `torch` / CUDA out of the worker
until `train()` is called. Metrics → TensorBoard. No `cancel()` in v1.

---

## Blueprint

```python
# dimos/learning/training/blueprint.py
learning_train = autoconnect(
    TrainerModule.blueprint(
        dataset_path="data/datasets/pick_red/",
        output_dir="data/runs/act_pick_red",
        auto_run=True,
    ),
).transports({})
```

## Module

```python
# dimos/learning/training/trainer_module.py

class TrainerModuleConfig(ModuleConfig):
    dataset_path:     str = ""
    output_dir:       str = ""
    config_path:      str | None = None     # optional BCConfig YAML override
    auto_run:         bool = False
    tensorboard_port: int = 6006


class TrainerModule(Module):
    config: TrainerModuleConfig

    @rpc
    def train(
        self,
        dataset_path:     str | None = None,
        output_dir:       str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None: ...
    @rpc
    def get_status(self) -> dict[str, Any]: ...
```

## Training entry point

```python
# dimos/learning/training/train.py

def train_bc(
    dataset_path:     str | Path,
    cfg:              BCConfig,
    output_dir:       str | Path,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Lazy-import lerobot, build Hydra-style argv from BCConfig, call
    lerobot's training entry point, append `dimos_meta.json` to output_dir,
    return the checkpoint path."""
```

`BCConfig` (ACT hyperparams) lives in `training/configs.py`.
`train_val_split()` lives next to `train_bc` in `training/train.py`.

## Run

```bash
dimos run learning-train
tensorboard --logdir data/runs/act_pick_red
```

Artifact: `data/runs/act_pick_red/` = `*.safetensors` + `dimos_meta.json`
(dataset snapshot + `joint_names`, `chunk_size`, `policy_type`).

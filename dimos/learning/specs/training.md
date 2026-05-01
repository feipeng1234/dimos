# Stage 2 — Training

Offline. Reads `dataset/`, writes `checkpoint/` + `dimos_meta.json`.

`TrainerModule` is an RPC façade over a training subprocess. Metrics →
TensorBoard. Lifecycle → `get_status()`.

---

## Blueprint

```python
# dimos/learning/training/blueprint.py
learning_train = autoconnect(
    TrainerModule.blueprint(auto_run=True),
).transports({})
```

## Module

```python
class TrainerModuleConfig(ModuleConfig):
    dataset_path:     str = ""
    output_dir:       str = ""
    config_kind:      Literal["bc", "vla"] = "bc"
    config_path:      str | None = None
    auto_run:         bool = False
    tensorboard_port: int = 6006


class TrainerModule(Module):
    config: TrainerModuleConfig

    @rpc
    def train(
        self,
        dataset_path:     str | None = None,
        output_dir:       str | None = None,
        config_kind:      Literal["bc", "vla"] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None: ...
    @rpc
    def cancel(self) -> bool: ...
    @rpc
    def get_status(self) -> dict[str, Any]: ...
```

## Run

```bash
dimos run learning-train \
  --dataset-path dataset/ \
  --output-dir   runs/act_pick_red \
  --config-kind  bc

tensorboard --logdir runs/act_pick_red
```

Artifact: `checkpoint/` = `*.safetensors` + `dimos_meta.json` (spec snapshot,
`joint_names`, `chunk_size`, `policy_type`, `expects_language`).

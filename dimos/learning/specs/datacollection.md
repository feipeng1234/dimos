# Stage 1 — Data

Two phases:

1. **Recording** — live; operator drives the robot. `RecordReplay` writes streams to `session.db`.
2. **DataPrep** — offline; convert `session.db` → `dataset/`.

One `dataset.yaml` drives both, parsed once into a `DatasetConfig`.

---

## Phase A — Recording

### Blueprint

```python
# dimos/learning/collection/blueprint.py
spec = DatasetConfig.from_file("datasets/pick_red.yaml")

learning_collect_quest_xarm7 = autoconnect(
    teleop_quest_xarm7,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(spec=spec),
).transports({
    ("buttons",     Buttons):       LCMTransport("/teleop/buttons",          Buttons),
    ("color_image", Image):         LCMTransport("/camera/color_image",      Image),
    ("status",      EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
})
```

`RecordReplay` (`--record-path`) captures every transport above, including `episode_status`.

---

### Dataset spec — YAML

```yaml
# datasets/pick_red.yaml
source: session.db

episodes:
  extractor: episode_status
  status_stream: episode_status
  default_task_label: pick_red_cube
  button_map:    {start: A,     save: B, discard: X}
  keyboard_map:  {start: space, save: s, discard: d}

observation:
  cam:
    stream: camera_color_image
    field: image
  joint_pos:
    stream: coordinator_joint_state
    field: position

action:
  joint_target:
    stream: coordinator_joint_command
    field: position

sync:
  anchor: cam
  rate_hz: 30
  tolerance_ms: 50
  strategy: nearest

output:
  format: lerobot
  path: datasets/pick_red/
  metadata: {fps: 30, robot: xarm7}
```

---

### Dataset spec — pydantic classes

```python
# dimos/learning/config.py
from dimos.protocol.service.spec import BaseConfig    # extra="forbid"


class DatasetConfig(BaseConfig):
    source:      str
    episodes:    EpisodeConfig
    observation: dict[str, StreamField]
    action:      dict[str, StreamField]
    sync:        SyncConfig
    output:      OutputConfig

    @classmethod
    def from_file(cls, path: str | Path) -> DatasetConfig: ...


class EpisodeConfig(BaseConfig):
    extractor:     Literal["episode_status", "ranges", "whole_session"] = "episode_status"
    status_stream: str = "episode_status"
    ranges:        list[tuple[float, float]] | None = None
    default_task_label: str | None = None
    button_map:    dict[Literal["start", "save", "discard"], str] = {"start": "A", "save": "B", "discard": "X"}
    keyboard_map:  dict[Literal["start", "save", "discard"], str] = {}


class StreamField(BaseConfig):
    stream: str
    field:  str | None = None


class SyncConfig(BaseConfig):
    anchor:       str
    rate_hz:      float
    tolerance_ms: float
    strategy:     Literal["nearest", "interp"] = "nearest"


class OutputConfig(BaseConfig):
    format:   Literal["lerobot", "hdf5", "rlds"] = "lerobot"
    path:     Path
    metadata: dict[str, Any] = {}
```

| Module | Reads |
|---|---|
| `EpisodeMonitorModule` | `spec.episodes.button_map` / `keyboard_map` / `default_task_label` |
| `DataPrepModule`       | full spec |
| `ChunkPolicyModule`    | `spec.observation`, `spec.sync` |

---

### EpisodeMonitorModule

Translates teleop input (buttons, keyboard, future inputs) into a canonical
`EpisodeStatus` stream. `DataPrep` reads only that stream — never raw inputs.

```python
# dimos/learning/collection/episode_monitor.py

class EpisodeStatus(BaseModel):
    state: Literal["idle", "recording"]
    episodes_saved:           int
    episodes_discarded:       int
    current_episode_start_ts: float | None
    last_event: Literal["start", "save", "discard", "init"] = "init"
    task_label: str | None = None


class EpisodeMonitorModuleConfig(ModuleConfig):
    spec: DatasetConfig


class EpisodeMonitorModule(Module):
    config: EpisodeMonitorModuleConfig

    buttons:  In[Buttons]
    keyboard: In[KeyPress]
    status:   Out[EpisodeStatus]

    @rpc
    def reset_counters(self) -> EpisodeStatus: ...
    @rpc
    def get_status(self)     -> EpisodeStatus: ...

    def _on_buttons(self,  msg: Buttons)  -> None: ...
    def _on_keyboard(self, msg: KeyPress) -> None: ...
```

State machine:

```
IDLE      --start-->     RECORDING
RECORDING --save-->      IDLE        (commit)
RECORDING --discard-->   IDLE        (drop)
RECORDING --start-->     RECORDING   (auto-commit prev)
session end mid-episode: always discard
```

---

### Run

```bash
dimos run learning-collect-quest-xarm7 \
  --spec-path   datasets/pick_red.yaml \
  --record-path data/pick_red.db
```

---

## Phase B — DataPrep

Reads `session.db`, slices on `episode_status`, syncs streams, writes
`dataset/`. Heavy deps run in a subprocess.

### Blueprint

```python
# dimos/learning/dataprep/blueprint.py
spec = DatasetConfig.from_file("datasets/pick_red.yaml")

learning_dataprep = autoconnect(
    DataPrepModule.blueprint(spec=spec, auto_run=True),
).transports({})
```

### DataPrepModule

```python
# dimos/learning/dataprep_module.py

class DataPrepModuleConfig(ModuleConfig):
    spec:       DatasetConfig
    output_dir: str | None = None
    auto_run:   bool       = False


class DataPrepModule(Module):
    config: DataPrepModuleConfig

    @rpc
    def build(self, output_dir: str | None = None) -> None: ...
    @rpc
    def cancel(self) -> bool: ...
    @rpc
    def get_status(self) -> dict[str, Any]: ...
    @rpc
    def inspect(self) -> dict[str, Any]: ...
```

### Run

```bash
dimos run learning-dataprep --spec-path datasets/pick_red.yaml
```

---

## End-to-end

```bash
SPEC=datasets/pick_red.yaml

dimos run learning-collect-quest-xarm7 --spec-path $SPEC --record-path data/pick_red.db
dimos run learning-dataprep            --spec-path $SPEC
```

```
session.db ─► dataset/ + meta/stats.json
```

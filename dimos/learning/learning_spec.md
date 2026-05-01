### Data collection

Two phases:
A. Recording - teleop/drive robot, streams recorded at `session.db`
B. DataPrep - convert `session.db` to `dataset/`

Phase A - Recording
``` python

spec = DatasetConfig.from_file("datasets/pick_cube.yaml")

learning_collect_quest_xarm7 = autoconnect(
    teleop_quest_xarm7,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(spec=spec),
)

# hardware
# sim
```

`RecordReplay` is a transport-layer hook (`--record-path`); captures every
transport in the blueprint (including `episode_status`) into `session.db`.

Open: feedback to operators — display episode number / status in the VR
headset. Unify episode-boundary declaration across VR / keyboard / active+passive
/ future inputs into one `episode_status` stream in the `.db` file.

``` python
from dimos.learning.config import DatasetConfig, EpisodeStatus

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

Config
```yaml
source: session.db

episodes:
  extractor: episode_status
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

`learning/config.py`

``` python
from dimos.protocol.service.spec import BaseConfig


class EpisodeStatus(BaseModel):           # runtime message (not BaseConfig — built in code, not from YAML)
    state: Literal["idle", "recording"]
    episodes_saved:           int
    episodes_discarded:       int
    current_episode_start_ts: float | None
    last_event: Literal["start", "save", "discard", "init"] = "init"
    task_label: str | None = None


class EpisodeConfig(BaseConfig):
    extractor:     Literal["episode_status", "ranges", "whole_session"]
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


class DatasetConfig(BaseConfig):
    source:      str
    episodes:    EpisodeConfig
    observation: dict[str, StreamField]
    action:      dict[str, StreamField]
    sync:        SyncConfig
    output:      OutputConfig

    @classmethod
    def from_file(cls, path: str | Path) -> DatasetConfig: ...
```

Phase B - DataPrep

```
spec = DatasetConfig.from_file("datasets/pick_red.yaml")

learning_dataprep = autoconnect(
    DataPrepModule.blueprint(spec=spec),
).transports({})
```

- DataPrepModule

``` python
class DataPrepModuleConfig(ModuleConfig):
    spec:       DatasetConfig
    output_dir: str | None = None

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

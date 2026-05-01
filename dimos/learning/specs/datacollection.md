# Stage 1 — Data

1. **Recording** — live; `RecordReplay` writes streams to `session.db`.
2. **DataPrep** — offline; `session.db` → `dataset/` (LeRobot v2).

---

## Phase A — Recording

### Blueprint

```python
# dimos/learning/collection/blueprint.py
learning_collect_quest_xarm7 = autoconnect(
    teleop_quest_xarm7,
    RealSenseCamera.blueprint(enable_pointcloud=False),
    EpisodeMonitorModule.blueprint(
        button_map={"start": "A", "save": "B", "discard": "X"},
        default_task_label="pick_red_cube",
    ),
).transports({
    ("buttons",     Buttons):       LCMTransport("/teleop/buttons",          Buttons),
    ("color_image", Image):         LCMTransport("/camera/color_image",      Image),
    ("status",      EpisodeStatus): LCMTransport("/learning/episode_status", EpisodeStatus),
})
```

`RecordReplay` (`--record-path`) captures every transport above into `session.db`.

### EpisodeMonitorModule

Translates teleop input (buttons, keyboard) into the canonical
`EpisodeStatus` stream. DataPrep reads only this stream — never raw inputs.

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
    button_map:   dict[Literal["start", "save", "discard"], str] = {"start": "A", "save": "B", "discard": "X"}
    keyboard_map: dict[Literal["start", "save", "discard"], str] = {}
    default_task_label: str | None = None


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

### Run

```bash
dimos run learning-collect-quest-xarm7 --record-path data/sessions/pick_red.db
```

---

## Phase B — DataPrep

### Blueprint

```python
# dimos/learning/dataprep/blueprint.py
learning_dataprep = autoconnect(
    DataPrepModule.blueprint(
        source="data/sessions/pick_red.db",
        episodes=EpisodeExtractor(),
        observation={
            "cam":       StreamField(stream="camera_color_image",      field="image"),
            "joint_pos": StreamField(stream="coordinator_joint_state", field="position"),
        },
        action={
            "joint_target": StreamField(stream="coordinator_joint_command", field="position"),
        },
        sync=SyncConfig(anchor="cam", rate_hz=30, tolerance_ms=50),
        output=OutputConfig(format="lerobot", path=Path("data/datasets/pick_red/")),
        auto_run=True,
    ),
).transports({})
```

### DataPrepModule

```python
# dimos/learning/dataprep_module.py
from dimos.protocol.service.spec import BaseConfig


class EpisodeExtractor(BaseConfig):
    extractor:     Literal["episode_status", "ranges", "whole_session"] = "episode_status"
    status_stream: str = "episode_status"
    ranges:        list[tuple[float, float]] | None = None


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


class DataPrepModuleConfig(ModuleConfig):
    source:      str
    episodes:    EpisodeExtractor
    observation: dict[str, StreamField]
    action:      dict[str, StreamField]
    sync:        SyncConfig
    output:      OutputConfig
    auto_run:    bool = False


class DataPrepModule(Module):
    config: DataPrepModuleConfig

    @rpc
    def build(self) -> None: ...
    @rpc
    def get_status(self) -> dict[str, Any]: ...
    @rpc
    def inspect(self) -> dict[str, Any]: ...
```

`build()` iterates samples, hands them to the format writer, and snapshots
`config.model_dump()` into `<output.path>/dimos_meta.json`. Stats are
written into `meta/stats.json` by `DataPrep.compute_stats`.

### Run

```bash
dimos run learning-dataprep
```

---

## End-to-end

```bash
dimos run learning-collect-quest-xarm7 --record-path data/sessions/pick_red.db
dimos run learning-dataprep
```

```
data/sessions/pick_red.db ─► data/datasets/pick_red/
                                  ├── data/    (parquet)
                                  ├── videos/  (MP4)
                                  └── meta/    (info.json, episodes.jsonl,
                                                stats.json, dimos_meta.json)
```

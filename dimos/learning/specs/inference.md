# Stage 3 — Inference

ACT only. Two pieces:

- **`ChunkPolicyModule`** (`learning/inference/`) — Module @ ~30 Hz.
  Builds obs, calls `policy.predict_chunk(obs)`, emits `ActionChunk`.
- **`ActionReplayer`** (`control/tasks/`) — `BaseControlTask` in the
  100 Hz `ControlCoordinator` tick loop. Buffers chunks, interpolates
  to `state.now`, emits `JointCommandOutput`.

```
ChunkPolicyModule (~15-30 Hz)
    │ ActionChunk (LCM)
    ▼
ControlCoordinator @ 100 Hz
    └─ ActionReplayer.compute(state) → JointCommandOutput → hardware
```

---

## Blueprint

```python
# dimos/learning/inference/blueprint.py
from dimos.learning.policy.base import ActionChunk

learning_infer_xarm7 = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(
        policy_path="data/runs/act_pick_red",
        inference_rate_hz=30.0,
    ),
    coordinator_action_replayer_xarm7,    # registers ActionReplayer with the coordinator
).transports({
    ("color_image",  Image):       LCMTransport("/camera/color_image",      Image),
    ("joint_state",  JointState):  LCMTransport("/coordinator/joint_state", JointState),
    ("action_chunk", ActionChunk): LCMTransport("/learning/action_chunk",   ActionChunk),
})
```

## Message types

```python
# dimos/learning/policy/base.py

class ActionChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ts:          float
    joint_names: list[str]
    positions:   np.ndarray   # (T, N)
    dt:          float
    chunk_id:    int


@runtime_checkable
class Policy(Protocol):
    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> Policy: ...
    @property
    def chunk_size(self)  -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...
```

`LeRobotPolicy` (`policy/lerobot_policy.py`) is the v1 implementation.

## ChunkPolicyModule

```python
# dimos/learning/inference/chunk_policy_module.py
from dimos.learning.dataprep import StreamField, SyncConfig, resolve_field


class ChunkPolicyModuleConfig(ModuleConfig):
    policy_path:       str
    inference_rate_hz: float = 30.0
    device:            str = "cuda"


class ChunkPolicyModule(Module):
    config: ChunkPolicyModuleConfig

    color_image:  In[Image]
    joint_state:  In[JointState]
    action_chunk: Out[ActionChunk]

    @rpc
    def reload_policy(self, policy_path: str, device: str | None = None) -> None: ...
    @rpc
    def get_status(self) -> dict[str, Any]: ...

    def _run_loop(self) -> None:
        period = 1.0 / self.config.inference_rate_hz
        while not self._stop.is_set():
            t0  = time.monotonic()
            obs = self._build_live_obs()
            if obs is None:
                time.sleep(period); continue

            positions = self.policy.predict_chunk(obs)
            self.action_chunk.publish(ActionChunk(
                ts=time.time(),
                joint_names=self.policy.joint_names,
                positions=positions,
                dt=period,
                chunk_id=self._next_chunk_id(),
            ))
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _build_live_obs(self) -> dict[str, np.ndarray] | None:
        # snapshot latched In[Image] / In[JointState], project via
        # resolve_field using self._observation (StreamField map
        # loaded from <policy_path>/dimos_meta.json at start()).
        ...
```

`start()` reads `<policy_path>/dimos_meta.json`, reconstructs
`observation: dict[str, StreamField]` and `sync: SyncConfig`, and stores
them as instance state. `_build_live_obs` calls `resolve_field`
on each entry — same projection as training, no train/serve skew.

## ActionReplayer

```python
# dimos/control/tasks/action_replayer_task.py

@dataclass
class ActionReplayerConfig:
    joint_names:       list[str]
    priority:          int = 10
    max_chunk_age_s:   float = 0.5
    hold_on_stall:     bool = True
    temporal_ensemble: bool = False


class ActionReplayer(BaseControlTask):
    def __init__(self, name: str, config: ActionReplayerConfig) -> None: ...

    @property
    def name(self) -> str: ...
    def claim(self)     -> ResourceClaim: ...
    def is_active(self) -> bool: ...
    def compute(self, state: CoordinatorState) -> JointCommandOutput | None: ...
    def on_action_chunk(self, msg: ActionChunk) -> None: ...
```

## ControlCoordinator wiring

`ControlCoordinator` gains a new port + dispatcher (mirrors how
`cartesian_command` / `twist_command` are routed):

```python
class ControlCoordinator(Module):
    # ... existing ports ...
    action_chunk: In[ActionChunk]

    def _on_action_chunk(self, msg: ActionChunk) -> None:
        for task in self._tasks:
            if isinstance(task, ActionReplayer):
                task.on_action_chunk(msg)
```

---

## Run

```bash
dimos run learning-infer-xarm7
```

## End-to-end

```bash
dimos run learning-collect-quest-xarm7 --record-path data/sessions/pick_red.db
dimos run learning-dataprep
dimos run learning-train
dimos run learning-infer-xarm7
```

```
data/sessions/pick_red.db ─► data/datasets/pick_red/ ─► data/runs/act_pick_red/ ─► live policy
```

# Stage 3 — Inference

- **`ChunkPolicyModule`** — Module @ 1–30 Hz. Builds obs via `ObsBuilder`,
  calls `policy.predict_chunk(obs)`, emits `ActionChunk`.
- **`ActionReplayer`** — `BaseControlTask` in the 100 Hz `ControlCoordinator`
  tick loop. Buffers chunks (latest-wins), interpolates to `state.now`,
  emits `JointCommandOutput`. Holds last position on stall.

---

## Blueprint

```python
# dimos/learning/inference/blueprint.py
from dimos.learning.config import ActionChunk

learning_infer_xarm7 = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(
        policy_path="runs/act_pick_red",
        inference_rate_hz=30.0,
    ),
    coordinator_action_replayer_xarm7,
).transports({
    ("color_image",   Image):        LCMTransport("/camera/color_image",      Image),
    ("joint_state",   JointState):   LCMTransport("/coordinator/joint_state", JointState),
    ("language_text", str):          LCMTransport("/learning/language_text",  str),
    ("action_chunk",  ActionChunk):  LCMTransport("/learning/action_chunk",   ActionChunk),
})
```

## Message types

`ActionChunk` lives in `dimos/learning/config.py` next to `EpisodeStatus`
and `DatasetConfig` — single import for all cross-stage contracts.
`Policy` is the backend Protocol; lives in `policy/base.py`.

```python
# dimos/learning/config.py

class ActionChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ts:          float
    joint_names: list[str]
    positions:   np.ndarray   # (T, N)
    dt:          float
    chunk_id:    int
```

```python
# dimos/learning/policy/base.py

@runtime_checkable
class Policy(Protocol):
    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> Policy: ...

    @property
    def chunk_size(self)       -> int: ...
    @property
    def joint_names(self)      -> list[str]: ...
    @property
    def expects_language(self) -> bool: ...

    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...
```

## ChunkPolicyModule

```python
# dimos/learning/inference/chunk_policy_module.py

class ChunkPolicyModuleConfig(ModuleConfig):
    policy_path:       str             # spec read from <policy_path>/dimos_meta.json
    inference_rate_hz: float = 5.0
    device:            str = "cuda"
    default_language:  str = ""


class ChunkPolicyModule(Module):
    config: ChunkPolicyModuleConfig

    color_image:   In[Image]
    joint_state:   In[JointState]
    language_text: In[str]
    action_chunk:  Out[ActionChunk]

    @rpc
    def set_language(self, text: str) -> None: ...
    @rpc
    def reload_policy(self, policy_path: str, device: str | None = None) -> None: ...
    @rpc
    def get_status(self) -> dict[str, Any]: ...

    # Lifecycle: start() loads policy + spawns the loop thread; stop() joins it.
    def _run_loop(self) -> None:
        period = 1.0 / self.config.inference_rate_hz
        while not self._stop.is_set():
            t0  = time.monotonic()
            obs = self._build_live_obs()
            if obs is None:                       # waiting for first frames
                time.sleep(period); continue

            positions = self.policy.predict_chunk(obs)        # (T, action_dim)
            self.action_chunk.publish(ActionChunk(
                ts=time.time(),
                joint_names=self.policy.joint_names,
                positions=positions,
                dt=period,
                chunk_id=self._next_chunk_id(),
            ))
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _build_live_obs(self) -> dict[str, np.ndarray] | None:
        # snapshot latched In[Image] / In[JointState] / In[str] under a lock,
        # hand to ObsBuilder.build(...) → returns obs dict or None if not ready
        ...
```

## ObsBuilder

`ChunkPolicyModule.start()` reads the embedded spec from
`<policy_path>/dimos_meta.json` and constructs the `ObsBuilder` from it
— no `--spec-path` needed at inference.

```python
# dimos/learning/inference/obs_builder.py

class ObsBuilder:
    def __init__(self, spec: DatasetConfig) -> None: ...
    def build(self, live_messages: dict[str, Any]) -> dict[str, np.ndarray]: ...
    def required_streams(self) -> set[str]: ...
```

## ActionReplayer

```python
# dimos/learning/inference/action_replayer.py

@dataclass
class ActionReplayerConfig:
    joint_names:       list[str]
    chunk_topic:       str = "action_chunk"
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

---

## Run

```bash
dimos run learning-infer-xarm7 --policy-path runs/act_pick_red
```

---

## End-to-end

```bash
SPEC=datasets/pick_red.yaml

dimos run learning-collect-quest-xarm7 --spec-path $SPEC --record-path data/pick_red.db
dimos run learning-dataprep            --spec-path $SPEC
dimos run learning-train               --dataset-path dataset/ --output-dir runs/act_pick_red
dimos run learning-infer-xarm7         --policy-path runs/act_pick_red
```

```
session.db ─► dataset/ ─► checkpoint/ ─► live policy
```

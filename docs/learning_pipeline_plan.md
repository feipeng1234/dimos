# Learning Pipeline for DimOS

## Context

DimOS has a mature control stack (100Hz tick loop, per-joint arbitration, hardware adapters, simulation backends) and observation storage (memory2), but **zero learning infrastructure**. This plan adds a complete learning pipeline: data collection, dataset management, training, and policy deployment.

**Key design principle:** The recorder is a **fully independent Module** that subscribes to topics. No coupling to the coordinator internals — just include the module, configure which topics to record, and you're good.

---

## Package Structure

```
dimos/learning/
    __init__.py
    types.py                        # Timestep, RawTimestep, EpisodeMeta, DatasetMeta
    collect/
        __init__.py
        recorder.py                 # DemonstrationRecorder (independent Module)
        episode.py                  # RawEpisode (stores raw messages)
    dataset/
        __init__.py
        dataset.py                  # Dataset (collection of episodes, numpy arrays)
        schema.py                   # RecordingSchema + field extractors
        export.py                   # LeRobot + RLDS export
        loader.py                   # PyTorch DataLoader integration
    train/
        __init__.py
        trainer.py                  # TrainerModule (wraps LeRobot backend)
        checkpoint.py               # Policy loading utilities
    deploy/
        __init__.py
        policy_module.py            # PolicyModule (standalone Module, publishes joint_command)
        safety.py                   # ActionClipper, RateLimiter
    eval/
        __init__.py
        evaluator.py                # Sim rollout runner
    blueprints.py                   # Pre-configured learning workflows
```

---

## 1. Data Collection (Topic-Based Recorder)

### Design

The `DemonstrationRecorder` is a standalone DimOS `Module`. It discovers all available topics in the system, then records the ones the user selects. **No modification to tick_loop.py or coordinator.py needed for collection.**

### 1.1 Two-Layer Data Architecture

**Layer 1 — Raw Recording:** The recorder stores raw DimOS messages (JointState, Image, etc.) in memory2. Preserves all metadata, no information loss. This is what the recorder produces.

**Layer 2 — Structured Dataset:** The dataset layer converts raw messages into named numpy arrays using a user-defined `RecordingSchema`. This is what the training pipeline consumes.

### 1.2 Types

**File:** `dimos/learning/types.py`

```python
@dataclass
class Timestep:
    """Training-ready timestep. Named numpy arrays for obs and actions."""
    t: float
    obs: dict[str, np.ndarray]      # e.g. {"joint_pos": array(7,), "cam0": array(480,640,3)}
    action: dict[str, np.ndarray]   # e.g. {"joint_pos_cmd": array(7,)}

@dataclass
class RawTimestep:
    """Raw recorded timestep. Preserves original DimOS messages."""
    t: float
    topics: dict[str, Any]          # topic_name -> raw message (JointState, Image, etc.)

@dataclass
class EpisodeMeta:
    episode_id: str
    task_description: str
    success: bool | None                        # None = unlabeled
    topic_names: list[str]                      # which topics were recorded
    start_time: float
    end_time: float
    num_timesteps: int
    hz: float                                   # recording frequency
    tags: dict[str, str]

@dataclass
class DatasetMeta:
    name: str
    description: str
    episodes: list[EpisodeMeta]
    created_at: float
```

### 1.3 DemonstrationRecorder Module

**File:** `dimos/learning/collect/recorder.py`

**Key design:** The recorder discovers ALL available topics in the system, then records the ones the user selects via config. It uses the transport layer directly (not fixed `In` ports) so it can dynamically subscribe to any topic.

```python
class DemonstrationRecorderConfig(ModuleConfig):
    store_path: str = "./demonstrations"        # SqliteStore path
    record_hz: float = 50.0                     # Recording rate
    image_hz: float = 10.0                      # Image capture rate (for high-bandwidth topics)
    record_topics: list[str] | None = None      # Topics to record. None = record all available.
    auto_episode_on_teleop: bool = True         # Auto start/stop on teleop engage/disengage

class DemonstrationRecorder(Module[DemonstrationRecorderConfig]):
    # No fixed In ports — discovers and subscribes dynamically

    # --- RPCs ---
    @rpc
    def list_available_topics(self) -> list[dict]: ...   # Discover all topics in the system
    @rpc
    def start_recording(self, task_description: str = "") -> str: ...  # returns episode_id
    @rpc
    def stop_recording(self) -> EpisodeMeta: ...
    @rpc
    def discard_recording(self) -> bool: ...
    @rpc
    def label_episode(self, episode_id: str, success: bool) -> bool: ...
    @rpc
    def list_episodes(self) -> list[EpisodeMeta]: ...
    @rpc
    def set_record_topics(self, topics: list[str]) -> bool: ...  # Change topics at runtime
```

**How it works:**
- On `start()`, discovers available topics via the transport layer
- Subscribes to topics matching `record_topics` config (or all if None)
- Each subscriber caches the **latest raw message** per topic (thread-safe)
- A timer at `record_hz` snapshots all cached messages into a `RawTimestep` and appends to the episode buffer
- High-bandwidth topics (images) captured at `image_hz`, stored via `JpegCodec`
- `stop_recording()` flushes the buffer to a memory2 `SqliteStore` as raw messages
- **No conversion to numpy at this stage** — raw messages preserve all metadata

**Auto-episode on teleop:** If `auto_episode_on_teleop=True`, the recorder also subscribes to the `Buttons` topic. When teleop engages -> auto `start_recording()`. When teleop disengages -> auto `stop_recording()`. Manual RPC still works alongside.

**Blueprint usage:**
```python
recording_blueprint = autoconnect(
    coordinator_teleop_xarm7,
    DemonstrationRecorder.blueprint(
        store_path="./demos",
        record_topics=["joint_state", "joint_command", "images"],  # or None for all
    ),
)
```

### 1.4 RawEpisode

**File:** `dimos/learning/collect/episode.py`

- `RawEpisodeBuilder`: Accumulates `RawTimestep` objects in memory, flushes to memory2 store
- `RawEpisode`: Read-only view — `.meta`, `.raw_timesteps() -> list[RawTimestep]`
- Each episode = a named stream in the SqliteStore, storing raw DimOS messages

---

## 2. Dataset Management (Raw -> Structured Conversion)

### 2.1 RecordingSchema (the bridge between raw and structured)

**File:** `dimos/learning/dataset/schema.py`

The schema defines how to extract named numpy arrays from raw topic messages. Users define one per task/robot config.

```python
@dataclass
class FieldExtractor:
    """Extracts a numpy array from a raw topic message."""
    topic: str                          # which topic to read from
    field: str | None = None            # attribute name (e.g. "position"), None = whole msg
    convert: Callable | None = None     # custom conversion fn (e.g. image_to_array)

@dataclass
class RecordingSchema:
    """Maps raw recorded topics -> named numpy arrays for training."""
    obs: dict[str, FieldExtractor]
    action: dict[str, FieldExtractor]

# Example usage:
manipulation_schema = RecordingSchema(
    obs={
        "joint_pos":  FieldExtractor("joint_state", field="position"),
        "joint_vel":  FieldExtractor("joint_state", field="velocity"),
        "cam0":       FieldExtractor("images", convert=image_to_array),
    },
    action={
        "joint_pos_cmd": FieldExtractor("joint_command", field="position"),
    },
)
```

Built-in converters provided for common message types (JointState -> array, Image -> array).

### 2.2 Dataset

**File:** `dimos/learning/dataset/dataset.py`

Converts raw episodes into structured `Timestep` data using a schema:

```python
class Dataset:
    def __init__(self, raw_episodes: list[RawEpisode], schema: RecordingSchema): ...

    def __len__(self) -> int: ...
    def get_timestep(self, episode_idx: int, step_idx: int) -> Timestep: ...
    def iter_timesteps(self) -> Iterator[Timestep]: ...
    def split(self, train_ratio: float = 0.8) -> tuple[Dataset, Dataset]: ...
    def stats(self) -> dict[str, dict[str, float]]: ...  # per-field min/max/mean/std
```

### 2.3 Export

**File:** `dimos/learning/dataset/export.py`

Two export targets — the two dominant robot learning dataset standards:

**LeRobot (HuggingFace Dataset):**
```
data/
  episode_000000.parquet
meta/
  info.json, episodes.jsonl, tasks.jsonl
videos/ (optional)
```
- Community sharing via HuggingFace Hub
- Python-native, Parquet + JSON, streaming-friendly
- Growing ecosystem (LeRobot library, community models)

**RLDS (Open X-Embodiment / Google DeepMind):**
```
dataset_name/
  1.0.0/
    dataset_info.json
    features.json
    dataset_name-train.tfrecord-00000-of-00001
```
- Standard for Open X-Embodiment (1M+ episodes, 22 robots)
- Required for pretrained models: Octo, RT-X
- TFRecord-based, works with `tensorflow_datasets`

### 2.4 PyTorch DataLoader

**File:** `dimos/learning/dataset/loader.py`

`DemonstrationDataset(torch.utils.data.Dataset)`:
- Observation horizon (stack last N frames)
- Action chunking (predict next K actions)
- Image preprocessing (resize, normalize, augment)
- Per-field normalization from dataset stats

---

## 3. Training Pipeline (LeRobot Backend)

**No custom training loops.** We wrap [LeRobot](https://github.com/huggingface/lerobot) as the training backend. It provides battle-tested implementations of ACT, Diffusion Policy, VQ-BeT, and more — with good defaults that work out of the box.

### 3.1 Available Algorithms (via LeRobot)

| Algorithm | Best For | Data Efficiency |
|-----------|----------|-----------------|
| **ACT** (Action Chunking Transformer) | General manipulation, beginners | ~50 demos |
| **Diffusion Policy** | Complex, multimodal trajectories | ~100+ demos |
| **VQ-BeT** | Discrete action spaces | ~100+ demos |
| **pi0 / SmolVLA** | Vision-language-action | Large datasets |

### 3.2 TrainerModule (DimOS Module wrapping LeRobot)

**File:** `dimos/learning/train/trainer.py`

A thin wrapper around LeRobot's training pipeline. The TrainerModule handles:
- Converting DimOS raw episodes -> LeRobot dataset format (via export)
- Calling `lerobot-train` with the right config
- Tracking training status and metrics
- Loading trained policies for deployment

```python
class TrainerModuleConfig(ModuleConfig):
    # Data
    dataset_path: str                           # Path to recorded episodes (SqliteStore)
    schema: RecordingSchema                     # How to convert raw -> arrays
    dataset_repo_id: str | None = None          # HF Hub repo (optional, for sharing)
    # Algorithm (LeRobot policy types)
    policy_type: str = "act"                    # "act" | "diffusion" | "vq_bet" | "tdmpc"
    # Output
    output_dir: str = "./outputs/train"
    job_name: str = "dimos_train"
    device: str = "cuda"
    # Monitoring
    wandb_enable: bool = False
    # Override any LeRobot config key
    overrides: dict[str, Any] = field(default_factory=dict)

class TrainerModule(Module[TrainerModuleConfig]):
    @rpc
    def start_training(self) -> str: ...        # Export data + launch lerobot-train, returns run_id
    @rpc
    def get_metrics(self) -> dict: ...          # Current loss, epoch, etc.
    @rpc
    def get_status(self) -> str: ...            # "idle" | "exporting" | "training" | "done" | "error"
    @rpc
    def stop_training(self) -> bool: ...
    @rpc
    def get_best_checkpoint(self) -> str: ...   # Path to best checkpoint dir
    @rpc
    def push_to_hub(self, repo_id: str) -> str: ...  # Push trained policy to HF Hub
```

**What `start_training()` does internally:**
1. Load raw episodes from `dataset_path`
2. Convert to LeRobot dataset format using `schema` (calls export pipeline)
3. Call LeRobot's training API with `policy_type` + `overrides`
4. Monitor training progress, update metrics
5. On completion, checkpoint is saved to `output_dir`

**CLI integration:**
```bash
dimos train --config configs/act_xarm7.yaml
# or as a blueprint:
dimos run train-act-xarm7
```

### 3.3 Policy Loading (for deployment)

**File:** `dimos/learning/train/checkpoint.py`

Wraps LeRobot's `PreTrainedPolicy` for loading:

```python
def load_policy(checkpoint_path: str) -> PreTrainedPolicy:
    """Load a trained LeRobot policy from checkpoint or HF Hub."""
    return PreTrainedPolicy.from_pretrained(checkpoint_path)

def load_policy_from_hub(repo_id: str) -> PreTrainedPolicy:
    """Load a trained policy from HuggingFace Hub."""
    return PreTrainedPolicy.from_pretrained(repo_id)
```

The PolicyModule (deploy) calls `policy.select_action(observation)` — LeRobot handles normalization, action chunking, and architecture-specific inference internally.

---

## 4. Policy Deployment

### 4.1 PolicyModule (standalone Module, NOT a ControlTask)

**File:** `dimos/learning/deploy/policy_module.py`

The policy runner is a regular DimOS `Module` — just like the teleop modules. It reads joint state, runs inference, and publishes joint commands on a topic. The coordinator's existing `JointServoTask` picks up those commands via `joint_command: In[JointState]`. **No coordinator changes needed.**

```python
class PolicyModuleConfig(ModuleConfig):
    checkpoint_path: str                    # Path to .pt checkpoint or HF repo_id
    predict_hz: float = 50.0               # Inference rate
    device: str = "cuda"

class PolicyModule(Module[PolicyModuleConfig]):
    # Reads observations
    joint_state: In[JointState]             # Subscribe to coordinator's joint state
    images: In[Image]                       # Subscribe to camera (visuomotor, optional)

    # Publishes actions — coordinator's joint_command picks this up
    joint_command: Out[JointState]

    @rpc
    def start(self) -> None: ...            # Load checkpoint, start inference loop
    @rpc
    def stop(self) -> None: ...
    @rpc
    def get_inference_stats(self) -> dict: ...  # latency, hz, etc.
```

**How it works:**
1. Loads a LeRobot `PreTrainedPolicy` from checkpoint or HF Hub
2. Subscribes to `joint_state` (and optionally `images`) topics
3. Runs an inference loop at `predict_hz`:
   - Build observation dict from latest joint state + image
   - `policy.select_action(observation)` — LeRobot handles normalization, chunking, architecture
   - Safety clip (joint limits + rate limit)
   - Publish as `JointState` on `joint_command` topic
4. Coordinator's existing `JointServoTask` receives it — same path as teleop

**This is the same pattern as teleop:** Phone teleop publishes Twist -> coordinator reads it. Policy publishes JointState -> coordinator reads it. No special integration needed.

**Blueprint usage:**
```python
# Deploy a trained policy — just wire it in like teleop
policy_deploy = autoconnect(
    coordinator_xarm7,                          # has a servo task configured
    PolicyModule.blueprint(checkpoint_path="./checkpoints/act_best"),
)
```

### 4.2 Safety

**File:** `dimos/learning/deploy/safety.py`

- `ActionClipper`: Clamp to joint limits
- `RateLimiter`: Max per-tick joint delta (e.g., 0.05 rad at 100Hz)
- Applied inside PolicyModule before publishing commands

---

## 5. Evaluation

**File:** `dimos/learning/eval/evaluator.py`

- Run N rollouts in simulation (MockAdapter or MuJoCo)
- Reset to initial state, run PolicyModule, collect metrics
- Reports: success rate, mean episode length, action smoothness

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
learning = [
    "torch>=2.0",
    "lerobot>=0.1",
    "tensorflow-datasets>=4.9",
]
```

---

## Files Modified (Existing)

| File | Change |
|------|--------|
| `pyproject.toml` | Add `learning` optional dependency group |

**That's it.** No changes to coordinator, tick loop, or any existing module. Everything else is new files under `dimos/learning/`. The policy module publishes on existing topics — the coordinator already knows how to read `joint_command`.

---

## Existing Code to Reuse

| What | Where |
|------|-------|
| `JointServoTask` | `dimos/control/tasks/servo_task.py` — receives joint commands published by PolicyModule |
| `SqliteStore` | `dimos/memory2/store/sqlite.py` — episode persistence |
| `Stream[T]` | `dimos/memory2/stream.py` — lazy episode querying |
| `PickleCodec`, `JpegCodec` | `dimos/memory2/codecs/` — serialization |
| `Module`, `ModuleConfig`, `In`, `Out` | `dimos/core/` — recorder and policy as Modules |
| `MockAdapter` | `dimos/hardware/manipulators/mock/adapter.py` — testing |
| `autoconnect` | `dimos/core/blueprints.py` — wiring modules into blueprints |

---

## Implementation Order

### Phase 1 — Types + Collection (no new deps)
1. `learning/__init__.py`, `learning/types.py`
2. `learning/collect/episode.py` — RawEpisodeBuilder + RawEpisode
3. `learning/collect/recorder.py` — DemonstrationRecorder Module
4. Unit tests

### Phase 2 — Dataset + Export (adds lerobot, tensorflow-datasets)
5. `learning/dataset/schema.py` — RecordingSchema + FieldExtractor + built-in converters
6. `learning/dataset/dataset.py` — Dataset (raw -> structured via schema)
7. `learning/dataset/export.py` — LeRobot + RLDS
8. `learning/dataset/loader.py` — PyTorch Dataset
9. Tests

### Phase 3 — Training (uses lerobot)
10. `learning/train/trainer.py` — TrainerModule (wraps LeRobot)
11. `learning/train/checkpoint.py` — Policy loading utilities
12. Tests

### Phase 4 — Deployment + Eval
13. `learning/deploy/safety.py`
14. `learning/deploy/policy_module.py` — PolicyModule (standalone Module, publishes joint_command)
15. `learning/eval/evaluator.py`
16. `learning/blueprints.py`
17. Integration test: collect -> train -> deploy on MockAdapter

---

## Verification

1. **Unit tests**: Each module tested independently with MemoryStore / MockAdapter
2. **Integration test**: Full loop — record episodes via mock teleop, export to LeRobot format, train ACT policy, deploy PolicyModule, verify it publishes valid JointState commands
3. **Manual test**: Wire recorder into a teleop blueprint, record episodes, export to LeRobot format, inspect with `lerobot` CLI or push to HF Hub
4. **CLI test**: `dimos train --config configs/act_xarm7.yaml` runs end-to-end
5. **Existing tests pass**: `pytest dimos/control/` — confirm no regressions

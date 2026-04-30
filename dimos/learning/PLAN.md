# DimOS Learning Framework — v1 Plan

## v1 Scope

**Goal:** end-to-end pipeline that lets a DimOS user collect teleop demos, train a policy, and run it on a real arm — for two concrete targets:

1. **BC / ACT** — train ACT (Action Chunking Transformer) on a pick-and-place demo set on xArm7.
2. **VLA finetune** — finetune a pretrained π₀ / π₀.₅ checkpoint on the same demo set.

Both targets share a single `DatasetSpec`, a single LeRobot dataset on disk, and a single inference module. The choice between ACT and a VLA is just a different training entry point and a different policy class at inference time.

**v1 architectural mandate — fully DimOS-native:**

Every phase of the pipeline (collection, training, inference) is exposed as a **Module + Blueprint** with RPC surfaces. There is **one user-facing UX**: `dimos --blueprint <phase>` for everything. Agent skills can drive any phase via @rpc. Composition between phases is just port wiring (`builder.done → trainer.builder_done`).

For collection and training — where the actual work is offline batch processing — the Module is an **orchestrator over a subprocess**: it spawns `python -m dimos.learning.dataprep build` or `python -m ...training.train`, parses its progress lines, and republishes them as typed events. The work itself stays in the subprocess (heavy deps isolated, process-cancellable, separately testable). Inference Modules do real live work because there is real live data flow.

Every Module exposes:
- `@rpc start()` / `@rpc stop()` — lifecycle
- `@rpc <action>(...)` — at least one agent-callable action
- `@rpc get_status()` — observability
- typed `In[...]` / `Out[...]` ports for blueprint composition

**Out of v1 (deferred to v2 — see bottom of file):**
- RL (online + offline)
- Pure proprioceptive policies in the 100 Hz tick loop (`PolicyControlTask`)
- Multi-embodiment / cross-task training
- Distributed / multi-GPU training
- Live recording of episode boundaries (we keep post-hoc button extraction)

**Design principle:** lean on existing infrastructure — `RecordReplay` (PR #1708), memory2, the Module/Blueprint system, **and** the `lerobot` library which already implements ACT and π₀/π₀.₅ end-to-end. We add only the DimOS glue.

---

## Architecture Overview

```
 COLLECT                          TRAIN                       INFER
 ───────                          ─────                       ─────
 Teleop + Camera                  load_dataset(spec)          ChunkPolicyModule (1–30 Hz)
       ↓                                ↓                            ↓ (action chunks)
 RecordReplay --record-path       train_bc / finetune_vla     ActionReplayer (100 Hz)
       ↓                                ↓                            ↓
 session.db                       checkpoint (.safetensors)   Coordinator joint_command
       ↓                          + stats.json                       ↓
 dataset.yaml + dataprep.py                                    Hardware
       ↓
 LeRobot v2 dataset on disk
```

The **same `dataset.yaml`** is the contract between collection (export), training (load), and inference (live obs construction). It is the single source of truth for what counts as observation/action, episode boundaries, sync strategy, and feature shapes.

---

## 1. Data Collection — STATUS: IMPLEMENTED, NEEDS POLISH

### 1.1 Recording (no new code)

Use Sam's `RecordReplay` from PR #1708:

```bash
dimos --blueprint quest_teleop_xarm7 --record-path session.db
```

Captures every LCM topic — joint states, joint commands, camera images, controller `Buttons`, IMU, etc. One stream per topic in a single `SqliteStore`. Episode boundaries are not marked at record time; they are recovered offline from button presses.

### 1.2 The dataset spec

Implemented in `dimos/learning/spec.py`. Schema is pydantic v2 → round-trips YAML/JSON. Eight typed classes: `EpisodeConfig`, `FieldRef`, `SyncConfig`, `FilterConfig`, `OutputConfig`, `DatasetSpec`, `Episode`, `Sample`. Friendly Quest button names (`A`/`B`/`X`/...) resolve to `Buttons` bit fields via `BUTTON_ALIASES`.

Example (see `dimos/learning/dataset.example.yaml` for the live template):

```yaml
source: session.db

episodes:
  extractor: buttons        # buttons | ranges | whole_session
  start: A                  # press to begin
  save:  B                  # press to commit
  discard: X                # press to drop
  default_task_label: pick_red_cube

observation:
  cam_high:
    stream: camera_color_image
    preprocess: jpeg_decode
  cam_wrist:
    stream: camera_wrist_color_image
    preprocess: jpeg_decode
  joint_pos:
    stream: coordinator_joint_state
    field: position

action:
  joint_target:
    stream: coordinator_joint_command
    field: position

sync:
  anchor: cam_high
  rate_hz: 30
  tolerance_ms: 50
  strategy: nearest

filters:
  success_only: true
  min_duration_s: 1.0
  task_labels: [pick_red_cube]

output:
  format: lerobot           # primary v1 target
  path: datasets/pick_red/
  metadata:
    fps: 30
    robot: xarm7
```

### 1.3 The pipeline file: `dimos/learning/dataprep.py`

Implemented. Does everything: read raw `session.db`, extract episodes from button events, sync streams, dispatch to the chosen format writer. Same module exposes `load_dataset(spec)` for training.

Public functions (all done):
- `load_spec(path)` / `save_spec(spec, path)` — YAML/JSON I/O
- `extract_episodes(store, cfg)` — three strategies (buttons/ranges/whole_session)
- `filter_episodes(eps, cfg)` — success / duration / label whitelist
- `iter_samples(store, episode, spec)` — anchor-rate timestep walker w/ bisect nearest-search
- `build_dataset(spec)` — full session.db → on-disk dataset
- `load_dataset(spec)` — returns a `torch.utils.data.Dataset[Sample]`
- `inspect(spec)` — episode/duration/per-stream stats
- `main()` — CLI: `build` / `inspect` / `review` (review is a stub)

### 1.4 Format writers (in `dimos/learning/formats/`)

| Format    | v1 priority | Status | Why |
|-----------|-------------|--------|-----|
| `lerobot` | **primary** | done   | Native input for both ACT and π₀/π₀.₅ via `lerobot` lib |
| `hdf5`    | secondary   | done   | ACT-original codebase, debugging, smaller deps |
| `rlds`    | v2          | done (gated on TF) | RT-X / OpenX-Embodiment compat — not needed for v1 |

### 1.5 Gaps to close in v1

The collection pipeline is functional, but a few things are needed before LeRobot training works cleanly. Each is small.

**(a) Per-episode task description** — π₀ is language-conditioned; LeRobot v2 has a `tasks.jsonl` table. Currently `Episode.task_label: str | None` is a tag; we need a free-form string per episode. Extend with:
```python
class Episode:
    task_description: str | None = None   # e.g. "pick up the red cube and place it on the blue plate"
```
The LeRobot writer already emits `tasks.jsonl` keyed on `task_label` — switch it to use `task_description` (fall back to `task_label`). Population: `EpisodeConfig.default_task_description` for single-task sessions, or set per episode in the `review` CLI.

**(b) Dataset statistics** — LeRobot training requires `meta/stats.json` (per-feature mean/std/min/max/q01/q99). Add a streaming stats accumulator inside `formats/lerobot.py::write` so we don't need a second pass over the data. Image stats are computed on a subsample (every Nth frame) to bound cost.

**(c) Train/val split** — LeRobot v2 supports filtering by episode index at training time, so we don't need to materialize two datasets. Add `FilterConfig.val_episode_ids: list[int] | None` and `FilterConfig.val_ratio: float | None` (deterministic seeded split). Trainer reads these.

**(d) Image format on disk** — LeRobot v2 stores images as MP4 videos by default (`videos/chunk-NNN/<key>/episode_NNNNNN.mp4`). Current writer writes them as parquet tensor columns, which works but inflates disk size. Switch to MP4 encoding via `imageio[ffmpeg]` for image streams ≥2D + uint8. Parquet cells then store frame indices, not pixels.

**(e) `review` CLI** — currently a stub. Implement a minimal non-interactive form first: load spec, list episodes with metadata, allow batch retag via `--set-label PICK_RED --episode-ids 0,1,2,5`. Interactive TUI is v2.

These five items + the existing skeleton complete the collection side for v1.

---

## 2. Training — NEW v1 WORK

### 2.1 Strategy: thin wrappers around `lerobot`

The `lerobot` library (HuggingFace + Tesla-PI fork) already implements ACT, Diffusion Policy, π₀, π₀.₅ — including dataloaders for the LeRobot v2 format, normalization, action chunking, language tokenization, training loop, checkpointing, and ONNX export.

**We do NOT reimplement these.** The v1 training pipeline is two thin Python wrappers that:
1. Take a DimOS `DatasetSpec`,
2. Translate it into a LeRobot config,
3. Invoke `lerobot.scripts.train.train()`,
4. Save the resulting checkpoint to a path that the inference module knows how to read.

This keeps `dimos/learning/training/` short, rides on a maintained upstream, and means a `pi0.5` upgrade is a config bump rather than a code change.

### 2.2 File layout: `dimos/learning/training/`

```
dimos/learning/training/
    train.py        # train_bc, finetune_vla — public entry points
    configs.py      # BCConfig, VLAConfig
    stats.py        # compute_stats(spec) -> dict (used by build_dataset too)
    split.py        # train/val episode split helper
```

### 2.3 The two entry points

```python
def train_bc(spec: DatasetSpec, cfg: BCConfig, output_dir: Path) -> Path:
    """Train an ACT (or other BC) policy on `spec`. Returns checkpoint path."""

def finetune_vla(spec: DatasetSpec, cfg: VLAConfig, output_dir: Path) -> Path:
    """Finetune a pretrained π₀ / π₀.₅ on `spec`. Returns checkpoint path."""
```

Both:
- Materialize the dataset via `build_dataset(spec)` if `spec.output.path` doesn't already exist (idempotent).
- Build a `lerobot.LeRobotDataset(spec.output.path)`.
- Build a LeRobot policy from `cfg`.
- Call the LeRobot training loop with `cfg.steps`, `cfg.batch_size`, `cfg.lr`, etc.
- Save final checkpoint + a sidecar `dimos_meta.json` with `{spec_path, dataset_path, dimos_version}` so inference can recover everything.

### 2.4 `BCConfig` (ACT-focused for v1)

```python
class BCConfig(BaseModel):
    policy_type: Literal["act", "diffusion"] = "act"

    # ACT model arch — defaults match the original ACT pick-and-place setup
    chunk_size: int = 50          # action_horizon
    n_obs_steps: int = 1
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    use_vae: bool = True
    kl_weight: float = 10.0

    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained: bool = True

    # Optim
    steps: int = 100_000
    batch_size: int = 8
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    # Eval
    val_ratio: float = 0.1
    save_every: int = 10_000
```

### 2.5 `VLAConfig` (π₀ / π₀.₅ finetune)

```python
class VLAConfig(BaseModel):
    policy_type: Literal["pi0", "pi0_5"] = "pi0_5"
    pretrained_path: str          # HF hub id or local path
    finetune_mode: Literal["full", "lora"] = "lora"
    lora_rank: int = 16
    freeze_vision: bool = True
    freeze_language: bool = True

    chunk_size: int = 50          # default π₀ action horizon

    steps: int = 30_000
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-4
    save_every: int = 5_000

    # The spec's task_description per episode is the language conditioning at train time.
    # No additional config needed.
```

### 2.6 Stats and split — pulled out so they're reusable

`stats.compute_stats(spec)` walks the materialized dataset once, accumulating Welford mean/std for joint vectors and per-channel image stats on a subsample. Writes `meta/stats.json`. Called from both `build_dataset` (so the disk-resident dataset is self-describing) and `train_bc` / `finetune_vla` (idempotent — skip if `stats.json` already exists).

`split.train_val_split(spec, val_ratio, seed=0)` returns two episode-id lists. Deterministic. Trainer passes these to LeRobot via its episode filter.

### 2.7 CLI

```bash
# Train ACT on a built dataset
python -m dimos.learning.training.train bc dataset.yaml \
    --output runs/act_pick_red \
    --steps 100000 --batch-size 8

# Finetune π₀.₅
python -m dimos.learning.training.train vla dataset.yaml \
    --output runs/pi05_pick_red \
    --pretrained lerobot/pi0_5 \
    --finetune-mode lora --lora-rank 16
```

The CLI is a tiny argparse wrapper that builds `BCConfig`/`VLAConfig` and calls the function.

### 2.8 Dependencies

Adds to `pyproject.toml` (under a `[project.optional-dependencies]` `learning` extra so default installs aren't bloated):
- `lerobot >= 0.2`
- `torch >= 2.3` (already implied by `lerobot`)
- `imageio[ffmpeg]` (MP4 image encoding)
- For VLA only: `transformers`, `accelerate`, `peft` (LoRA)

User installs with `pip install -e .[learning]`.

---

## 3. Inference — NEW v1 WORK

### 3.1 The two paths, simplified for v1

For v1 we need exactly **one** inference module: `ChunkPolicyModule`. Both ACT and π₀/π₀.₅ produce action chunks (sequences of length `chunk_size`), so the same module handles them. The model runs slow (1–30 Hz depending on whether it's ACT or VLA); a separate `ActionReplayer` plays the chunk back at the coordinator's 100 Hz tick rate.

```
       ┌──────────────────────────┐
       │   ChunkPolicyModule      │   ← runs in its own thread/process at policy rate
       │   In:  color_image       │
       │       joint_state        │
       │       language_text      │
       │   Out: action_chunk      │
       └────────────┬─────────────┘
                    │ (T, action_dim)
                    ▼
       ┌──────────────────────────┐
       │   ActionReplayer         │   ← part of the ControlTask graph
       │   ControlTask @ 100 Hz   │
       │   pops next action,      │
       │   emits JointCommand     │
       └──────────────────────────┘
```

`PolicyControlTask` (joint-only, in-tick-loop) is **deferred to v2** — it's only useful for proprioceptive policies, which we don't train in v1.

### 3.2 File layout: `dimos/learning/inference/`

```
dimos/learning/inference/
    chunk_policy_module.py    # ChunkPolicyModule
    action_replayer.py        # ActionReplayer (subclass of BaseControlTask)
    obs_builder.py            # spec.observation -> live obs dict, decoupled
    blueprints.py             # autoconnect helpers
```

### 3.3 `dimos/learning/policy/` — Policy protocol

```python
class Policy(Protocol):
    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "Policy": ...
    def predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return shape (chunk_size, action_dim)."""

class LeRobotPolicy(Policy):
    """Wraps any lerobot.PreTrainedPolicy (ACT, π₀, π₀.₅, Diffusion).
       Detects policy_type from the checkpoint metadata."""
```

v1 ships `LeRobotPolicy` only. `OnnxPolicy`, custom `TorchPolicy` are v2.

### 3.4 `ChunkPolicyModule` skeleton

```python
class ChunkPolicyModule(Module):
    color_image: In[Image]
    joint_state: In[JointState]
    language_text: In[str]            # optional; ignored if policy doesn't use it

    action_chunk: Out[ActionChunk]    # new typed message: ts, joint_names, positions[T, N]

    def __init__(self, *, spec_path: str, policy_path: str,
                 inference_rate_hz: float, device: str = "cuda"):
        self._spec = load_spec(spec_path)
        self._policy = LeRobotPolicy.load(Path(policy_path), device)
        self._obs_builder = ObsBuilder(self._spec)
        self._stats = load_stats(Path(policy_path).parent / "stats.json")

    @rate_limited(inference_rate_hz)
    def on_tick(self):
        obs = self._obs_builder.build(
            color_image=self.color_image.latest(),
            joint_state=self.joint_state.latest(),
            language=self.language_text.latest_or("default task"),
        )
        chunk = self._policy.predict_chunk(self._stats.normalize(obs))
        chunk = self._stats.unnormalize_actions(chunk)
        self.action_chunk.publish(ActionChunk(positions=chunk, ts=time.time(), ...))
```

### 3.5 `ActionReplayer` — a `BaseControlTask`

Lives in the tick loop. Subscribes to `action_chunk`. Maintains a buffer of pending actions with their relative timestamps; `compute(state)` interpolates to the current tick time.

```python
class ActionReplayer(BaseControlTask):
    name = "policy_replay"

    def __init__(self, joint_names: list[str], chunk_topic: str, policy_dt: float):
        self._joint_names = joint_names
        self._buffer: deque[tuple[float, np.ndarray]] = deque()  # (target_ts, positions)
        ...

    def on_action_chunk(self, msg: ActionChunk) -> None:
        # Push new chunk, drop overlap with current buffer (latest chunk wins)
        ...

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        if not self._buffer:
            return None
        target = self._lookup_or_interp(state.now)
        return JointCommandOutput(joint_names=self._joint_names, positions=target)
```

Key behaviors:
- **Latest chunk wins**: when a new chunk arrives, drop any buffered actions ≥ the new chunk's start time. Smooth (no gap) because the new chunk's first action is conditioned on near-current obs.
- **Lookback safety**: if the policy module stalls and the buffer empties, hold last commanded position (don't fall to zero). Log a warning.
- **Temporal ensembling (optional v1 nice-to-have)**: when ACT publishes overlapping chunks, exponentially weight predictions for the same target timestamp. Off by default.

### 3.6 Live observation construction

`ObsBuilder` reads `spec.observation` and exposes `build(**live_streams) -> dict[str, np.ndarray]`. Reuses the same `_resolve_field` and `preprocess` registry from `dataprep.py` so train and infer share normalization. This is the single most important consistency guarantee in the framework.

### 3.7 Inference blueprint

```python
# Vision policy (ACT or π₀.₅) → Coordinator
pick_red_cube = autoconnect(
    RealSenseCamera.blueprint(camera_id="cam_high"),
    ChunkPolicyModule.blueprint(
        spec_path="datasets/pick_red/dataset.yaml",
        policy_path="runs/pi05_pick_red/checkpoint",
        inference_rate_hz=5.0,
    ),
    ControlCoordinator.blueprint(
        hardware=[xarm7],
        tasks=[
            TaskConfig(name="policy_replay", type="action_replayer",
                       chunk_topic="action_chunk", policy_dt=0.2),
        ],
    ),
)
```

### 3.8 Coordinator change (one line)

In `dimos/control/coordinator.py::_create_task_from_config` add a case for `type == "action_replayer"` that constructs the `ActionReplayer`. Plus a new `ActionReplayerConfig` in `dimos/control/task.py`'s task config union. Same pattern as existing tasks.

---

## 4. File Structure (v1)

```
dimos/learning/
    spec.py                       # ✅ DatasetSpec (DataPrep-callable types)
    dataprep.py                   # ✅ DataPrep class + CLI
    dataset.example.yaml          # ✅

    formats/
        lerobot.py                # ✅ (needs MP4 + stats — §1.5)
        hdf5.py                   # ✅
        rlds.py                   # ✅ (v2 priority)

    collection/                   # 🆕 v1 — collection blueprint
        episode_monitor.py        # EpisodeMonitorModule (live counters via @rpc)
        blueprint.py              # collection_blueprint(...)

    training/                     # 🆕 v1 — training scripts + orchestrator Modules
        configs.py                # BCConfig, VLAConfig                    [script]
        stats.py                  # Stats class + compute_stats             [script]
        split.py                  # train_val_split                         [script]
        train.py                  # train_bc, finetune_vla, CLI             [script]
        trainer_module.py         # TrainerModule — wraps build+train       [Module]
        monitor_module.py         # LearningMonitorModule (rerun + JSONL)   [Module]
        blueprint.py              # learning_train_{act,vla,idle}

    inference/                    # 🆕 v1 — inference blueprint (real live Modules)
        obs_builder.py            # ObsBuilder (uses DataPrep.resolve_field)
        chunk_policy_module.py    # ChunkPolicyModule (real Module)
        action_replayer.py        # ActionReplayer (BaseControlTask)
        blueprint.py              # policy_blueprint(...)

    policy/                       # 🆕 v1 — Policy abstraction
        base.py                   # Policy protocol + ActionChunk message
        lerobot_policy.py         # LeRobotPolicy (ACT, π₀, π₀.₅)
```

**Note on script-vs-Module split inside `training/`:** The four `[script]`
files (`configs.py`, `stats.py`, `split.py`, `train.py`) hold the actual
training logic and are independently usable from notebooks/CI/tests. The
three `[Module]` files wrap them as DimOS Modules that spawn the scripts
as subprocesses — that's the dual-surface UX the v1 mandate requires.

Critical files outside `dimos/learning/`:

| File | Change |
|------|--------|
| `dimos/control/coordinator.py` | Add `"action_replayer"` case in `_create_task_from_config` |
| `dimos/control/task.py` | Add `ActionReplayerConfig` to task config union |
| `pyproject.toml` | Add `[project.optional-dependencies].learning` extra |
| `dimos/messages/` (or wherever DimOS LCM types live) | New `ActionChunk` type: `(joint_names, positions[T,N], ts, dt)` |

---

## 5. End-to-End Demo Recipe

Two flows, one command list each. Both assume `pip install -e .[learning]` is done.

### 5.1 ACT pick-and-place on xArm7 — blueprint-first UX

Each phase is `dimos --blueprint <name>`. Underlying scripts (`python -m
dimos.learning.dataprep`, `python -m dimos.learning.training.train`) are
still callable directly for CI / notebooks / debugging — but the
default flow is the blueprint surface.

```bash
# 1. Collect — teleop + camera + RecordReplay + EpisodeMonitorModule
dimos --blueprint learning_collect_quest_xarm7 --record-path data/pick_red.db
# (operator presses A=start / B=save / X=discard;
#  EpisodeMonitorModule.status streams "episodes_saved: N" live)

# 2. Train — DatasetBuilderModule + TrainerModule + LearningMonitorModule
dimos --blueprint learning_train_act \
    --spec dataset.yaml --output runs/act_pick_red
# Inside: builder runs first (subprocess: dataprep build), trainer
# auto-fires on builder.done (subprocess: train bc), monitor logs to rerun.

# 3. Infer — Camera + ChunkPolicyModule + ActionReplayer + Coordinator
dimos --blueprint learning_infer_pick_red \
    --policy-path runs/act_pick_red
```

### 5.2 π₀.₅ finetune on the same data

Steps 1 + 3 unchanged. Step 2 is the same `learning_train_*` blueprint
with `--kind vla` and a `--pretrained` flag — agent or human just changes
the trigger payload, not the blueprint.

```bash
dimos --blueprint learning_train_vla \
    --spec dataset.yaml --output runs/pi05_pick_red \
    --pretrained lerobot/pi0_5 --finetune-mode lora --lora-rank 16
```

### 5.3 Agent-driven flow (same Modules, no `auto_run`)

Demonstrates the @rpc surface. Run a single training blueprint with auto-run
disabled; a chat agent then drives every phase:

```bash
dimos --blueprint learning_train_idle  # builder + trainer + monitor, all idle
```

```
agent: "build the dataset for pick_red"
  → DatasetBuilderModule.build(spec_path="dataset.yaml")
  ← BuildProgress events stream back to the chat
  ← BuildDone(success=True, dataset_path="datasets/pick_red/")

agent: "train ACT on it for 100k steps"
  → TrainerModule.train(
      spec_path="dataset.yaml",
      output_dir="runs/act_pick_red",
      config_kind="bc",
      config_overrides={"steps": 100_000},
    )
  ← TrainProgress events stream loss/step
  ← TrainDone(success=True, checkpoint_dir="runs/act_pick_red/...")

agent: "deploy it on the xarm"
  → launches `dimos --blueprint learning_infer_pick_red --policy-path ...`
```

The fact that the same Modules drive both the "everything-auto" CLI flow
(§5.1) and the "agent-driven" flow (§5.3) is the v1 architectural payoff.

---

## 6. Verification (what we test before declaring v1 done)

1. **Recording** — teleop blueprint with `--record-path` produces a session.db whose stream listing matches the spec.
2. **Build** — `python -m dimos.learning.dataprep build dataset.yaml` against a real session, then `lerobot.LeRobotDataset(path)` opens it without error and `len(ds) > 0`.
3. **Stats** — `meta/stats.json` exists and has finite, non-degenerate values for every observation/action key.
4. **Train** — `train_bc` runs ≥1k steps end-to-end on a real session; loss decreases; checkpoint loads back via `LeRobotPolicy.load`.
5. **VLA finetune** — `finetune_vla` runs ≥500 steps with LoRA on top of a downloaded π₀.₅ checkpoint; no OOM at batch=4 on a 24 GB GPU; loss decreases.
6. **Live obs parity** — `ObsBuilder.build(...)` on a fake live stream and `iter_samples(...)` on the same data give bit-identical observation dicts.
7. **Inference (sim)** — `ChunkPolicyModule` + `ActionReplayer` + `ControlCoordinator` with MuJoCo xArm7 produces non-NaN joint commands at 100 Hz, replays a 50-step chunk smoothly, recovers when policy module stalls.
8. **Inference (hw)** — same blueprint on real xArm7 produces a successful pick-and-place at ≥30% success rate after 50 demos. (Success rate is informational; the test is "no crashes, no jerks, no diverging commands.")

---

## 7. Key Design Decisions (v1)

| Decision | Rationale |
|----------|-----------|
| LeRobot v2 is the canonical on-disk format | Both ACT and π₀/π₀.₅ train from it natively; no custom dataloaders |
| `lerobot` library does the heavy lifting | We don't reimplement ACT, π₀, dataloader, normalization, or training loop |
| One inference module (`ChunkPolicyModule`), not three | ACT and VLA both produce chunks; only the model class differs |
| `ActionReplayer` lives in the tick loop, model lives in a Module | Decouple slow inference (1–5 Hz VLA) from fast control (100 Hz) |
| `ObsBuilder` reused between train and infer | Single source of truth for observation construction — eliminates train/serve skew |
| Episode metadata carries `task_description` | π₀/π₀.₅ are language-conditioned; `task_label` alone is too narrow |
| Stats computed at build time, written to disk | Trainers and inference both read from `meta/stats.json` — no recompute |
| Coordinator change is one new task type (`action_replayer`) | Minimal, additive |
| RLDS, ONNX, RL, proprio-only policy task → v2 | Deliberately not in scope |

---

## 8. Risks & Open Questions for v1

- **`lerobot` API stability.** We're pinning to a specific minor version. If their training entry point changes, our wrapper breaks. Mitigation: pin tightly in `pyproject.toml`, add an integration test that exercises the wrapper.
- **π₀.₅ checkpoint availability.** Depends on the public release. Fallback: ship v1 with π₀ only, add π₀.₅ when it lands (config bump).
- **Action space match.** π₀ assumes a 7-DoF EEF action by default; xArm7 joint-position control is 7-DoF joint. Need to either (a) keep π₀'s action head and use joint targets in its expected layout, or (b) retrain the action head. v1 chooses (a) via the spec's action key naming.
- **Real-time perf of chunk replay.** First action of a chunk is conditioned on `t = chunk_arrival_time`, but it executes some ms later. With 50-step chunks at 30 Hz this is ~1.6 s of buffer; if the policy module stalls, the replayer drifts. Mitigation: replayer rejects stale chunks (`now − chunk.ts > policy_dt × 1.5`) and re-requests.
- **Camera calibration in the spec.** ACT/π₀ are sensitive to camera placement. The spec doesn't currently encode camera intrinsics/extrinsics. v1 punt: rely on the operator to record from the same physical setup at infer time. Add `metadata.cameras` schema in v1.5 if it bites.

---

---

# v2 Considerations (deferred)

These are explicitly out of v1 scope. Pulled here so we don't lose track. Anything from this list that becomes cheap during v1 implementation gets promoted up.

### Training
- **RL** — `train_rl(env_cfg, model_cfg)` for online (PPO/SAC) and offline (CQL/IQL/AWAC) RL. Needs an env wrapper around DimOS (sim primarily — MuJoCo via the existing `MujocoCamera` work).
- **Multi-task / multi-embodiment training** — train one policy on demos from xArm7 + Piper + Mock. Needs URDF retargeting, embodiment-id conditioning.
- **Distributed training** — `accelerate` / FSDP for VLA full-finetune on multi-GPU.
- **Curriculum / dataset weighting** — sample harder episodes more often, weight by reward, etc.
- **Diffusion Policy** as a first-class BC option (lerobot supports it; just a config).
- **Active data collection** — uncertainty-based suggestion of which demos to collect next.

### Inference
- **`PolicyControlTask`** — joint-only proprioceptive policies in the tick loop (100 Hz, no Module overhead). Useful for residual policies, locomotion.
- **`OnnxPolicy` / `TorchScriptPolicy`** — alternate Policy backends for deployment without `lerobot` runtime dep.
- **Cross-embodiment retargeting at inference** — train on xArm7 demos, deploy on Piper.
- **Temporal ensembling on by default** — currently nice-to-have in v1, make it the default after measuring its effect on jerk.
- **Async chunk pipelining** — request chunk N+1 while replaying chunk N to hide policy latency completely.
- **Real-time safety layer** — collision check + joint-limit clamp downstream of `ActionReplayer`.

### Data Collection
- **Live `EpisodeManagerModule`** — annotate episode boundaries at record time instead of post-hoc. Useful when the operator wants to pause/resume and the recording length is huge. Currently overkill.
- **RLDS / TFDS writer** — needed for OpenX-Embodiment contributions and RT-X-style training. Skeleton already exists.
- **Interactive `review` TUI** — scrub through episodes, watch the camera stream, retag/discard. v1 ships a non-interactive batch retag CLI only.
- **Camera intrinsics/extrinsics in spec metadata** — required for any cross-embodiment or sim-real work.
- **Force/torque streams as observation** — schema already supports it; need preprocess hooks for FT data.
- **Imitation-from-observation** — episodes without action streams (just video). Needs an inverse dynamics model.
- **Custom transports for streaming** — record directly to a remote SqliteStore over network, bypass local disk. Probably never needed.

### Training Module / DimOS-native
- **`TrainingModule` agent skill** — expose `train_bc` / `finetune_vla` as RPCs so the LLM agent can trigger training from a chat session. Requires sandboxing (long-running subprocess management, GPU resource gating).
- **`PolicyRegistry` Module** — track all trained policies, their specs, eval results. A "model zoo" served as a Module for blueprint composition.

### Tooling
- **W&B / TensorBoard integration** standardized across all trainers.
- **Eval harness** — replay validation episodes through the trained policy in sim, report success rate. Needed for any kind of automated training pipeline.
- **HuggingFace Hub upload** — `dimos.learning.training.publish(checkpoint, repo_id)` so trained policies are sharable.

### Possibly to promote into v1 if cheap
- **Stats subsampling for image features** — needed in v1 anyway; making it configurable is a 5-line addition.
- **Episode-level train/val split** — already needed in v1; might as well expose `--split-by hash(episode_id)` as a third strategy.
- **Diffusion Policy via the same `train_bc` entry point** — `lerobot` already has it, the only additional work is one more policy_type literal in `BCConfig`. Probably ship it.
- **`OnnxPolicy`** — only worth promoting if a deployment target without `lerobot` install exists. None in v1; defer.

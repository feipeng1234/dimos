# Folder Structure

The four spec docs in this directory are the source of truth. The code
tree below is the implementation layout — each file maps to a section in
one of the three stage docs.

```
dimos/learning/
│
├── specs/                          # ← spec docs (you are here)
│   ├── structure.md                # this file — folder layout
│   ├── datacollection.md           # Stage 1 — recording + dataprep + inspect
│   ├── training.md                 # Stage 2 — TrainerModule
│   └── inference.md                # Stage 3 — ChunkPolicyModule + ActionReplayer
│
├── __init__.py
├── config.py                       # DatasetConfig + sub-configs (pydantic BaseConfig)
├── dataset.example.yaml            # annotated example spec
│
├── dataprep.py                     # DataPrep façade + resolve_field staticmethod
│                                   #   `python -m dimos.learning.dataprep build|inspect`
├── dataprep_module.py              # DataPrepModule (wraps the subprocess for blueprint UX)
│
├── collection/                     # ── Stage 1 / Phase A: live recording ──
│   ├── __init__.py
│   ├── episode_monitor.py          # EpisodeStatus, EpisodeMonitorModule(Config)
│   └── blueprint.py                # learning_collect_quest_{xarm7,xarm6,piper,dual}
│
├── formats/                        # ── dataset writers (DataPrep._get_writer dispatches) ──
│   ├── __init__.py
│   ├── lerobot.py                  # LeRobot v2 (parquet + MP4 + meta/stats.json)
│   ├── hdf5.py                     # flat HDF5
│   └── rlds.py                     # RLDS / TFDS
│
├── training/                       # ── Stage 2: offline training ──
│   ├── __init__.py
│   ├── trainer_module.py           # TrainProgress, TrainDone, TrainerModule(Config)
│   ├── train.py                    # subprocess CLI
│                                   #   `python -m dimos.learning.training.train {bc|vla}`
│   ├── configs.py                  # bc / vla training configs
│   ├── split.py                    # train/val episode-level split
│   ├── stats.py                    # meta/stats.json computation (norm/unnorm)
│   └── blueprint.py                # learning_train
│
├── policy/                         # ── policy backends (live + checkpoint loading) ──
│   ├── __init__.py
│   ├── base.py                     # ActionChunk pydantic + Policy Protocol
│   └── lerobot_policy.py           # LeRobotPolicy.load → reads dimos_meta.json + stats.json
│
└── inference/                      # ── Stage 3: live policy serving ──
    ├── __init__.py
    ├── chunk_policy_module.py      # ChunkPolicyModule(Config); slow Module @ 1–30 Hz
    ├── obs_builder.py              # ObsBuilder; calls DataPrep.resolve_field
    ├── action_replayer.py          # ActionReplayer (BaseControlTask, NOT a Module)
    └── blueprint.py                # learning_infer_{xarm7,xarm6,piper}
                                    #   + learning_infer_vla_{xarm7,...}
```

---

## Where each artifact is produced / consumed

| Artifact                | Producer                                          | Consumer                                      |
|---|---|---|
| `dataset.yaml`          | human (operator)                                  | `DataPrep`, `ObsBuilder`                       |
| `session.db`            | `RecordReplay` (transport hook, `--record-path`)  | `DataPrep`                                     |
| `dataset/` + stats      | `dataprep build` → `formats/<fmt>.py`             | `lerobot.LeRobotDataset`, `train.py`           |
| `checkpoint/` + meta    | `train.py`                                        | `LeRobotPolicy.load`, `ChunkPolicyModule`      |
| `ActionChunk` (live)    | `ChunkPolicyModule` (Module, LCM)                 | `ActionReplayer` (BaseControlTask)             |
| `JointCommandOutput`    | `ActionReplayer` (in 100 Hz tick loop)            | `ControlCoordinator` → hardware                |

---

## `DatasetConfig` as the single source of truth

`DatasetConfig` (loaded once from `dataset.yaml`) drives module configs
across stages — same instance, no drift between train and serve.

```python
# Top-level, in each blueprint factory:
spec = DatasetConfig.from_file(spec_path)

# Passed as a typed field on each module's config:
EpisodeMonitorModule.blueprint(spec=spec)         # Stage 1: spec.episodes
DataPrepModule.blueprint(spec=spec)               # Stage 1: full spec
ChunkPolicyModule.blueprint(spec=spec, ...)       # Stage 3: spec.observation, spec.sync
```

| Stage | Module | How it gets the spec |
|---|---|---|
| 1A    | `EpisodeMonitorModule` | passed in via blueprint (`spec=spec`); reads `spec.episodes` for button maps |
| 1B    | `DataPrepModule`       | passed in via blueprint; reads full spec. **DataPrep snapshots the spec into `dataset/dataset.yaml`** so downstream stages don't need the YAML. |
| 2     | `TrainerModule`        | reads `dataset/dataset.yaml` + LeRobot `info.json`; copies spec snapshot into `checkpoint/dimos_meta.json` |
| 3     | `ChunkPolicyModule`    | reads `<policy_path>/dimos_meta.json` at `start()`; constructs `ObsBuilder` from the embedded spec. **No `--spec-path` flag needed at inference.** |

The operator only ever passes `--spec-path` for Recording and DataPrep
(stages where the spec is the input). After DataPrep, the spec rides
with the data.

Same `resolve_field` is invoked from `DataPrep.iter_episode_samples`
(Stage 1B) and `ObsBuilder.build` (Stage 3). One source of truth →
no train/serve skew.

---

## What's deliberately not in this tree

- **`RecordReplay`** — transport-layer hook (in `dimos/core/`), not a
  `learning/` Module. Enabled by `--record-path` at the CLI; unaware of
  what's recording.
- **`coordinator_action_replayer_<robot>`** — per-robot coordinator
  blueprints that register the `ActionReplayer` task. These live next
  to the rest of the per-robot wiring (likely
  `dimos/robot/<robot>/blueprints.py`), not under `learning/`.
- **A second `ControlCoordinator`** — the existing one is reused. We add
  one task type (`ActionReplayer`), not a parallel control stack.
- **New transports** — v1 is LCM-only on the wire.
- **New LCM message types** — `ActionChunk` is local-only pydantic in v1.
  Promote to a generated LCM type in v2 only if cross-language consumers
  need it.

---

## Module / non-Module split (one rule)

A class becomes a **Module** when it:
- has long-lived state worth `start()/stop()` lifecycle, **and**
- needs typed I/O ports across process boundaries.

Otherwise it stays a plain class or a `BaseControlTask`:

| Class | Type | Why |
|---|---|---|
| `EpisodeMonitorModule` | Module | Long-lived; subscribes to buttons; publishes status |
| `DataPrepModule`       | Module | Wraps subprocess; agent-callable via `@skill` |
| `TrainerModule`        | Module | Wraps subprocess; long-running; agent-callable |
| `ChunkPolicyModule`    | Module | Long-lived inference thread; latched In ports |
| `DataPrep`             | plain class | Stateless façade over static helpers; no ports |
| `ObsBuilder`           | plain class | Pure function over latched messages |
| `ActionReplayer`       | `BaseControlTask` | Must run in coordinator's 100 Hz thread, not via transport |
| `RecordReplay`         | transport hook | Captures every stream uniformly; not a Module |

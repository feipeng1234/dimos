# Folder Structure

Per-producer types: each Module owns its config + emitted message types
in its own file. No shared `config.py`, no umbrella class, no shared YAML.

```
dimos/learning/
│
├── specs/
│   ├── structure.md
│   ├── datacollection.md           # Stage 1
│   ├── training.md                 # Stage 2
│   └── inference.md                # Stage 3
│
├── dataprep.py                     # types + pure helpers (no Module)
│                                   #   - Episode, Sample
│                                   #   - StreamField, SyncConfig, OutputConfig, EpisodeExtractor
│                                   #   - resolve_field, compute_stats,
│                                   #     extract_episodes, iter_episode_samples
├── dataprep_module.py              # DataPrepModule(Config) only
│
├── collection/
│   ├── episode_monitor.py          # EpisodeStatus + EpisodeMonitorModule(Config)
│   └── blueprint.py                # learning_collect_quest_<robot>
│
├── formats/                        # dataset writers; each calls DataPrep.compute_stats
│   ├── lerobot.py                  # LeRobot v2 (parquet + MP4 + meta/stats.json)
│   ├── hdf5.py
│   └── rlds.py
│
├── training/
│   ├── trainer_module.py           # TrainerModule(Config); runs train_bc on a thread
│   ├── train.py                    # train_bc + train_val_split (lazy lerobot/torch)
│   ├── configs.py                  # BCConfig
│   └── blueprint.py                # learning_train
│
├── policy/
│   ├── base.py                     # ActionChunk + Policy Protocol
│   └── lerobot_policy.py           # LeRobotPolicy.load
│
└── inference/
    ├── chunk_policy_module.py      # ChunkPolicyModule(Config); ~30 Hz
    │                               #   (obs construction is a private method;
    │                               #    uses DataPrep.resolve_field)
    └── blueprint.py                # learning_infer_<robot>
```

`ActionReplayer` is a `ControlTask`, not a learning Module — it lives
with the other coordinator tasks:

```
dimos/control/
├── coordinator.py                  # adds action_chunk: In[ActionChunk]
│                                   #      _on_action_chunk → ActionReplayer
└── tasks/
    ├── teleop_task.py
    ├── ...
    └── action_replayer_task.py     # NEW; imports ActionChunk from learning/policy/base.py
```

Dependency: `control → learning.policy` (one-way).

---

## Per-producer typed contracts

| Class | Lives in | Used by |
|---|---|---|
| `EpisodeStatus`, `EpisodeMonitorModuleConfig` | `learning/collection/episode_monitor.py` | `EpisodeMonitorModule`; `DataPrep` |
| `EpisodeExtractor`, `StreamField`, `SyncConfig`, `OutputConfig`, `Episode`, `Sample` | `learning/dataprep.py` | `DataPrepModule`, `ChunkPolicyModule`, format writers |
| `DataPrepModuleConfig` | `learning/dataprep_module.py` | `DataPrepModule` |
| `BCConfig` | `learning/training/configs.py` | `train_bc` |
| `TrainerModuleConfig` | `learning/training/trainer_module.py` | `TrainerModule` |
| `ActionChunk`, `Policy` Protocol | `learning/policy/base.py` | `ChunkPolicyModule`, `ActionReplayer`, `ControlCoordinator` |
| `ChunkPolicyModuleConfig` | `learning/inference/chunk_policy_module.py` | `ChunkPolicyModule` |
| `ActionReplayerConfig` | `control/tasks/action_replayer_task.py` | `ActionReplayer` |

---

## Artifact flow

All generated artifacts live under `data/` (gitignored at repo root):

```
data/
├── sessions/<name>.db              ← RecordReplay
├── datasets/<name>/                ← DataPrepModule.build()
│   ├── data/        (parquet)
│   ├── videos/      (MP4)
│   └── meta/
│       ├── info.json
│       ├── episodes.jsonl
│       ├── stats.json              (DataPrep.compute_stats)
│       └── dimos_meta.json         (DataPrepModuleConfig.model_dump())
└── runs/<name>/                    ← train_bc
    ├── *.safetensors
    └── dimos_meta.json             (dataset snapshot + policy fields)
```

`dimos_meta.json` rides with the data: DataPrep writes it; training
copies it forward + adds policy fields; inference reads it at `start()`.
Operator never passes a spec path.

---

## Configuration

All module config is set as kwargs in the blueprint. No CLI flags on
our modules. Framework CLI surface is `GlobalConfig` only (env vars,
`.env`, things like `--record-path`).

---

## Module / non-Module split

A class becomes a **Module** when it has long-lived state with
`start()/stop()` lifecycle **and** typed I/O ports.

| Class | Type | Why |
|---|---|---|
| `EpisodeMonitorModule` | Module | Long-lived; subscribes to inputs; publishes status |
| `DataPrepModule`       | Module | Long-running build job |
| `TrainerModule`        | Module | Runs training on a daemon thread |
| `ChunkPolicyModule`    | Module | Long-lived inference thread |
| `ActionReplayer`       | `BaseControlTask` | Runs in coordinator's 100 Hz thread |
| `RecordReplay`         | transport hook | Captures every stream uniformly |

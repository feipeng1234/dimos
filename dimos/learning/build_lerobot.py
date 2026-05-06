# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a LeRobot v2 dataset from one or more DimOS recording .db files.

Each `dimos --record-path X.db run learning-collect-quest-...` produces one
.db. Pressing A starts an episode and B saves it. To merge N such files into
one LeRobot dataset, drive each through `extract_episodes` +
`iter_episode_samples` and feed the combined sample iterator into the
LeRobot writer in a single pass so episode/frame indices and per-feature
stats are computed once.

CLI::

    python -m dimos.learning.build_lerobot \\
        data/recordings/ep1.db data/recordings/ep2.db data/recordings/ep3.db \\
        --output data/datasets/pickplace_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dimos.learning.dataprep import (
    Episode,
    EpisodeExtractor,
    OutputConfig,
    Sample,
    StreamField,
    SyncConfig,
    extract_episodes,
    get_writer,
    iter_episode_samples,
)
from dimos.memory2.store.sqlite import SqliteStore
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def build_lerobot_from_dbs(
    sources: list[Path],
    output: Path,
    observation: dict[str, StreamField],
    action: dict[str, StreamField],
    sync: SyncConfig,
    status_stream: str = "status",
    metadata: dict[str, Any] | None = None,
    default_task_label: str | None = None,
) -> Path:
    """Merge episodes from N recording .db files into one LeRobot v2 dataset.

    Each source .db is opened, scanned for successful episodes via the
    recorded EpisodeStatus stream, then drained one-by-one through
    `iter_episode_samples` while episode IDs are renumbered globally.

    Returns the dataset root path (same as `output`).
    """
    if not sources:
        raise ValueError("Need at least one source .db path")

    streams = {**observation, **action}
    obs_keys = set(observation)
    action_keys = set(action)
    extractor_cfg = EpisodeExtractor(
        extractor="episode_status", status_stream=status_stream
    )

    plan: list[tuple[Path, SqliteStore, list[Episode]]] = []
    global_episodes: list[Episode] = []
    counter = 0
    try:
        for src in sources:
            store = SqliteStore(path=str(src), must_exist=True)
            try:
                if status_stream not in store.list_streams():
                    raise RuntimeError(
                        f"{src}: status stream {status_stream!r} not present "
                        f"(streams: {store.list_streams()}). This .db was likely "
                        "produced by `dimos recorder` rather than `--record-path`."
                    )
                eps = [e for e in extract_episodes(store, extractor_cfg) if e.success]
            except Exception:
                store.stop()
                raise

            renumbered: list[Episode] = []
            for e in eps:
                new_id = f"ep_{counter:06d}"
                counter += 1
                label = e.task_label or default_task_label
                renumbered.append(e.model_copy(update={"id": new_id, "task_label": label}))

            logger.info(
                "[build_lerobot] %s -> %d successful episode(s)",
                src, len(renumbered),
            )
            plan.append((src, store, renumbered))
            global_episodes.extend(renumbered)

        if not global_episodes:
            raise RuntimeError(
                "No successful episodes across sources. Make sure you pressed "
                "B (save) before stopping each `dimos run`."
            )

        logger.info(
            "[build_lerobot] %d source(s), %d total episode(s) -> %s",
            len(sources), len(global_episodes), output,
        )

        def _all_samples() -> Iterator[Sample]:
            for src, store, eps in plan:
                for ep in eps:
                    try:
                        yield from iter_episode_samples(
                            store=store,
                            episode=ep,
                            streams=streams,
                            sync=sync,
                            obs_keys=obs_keys,
                            action_keys=action_keys,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"{src} episode {ep.id}: {type(e).__name__}: {e}"
                        ) from e

        out_cfg = OutputConfig(format="lerobot", path=output, metadata=metadata or {})
        dataset_path = get_writer("lerobot")(_all_samples(), out_cfg)

        _write_dimos_meta(
            Path(dataset_path),
            sources=[Path(s) for s in sources],
            observation=observation,
            action=action,
            sync=sync,
            episodes=global_episodes,
            metadata=metadata or {},
        )

        logger.info(
            "[build_lerobot] succeeded — wrote %d episode(s) to %s",
            len(global_episodes), dataset_path,
        )
        return Path(dataset_path)
    finally:
        for _src, store, _eps in plan:
            store.stop()


def _write_dimos_meta(
    dataset_path: Path,
    sources: list[Path],
    observation: dict[str, StreamField],
    action: dict[str, StreamField],
    sync: SyncConfig,
    episodes: list[Episode],
    metadata: dict[str, Any],
) -> None:
    """Sidecar describing how this dataset was built. Mirrors the schema
    written by DataPrepModule._write_dimos_meta, plus a `sources` list."""
    meta = {
        "sources":     [str(s) for s in sources],
        "observation": {k: v.model_dump() for k, v in observation.items()},
        "action":      {k: v.model_dump() for k, v in action.items()},
        "sync":        sync.model_dump(),
        "episodes": [
            {
                "id": e.id,
                "start_ts": e.start_ts,
                "end_ts": e.end_ts,
                "task_label": e.task_label,
                "success": e.success,
            }
            for e in episodes
        ],
        "format":   "lerobot",
        "metadata": metadata,
    }
    with open(dataset_path / "dimos_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _default_streams(with_images: bool) -> tuple[dict[str, StreamField], dict[str, StreamField]]:
    """Defaults for `learning-collect-quest-quest-xarm7[-sim]` recordings.

    Observation: joint_state (+ wrist color_image when available).
    Action: joint_state.position (next-frame state — standard ACT behavior cloning).
    """
    observation: dict[str, StreamField] = {
        "joint_state": StreamField(stream="joint_state", field="position"),
    }
    if with_images:
        observation["wrist"] = StreamField(stream="color_image", field="data")
    action = {
        "joint_target": StreamField(stream="joint_state", field="position"),
    }
    return observation, action


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m dimos.learning.build_lerobot",
        description=(
            "Build one LeRobot v2 dataset from N DimOS sqlite recording files. "
            "Defaults match `learning-collect-quest-xarm7-sim` recordings."
        ),
    )
    p.add_argument(
        "sources", nargs="+", type=Path,
        help="Paths to recording .db files (one or more)",
    )
    p.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output dataset directory (will be created)",
    )
    p.add_argument(
        "--rate-hz", type=float, default=30.0,
        help="Sample rate in Hz (default: 30)",
    )
    p.add_argument(
        "--tolerance-ms", type=float, default=50.0,
        help="Per-stream sync tolerance in ms (default: 50)",
    )
    p.add_argument(
        "--anchor", default="joint_state",
        help="Anchor stream for sync (default: joint_state)",
    )
    p.add_argument(
        "--no-images", action="store_true",
        help="Skip the color_image observation (use for state-only datasets)",
    )
    p.add_argument(
        "--status-stream", default="status",
        help="Stream carrying EpisodeStatus events (default: status)",
    )
    p.add_argument(
        "--task-label", default="pick_and_place",
        help="Default task label for episodes without one (default: pick_and_place)",
    )
    p.add_argument(
        "--robot", default="xarm7",
        help="Robot name written into LeRobot info.json (default: xarm7)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    missing = [str(s) for s in args.sources if not s.exists()]
    if missing:
        logger.error("Source file(s) not found: %s", missing)
        return 2

    observation, action = _default_streams(with_images=not args.no_images)
    sync = SyncConfig(
        anchor=args.anchor,
        rate_hz=args.rate_hz,
        tolerance_ms=args.tolerance_ms,
    )
    metadata = {"fps": int(round(args.rate_hz)), "robot": args.robot}

    try:
        build_lerobot_from_dbs(
            sources=list(args.sources),
            output=args.output,
            observation=observation,
            action=action,
            sync=sync,
            status_stream=args.status_stream,
            metadata=metadata,
            default_task_label=args.task_label,
        )
    except Exception as e:
        logger.error("[build_lerobot] FAILED: %s: %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

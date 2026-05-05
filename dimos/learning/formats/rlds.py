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

"""RLDS / TFDS dataset writer.

Emits a single TFRecord shard following the RLDS Episode/Step protocol.
Each example is a ``tf.train.SequenceExample`` whose feature_lists hold
per-step observation / action / reward / discount / is_first / is_last /
is_terminal arrays, plus a context for episode metadata.

Layout::

    <output.path>/
        features.json            schema (feature names + shapes + dtypes)
        dataset_info.json        episode count, step count, fps, robot, stats
        rlds-00000-of-00001.tfrecord

This is RLDS-shaped on the wire; loading it as a `tfds.builder` requires
matching the schema in a TFDS DatasetBuilder. See the RLDS docs for that
glue. For OpenX-Embodiment contributions, point your TFDS builder at
this directory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from dimos.learning.dataprep import OutputConfig, Sample
from dimos.learning.formats._stats import StreamingStats


def write(samples: Iterator[Sample], output: OutputConfig) -> Path:
    """Drain `samples` into RLDS-style TFRecord shards. Returns the dataset dir."""
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "RLDS writer requires tensorflow — install with "
            "`pip install tensorflow_datasets` (pulls in tf)"
        ) from e

    root = Path(output.path)
    root.mkdir(parents=True, exist_ok=True)

    stats = StreamingStats(
        image_subsample=int(output.metadata.get("image_subsample", 10)),
        quantile_reservoir=int(output.metadata.get("quantile_reservoir", 10_000)),
        seed=int(output.metadata.get("stats_seed", 0)),
    )

    default_task_label = output.metadata.get("default_task_label", "task")
    fps = float(output.metadata.get("fps", 30.0))

    feature_shapes: dict[str, tuple[int, ...]] = {}
    feature_dtypes: dict[str, str] = {}
    tasks_index: dict[str, int] = {}

    # Per-episode buffers.
    cur_id: str | None = None
    cur_idx = -1
    cur_start_ts: float | None = None
    buf_ts: list[float] = []
    buf_obs: dict[str, list[np.ndarray]] = {}
    buf_act: dict[str, list[np.ndarray]] = {}

    total_frames = 0
    episodes_meta: list[dict[str, Any]] = []
    shard_path = root / "rlds-00000-of-00001.tfrecord"

    def _bytes(arr: np.ndarray) -> "tf.train.Feature":
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr.tobytes()]))

    def _flush(writer: "tf.io.TFRecordWriter") -> None:
        nonlocal cur_idx, cur_start_ts
        if cur_idx < 0 or not buf_ts:
            return

        T = len(buf_ts)
        feature_lists: dict[str, "tf.train.FeatureList"] = {}

        def _make_list(arrs: list[np.ndarray]) -> "tf.train.FeatureList":
            return tf.train.FeatureList(feature=[_bytes(a) for a in arrs])

        for k, frames in buf_obs.items():
            feature_lists[f"observation/{k}"] = _make_list(frames)
        for k, frames in buf_act.items():
            feature_lists[f"action/{k}"] = _make_list(frames)

        ts_arr = np.asarray(buf_ts, dtype=np.float32)
        feature_lists["timestamp"] = tf.train.FeatureList(
            feature=[tf.train.Feature(float_list=tf.train.FloatList(value=[t])) for t in ts_arr]
        )
        # RLDS step booleans.
        is_first = [i == 0     for i in range(T)]
        is_last  = [i == T - 1 for i in range(T)]
        for name, vals in (("is_first", is_first), ("is_last", is_last), ("is_terminal", is_last)):
            feature_lists[name] = tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)])) for v in vals
            ])
        # Default reward / discount per RLDS convention.
        feature_lists["reward"] = tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])) for _ in range(T)
        ])
        feature_lists["discount"] = tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])) for _ in range(T)
        ])

        ctx = tf.train.Features(feature={
            "episode_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[cur_idx])),
            "length":        tf.train.Feature(int64_list=tf.train.Int64List(value=[T])),
            "start_ts":      tf.train.Feature(float_list=tf.train.FloatList(value=[float(cur_start_ts or 0.0)])),
            "task":          tf.train.Feature(bytes_list=tf.train.BytesList(value=[default_task_label.encode()])),
        })
        ex = tf.train.SequenceExample(
            context=ctx,
            feature_lists=tf.train.FeatureLists(feature_list=feature_lists),
        )
        writer.write(ex.SerializeToString())

        episodes_meta.append({
            "episode_index": cur_idx,
            "length":        T,
            "start_ts":      float(cur_start_ts or 0.0),
            "task":          default_task_label,
        })
        buf_ts.clear()
        buf_obs.clear()
        buf_act.clear()

    with tf.io.TFRecordWriter(str(shard_path)) as writer:
        for sample in samples:
            if sample.episode_id != cur_id:
                _flush(writer)
                cur_id = sample.episode_id
                cur_idx += 1
                cur_start_ts = float(sample.ts)
                if default_task_label not in tasks_index:
                    tasks_index[default_task_label] = len(tasks_index)

            buf_ts.append(float(sample.ts) - (cur_start_ts or 0.0))
            for k, v in sample.observation.items():
                a = np.asarray(v)
                buf_obs.setdefault(k, []).append(a)
                stats.update(f"observation.{k}", a)
                if k not in feature_shapes:
                    feature_shapes[f"observation/{k}"] = tuple(a.shape)
                    feature_dtypes[f"observation/{k}"] = str(a.dtype)
            for k, v in sample.action.items():
                a = np.asarray(v)
                buf_act.setdefault(k, []).append(a)
                stats.update(f"action.{k}", a)
                if k not in feature_shapes:
                    feature_shapes[f"action/{k}"] = tuple(a.shape)
                    feature_dtypes[f"action/{k}"] = str(a.dtype)
            total_frames += 1

        _flush(writer)

    # ── sidecar metadata ─────────────────────────────────────────────────────
    features_meta = {
        name: {"shape": list(shape), "dtype": feature_dtypes[name]}
        for name, shape in feature_shapes.items()
    }
    features_meta["timestamp"] = {"shape": [], "dtype": "float32"}
    features_meta["is_first"]  = {"shape": [], "dtype": "int64"}
    features_meta["is_last"]   = {"shape": [], "dtype": "int64"}
    features_meta["is_terminal"] = {"shape": [], "dtype": "int64"}
    features_meta["reward"]    = {"shape": [], "dtype": "float32"}
    features_meta["discount"]  = {"shape": [], "dtype": "float32"}

    info = {
        "format_version": "rlds-1.0",
        "robot":          output.metadata.get("robot", "unknown"),
        "fps":            fps,
        "num_episodes":   len(episodes_meta),
        "num_steps":      total_frames,
        "num_tasks":      len(tasks_index),
        "tasks":          {idx: task for task, idx in tasks_index.items()},
        "episodes":       episodes_meta,
        "stats":          stats.finalize(),
    }
    with open(root / "features.json", "w") as f:
        json.dump(features_meta, f, indent=2)
    with open(root / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return root

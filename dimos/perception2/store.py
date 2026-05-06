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

"""ObjectStore — clustered persistent record of perceived objects.

Wraps a memory2 ``Stream[Object3D]`` to give the perception module:

* ``observe(name, world_pts, ts, confidence)`` — find-or-create with
  same-label-within-radius merging, accumulating pointcloud + bbox.
* ``snapshot()`` — list of currently-tracked Object3D records, freshest
  first, bounded by ``max_objects``.
* ``find_by_text(query, k)`` — substring score for v1; replace with
  CLIP/text-embed similarity in a follow-up.
* ``find_by_name(name)`` — exact-name lookup for ``goto_object``.

memory2 is append-only, so we keep the source-of-truth in an
in-process dict ``self._objects: dict[cluster_id, Object3D]`` and use
the Stream as a durable change log + cross-restart replay.  On
``observe``, we update the dict AND append the new state to the
Stream.  On startup, ``replay()`` rebuilds the dict from the latest
entry per cluster_id in the Stream.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import time
from typing import Any

import numpy as np

from dimos.memory2.stream import Stream
from dimos.perception2.object3d import Object3D
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ObjectStore:
    """In-memory dict + memory2-backed change log of perceived objects."""

    def __init__(
        self,
        stream: Stream[Object3D],
        *,
        merge_radius_m: float = 0.5,
        max_objects: int = 500,
        accumulator_max_points: int = 4096,
    ) -> None:
        """Construct the store.

        Parameters
        ----------
        stream:
            memory2 ``Stream[Object3D]`` for persistence.  Created
            via ``store.stream("objects", Object3D)``.
        merge_radius_m:
            Two detections with the same label whose centers are
            within this distance get merged into one record.
            Different labels never merge regardless of distance.
        max_objects:
            Hard cap on the in-memory dict — LRU evicts on overflow.
            The memory2 stream is uncapped (append-only log).
        accumulator_max_points:
            Per-object pointcloud cap; merged() downsamples beyond
            this.  Keeps storage bounded even for long-lived objects
            seen many times.
        """
        self._stream = stream
        self._merge_radius_m = merge_radius_m
        self._max_objects = max_objects
        self._accumulator_max_points = accumulator_max_points
        # OrderedDict so popitem(last=False) gives true LRU eviction;
        # cluster_id == "<sanitized_name>_<seq>" — stable across the
        # session, not globally unique across restarts (replay rebuilds).
        self._objects: OrderedDict[str, Object3D] = OrderedDict()
        self._next_seq: int = 0

    def replay(self) -> None:
        """Rebuild the in-memory dict from the underlying Stream.

        Called once at module start.  Walks every persisted observation
        and keeps the latest (by ``ts``) entry per ``cluster_id`` tag.
        Stream is iterated lazily so this is cheap even with a long
        history.
        """
        latest: dict[str, tuple[float, Object3D]] = {}
        try:
            for obs in self._stream.to_list():
                cluster_id = obs.tags.get("cluster_id") if obs.tags else None
                if not isinstance(cluster_id, str):
                    continue
                payload: Object3D = obs.data
                cur = latest.get(cluster_id)
                if cur is None or payload.ts >= cur[0]:
                    latest[cluster_id] = (payload.ts, payload)
        except Exception as e:
            logger.warning("ObjectStore.replay failed (continuing fresh): %s", e)
            return
        for cluster_id, (_, obj) in sorted(latest.items(), key=lambda kv: kv[1][0]):
            self._objects[cluster_id] = obj
            num = _seq_from_cluster_id(cluster_id)
            if num is not None and num >= self._next_seq:
                self._next_seq = num + 1
        if self._objects:
            logger.info("ObjectStore replay: %d objects restored", len(self._objects))

    def observe(
        self,
        name: str,
        world_pts: np.ndarray,
        ts: float | None = None,
        confidence: float = 1.0,
    ) -> Object3D:
        """Find-or-create + merge.  Returns the (possibly updated) Object3D."""
        ts = ts if ts is not None else time.time()
        candidate = Object3D.from_pointcloud(name, world_pts, ts=ts, confidence=confidence)

        merged_id = self._find_match(name, candidate.center)
        if merged_id is not None:
            existing = self._objects[merged_id]
            updated = existing.merged(world_pts, ts=ts, max_points=self._accumulator_max_points)
            # Bump confidence to the max across observations — single
            # cautious detection shouldn't drag down a confident track.
            updated = updated.model_copy(
                update=dict(confidence=max(existing.confidence, confidence))
            )
            self._objects[merged_id] = updated
            self._objects.move_to_end(merged_id)  # mark MRU
            self._append(merged_id, updated)
            return updated

        cluster_id = self._next_id(name)
        self._objects[cluster_id] = candidate
        self._evict_if_full()
        self._append(cluster_id, candidate)
        return candidate

    def snapshot(self) -> list[Object3D]:
        """All currently-tracked objects, freshest (by ts) first."""
        return sorted(self._objects.values(), key=lambda o: o.ts, reverse=True)

    def find_by_text(self, query: str, k: int = 3) -> list[Object3D]:
        """Top-k objects whose names contain the query (case-insensitive).

        v1: simple substring score.  Future: replace with CLIP
        text-embed cosine via ``self._stream.search(...)`` once we
        embed object names on insert.
        """
        q = query.strip().lower()
        if not q:
            return []
        scored = []
        for obj in self._objects.values():
            name_lc = obj.name.lower()
            if q in name_lc:
                # Shorter names matching are better matches; ts breaks ties.
                score = (-len(name_lc), obj.ts)
                scored.append((score, obj))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [obj for _, obj in scored[:k]]

    def find_by_name(self, name: str) -> Object3D | None:
        """Most recently seen object whose name matches (case-insensitive)."""
        target = name.strip().lower()
        candidates = [obj for obj in self._objects.values() if obj.name.lower() == target]
        if not candidates:
            # Fall back to substring — agent may say "chair" for a "wooden chair"
            sub = self.find_by_text(target, k=1)
            return sub[0] if sub else None
        candidates.sort(key=lambda o: o.ts, reverse=True)
        return candidates[0]

    def _find_match(self, name: str, center: tuple[float, float, float]) -> str | None:
        """Existing cluster_id for same-label entry within merge_radius_m, or None."""
        target = name.strip().lower()
        cx, cy, cz = center
        best: tuple[float, str] | None = None
        for cluster_id, obj in self._objects.items():
            if obj.name.strip().lower() != target:
                continue
            ox, oy, oz = obj.center
            d = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2 + (cz - oz) ** 2)
            if d <= self._merge_radius_m and (best is None or d < best[0]):
                best = (d, cluster_id)
        return best[1] if best is not None else None

    def _next_id(self, name: str) -> str:
        sanitized = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
        seq = self._next_seq
        self._next_seq += 1
        return f"{sanitized}_{seq}"

    def _evict_if_full(self) -> None:
        while len(self._objects) > self._max_objects:
            evicted_id, evicted = self._objects.popitem(last=False)
            logger.debug("ObjectStore evicted %s (%s)", evicted_id, evicted.name)

    def _append(self, cluster_id: str, obj: Object3D) -> None:
        try:
            self._stream.append(
                obj,
                ts=obj.ts,
                pose=obj.center,
                tags={
                    "cluster_id": cluster_id,
                    "name": obj.name,
                    "n_obs": obj.n_obs,
                },
            )
        except Exception as e:
            # Store failure shouldn't kill perception — log and keep going
            # with the in-memory dict; we lose persistence for this entry only.
            logger.warning("ObjectStore append failed: %s", e)


def _seq_from_cluster_id(cluster_id: str) -> int | None:
    tail = cluster_id.rsplit("_", 1)
    if len(tail) != 2:
        return None
    try:
        return int(tail[1])
    except ValueError:
        return None


# Suppress pyright on the pose tag being "Any" — memory2 stores it
# opaquely and we round-trip through replay() rather than relying on
# .near() spatial filters in v1.
_ = Any


__all__ = ["ObjectStore"]

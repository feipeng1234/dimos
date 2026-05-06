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

"""ObjectPerception — Qwen-VL detector + lidar back-projection + memory2.

Drop-in replacement for the legacy ``perception/detection/moduleDB.py``
``ObjectDBModule``.  Subscribes to the same inputs (color_image +
pointcloud), publishes a new ``ObjectBoundingBoxes`` topic for the
viser overlay, and persists every detection to a memory2 SqliteStore.

Wiring per perception tick:
    image + pointcloud + tf("camera_optical" -> world) + camera_info
        |
        v   (1) detector.detect(image) -> [Detection2DBBox]
        |
        |   (2) for each bbox:
        |       back_project_bbox(...) -> world-frame Nx3
        |       store.observe(name, pts, ts, conf) -> Object3D
        |
        v   (3) publish ObjectBoundingBoxes from store.snapshot()

Tick rate is config.period_s (default 1 Hz).  Detection only fires
when a fresh image is available (sharpness barrier built into
upstream camera modules already throttles per-pixel duplicates).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from pydantic import ConfigDict
from reactivex import interval
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.memory2.module import MemoryModule, MemoryModuleConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import make_vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.navigation_spec import NavigationInterfaceSpec
from dimos.perception2.back_projection import back_project_bbox
from dimos.perception2.detector import Detector, VlDetector
from dimos.perception2.msgs import ObjectBoundingBox, ObjectBoundingBoxes
from dimos.perception2.object3d import Object3D
from dimos.perception2.store import ObjectStore
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ObjectPerceptionConfig(MemoryModuleConfig):
    """Tunables for ``ObjectPerception``.

    Inherits ``db_path`` from ``MemoryModuleConfig`` (default
    ``"recording.db"`` resolved against ``DIMOS_PROJECT_ROOT``); pass
    a perception-specific path like ``"perception2_objects.db"``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: str = "perception2_objects.db"

    camera_info: CameraInfo
    """Pinhole intrinsics matching the camera that produces
    ``color_image``.  Required because ``CameraInfo`` doesn't flow as
    a port (the upstream publisher's rate is too low to align on
    every detection frame); pass it once at construction time."""

    period_s: float = 1.0
    """Seconds between perception ticks.  1 Hz default keeps Qwen-VL
    cost reasonable (~¢/image) and matches the rate at which a slowly
    walking robot sees genuinely new content."""

    merge_radius_m: float = 0.5
    """Same-label detections within this distance merge into one
    ``Object3D``.  Tune up for sparser scenes, down for cluttered."""

    max_objects: int = 500
    """LRU cap on the in-memory dict (memory2 stream is uncapped log)."""

    publish_min_obs: int = 1
    """Only objects with ``n_obs >= this`` are published to the viser
    overlay.  Set to 2+ to suppress single-frame flicker."""


class ObjectPerception(MemoryModule):
    """Lidar + VL-detector → memory2-backed 3D object DB.

    Inputs
    ------
    color_image : In[Image]
        RGB feed; one frame per perception tick is consumed.
    pointcloud : In[PointCloud2]
        World-frame lidar — the fused 360° cloud from
        ``MujocoSimModule`` is what we expect.

    Outputs
    -------
    object_bboxes : Out[ObjectBoundingBoxes]
        Per-tick snapshot of all currently-tracked objects, for the
        viser overlay.  Published at ``config.period_s``.
    """

    config: ObjectPerceptionConfig

    color_image: In[Image]
    pointcloud: In[PointCloud2]
    object_bboxes: Out[ObjectBoundingBoxes]

    # Auto-injected by the dimos coordinator from any module satisfying
    # the NavigationInterfaceSpec protocol — same pattern as
    # NavigationSkillContainer.  Used by goto_object().
    _navigation: NavigationInterfaceSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._detector: Detector | None = None
        self._object_store: ObjectStore | None = None
        self._latest_image: Image | None = None
        self._latest_pointcloud: PointCloud2 | None = None
        self._image_count: int = 0
        self._pc_count: int = 0
        self._tick_count: int = 0
        self._image_lock = threading.Lock()
        self._tick_in_flight = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        # Detector is constructed lazily here (not __init__) so blueprint
        # validation doesn't trigger an OpenAI client + API key check at
        # import time — important for offline test runs.
        self._detector = VlDetector()
        stream = self.store.stream("perception2_objects", Object3D)
        self._object_store = ObjectStore(
            stream,
            merge_radius_m=self.config.merge_radius_m,
            max_objects=self.config.max_objects,
        )
        self._object_store.replay()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
        self.register_disposable(Disposable(self.pointcloud.subscribe(self._on_pointcloud)))
        # Periodic tick.  reactivex.interval emits on a thread pool
        # scheduler — fine because _process_tick guards against
        # re-entry with a non-blocking lock.
        self.register_disposable(
            interval(self.config.period_s).subscribe(lambda _: self._process_tick())
        )
        logger.info(
            "ObjectPerception started: period=%.2fs, merge_radius=%.2fm, max_objects=%d",
            self.config.period_s,
            self.config.merge_radius_m,
            self.config.max_objects,
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_image(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_image = msg
            self._image_count += 1
            if self._image_count == 1 or self._image_count % 50 == 0:
                logger.info("ObjectPerception: image #%d received", self._image_count)

    def _on_pointcloud(self, msg: PointCloud2) -> None:
        with self._image_lock:
            self._latest_pointcloud = msg
            self._pc_count += 1
            if self._pc_count == 1 or self._pc_count % 10 == 0:
                logger.info(
                    "ObjectPerception: pointcloud #%d received (frame_id=%s, n_points=%d)",
                    self._pc_count,
                    msg.frame_id,
                    len(msg.pointcloud.points) if msg.pointcloud is not None else 0,
                )

    def _process_tick(self) -> None:
        self._tick_count += 1
        if self._tick_count <= 3 or self._tick_count % 10 == 0:
            logger.info(
                "ObjectPerception: tick #%d (image=%s, pc=%s, lock_held=%s)",
                self._tick_count,
                self._latest_image is not None,
                self._latest_pointcloud is not None,
                not self._tick_in_flight.acquire(blocking=False)
                or (self._tick_in_flight.release() or True),
            )
        # Skip if a previous tick is still mid-detection — Qwen calls
        # are slow (~500ms+), and we'd rather drop ticks than queue
        # them.  Non-blocking acquire keeps scheduler threads free.
        if not self._tick_in_flight.acquire(blocking=False):
            logger.info(
                "ObjectPerception: tick #%d skipped (previous still in flight)", self._tick_count
            )
            return
        try:
            self._do_tick()
        except Exception as e:
            logger.error("ObjectPerception tick failed: %s", e, exc_info=True)
        finally:
            self._tick_in_flight.release()

    def _do_tick(self) -> None:
        if self._detector is None or self._object_store is None:
            logger.warning("ObjectPerception _do_tick called before start finished")
            return
        with self._image_lock:
            image = self._latest_image
            pointcloud = self._latest_pointcloud
        if image is None or pointcloud is None:
            logger.info(
                "ObjectPerception: skipping tick — image=%s, pointcloud=%s",
                image is not None,
                pointcloud is not None,
            )
            return

        ts = time.time()

        # Use the IMAGE's frame_id as the target — that's the camera
        # the bboxes were detected from.  Decoupling from a hardcoded
        # frame name lets the splat camera (forward when
        # DIMOS_CAMERA_FORWARD=1) and any future camera publish their
        # own tf without us touching this code.
        transform = self.tf.get(image.frame_id, pointcloud.frame_id, image.ts, 5.0)
        if transform is None:
            logger.warning(
                "ObjectPerception tf lookup failed: %s -> %s @ %.3f",
                image.frame_id,
                pointcloud.frame_id,
                image.ts,
            )
            return

        logger.info("ObjectPerception: calling detector...")
        detections = self._detector.detect(image)
        logger.info("ObjectPerception: detector returned %d boxes", len(detections))
        if not detections:
            self._publish_snapshot(ts)
            return

        for det in detections:
            try:
                pts = back_project_bbox(
                    bbox_xyxy=tuple(det.bbox),  # type: ignore[arg-type]
                    world_pointcloud=pointcloud,
                    camera_info=self.config.camera_info,
                    world_to_optical=transform,
                )
            except Exception as e:
                logger.debug("back_project_bbox failed for %s: %s", det.name, e)
                continue
            if pts is None or pts.size == 0:
                continue
            self._object_store.observe(
                name=det.name,
                world_pts=pts,
                ts=ts,
                confidence=float(det.confidence),
            )

        self._publish_snapshot(ts)

    def _publish_snapshot(self, ts: float) -> None:
        if self._object_store is None:
            return
        boxes = [
            ObjectBoundingBox(
                name=obj.name,
                center=obj.center,
                extent=obj.extent,
                orientation_wxyz=obj.orientation_wxyz,
                confidence=obj.confidence,
                n_obs=obj.n_obs,
            )
            for obj in self._object_store.snapshot()
            if obj.n_obs >= self.config.publish_min_obs
        ]
        self.object_bboxes.publish(ObjectBoundingBoxes(boxes=boxes, ts=ts))

    # --- agent-callable skills ------------------------------------------

    @skill
    def list_objects(self) -> str:
        """List every object the robot has detected and remembers.

        Returns a markdown bullet list with the object's name, world
        position, and how many times it's been re-observed.  Use this
        to find out what's in the scene before deciding where to go.
        """
        if self._object_store is None:
            return "Object perception is not started yet."
        snap = self._object_store.snapshot()
        if not snap:
            return "No objects detected yet."
        lines = []
        for obj in snap:
            x, y, _ = obj.center
            lines.append(f"- {obj.name} @ ({x:+.2f}, {y:+.2f}) seen {obj.n_obs}x")
        return "\n".join(lines)

    @skill
    def find_object(self, query: str) -> str:
        """Find detected objects whose names match a description.

        Searches the perceived-object database for entries whose names
        contain the query (case-insensitive).  Returns up to 3 matches
        sorted by recency.  Use this when the agent has been asked
        about a specific kind of object.

        Args:
            query: The text to search for, e.g. ``"chair"`` or
                ``"american flag"``.
        """
        if self._object_store is None:
            return "Object perception is not started yet."
        hits = self._object_store.find_by_text(query, k=3)
        if not hits:
            return f"No objects matching '{query}'."
        lines = []
        for obj in hits:
            x, y, z = obj.center
            lines.append(f"- {obj.name} @ ({x:+.2f}, {y:+.2f}, {z:+.2f}) seen {obj.n_obs}x")
        return "\n".join(lines)

    @skill
    def goto_object(self, name: str) -> str:
        """Navigate to a previously detected object by name.

        Looks up the object in the perceived-object database and sets
        a navigation goal at its center.  Falls back to substring
        match if no exact name hits.  To cancel, call
        ``stop_navigation``.

        Args:
            name: The object name as it appears in ``list_objects``.
        """
        if self._object_store is None:
            return "Object perception is not started yet."
        target = self._object_store.find_by_name(name)
        if target is None:
            return f"No object called '{name}' has been detected."
        x, y, z = target.center
        goal = PoseStamped(
            position=make_vector3(x, y, z),
            orientation=Quaternion(),
            frame_id="map",
        )
        accepted = self._navigation.set_goal(goal)
        if not accepted:
            return (
                f"Found {target.name} at ({x:+.2f}, {y:+.2f}) but the "
                f"navigation stack rejected the goal."
            )
        return f"Heading to {target.name} at ({x:+.2f}, {y:+.2f})."


__all__ = ["ObjectPerception", "ObjectPerceptionConfig"]

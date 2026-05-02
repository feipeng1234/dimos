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

"""E2E integration test: cross-wall planning in the MuJoCo HSSD-house sim.

Drives the G1 nav stack through 5 waypoints spread across the converted
SceneSmith ``scene_209`` floor plan (4 bedrooms, 4 bathrooms, kitchen,
dining, gym, hallway, two living rooms - ~30 m N-S x 14 m E-W).

The test:

1. Boots the ``unitree_g1_nav_mujoco_sim`` blueprint with
   ``mujoco_headless=True`` so no GUI is required.
2. Subscribes to ``/odometry`` and the latest robot pose.
3. Sequentially publishes 5 goals on ``/clicked_point``, one in a
   different room each time.  For each, waits up to ``timeout_sec`` for
   the robot to arrive within ``threshold`` metres.
4. After every waypoint, snapshots a top-down render of the scene with
   the camera centred on the robot's latest pose.  We render in-process
   via ``mujoco.Renderer`` against the same MJCF the sim subprocess
   loaded — the camera feed therefore reflects the same world the nav
   stack sees, just from a documentation-friendly birds-eye angle.
5. Asserts the saved PNG decodes cleanly and that the robot's base z is
   ≥ ``_UPRIGHT_Z_MIN`` (so a frame where the G1 has fallen on its face
   fails the test).
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time

import pytest

# Heavy deps — skip the whole module if they're missing.
pytest.importorskip("gtsam")
pytest.importorskip("mujoco")
pytest.importorskip("PIL")

import lcm as lcmlib
import mujoco
from PIL import Image as PILImage

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.protocol.service.lcmservice import _DEFAULT_LCM_URL
from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_nav_mujoco_sim import (
    unitree_g1_nav_mujoco_sim,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# ── Topics ─────────────────────────────────────────────────────────────────
_ODOM_TOPIC = "/odometry#nav_msgs.Odometry"
_GOAL_TOPIC = "/clicked_point#geometry_msgs.PointStamped"

# Robot upright sanity bound — G1 base sits ~0.7 m when standing; if the
# snapshot pose has it below 0.4 m something has gone wrong (faceplant,
# fell through floor, etc.).
_UPRIGHT_Z_MIN = 0.4

# Seconds to give the nav stack to build the terrain / visibility graph
# before issuing the first goal.
_WARMUP_SEC = 15.0

# Path to the live HSSD-house MJCF (scene_209-derived).
# parents: tests/ → nav_stack/ → navigation/ → dimos/ → <repo root>/.
_SCENE_XML = Path(__file__).resolve().parents[4] / "data" / "hssd_house" / "scene_hssd_house.xml"


# Five waypoints spread across distinct rooms in the scene_209 floor plan.
# Coordinates are taken from the SceneSmith YAML's room frame translations
# (``combined_house/house.dmd.yaml``).
#
# Sequence forms a serpentine N-bound traversal — useful as a pre-test for
# loop-closure once we close the loop south again:
#
#     spawn(1.2, -1.0)
#         → dining (0.2, -3.2)        [step south to enter]
#         → kitchen (-3.0, 2.3)
#         → living_room (5.9, 2.8)
#         → bedroom_2 (4.65, 16.9)
#         → bedroom_4 (2.82, 22.25)   [far north]
#
# Tuple format: (label, x, y, z, timeout_sec, reach_threshold_m).
_WAYPOINTS: list[tuple[str, float, float, float, float, float]] = [
    ("dining_room", 0.2, -3.2, 0.0, 90.0, 1.5),
    ("kitchen", -3.0, 2.3, 0.0, 120.0, 1.5),
    ("living_room", 5.9, 2.8, 0.0, 120.0, 1.5),
    ("bedroom_2", 4.65, 16.9, 0.0, 240.0, 2.0),
    ("bedroom_4", 2.82, 22.25, 0.0, 240.0, 2.0),
]


pytestmark = [pytest.mark.slow, pytest.mark.mujoco]


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


class _OdomTracker:
    """Thread-safe latest-odom holder + LCM pump."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._lcm = lcmlib.LCM(_DEFAULT_LCM_URL)
        self._sub = self._lcm.subscribe(_ODOM_TOPIC, self._on_odom)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _on_odom(self, _channel: str, data: bytes) -> None:
        msg = Odometry.lcm_decode(data)
        with self._lock:
            self._count += 1
            self._x = msg.x
            self._y = msg.y
            self._z = msg.pose.position.z

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._lcm.handle_timeout(100)
            except Exception:
                pass

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def snapshot(self) -> tuple[float, float, float]:
        with self._lock:
            return self._x, self._y, self._z

    def publish_goal(self, x: float, y: float, z: float) -> None:
        goal = PointStamped(x=x, y=y, z=z, ts=time.time(), frame_id="map")
        self._lcm.publish(_GOAL_TOPIC, goal.lcm_encode())

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self._lcm.unsubscribe(self._sub)


class _SceneRenderer:
    """Birds-eye snapshot helper, centred on the robot's xy each call.

    Loads the same MJCF the sim subprocess loaded; we don't replicate
    the robot's joint state because the snapshot is for human review of
    the surrounding scene + the captured base pose, not a full ground-
    truth playback.
    """

    def __init__(self, scene_xml_path: Path, width: int = 640, height: int = 480) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        self._cam = mujoco.MjvCamera()

    def render_topdown(
        self,
        robot_xy: tuple[float, float],
        robot_z: float,
        out_path: Path,
    ) -> None:
        x, y = robot_xy
        self._cam.lookat[:] = (x, y, max(robot_z, 0.5))
        self._cam.distance = 6.0
        self._cam.azimuth = 90.0
        self._cam.elevation = -55.0
        self._renderer.update_scene(self._data, camera=self._cam)
        img = self._renderer.render()
        PILImage.fromarray(img).save(out_path)


def _wait_for_first_odom(tracker: _OdomTracker, timeout_sec: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if tracker.count > 0:
            return
        time.sleep(0.5)
    raise AssertionError(f"No odometry received after {timeout_sec}s — sim not running?")


def _wait_for_arrival(
    tracker: _OdomTracker,
    target_xy: tuple[float, float],
    threshold: float,
    timeout_sec: float,
) -> tuple[bool, tuple[float, float, float], float]:
    """Poll odom until robot is within ``threshold`` of ``target_xy`` or timeout."""
    gx, gy = target_xy
    t0 = time.monotonic()
    while True:
        x, y, z = tracker.snapshot()
        dist = _distance(x, y, gx, gy)
        if dist <= threshold:
            return True, (x, y, z), dist
        if time.monotonic() - t0 >= timeout_sec:
            return False, (x, y, z), dist
        time.sleep(0.2)


@pytest.mark.skipif(
    not _SCENE_XML.exists(),
    reason=f"scene XML missing: {_SCENE_XML} — run scenesmith_to_mjcf converter first",
)
def test_5_waypoint_loop_in_hssd_house(tmp_path: Path) -> None:
    """Drive the G1 through 5 waypoints in the converted scene_209 house.

    Asserts each waypoint reached within its budget and saves a PNG of the
    scene centred on the robot at every arrival.  Each frame must decode
    as a valid PNG and the robot's base z must be ≥ ``_UPRIGHT_Z_MIN`` so
    a faceplant fails the test.
    """
    # Kinematic-robot mode skips the ONNX walking policy: the robot's
    # floating base is integrated each tick from the latest cmd_vel and
    # joints stay frozen at home.  Removes gait dynamics from the test
    # — we only care about the planner driving the robot from waypoint
    # to waypoint, not whether the G1 actually walks there.
    blueprint = unitree_g1_nav_mujoco_sim.global_config(
        mujoco_headless=True,
        mujoco_kinematic_robot=True,
    )
    coordinator = ModuleCoordinator.build(blueprint)

    tracker = _OdomTracker()
    renderer = _SceneRenderer(_SCENE_XML)

    saved: list[Path] = []

    try:
        logger.info("[mujoco-cross-wall] coordinator built; waiting for odom…")
        _wait_for_first_odom(tracker, timeout_sec=60.0)

        x0, y0, z0 = tracker.snapshot()
        logger.info(
            "[mujoco-cross-wall] odom online; robot at "
            f"({x0:.2f}, {y0:.2f}, {z0:.2f}); warming up {_WARMUP_SEC}s…"
        )
        time.sleep(_WARMUP_SEC)

        for label, gx, gy, gz, timeout_sec, threshold in _WAYPOINTS:
            sx, sy, sz = tracker.snapshot()
            logger.info(
                f"[mujoco-cross-wall] === {label}: goal ({gx}, {gy}) | "
                f"robot ({sx:.2f}, {sy:.2f}) | "
                f"dist={_distance(sx, sy, gx, gy):.2f}m | "
                f"budget={timeout_sec}s ==="
            )
            tracker.publish_goal(gx, gy, gz)

            reached, (cx, cy, cz), final_dist = _wait_for_arrival(
                tracker, (gx, gy), threshold, timeout_sec
            )

            assert cz >= _UPRIGHT_Z_MIN, (
                f"{label}: robot base z={cz:.2f}m below upright bound "
                f"{_UPRIGHT_Z_MIN}m — likely fallen / clipping floor"
            )

            png_path = tmp_path / f"waypoint_{label}.png"
            renderer.render_topdown((cx, cy), cz, png_path)
            saved.append(png_path)

            assert png_path.exists() and png_path.stat().st_size > 0, (
                f"{label}: snapshot {png_path} not written"
            )
            with PILImage.open(png_path) as img:
                w, h = img.size
                assert w >= 320 and h >= 240, f"{label}: snapshot too small ({w}x{h})"
                assert img.mode in ("RGB", "RGBA"), (
                    f"{label}: snapshot mode {img.mode!r} not RGB/RGBA"
                )

            assert reached, (
                f"{label}: robot did not reach ({gx}, {gy}) within {timeout_sec}s. "
                f"Final pos=({cx:.2f}, {cy:.2f}), dist={final_dist:.2f}m. "
                f"snapshot saved at {png_path}"
            )
            logger.info(
                f"[mujoco-cross-wall] {label} reached at ({cx:.2f}, {cy:.2f}) — "
                f"dist={final_dist:.2f}m, snapshot {png_path}"
            )

        # Surface the saved-PNG paths so a human can locate the shots
        # in pytest's tmp dir output.
        logger.info(f"[mujoco-cross-wall] saved {len(saved)} snapshots in {tmp_path}")
        for p in saved:
            logger.info(f"  - {p}")

    finally:
        tracker.close()
        coordinator.stop()

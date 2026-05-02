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

"""Unit tests for MujocoPoseToOdometryAdapter.

The adapter converts a `PoseStamped` (mujoco connection convention,
frame_id="world") to a `nav_msgs/Odometry` (nav_stack convention,
frame_id=_FRAME_PARENT, child_frame_id=_FRAME_CHILD) and publishes a
matching `odom→body` Transform.

These tests bypass the full module lifecycle and exercise the
conversion logic directly so we can verify frame names, pose values,
and timestamps without standing up a coordinator.
"""

from __future__ import annotations

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.robot.unitree.g1.blueprints.navigation._mujoco_pose_adapter import (
    MujocoPoseToOdometryAdapter,
)

# Match the per-module string literals the adapter uses (frames.py
# was deleted on rosnav8 in favour of inline strings).
_FRAME_PARENT = "map"
_FRAME_CHILD = "sensor"


class _FakePort:
    """Minimal stand-in for an Out[T] port — captures published messages."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


def _make_adapter() -> tuple[MujocoPoseToOdometryAdapter, _FakePort, _FakePort]:
    """Build an adapter with stubbed `odometry` and `tf` ports.

    Avoids `MujocoPoseToOdometryAdapter()` (which would auto-instantiate
    `odom: In[PoseStamped]` and `odometry: Out[Odometry]` from the type
    annotations) so we can isolate the conversion logic.
    """
    adapter = MujocoPoseToOdometryAdapter.__new__(MujocoPoseToOdometryAdapter)
    odometry_port = _FakePort()
    tf_port = _FakePort()
    adapter.odometry = odometry_port  # type: ignore[assignment]
    adapter._tf = tf_port  # type: ignore[assignment]
    return adapter, odometry_port, tf_port


def test_publishes_odometry_with_nav_stack_frame_names() -> None:
    adapter, odometry_port, _ = _make_adapter()

    pose = PoseStamped(
        ts=42.0,
        frame_id="world",
        position=Vector3(1.0, 2.0, 0.5),
        orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )

    adapter._on_pose(pose)

    assert len(odometry_port.messages) == 1
    odom = odometry_port.messages[0]
    assert isinstance(odom, Odometry)
    assert odom.frame_id == _FRAME_PARENT
    assert odom.child_frame_id == _FRAME_CHILD
    assert odom.ts == 42.0
    assert odom.position.x == 1.0
    assert odom.position.y == 2.0
    assert odom.position.z == 0.5
    assert odom.orientation.w == 1.0


def test_publishes_tf_transform_alongside_odometry() -> None:
    adapter, _, tf_port = _make_adapter()

    pose = PoseStamped(
        ts=7.0,
        frame_id="world",
        position=Vector3(-3.0, 4.0, 1.0),
        orientation=Quaternion(0.0, 0.0, 0.7071, 0.7071),
    )

    adapter._on_pose(pose)

    assert len(tf_port.messages) == 1
    tf_msg = tf_port.messages[0]
    assert isinstance(tf_msg, Transform)
    assert tf_msg.frame_id == _FRAME_PARENT
    assert tf_msg.child_frame_id == _FRAME_CHILD
    assert tf_msg.ts == 7.0
    assert tf_msg.translation.x == -3.0
    assert tf_msg.translation.y == 4.0
    assert tf_msg.translation.z == 1.0


def test_each_pose_emits_one_odom_and_one_tf() -> None:
    adapter, odometry_port, tf_port = _make_adapter()

    # Note: PoseStamped substitutes ``time.time()`` for a literal ``ts=0``,
    # so we start at 1.0 to keep the assertion deterministic.
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
    for ts in timestamps:
        pose = PoseStamped(
            ts=ts,
            frame_id="world",
            position=Vector3(ts, 0.0, 0.0),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        adapter._on_pose(pose)

    assert len(odometry_port.messages) == 5
    assert len(tf_port.messages) == 5
    assert [m.ts for m in odometry_port.messages] == timestamps  # type: ignore[attr-defined]
    assert [m.ts for m in tf_port.messages] == timestamps  # type: ignore[attr-defined]

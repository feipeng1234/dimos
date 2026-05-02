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

The adapter converts a ``PoseStamped`` (the mujoco subprocess publishes
these on ``G1SimConnection.odom`` with frame_id="world") to a
``nav_msgs/Odometry``.  Frame names: parent passes through from the
input (so consumers see the same world frame the source labelled it
in), and child is hardcoded to ``"base_link"`` — G1's canonical body
frame, used throughout the rest of the G1 module tree
(``mujoco_sim.py::_publish_tf``, the camera modules, etc).
"""

from __future__ import annotations

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.robot.unitree.g1.blueprints.navigation._mujoco_pose_adapter import (
    MujocoPoseToOdometryAdapter,
)


class _FakePort:
    """Minimal stand-in for an Out[T] port — captures published messages."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


def _make_adapter() -> tuple[MujocoPoseToOdometryAdapter, _FakePort]:
    """Build an adapter with a stubbed ``odometry`` port.

    Avoids ``MujocoPoseToOdometryAdapter()`` (which would auto-instantiate
    ``odom: In[PoseStamped]`` and ``odometry: Out[Odometry]`` from the
    type annotations) so we can isolate the conversion logic.
    """
    adapter = MujocoPoseToOdometryAdapter.__new__(MujocoPoseToOdometryAdapter)
    odometry_port = _FakePort()
    adapter.odometry = odometry_port  # type: ignore[assignment]
    return adapter, odometry_port


def test_passes_through_input_frame_id_with_g1_body_child() -> None:
    adapter, odometry_port = _make_adapter()

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
    assert odom.frame_id == "world"  # passed through from PoseStamped
    assert odom.child_frame_id == "base_link"  # G1 canonical body frame
    assert odom.ts == 42.0
    assert odom.position.x == 1.0
    assert odom.position.y == 2.0
    assert odom.position.z == 0.5
    assert odom.orientation.w == 1.0


def test_inherits_alternate_frame_id_when_source_changes() -> None:
    adapter, odometry_port = _make_adapter()

    # If anyone ever changes G1SimConnection's PoseStamped frame_id, the
    # adapter inherits it instead of silently overriding to a hardcoded
    # name.
    pose = PoseStamped(
        ts=1.0,
        frame_id="some_other_world",
        position=Vector3(0.0, 0.0, 0.0),
        orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )

    adapter._on_pose(pose)

    odom = odometry_port.messages[0]
    assert isinstance(odom, Odometry)
    assert odom.frame_id == "some_other_world"
    assert odom.child_frame_id == "base_link"


def test_each_pose_emits_one_odom() -> None:
    adapter, odometry_port = _make_adapter()

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
    assert [m.ts for m in odometry_port.messages] == timestamps  # type: ignore[attr-defined]

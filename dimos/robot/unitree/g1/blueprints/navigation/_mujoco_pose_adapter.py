# Copyright 2025-2026 Dimensional Inc.
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

"""MuJoCo G1 → nav_stack pose/odometry adapter.

`G1SimConnection.odom` publishes a ``PoseStamped`` (frame_id="world",
the mujoco subprocess's world frame) on a stream consumed only by this
adapter.  The nav_stack consumers (PGO, SimplePlanner, FarPlanner,
MovementManager, TerrainAnalysis) want a ``nav_msgs/Odometry`` instead.

Frame conventions: we pass ``msg.frame_id`` straight through (so the
output Odometry inherits whatever world frame G1SimConnection's source
labels its pose in — currently "world"), and use ``"base_link"`` as the
child frame to match the existing G1 TF tree
(``mujoco_sim.py::_publish_tf`` publishes ``world → base_link``,
``base_link → camera_link``, etc).  We do NOT republish a transform on
``self.tf`` — that edge is already in the TF tree.

PGO's ``_on_odom`` only reads ``msg.pose`` + ``msg.ts``; the frame
strings are informational for downstream consumers that match against
their own ``ModuleConfig.world_frame`` / ``body_frame`` knobs (see
``SimplePlannerConfig`` / ``PGOConfig``).
"""

from __future__ import annotations

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry

# G1's canonical body frame, used by every other G1 module that
# publishes a transform with the robot as its child (see
# ``dimos/robot/unitree/g1/mujoco_sim.py::_publish_tf`` and
# ``dimos/robot/unitree/g1/blueprints/primitive/...``).  Hardcoding
# this matches the de-facto convention rather than introducing a
# fourth name.
_G1_BODY_FRAME = "base_link"


class MujocoPoseToOdometryAdapterConfig(ModuleConfig):
    pass


class MujocoPoseToOdometryAdapter(Module):
    """Convert mujoco PoseStamped → nav_msgs/Odometry, frame-passthrough."""

    config: MujocoPoseToOdometryAdapterConfig

    odom: In[PoseStamped]
    odometry: Out[Odometry]

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_pose)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_pose(self, msg: PoseStamped) -> None:
        # Twist fields stay zero — the mujoco G1 connection doesn't expose
        # linear/angular velocity.  Downstream PathFollower / LocalPlanner
        # treat absence of twist as zero, which is correct here.
        self.odometry.publish(
            Odometry(
                ts=msg.ts,
                frame_id=msg.frame_id,
                child_frame_id=_G1_BODY_FRAME,
                pose=Pose(position=msg.position, orientation=msg.orientation),
            )
        )


__all__ = ["MujocoPoseToOdometryAdapter", "MujocoPoseToOdometryAdapterConfig"]

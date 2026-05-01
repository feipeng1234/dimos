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

"""PoseStamped → nav_msgs/Odometry adapter for the MuJoCo G1 connection.

Nav stack expects `odometry: In[Odometry]` (nav_msgs.Odometry with twist),
but `G1SimConnection.odom` publishes `PoseStamped`. This module bridges
the two: it subscribes to a PoseStamped stream, wraps each pose in an
Odometry message with zero twist (mujoco G1's connection doesn't expose
linear/angular velocity), and republishes it.
"""

from __future__ import annotations

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry


class MujocoPoseToOdometryAdapterConfig(ModuleConfig):
    child_frame_id: str = "base_link"


class MujocoPoseToOdometryAdapter(Module):
    """Convert PoseStamped (mujoco connection) → nav_msgs/Odometry (nav stack)."""

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
        self.odometry.publish(
            Odometry(
                ts=msg.ts,
                frame_id=msg.frame_id or "world",
                child_frame_id=self.config.child_frame_id,
                pose=Pose(position=msg.position, orientation=msg.orientation),
            )
        )


__all__ = ["MujocoPoseToOdometryAdapter", "MujocoPoseToOdometryAdapterConfig"]

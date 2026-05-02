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

`G1SimConnection.odom` publishes `PoseStamped` with `frame_id="world"`
and the connection itself publishes the TF `world → base_link`. The
nav_stack consumers (SimplePlanner, FarPlanner, MovementManager,
TerrainAnalysis) expect:

- `odometry` as `nav_msgs/Odometry` with `frame_id=_FRAME_PARENT` (`"odom"`)
  and `child_frame_id=_FRAME_CHILD` (`"body"`).
- A TF chain that includes `odom → body` so `(map, body)` and
  `(odom, body)` lookups resolve.

This module bridges both:
- Subscribes to a PoseStamped stream, republishes as
  `nav_msgs/Odometry` with the nav_stack frame names (twist zeroed —
  the mujoco connection doesn't expose linear/angular velocity).
- Publishes the matching `odom → body` Transform on the TF tree.

PGO already publishes `map → odom` (loop-closure correction), so the
chain `map → odom → body` becomes resolvable end-to-end.
"""

from __future__ import annotations

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.nav_msgs.Odometry import Odometry

# Frame names match the convention the Unity sim's UnityBridgeModule
# publishes (``frame_id="map"``, ``child_frame_id="sensor"``) so the
# downstream nav_stack consumers (SimplePlanner / FarPlanner /
# MovementManager / TerrainAnalysis) see the same TF tree shape from
# either sim source.  Was previously
# ``from dimos.navigation.nav_stack.frames import _FRAME_CHILD, _FRAME_PARENT``
# but ``frames.py`` was deleted on rosnav8 in favour of per-module
# string literals.
_FRAME_PARENT = "map"
_FRAME_CHILD = "sensor"


class MujocoPoseToOdometryAdapterConfig(ModuleConfig):
    pass


class MujocoPoseToOdometryAdapter(Module):
    """Convert mujoco PoseStamped → nav_stack-conventioned Odometry + TF."""

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
                frame_id=_FRAME_PARENT,
                child_frame_id=_FRAME_CHILD,
                pose=Pose(position=msg.position, orientation=msg.orientation),
            )
        )

        self.tf.publish(
            Transform(
                translation=msg.position,
                rotation=msg.orientation,
                frame_id=_FRAME_PARENT,
                child_frame_id=_FRAME_CHILD,
                ts=msg.ts,
            )
        )


__all__ = ["MujocoPoseToOdometryAdapter", "MujocoPoseToOdometryAdapterConfig"]

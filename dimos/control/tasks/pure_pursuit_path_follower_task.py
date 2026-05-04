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

"""Classic Pure Pursuit path-follower as a passive ControlTask.

Geometric controller: find a fixed-distance lookahead point on the path,
compute the arc curvature κ to reach it, output ``(vx = target_speed,
wz = vx · κ)``. No adaptive lookahead, no curvature-based speed
regulation, no cross-track PID — that's :mod:`path_follower_task`
(Regulated Pure Pursuit).

For ``Pure Pursuit + FF`` cohort, supply ``ff_config`` — the static
plant-gain compensator divides cmd by ``K_plant`` before output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from dimos.control.task import (
    BaseControlTask,
    ControlMode,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
)
from dimos.control.tasks.feedforward_gain_compensator import (
    FeedforwardGainCompensator,
    FeedforwardGainConfig,
)
from dimos.control.tasks.path_controllers import PurePursuitController
from dimos.control.tasks.path_distancer import PathDistancer
from dimos.utils.logging_config import setup_logger
from dimos.utils.trigonometry import angle_diff

if TYPE_CHECKING:
    from dimos.core.global_config import GlobalConfig
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.nav_msgs.Path import Path

logger = setup_logger()

PurePursuitState = Literal[
    "idle", "initial_rotation", "path_following", "final_rotation", "arrived", "aborted"
]


@dataclass
class PurePursuitPathFollowerTaskConfig:
    joint_names: list[str] = field(default_factory=lambda: ["base/vx", "base/vy", "base/wz"])
    priority: int = 20
    target_speed: float = 0.55
    lookahead_distance: float = 0.5  # constant; classic PP
    control_frequency: float = 10.0
    k_angular: float = 0.6  # heading-correction term in PurePursuitController
    max_angular_velocity: float = 1.5
    goal_tolerance: float = 0.2
    orientation_tolerance: float = 0.35
    ff_config: FeedforwardGainConfig | None = None


class PurePursuitPathFollowerTask(BaseControlTask):
    """Classic Pure Pursuit. State machine mirrors BaselinePathFollowerTask
    so this slots into the same benchmark harness."""

    def __init__(
        self,
        name: str,
        config: PurePursuitPathFollowerTaskConfig,
        global_config: GlobalConfig,
    ) -> None:
        if len(config.joint_names) != 3:
            raise ValueError(
                f"PurePursuitPathFollowerTask '{name}' needs 3 joints "
                f"(vx, vy, wz), got {len(config.joint_names)}"
            )

        self._name = name
        self._config = config
        self._joint_names_list = list(config.joint_names)
        self._joint_names = frozenset(config.joint_names)

        self._controller = PurePursuitController(
            global_config,
            control_frequency=config.control_frequency,
            min_lookahead=config.lookahead_distance,
            max_lookahead=config.lookahead_distance,  # constant ⟹ classic PP
            lookahead_gain=0.0,  # don't scale with speed
            max_linear_speed=config.target_speed,
            k_angular=config.k_angular,
            max_angular_velocity=config.max_angular_velocity,
        )
        self._ff: FeedforwardGainCompensator | None = (
            FeedforwardGainCompensator(config.ff_config) if config.ff_config else None
        )

        self._state: PurePursuitState = "idle"
        self._path: Path | None = None
        self._distancer: PathDistancer | None = None
        self._current_odom: PoseStamped | None = None
        self._max_progress_idx: int = 0

    # ------------------------------------------------------------------
    # ControlTask protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def claim(self) -> ResourceClaim:
        return ResourceClaim(
            joints=self._joint_names,
            priority=self._config.priority,
            mode=ControlMode.VELOCITY,
        )

    def is_active(self) -> bool:
        return self._state in ("initial_rotation", "path_following", "final_rotation")

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        if not self.is_active():
            return None
        if self._path is None or self._distancer is None or self._current_odom is None:
            return None

        match self._state:
            case "initial_rotation":
                vx, vy, wz = self._step_initial_rotation()
            case "path_following":
                vx, vy, wz = self._step_path_following()
            case "final_rotation":
                vx, vy, wz = self._step_final_rotation()
            case _:
                return None

        if self._ff is not None:
            vx, vy, wz = self._ff.compute(vx, vy, wz)

        return JointCommandOutput(
            joint_names=self._joint_names_list,
            velocities=[vx, vy, wz],
            mode=ControlMode.VELOCITY,
        )

    def on_preempted(self, by_task: str, joints: frozenset[str]) -> None:
        if joints & self._joint_names and self.is_active():
            logger.warning(f"PurePursuitPathFollowerTask '{self._name}' preempted by {by_task}")
            self._state = "aborted"

    # ------------------------------------------------------------------
    # State-machine bodies
    # ------------------------------------------------------------------

    def _windowed_closest(self, pos: np.ndarray, window: int = 20) -> int:
        assert self._path is not None
        n = len(self._path.poses)
        lo = self._max_progress_idx
        hi = min(n, lo + window + 1)
        best_idx = lo
        best_d_sq = float("inf")
        for i in range(lo, hi):
            p = self._path.poses[i].position
            d_sq = (p.x - pos[0]) ** 2 + (p.y - pos[1]) ** 2
            if d_sq < best_d_sq:
                best_d_sq = d_sq
                best_idx = i
        return best_idx

    def _step_initial_rotation(self) -> tuple[float, float, float]:
        assert self._path is not None and self._current_odom is not None
        first_yaw = self._path.poses[0].orientation.euler[2]
        robot_yaw = self._current_odom.orientation.euler[2]
        yaw_err = angle_diff(first_yaw, robot_yaw)

        if abs(yaw_err) < self._config.orientation_tolerance:
            self._state = "path_following"
            return self._step_path_following()

        twist = self._controller.rotate(yaw_err)
        return float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)

    def _step_path_following(self) -> tuple[float, float, float]:
        assert self._path is not None
        assert self._distancer is not None
        assert self._current_odom is not None

        pos = np.array([self._current_odom.position.x, self._current_odom.position.y])

        closest = self._windowed_closest(pos)
        if closest > self._max_progress_idx:
            self._max_progress_idx = closest

        # Closed-path arrival gate: require traversal of >= 70% of path
        # before allowing arrival, otherwise circle/figure-8 trip on tick 1.
        progress_threshold = max(1, int(0.7 * (len(self._path.poses) - 1)))
        if (
            self._max_progress_idx >= progress_threshold
            and self._distancer.distance_to_goal(pos) < self._config.goal_tolerance
        ):
            self._state = "final_rotation"
            return self._step_final_rotation()

        lookahead = self._distancer.find_lookahead_point(closest)
        twist = self._controller.advance(
            lookahead,
            self._current_odom,
            current_speed=self._config.target_speed,
            path_curvature=None,  # classic PP: no curvature-based speed limit
        )
        return float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)

    def _step_final_rotation(self) -> tuple[float, float, float]:
        assert self._path is not None and self._current_odom is not None
        goal_yaw = self._path.poses[-1].orientation.euler[2]
        robot_yaw = self._current_odom.orientation.euler[2]
        yaw_err = angle_diff(goal_yaw, robot_yaw)

        if abs(yaw_err) < self._config.orientation_tolerance:
            self._state = "arrived"
            logger.info(f"PurePursuitPathFollowerTask '{self._name}' arrived")
            return 0.0, 0.0, 0.0

        twist = self._controller.rotate(yaw_err)
        return float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_path(self, path: Path, current_odom: PoseStamped) -> bool:
        if path is None or len(path.poses) < 2:
            logger.warning(f"PurePursuitPathFollowerTask '{self._name}': invalid path")
            return False
        self._path = path
        self._distancer = PathDistancer(path)
        self._current_odom = current_odom
        self._max_progress_idx = 0
        if self._ff is not None:
            self._ff.reset()

        first_yaw = path.poses[0].orientation.euler[2]
        robot_yaw = current_odom.orientation.euler[2]
        yaw_err = angle_diff(first_yaw, robot_yaw)
        self._state = (
            "initial_rotation"
            if abs(yaw_err) >= self._config.orientation_tolerance
            else "path_following"
        )

        logger.info(
            f"PurePursuitPathFollowerTask '{self._name}' started "
            f"({len(path.poses)} poses, initial state={self._state})"
        )
        return True

    def update_odom(self, odom: PoseStamped) -> None:
        self._current_odom = odom

    def cancel(self) -> bool:
        if not self.is_active():
            return False
        self._state = "aborted"
        return True

    def reset(self) -> bool:
        if self.is_active():
            return False
        self._state = "idle"
        self._path = None
        self._distancer = None
        self._current_odom = None
        self._max_progress_idx = 0
        return True

    def get_state(self) -> PurePursuitState:
        return self._state


__all__ = [
    "PurePursuitPathFollowerTask",
    "PurePursuitPathFollowerTaskConfig",
]

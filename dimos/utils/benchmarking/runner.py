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

"""Sim benchmark runner.

Boots a real :class:`ControlCoordinator` in-process with the
:class:`Go2SimTwistBaseAdapter` on the bottom edge, installs the controller
as a :class:`ControlTask`, drives it through a reference path while pumping
adapter odometry into the task each iteration, and records the trajectory.

In-process (not via :class:`ModuleCoordinator`) so the runner can call
``task.update_odom()`` and ``adapter.read_odometry()`` directly without
crossing a subprocess boundary. This loses the sandboxing benefit but
keeps the same ControlCoordinator + TickLoop + adapter pipeline that
runs on hardware.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TYPE_CHECKING, Protocol

from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.baseline_path_follower_task import (
    BaselinePathFollowerTask,
    BaselinePathFollowerTaskConfig,
)
from dimos.control.tasks.feedforward_gain_compensator import FeedforwardGainConfig
from dimos.control.tasks.mpc_path_follower_task import (
    MPCPathFollowerTask,
    MPCPathFollowerTaskConfig,
)
from dimos.control.tasks.path_follower_task import (
    PathFollowerTask,
    PathFollowerTaskConfig,
)
from dimos.control.tasks.pure_pursuit_path_follower_task import (
    PurePursuitPathFollowerTask,
    PurePursuitPathFollowerTaskConfig,
)
from dimos.control.tasks.reactive_path_follower_task import (
    ReactivePathFollowerTask,
    ReactivePathFollowerTaskConfig,
)
from dimos.control.tasks.velocity_tracking_pid import VelocityTrackingConfig
from dimos.core.global_config import global_config
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.benchmarking.scoring import ExecutedTrajectory, TrajectoryTick
from dimos.utils.benchmarking.sim_blueprint import GO2_TICK_RATE_HZ, _base_joints, _go2_sim_base

if TYPE_CHECKING:
    from dimos.hardware.drive_trains.go2_sim.adapter import Go2SimTwistBaseAdapter
    from dimos.msgs.nav_msgs.Path import Path


_ARRIVED_STATES = frozenset({"arrived", "completed"})
_FAILED_STATES = frozenset({"aborted"})


class _PathFollowerLike(Protocol):
    """Common contract every controller task must satisfy for the sim loop."""

    def start_path(self, path: Path, current_odom: PoseStamped) -> bool: ...
    def update_odom(self, odom: PoseStamped) -> None: ...
    def compute(self, state) -> object: ...  # CoordinatorState → JointCommandOutput|None
    def get_state(self) -> str: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _odom_to_pose(odom: list[float]) -> PoseStamped:
    return PoseStamped(
        position=Vector3(odom[0], odom[1], 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, odom[2])),
    )


def _vels_to_twist(v: list[float]) -> Twist:
    return Twist(linear=Vector3(v[0], v[1], 0.0), angular=Vector3(0.0, 0.0, v[2]))


# ---------------------------------------------------------------------------
# Generic sim run
# ---------------------------------------------------------------------------


def _run_path_follower_sim(
    task_factory: Callable[[], _PathFollowerLike],
    path: Path,
    timeout_s: float,
    sample_rate_hz: float,
) -> ExecutedTrajectory:
    """Generic sim loop: any task implementing the path-follower contract."""
    coord = ControlCoordinator(
        tick_rate=GO2_TICK_RATE_HZ,
        hardware=[_go2_sim_base()],
        tasks=[
            TaskConfig(
                name="vel_base",
                type="velocity",
                joint_names=_base_joints,
                priority=10,
            ),
        ],
    )
    task = task_factory()

    coord.start()
    try:
        adapter: Go2SimTwistBaseAdapter = coord._hardware["base"].adapter
        # Reset plant to path start so sim odom matches path[0].
        start = path.poses[0]
        adapter.set_initial_pose(start.position.x, start.position.y, start.orientation.euler[2])
        adapter.connect()

        coord.add_task(task)
        task.start_path(path, _odom_to_pose(adapter.read_odometry()))

        ticks: list[TrajectoryTick] = []
        period = 1.0 / sample_rate_hz
        t0 = time.perf_counter()
        next_sample = t0
        arrived = False

        while True:
            now = time.perf_counter()
            t_rel = now - t0
            if t_rel > timeout_s:
                break

            pose = _odom_to_pose(adapter.read_odometry())
            task.update_odom(pose)

            ticks.append(
                TrajectoryTick(
                    t=t_rel,
                    pose=pose,
                    cmd_twist=_vels_to_twist(adapter._cmd),
                    actual_twist=_vels_to_twist(adapter.read_velocities()),
                )
            )

            s = task.get_state()
            if s in _ARRIVED_STATES:
                arrived = True
                break
            if s in _FAILED_STATES:
                break

            next_sample += period
            sleep_for = next_sample - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        return ExecutedTrajectory(ticks=ticks, arrived=arrived)
    finally:
        coord.stop()


# ---------------------------------------------------------------------------
# Per-controller thin wrappers
# ---------------------------------------------------------------------------


def run_baseline_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    k_angular: float = 0.5,
    pid_config: VelocityTrackingConfig | None = None,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "baseline_follower",
) -> ExecutedTrajectory:
    """Production LocalPlanner P-controller in sim."""

    def _make() -> BaselinePathFollowerTask:
        return BaselinePathFollowerTask(
            name=task_name,
            config=BaselinePathFollowerTaskConfig(
                speed=speed,
                k_angular=k_angular,
                pid_config=pid_config,
                ff_config=ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_lyapunov_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    pid_config: VelocityTrackingConfig | None = None,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "lyapunov_follower",
) -> ExecutedTrajectory:
    """Lyapunov reactive controller in sim."""

    def _make() -> ReactivePathFollowerTask:
        return ReactivePathFollowerTask(
            name=task_name,
            config=ReactivePathFollowerTaskConfig(
                joint_names=list(_base_joints),
                pid_config=pid_config,
                ff_config=ff_config,
            ),
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_pure_pursuit_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "pure_pursuit_follower",
) -> ExecutedTrajectory:
    """Classic Pure Pursuit (constant lookahead, no PID) in sim."""

    def _make() -> PurePursuitPathFollowerTask:
        return PurePursuitPathFollowerTask(
            name=task_name,
            config=PurePursuitPathFollowerTaskConfig(
                joint_names=list(_base_joints),
                target_speed=speed,
                ff_config=ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_rpp_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "rpp_follower",
) -> ExecutedTrajectory:
    """Regulated Pure Pursuit (PathFollowerTask: PurePursuit + adaptive
    lookahead + curvature-aware velocity profiler + cross-track PID) in sim.
    """

    def _make() -> PathFollowerTask:
        return PathFollowerTask(
            name=task_name,
            config=PathFollowerTaskConfig(
                joint_names=list(_base_joints),
                max_linear_speed=speed,
                ff_config=ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_mpc_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    task_name: str = "mpc_follower",
) -> ExecutedTrajectory:
    """MPC stub — raises NotImplementedError on first compute()."""
    from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED

    def _make() -> MPCPathFollowerTask:
        return MPCPathFollowerTask(
            name=task_name,
            config=MPCPathFollowerTaskConfig(
                joint_names=list(_base_joints),
                target_speed=speed,
                plant=GO2_PLANT_FITTED,
            ),
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


__all__ = [
    "run_baseline_sim",
    "run_lyapunov_sim",
    "run_mpc_sim",
    "run_pure_pursuit_sim",
    "run_rpp_sim",
]

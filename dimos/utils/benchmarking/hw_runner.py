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

"""Hardware benchmark runner for the Go2.

Talks to the production coordinator over LCM:
  /cmd_vel    (out) — Twist commands sent at 10 Hz from the controller task
  /go2/odom   (in)  — onboard Go2 odometry, used as pose feedback

Operator workflow:
  Terminal A: ``dimos run unitree-go2-webrtc-keyboard-teleop``
              (coordinator + teleop; teleop stays silent until a key is held)
  Terminal B: ``python -m dimos.utils.benchmarking.run_battery_hw …``

Caveats:
  - Onboard Go2 odom drifts on long / curvy paths. Use short paths
    (straight, single corner, small circle) for trustworthy CTE numbers.
  - Pre-roll is manual: the operator parks the robot at the path start and
    presses Enter. There is no automated reset.
  - Go2 over WebRTC publishes pose only — no twist. We derive body-frame
    velocities by finite-differencing pose with EMA smoothing
    (``_PoseVelocityEstimator``), so optional inner loops have a real
    actual signal instead of zero.
  - Safety: SIGINT (Ctrl+C) → 3x zero Twist published, exit. Saturation
    clamp enforced on every command.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import signal
import threading
import time
from typing import Protocol

from dimos.control.task import (
    CoordinatorState,
    JointStateSnapshot,
)
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
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.benchmarking.scoring import ExecutedTrajectory, TrajectoryTick
from dimos.utils.trigonometry import angle_diff

# Hardware-relevant safety limits (from Rung 1 saturation envelope)
VX_MAX = 1.0  # m/s
WZ_MAX = 1.5  # rad/s
TICK_HZ = 10.0  # matches Go2 control rate

# Any of these task states means "we're done"
_ARRIVED_STATES = frozenset({"arrived", "completed"})
_FAILED_STATES = frozenset({"aborted"})


class _PathFollowerLike(Protocol):
    """Common contract every controller task must satisfy for the hw loop.

    Matches BaseControlTask + the side-channel methods that all current
    path-follower tasks share.
    """

    def start_path(self, path: Path, current_odom: PoseStamped) -> bool: ...
    def update_odom(self, odom: PoseStamped) -> None: ...
    def compute(self, state: CoordinatorState): ...  # returns JointCommandOutput | None
    def get_state(self) -> str: ...


@dataclass
class HwRunOptions:
    timeout_s: float = 30.0
    speed: float = 0.55
    k_angular: float = 0.5
    pid_config: VelocityTrackingConfig | None = None
    ff_config: FeedforwardGainConfig | None = None
    odom_warmup_s: float = 2.0
    cmd_topic: str = "/cmd_vel"
    odom_topic: str = "/go2/odom"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _twist_clamped(vx: float, vy: float, wz: float) -> Twist:
    return Twist(
        linear=Vector3(_clamp(vx, -VX_MAX, VX_MAX), 0.0, 0.0),  # vy ignored on Go2
        angular=Vector3(0.0, 0.0, _clamp(wz, -WZ_MAX, WZ_MAX)),
    )


class _PoseVelocityEstimator:
    """Differentiate consecutive PoseStamped to derive body-frame (vx, vy, wz).

    Go2 over WebRTC publishes pose only — no velocity. EMA smooths the raw
    differences (single-pole, alpha tuned for 10 Hz so it follows real motion
    but suppresses sample-noise jitter).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self._prev_pose: PoseStamped | None = None
        self._prev_t: float | None = None
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._alpha = alpha

    def update(self, pose: PoseStamped, t: float) -> tuple[float, float, float]:
        if self._prev_pose is None or self._prev_t is None:
            self._prev_pose = pose
            self._prev_t = t
            return 0.0, 0.0, 0.0
        dt = t - self._prev_t
        if dt <= 0:
            return self._vx, self._vy, self._wz

        dx = pose.position.x - self._prev_pose.position.x
        dy = pose.position.y - self._prev_pose.position.y
        dyaw = angle_diff(pose.orientation.euler[2], self._prev_pose.orientation.euler[2])

        wx = dx / dt
        wy = dy / dt
        yaw = pose.orientation.euler[2]
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        bx = wx * cos_y + wy * sin_y
        by = -wx * sin_y + wy * cos_y
        bw = dyaw / dt

        self._vx = self._alpha * bx + (1 - self._alpha) * self._vx
        self._vy = self._alpha * by + (1 - self._alpha) * self._vy
        self._wz = self._alpha * bw + (1 - self._alpha) * self._wz

        self._prev_pose = pose
        self._prev_t = t
        return self._vx, self._vy, self._wz


def _shift_path_to_start_at_pose(path: Path, start_pose: PoseStamped) -> Path:
    """Rigid-transform ``path`` so its first pose aligns with ``start_pose``.

    Reference paths are defined in a robot-centric frame (path[0] at origin
    facing +x). On hardware we need them in the Go2 odom frame, anchored
    to wherever the robot is now.
    """
    px0 = path.poses[0].position.x
    py0 = path.poses[0].position.y
    pyaw0 = path.poses[0].orientation.euler[2]
    sx = start_pose.position.x
    sy = start_pose.position.y
    syaw = start_pose.orientation.euler[2]

    dyaw = syaw - pyaw0
    cos_d, sin_d = math.cos(dyaw), math.sin(dyaw)

    new_poses = []
    for p in path.poses:
        rx = p.position.x - px0
        ry = p.position.y - py0
        nx = rx * cos_d - ry * sin_d
        ny = rx * sin_d + ry * cos_d
        new_poses.append(
            PoseStamped(
                position=Vector3(sx + nx, sy + ny, 0.0),
                orientation=Quaternion.from_euler(Vector3(0.0, 0.0, p.orientation.euler[2] + dyaw)),
            )
        )
    return Path(poses=new_poses)


# ---------------------------------------------------------------------------
# Generic hardware run
# ---------------------------------------------------------------------------


def _run_path_follower_hw(
    task_factory: Callable[[], _PathFollowerLike],
    path: Path,
    opts: HwRunOptions,
    *,
    interactive: bool,
    label: str,
) -> tuple[Path, ExecutedTrajectory]:
    """Generic hardware loop: any task implementing the path-follower contract.

    Handles odom warm-up, pre-roll, path anchoring, the 10 Hz tick loop,
    SIGINT safety, and zero-Twist on exit.
    """
    cmd_pub = LCMTransport(opts.cmd_topic, Twist)
    odom_sub = LCMTransport(opts.odom_topic, PoseStamped)

    latest_pose: list[PoseStamped | None] = [None]
    last_odom_t: list[float] = [0.0]
    odom_lock = threading.Lock()

    def _on_odom(msg: PoseStamped) -> None:
        with odom_lock:
            latest_pose[0] = msg
            last_odom_t[0] = time.perf_counter()

    odom_sub.subscribe(_on_odom)

    # Wait for odom warmup so we have a real pose before starting.
    print(f"[hw {label}] waiting up to {opts.odom_warmup_s:.1f}s for odom on {opts.odom_topic}...")
    deadline = time.perf_counter() + opts.odom_warmup_s
    while time.perf_counter() < deadline:
        with odom_lock:
            if latest_pose[0] is not None:
                break
        time.sleep(0.05)
    with odom_lock:
        if latest_pose[0] is None:
            cmd_pub.broadcast(None, _twist_clamped(0, 0, 0))
            raise RuntimeError(
                f"No odom received on {opts.odom_topic} within {opts.odom_warmup_s}s. "
                "Is `dimos run unitree-go2-webrtc-keyboard-teleop` running?"
            )
        start_pose = latest_pose[0]

    if interactive:
        first = path.poses[0]
        print(
            f"[hw {label}] PRE-ROLL: park robot at path start "
            f"({first.position.x:.2f}, {first.position.y:.2f}, "
            f"yaw={first.orientation.euler[2]:.2f}) then press Enter…"
        )
        input()
        with odom_lock:
            start_pose = latest_pose[0]

    task = task_factory()
    path_world = _shift_path_to_start_at_pose(path, start_pose)
    print(
        f"[hw {label}] anchored path: start=({path_world.poses[0].position.x:.2f},"
        f"{path_world.poses[0].position.y:.2f}) "
        f"goal=({path_world.poses[-1].position.x:.2f},"
        f"{path_world.poses[-1].position.y:.2f})"
    )
    task.start_path(path_world, start_pose)

    # SIGINT → clean stop
    stop_flag = {"stop": False}

    def _sigint_handler(_signum, _frame):  # type: ignore[no-untyped-def]
        stop_flag["stop"] = True
        print(f"\n[hw {label}] SIGINT — stopping")

    prev = signal.signal(signal.SIGINT, _sigint_handler)

    ticks: list[TrajectoryTick] = []
    arrived = False
    period = 1.0 / TICK_HZ
    t0 = time.perf_counter()
    next_tick = t0
    vel_est = _PoseVelocityEstimator()

    try:
        while True:
            now = time.perf_counter()
            t_rel = now - t0

            if stop_flag["stop"]:
                break
            if t_rel > opts.timeout_s:
                print(f"[hw {label}] timeout after {opts.timeout_s:.1f}s")
                break

            with odom_lock:
                pose = latest_pose[0]
                last_pose_age = now - last_odom_t[0]
            if pose is None or last_pose_age > 1.0:
                print(f"[hw {label}] ABORT: stale odom ({last_pose_age:.2f}s)")
                break

            task.update_odom(pose)
            est_vx, est_vy, est_wz = vel_est.update(pose, now)
            state = CoordinatorState(
                joints=JointStateSnapshot(
                    joint_velocities={
                        "base/vx": est_vx,
                        "base/vy": est_vy,
                        "base/wz": est_wz,
                    },
                    timestamp=now,
                ),
                t_now=now,
                dt=period,
            )
            cmd = task.compute(state)
            if cmd is not None and cmd.velocities is not None:
                vx, vy, wz = cmd.velocities[0], cmd.velocities[1], cmd.velocities[2]
            else:
                vx = vy = wz = 0.0

            twist = _twist_clamped(vx, vy, wz)
            cmd_pub.broadcast(None, twist)

            ticks.append(
                TrajectoryTick(
                    t=t_rel,
                    pose=pose,
                    cmd_twist=twist,
                    actual_twist=Twist(
                        linear=Vector3(est_vx, est_vy, 0.0),
                        angular=Vector3(0.0, 0.0, est_wz),
                    ),
                )
            )

            s = task.get_state()
            if s in _ARRIVED_STATES:
                arrived = True
                print(f"[hw {label}] arrived in {t_rel:.2f}s")
                break
            if s in _FAILED_STATES:
                print(f"[hw {label}] task aborted")
                break

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        for _ in range(3):
            cmd_pub.broadcast(None, _twist_clamped(0, 0, 0))
            time.sleep(0.05)
        signal.signal(signal.SIGINT, prev)

    return path_world, ExecutedTrajectory(ticks=ticks, arrived=arrived)


# ---------------------------------------------------------------------------
# Per-controller thin wrappers
# ---------------------------------------------------------------------------


def run_baseline_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Production LocalPlanner P-controller (BaselinePathFollowerTask) on hw."""
    opts = opts or HwRunOptions()

    def _make() -> BaselinePathFollowerTask:
        return BaselinePathFollowerTask(
            name="baseline_hw_follower",
            config=BaselinePathFollowerTaskConfig(
                speed=opts.speed,
                k_angular=opts.k_angular,
                pid_config=opts.pid_config,
                ff_config=opts.ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="baseline")


def run_lyapunov_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Lyapunov reactive controller (ReactivePathFollowerTask) on hw."""
    opts = opts or HwRunOptions()

    def _make() -> ReactivePathFollowerTask:
        return ReactivePathFollowerTask(
            name="lyapunov_hw_follower",
            config=ReactivePathFollowerTaskConfig(
                joint_names=["base/vx", "base/vy", "base/wz"],
                pid_config=opts.pid_config,
                ff_config=opts.ff_config,
            ),
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="lyapunov")


def run_pure_pursuit_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Classic Pure Pursuit (constant lookahead, no PID) on hw."""
    opts = opts or HwRunOptions()

    def _make() -> PurePursuitPathFollowerTask:
        return PurePursuitPathFollowerTask(
            name="pure_pursuit_hw_follower",
            config=PurePursuitPathFollowerTaskConfig(
                joint_names=["base/vx", "base/vy", "base/wz"],
                target_speed=opts.speed,
                ff_config=opts.ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="pp")


def run_rpp_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Regulated Pure Pursuit (PathFollowerTask: PurePursuit + adaptive
    lookahead + curvature-aware velocity profiler + cross-track PID) on hw.
    """
    opts = opts or HwRunOptions()

    def _make() -> PathFollowerTask:
        return PathFollowerTask(
            name="rpp_hw_follower",
            config=PathFollowerTaskConfig(
                joint_names=["base/vx", "base/vy", "base/wz"],
                max_linear_speed=opts.speed,
                ff_config=opts.ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="rpp")


def run_mpc_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """MPC stub on hardware — raises NotImplementedError on first compute()."""
    opts = opts or HwRunOptions()
    from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED

    def _make() -> MPCPathFollowerTask:
        return MPCPathFollowerTask(
            name="mpc_hw_follower",
            config=MPCPathFollowerTaskConfig(
                joint_names=["base/vx", "base/vy", "base/wz"],
                target_speed=opts.speed,
                plant=GO2_PLANT_FITTED,
            ),
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="mpc")


__all__ = [
    "HwRunOptions",
    "run_baseline_hw",
    "run_lyapunov_hw",
    "run_mpc_hw",
    "run_pure_pursuit_hw",
    "run_rpp_hw",
]

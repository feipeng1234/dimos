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
  /cmd_vel    (out) — Twist commands sent at 10 Hz from BaselinePathFollowerTask
  /go2/odom   (in)  — onboard Go2 odometry, used as pose feedback

Operator workflow:
  Terminal A: ``dimos run unitree-go2-coordinator``        (brings up WebRTC + adapter)
  Terminal B: ``python -m dimos.utils.benchmarking.run_battery_hw --path straight_5m``

Caveats:
  - Onboard Go2 odom drifts on long / curvy paths. Use short paths (straight,
    single corner, small circle) for trustworthy CTE numbers; treat figure-8
    / slalom / square as "did the robot finish without crashing" smoke tests.
  - Pre-roll is manual: the operator parks the robot at the path start and
    presses Enter. There is no automated reset.
  - Safety: SIGINT (Ctrl+C) → zero Twist published, exit. Saturation clamp
    enforced on every published command.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass

from dimos.control.task import (
    CoordinatorState,
    JointStateSnapshot,
)
from dimos.control.tasks.baseline_path_follower_task import (
    BaselinePathFollowerTask,
    BaselinePathFollowerTaskConfig,
)
from dimos.control.tasks.velocity_tracking_pid import VelocityTrackingConfig
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.benchmarking.scoring import ExecutedTrajectory, TrajectoryTick

# Hardware-relevant safety limits (from handoff doc)
VX_MAX = 1.0   # m/s
WZ_MAX = 1.5   # rad/s
TICK_HZ = 10.0  # matches Go2 control rate


@dataclass
class HwRunOptions:
    timeout_s: float = 30.0
    speed: float = 0.55
    k_angular: float = 0.5
    pid_config: VelocityTrackingConfig | None = None
    odom_warmup_s: float = 2.0   # require odom messages before launching
    cmd_topic: str = "/cmd_vel"
    odom_topic: str = "/go2/odom"


class _SafetyStop(Exception):
    pass


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _twist_clamped(vx: float, vy: float, wz: float) -> Twist:
    return Twist(
        linear=Vector3(_clamp(vx, -VX_MAX, VX_MAX), 0.0, 0.0),  # vy ignored on Go2
        angular=Vector3(0.0, 0.0, _clamp(wz, -WZ_MAX, WZ_MAX)),
    )


def run_baseline_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> ExecutedTrajectory:
    """Drive the baseline path-follower through ``path`` on real Go2 hardware.

    Returns an :class:`ExecutedTrajectory` ready to score.
    """
    opts = opts or HwRunOptions()

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
    print(f"[hw] waiting up to {opts.odom_warmup_s:.1f}s for odom on {opts.odom_topic}...")
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
                "Is `dimos run unitree-go2-coordinator` running?"
            )
        start_pose = latest_pose[0]
    print(
        f"[hw] odom OK. current pose = "
        f"({start_pose.position.x:.2f}, {start_pose.position.y:.2f}, "
        f"yaw={start_pose.orientation.euler[2]:.2f})"
    )

    # Pre-roll: ask operator to position the robot at the path start.
    if interactive:
        first = path.poses[0]
        print(
            f"[hw] PRE-ROLL: park robot at path start "
            f"({first.position.x:.2f}, {first.position.y:.2f}, "
            f"yaw={first.orientation.euler[2]:.2f}) in the map frame, then press Enter…"
        )
        input()
        with odom_lock:
            start_pose = latest_pose[0]
        print(
            f"[hw] starting at "
            f"({start_pose.position.x:.2f}, {start_pose.position.y:.2f}, "
            f"yaw={start_pose.orientation.euler[2]:.2f})"
        )

    task = BaselinePathFollowerTask(
        name="baseline_hw_follower",
        config=BaselinePathFollowerTaskConfig(
            speed=opts.speed,
            k_angular=opts.k_angular,
            pid_config=opts.pid_config,
        ),
        global_config=global_config,
    )
    task.start_path(path, start_pose)

    # Install Ctrl+C handler that triggers a clean stop.
    stop_flag = {"stop": False}

    def _sigint_handler(_signum, _frame):  # type: ignore[no-untyped-def]
        stop_flag["stop"] = True
        print("\n[hw] SIGINT — stopping")

    prev = signal.signal(signal.SIGINT, _sigint_handler)

    ticks: list[TrajectoryTick] = []
    arrived = False
    period = 1.0 / TICK_HZ
    t0 = time.perf_counter()
    next_tick = t0
    last_cmd = (0.0, 0.0, 0.0)

    try:
        while True:
            now = time.perf_counter()
            t_rel = now - t0

            if stop_flag["stop"]:
                break
            if t_rel > opts.timeout_s:
                print(f"[hw] timeout after {opts.timeout_s:.1f}s")
                break

            with odom_lock:
                pose = latest_pose[0]
                last_pose_age = now - last_odom_t[0]
            if pose is None or last_pose_age > 1.0:
                # Odom dropped — safety stop, abort.
                print(f"[hw] ABORT: stale odom ({last_pose_age:.2f}s)")
                break

            task.update_odom(pose)
            # CoordinatorState only used for t_now; joint state empty since we
            # don't have CoordinatorState publishing pose into joint_velocities
            # in this in-process setup.
            state = CoordinatorState(joints=JointStateSnapshot(timestamp=now), t_now=now, dt=period)
            cmd = task.compute(state)
            if cmd is not None and cmd.velocities is not None:
                vx, vy, wz = cmd.velocities[0], cmd.velocities[1], cmd.velocities[2]
            else:
                vx = vy = wz = 0.0

            twist = _twist_clamped(vx, vy, wz)
            cmd_pub.broadcast(None, twist)
            last_cmd = (
                float(twist.linear.x),
                float(twist.linear.y),
                float(twist.angular.z),
            )

            ticks.append(
                TrajectoryTick(
                    t=t_rel,
                    pose=pose,
                    cmd_twist=twist,
                    actual_twist=Twist(),  # not measured — Go2 odom doesn't expose body twist here
                )
            )

            s = task.get_state()
            if s == "arrived":
                arrived = True
                print(f"[hw] arrived in {t_rel:.2f}s")
                break
            if s == "aborted":
                print("[hw] task aborted")
                break

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        # Always publish zero twist on exit (safety).
        for _ in range(3):
            cmd_pub.broadcast(None, _twist_clamped(0, 0, 0))
            time.sleep(0.05)
        signal.signal(signal.SIGINT, prev)

    return ExecutedTrajectory(ticks=ticks, arrived=arrived)


__all__ = ["HwRunOptions", "run_baseline_hw"]

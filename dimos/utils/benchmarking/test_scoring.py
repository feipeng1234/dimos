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

"""Synthetic-trajectory tests for the scoring library."""

from __future__ import annotations

import math

import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.benchmarking.scoring import (
    ExecutedTrajectory,
    TrajectoryTick,
    score_run,
)


def _pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def _straight_path(length: float = 5.0, step: float = 0.1) -> Path:
    n = int(length / step)
    poses = [_pose(i * step, 0.0, 0.0) for i in range(n + 1)]
    return Path(poses=poses)


def _zero_twist() -> Twist:
    return Twist()


def _const_twist(vx: float = 0.5, wz: float = 0.0) -> Twist:
    return Twist(linear=Vector3(vx, 0.0, 0.0), angular=Vector3(0.0, 0.0, wz))


def test_perfect_tracking_zero_error() -> None:
    """Executed pose exactly on the path → CTE and heading error are zero."""
    path = _straight_path(length=5.0, step=0.1)
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, 0.0, 0.0),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(50)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.cte_rms == pytest.approx(0.0, abs=1e-9)
    assert result.cte_max == pytest.approx(0.0, abs=1e-9)
    assert result.heading_err_rms == pytest.approx(0.0, abs=1e-9)
    assert result.heading_err_max == pytest.approx(0.0, abs=1e-9)
    assert result.arrived is True
    assert result.n_ticks == 50


def test_constant_lateral_offset() -> None:
    """Executed pose offset 0.1 m perpendicular to path → CTE = 0.1."""
    path = _straight_path(length=5.0, step=0.1)
    offset = 0.1
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, offset, 0.0),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(50)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.cte_rms == pytest.approx(offset, abs=1e-6)
    assert result.cte_max == pytest.approx(offset, abs=1e-6)
    assert result.heading_err_rms == pytest.approx(0.0, abs=1e-9)


def test_constant_heading_offset() -> None:
    """Executed pose on path but yawed 0.2 rad off → heading_err = 0.2."""
    path = _straight_path(length=5.0, step=0.1)
    yaw_off = 0.2
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, 0.0, yaw_off),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(50)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.heading_err_rms == pytest.approx(yaw_off, abs=1e-6)
    assert result.heading_err_max == pytest.approx(yaw_off, abs=1e-6)
    assert result.cte_rms == pytest.approx(0.0, abs=1e-9)


def test_command_metrics() -> None:
    """RMS speeds and time-to-complete reflect commanded values."""
    path = _straight_path(length=5.0, step=0.1)
    vx, wz = 0.4, 0.3
    n = 50
    dt = 0.1
    ticks = [
        TrajectoryTick(
            t=i * dt,
            pose=_pose(i * 0.1, 0.0, 0.0),
            cmd_twist=_const_twist(vx, wz),
            actual_twist=_const_twist(vx, wz),
        )
        for i in range(n)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.linear_speed_rms == pytest.approx(vx, abs=1e-9)
    assert result.angular_speed_rms == pytest.approx(wz, abs=1e-9)
    assert result.time_to_complete == pytest.approx((n - 1) * dt, abs=1e-9)
    # constant cmd → cmd_rate_integral = 0
    assert result.cmd_rate_integral == pytest.approx(0.0, abs=1e-9)


def test_cmd_rate_integral_picks_up_jumps() -> None:
    path = _straight_path(length=5.0, step=0.1)
    # Alternating linear-x command at 0.0 and 0.5 each tick → jump magnitude
    # is 0.5 between every adjacent pair; for 5 ticks we get 4 jumps × 0.5 = 2.0.
    vxs = [0.0, 0.5, 0.0, 0.5, 0.0]
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, 0.0, 0.0),
            cmd_twist=_const_twist(vx, 0.0),
            actual_twist=_const_twist(vx, 0.0),
        )
        for i, vx in enumerate(vxs)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))
    assert result.cmd_rate_integral == pytest.approx(2.0, abs=1e-9)


def test_empty_trajectory_returns_zeros() -> None:
    path = _straight_path()
    result = score_run(path, ExecutedTrajectory(ticks=[], arrived=False))
    assert result.n_ticks == 0
    assert result.cte_rms == 0.0
    assert result.arrived is False


def test_corner_path_segment_choice() -> None:
    """L-shaped path: a pose right at the corner is on both legs; pick whichever."""
    poses = [_pose(0.0, 0.0), _pose(1.0, 0.0), _pose(1.0, 1.0)]
    path = Path(poses=poses)
    ticks = [
        TrajectoryTick(
            t=0.0,
            pose=_pose(1.0, 0.0, 0.0),  # exactly on corner
            cmd_twist=_zero_twist(),
            actual_twist=_zero_twist(),
        ),
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=False))
    assert result.cte_rms == pytest.approx(0.0, abs=1e-9)


def test_off_axis_perpendicular_to_corner() -> None:
    """Pose 0.3 m above the L-corner: nearest distance is to corner point."""
    poses = [_pose(0.0, 0.0), _pose(1.0, 0.0), _pose(1.0, 1.0)]
    path = Path(poses=poses)
    ticks = [
        TrajectoryTick(
            t=0.0,
            pose=_pose(1.3, 0.0, 0.0),  # past the corner along leg-1's extension
            cmd_twist=_zero_twist(),
            actual_twist=_zero_twist(),
        ),
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=False))
    # nearest point on either segment is the corner (1.0, 0.0); distance = 0.3
    assert result.cte_rms == pytest.approx(0.3, abs=1e-6)

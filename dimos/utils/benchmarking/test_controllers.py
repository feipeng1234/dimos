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

"""Unit + integration tests for the controller suite."""

from __future__ import annotations

import pytest

from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.control.tasks.feedforward_gain_compensator import (
    FeedforwardGainCompensator,
    FeedforwardGainConfig,
)
from dimos.control.tasks.mpc_path_follower_task import (
    MPCPathFollowerTask,
    MPCPathFollowerTaskConfig,
)
from dimos.control.tasks.pure_pursuit_path_follower_task import (
    PurePursuitPathFollowerTask,
    PurePursuitPathFollowerTaskConfig,
)
from dimos.core.global_config import global_config
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.benchmarking.paths import circle, straight_line
from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED
from dimos.utils.benchmarking.runner import (
    run_baseline_sim,
    run_lyapunov_sim,
    run_pure_pursuit_sim,
    run_rpp_sim,
)
from dimos.utils.benchmarking.scoring import score_run


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def _state(joint_velocities: dict[str, float] | None = None, t_now: float = 0.0):
    return CoordinatorState(
        joints=JointStateSnapshot(joint_velocities=joint_velocities or {}, timestamp=t_now),
        t_now=t_now,
        dt=0.1,
    )


# ---------------------------------------------------------------------------
# FeedforwardGainCompensator
# ---------------------------------------------------------------------------


def test_ff_unity_gain_passthrough() -> None:
    ff = FeedforwardGainCompensator(FeedforwardGainConfig())  # K=1 default
    assert ff.compute(0.5, 0.0, 0.5) == (0.5, 0.0, 0.5)


def test_ff_divides_by_K() -> None:
    ff = FeedforwardGainCompensator(FeedforwardGainConfig(K_vx=2.0, K_wz=4.0))
    out = ff.compute(0.5, 0.0, 0.8)
    assert out[0] == pytest.approx(0.25)
    assert out[2] == pytest.approx(0.2)


def test_ff_clamps_to_output_limits() -> None:
    ff = FeedforwardGainCompensator(
        FeedforwardGainConfig(
            K_vx=0.5,
            K_wz=0.5,
            output_min_vx=-1.0,
            output_max_vx=1.0,
            output_min_wz=-1.5,
            output_max_wz=1.5,
        )
    )
    # 5.0 / 0.5 = 10.0 → clamped to 1.0
    out = ff.compute(5.0, 0.0, 5.0)
    assert out[0] == 1.0
    assert out[2] == 1.5


def test_ff_reset_is_noop() -> None:
    ff = FeedforwardGainCompensator()
    ff.reset()  # stateless — should not error


# ---------------------------------------------------------------------------
# PurePursuitPathFollowerTask
# ---------------------------------------------------------------------------


def test_pure_pursuit_constructs() -> None:
    t = PurePursuitPathFollowerTask("pp", PurePursuitPathFollowerTaskConfig(), global_config)
    assert t.get_state() == "idle"
    assert "base/wz" in t.claim().joints


def test_pure_pursuit_straight_yields_zero_wz() -> None:
    """On a straight path, robot pointed +x: cmd_wz should be ~0 in steady state."""
    t = PurePursuitPathFollowerTask("pp", PurePursuitPathFollowerTaskConfig(), global_config)
    path = straight_line(length=5.0)
    t.start_path(path, _pose(0.5, 0.0, 0.0))  # already aligned, mid-path
    out = t.compute(_state())
    assert out is not None
    assert out.velocities is not None
    assert abs(out.velocities[2]) < 0.05, f"expected ~0 wz on straight, got {out.velocities[2]}"


def test_pure_pursuit_with_ff_halves_wz() -> None:
    """Same controller, FF with K_wz=2 should halve commanded wz."""
    cfg_ff = PurePursuitPathFollowerTaskConfig(
        ff_config=FeedforwardGainConfig(K_vx=1.0, K_wz=2.0),
    )
    t_no_ff = PurePursuitPathFollowerTask("pp", PurePursuitPathFollowerTaskConfig(), global_config)
    t_ff = PurePursuitPathFollowerTask("pp_ff", cfg_ff, global_config)

    # Off-path pose so wz is non-zero.
    odom = _pose(0.0, 0.3, 0.0)
    path = straight_line(length=5.0)
    t_no_ff.start_path(path, odom)
    t_ff.start_path(path, odom)
    out_a = t_no_ff.compute(_state())
    out_b = t_ff.compute(_state())
    assert out_a is not None and out_a.velocities is not None
    assert out_b is not None and out_b.velocities is not None
    # vx unchanged (K_vx=1), wz halved
    assert out_b.velocities[0] == pytest.approx(out_a.velocities[0], rel=0.01)
    assert out_b.velocities[2] == pytest.approx(out_a.velocities[2] / 2.0, rel=0.05)


# ---------------------------------------------------------------------------
# MPCPathFollowerTask (stub)
# ---------------------------------------------------------------------------


def test_mpc_stub_raises_not_implemented() -> None:
    cfg = MPCPathFollowerTaskConfig(plant=GO2_PLANT_FITTED)
    t = MPCPathFollowerTask("mpc", cfg)
    t.start_path(straight_line(), _pose(0.0, 0.0, 0.0))
    with pytest.raises(NotImplementedError):
        t.compute(_state())


def test_mpc_requires_plant() -> None:
    t = MPCPathFollowerTask("mpc", MPCPathFollowerTaskConfig(plant=None))
    with pytest.raises(ValueError, match="plant must be set"):
        t.start_path(straight_line(), _pose(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# End-to-end sim integration (slow — one path each)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,runner",
    [
        ("baseline", run_baseline_sim),
        ("pure_pursuit", run_pure_pursuit_sim),
        ("rpp", run_rpp_sim),
        ("lyapunov", run_lyapunov_sim),
    ],
)
def test_each_controller_arrives_on_straight_5m(name: str, runner) -> None:
    path = straight_line(length=5.0)
    traj = runner(path, timeout_s=20)
    score = score_run(path, traj)
    assert score.arrived, f"{name} did not arrive on straight_5m"
    assert score.cte_rms < 0.10, f"{name} CTE too high: {score.cte_rms * 100:.1f} cm"


def test_rpp_handles_closed_path_without_false_arrival() -> None:
    """Regression for the closed-path bug: RPP on circle_R1.0 must not
    trip arrival on tick 1 just because path[0] == path[-1]."""
    path = circle(radius=1.0)
    traj = run_rpp_sim(path, timeout_s=30)
    score = score_run(path, traj)
    # Must have actually traversed the circle, not arrived in 0.1s.
    assert score.time_to_complete > 5.0, (
        f"RPP arrived too fast ({score.time_to_complete:.2f}s) — closed-path arrival gate is broken"
    )
    assert score.n_ticks > 50

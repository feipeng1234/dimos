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

"""Hardware benchmark: drive the baseline (or any cohort) on a real Go2.

Usage:
    Terminal A: dimos run unitree-go2-coordinator
    Terminal B: python -m dimos.utils.benchmarking.run_battery_hw \\
                  --path straight_5m \\
                  --k_angular 1.0 \\
                  [--pi]                # adds Session-3 wz inner loop
                  [--timeout 30]
                  [--out /tmp/hw_run]

Drop ``--path`` to run the full battery (operator gets prompted between
each path to re-park the robot).

Onboard odom drifts on long/curvy paths. Recommended hardware battery:
  straight_5m, single_corner, circle_R0.5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from dimos.control.tasks.feedforward_gain_compensator import FeedforwardGainConfig
from dimos.control.tasks.velocity_tracking_pid import (
    VelocityPIDConfig,
    VelocityTrackingConfig,
)
from dimos.utils.benchmarking.hw_runner import (
    HwRunOptions,
    run_baseline_hw,
    run_lyapunov_hw,
    run_mpc_hw,
    run_pure_pursuit_hw,
    run_rpp_hw,
)
from dimos.utils.benchmarking.paths import (
    default_battery,
    trajectory_to_svg,
)
from dimos.utils.benchmarking.scoring import score_run

SESSION3_WZ_PID = VelocityPIDConfig(
    kp=0.346,
    ki=1.343,
    kd=0.0,
    max_integral=0.5,
    output_min=-1.5,
    output_max=1.5,
)
PASSTHROUGH_PID = VelocityPIDConfig(kp=0.0, ki=0.0)
SESSION3_VELOCITY_TRACKING = VelocityTrackingConfig(
    vx=PASSTHROUGH_PID,
    vy=PASSTHROUGH_PID,
    wz=SESSION3_WZ_PID,
    dt=0.1,
)

# Strategy B: static plant-gain feedforward. Numbers from
# dimos.utils.benchmarking.plant_models (Session 3 fitted Go2 K values).
GO2_FEEDFORWARD = FeedforwardGainConfig(
    K_vx=1.008,
    K_vy=1.008,  # Go2 vy ≈ 0 anyway, K placeholder
    K_wz=2.175,
)


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path name from default_battery() to run; omit to run all paths.",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/benchmarking_hw"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--k_angular", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=0.55)
    parser.add_argument(
        "--pi", action="store_true", help="Enable Session-3 wz PI inner loop (Strategy D)"
    )
    parser.add_argument(
        "--ff",
        action="store_true",
        help="Enable static feedforward plant-gain compensation (Strategy B)",
    )
    parser.add_argument(
        "--controller",
        choices=["baseline", "lyapunov", "pure_pursuit", "rpp", "mpc"],
        default="baseline",
        help="Which controller to run",
    )
    args = parser.parse_args()
    if args.pi and args.ff:
        raise SystemExit("--pi and --ff are mutually exclusive")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sha = _git_short_sha()

    battery = default_battery()
    if args.path is not None:
        if args.path not in battery:
            raise SystemExit(f"Unknown path {args.path!r}; choices: {sorted(battery)}")
        battery = {args.path: battery[args.path]}

    opts = HwRunOptions(
        timeout_s=args.timeout,
        speed=args.speed,
        k_angular=args.k_angular,
        pid_config=SESSION3_VELOCITY_TRACKING if args.pi else None,
        ff_config=GO2_FEEDFORWARD if args.ff else None,
    )

    cohort_label = (
        f"hw_{args.controller}"
        f"{'_k' + str(args.k_angular) if args.controller == 'baseline' else ''}"
        f"{'_pi' if args.pi else ''}"
        f"{'_ff' if args.ff else ''}"
    )
    runner_fn = {
        "baseline": run_baseline_hw,
        "lyapunov": run_lyapunov_hw,
        "pure_pursuit": run_pure_pursuit_hw,
        "rpp": run_rpp_hw,
        "mpc": run_mpc_hw,
    }[args.controller]
    cohort_dir = out / cohort_label
    cohort_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, path in battery.items():
        print(f"\n{'#' * 60}\n# {cohort_label} :: {name} ({len(path.poses)} poses)\n{'#' * 60}")
        path_used, traj = runner_fn(path, opts)
        score = score_run(path_used, traj)
        last = traj.ticks[-1] if traj.ticks else None
        tail = f"  arrived={score.arrived}  ticks={score.n_ticks}  t={score.time_to_complete:.2f}s"
        if last is not None:
            tail += f"  last=({last.pose.position.x:.2f},{last.pose.position.y:.2f})"
        print(tail)
        print(
            f"  CTE rms={score.cte_rms * 100:.1f}cm max={score.cte_max * 100:.1f}cm  "
            f"head_err rms={score.heading_err_rms:.3f}rad  "
            f"v_lin_rms={score.linear_speed_rms:.2f}m/s"
        )
        results[name] = asdict(score)
        executed_xy = [(t.pose.position.x, t.pose.position.y) for t in traj.ticks]
        (cohort_dir / f"{name}.svg").write_text(trajectory_to_svg(path_used, executed_xy))

        # Raw per-tick dump for offline alpha-tuning / cmd-vs-actual analysis.
        ticks_dump = [
            {
                "t": tk.t,
                "x": tk.pose.position.x,
                "y": tk.pose.position.y,
                "yaw": tk.pose.orientation.euler[2],
                "cmd_vx": tk.cmd_twist.linear.x,
                "cmd_wz": tk.cmd_twist.angular.z,
                "est_vx": tk.actual_twist.linear.x,
                "est_vy": tk.actual_twist.linear.y,
                "est_wz": tk.actual_twist.angular.z,
            }
            for tk in traj.ticks
        ]
        (cohort_dir / f"{name}_ticks.json").write_text(json.dumps(ticks_dump))

    summary = out / f"hw_baseline_{sha}_{cohort_label}.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary}")
    print(f"Per-path SVGs in {cohort_dir}/")


if __name__ == "__main__":
    main()

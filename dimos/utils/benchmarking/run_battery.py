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

"""Run all path-follower cohorts against the full path battery in sim.

Cohorts:
  baseline_k0.5            : production PController (k_angular=0.5), no PI
  baseline_k1.0_tuned      : same outer loop, k_angular=1.0 (sweep optimum)
  baseline_k0.5_pi         : default outer + Session-3 wz PI inner loop
  lyapunov                 : Lyapunov reactive controller, no PI
  lyapunov_pi              : Lyapunov + Session-3 wz PI inner loop

Writes per-cohort scored JSON + per-cohort/per-path trajectory SVGs.

Usage:
    python -m dimos.utils.benchmarking.run_battery [--out PATH] [--timeout S]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from dimos.control.tasks.feedforward_gain_compensator import FeedforwardGainConfig
from dimos.control.tasks.velocity_tracking_pid import (
    VelocityPIDConfig,
    VelocityTrackingConfig,
)
from dimos.utils.benchmarking.paths import (
    default_battery,
    multi_trajectory_to_svg,
    trajectory_to_svg,
)
from dimos.utils.benchmarking.runner import (
    run_baseline_sim,
    run_lyapunov_sim,
    run_pure_pursuit_sim,
    run_rpp_sim,
)
from dimos.utils.benchmarking.scoring import ExecutedTrajectory, score_run

# Session 3 wz PI gains, from
# ~/char_runs/session_20260425-131525/modeling/tuning/tuning_summary.json
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
GO2_FF = FeedforwardGainConfig(K_vx=1.008, K_vy=1.008, K_wz=2.175)


COHORTS: dict[str, Callable[[Path, float], ExecutedTrajectory]] = {
    # Baseline P-controller cohorts (production LocalPlanner algorithm)
    "baseline_k0.5": lambda p, t: run_baseline_sim(p, timeout_s=t, k_angular=0.5),
    "baseline_k1.0_tuned": lambda p, t: run_baseline_sim(p, timeout_s=t, k_angular=1.0),
    "baseline_k0.5_pi": lambda p, t: run_baseline_sim(
        p, timeout_s=t, k_angular=0.5, pid_config=SESSION3_VELOCITY_TRACKING
    ),
    "baseline_k0.5_ff": lambda p, t: run_baseline_sim(
        p, timeout_s=t, k_angular=0.5, ff_config=GO2_FF
    ),
    # Classic Pure Pursuit
    "pure_pursuit": lambda p, t: run_pure_pursuit_sim(p, timeout_s=t),
    "pure_pursuit_ff": lambda p, t: run_pure_pursuit_sim(p, timeout_s=t, ff_config=GO2_FF),
    # Regulated Pure Pursuit (existing PathFollowerTask: PP + adaptive lookahead +
    # curvature speed reg + cross-track PID)
    "rpp": lambda p, t: run_rpp_sim(p, timeout_s=t),
    "rpp_ff": lambda p, t: run_rpp_sim(p, timeout_s=t, ff_config=GO2_FF),
    # Lyapunov reactive
    "lyapunov": lambda p, t: run_lyapunov_sim(p, timeout_s=t),
    "lyapunov_pi": lambda p, t: run_lyapunov_sim(
        p, timeout_s=t, pid_config=SESSION3_VELOCITY_TRACKING
    ),
    # MPC stub (raises NotImplementedError; included to reserve the cohort slot)
    # "mpc":                 lambda p, t: run_mpc_sim(p, timeout_s=t),
}


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/benchmarking_baseline"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sha = _git_short_sha()

    battery = default_battery()
    all_results: dict[str, dict[str, dict]] = {}
    # Per-path: cohort_name -> executed_xy
    per_path_executed: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: {} for name in battery
    }

    for cohort_name, runner_fn in COHORTS.items():
        print(f"\n{'#' * 60}\n# Cohort: {cohort_name}\n{'#' * 60}")
        cohort_dir = out / cohort_name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        cohort_results: dict[str, dict] = {}

        for name, path in battery.items():
            print(f"\n=== {name} ({len(path.poses)} poses) ===")
            traj = runner_fn(path, args.timeout)
            score = score_run(path, traj)
            last = traj.ticks[-1] if traj.ticks else None
            tail = (
                f"  arrived={score.arrived}  ticks={score.n_ticks}  t={score.time_to_complete:.2f}s"
            )
            if last is not None:
                tail += f"  last=({last.pose.position.x:.2f},{last.pose.position.y:.2f})"
            print(tail)
            print(
                f"  CTE rms={score.cte_rms * 100:.1f}cm max={score.cte_max * 100:.1f}cm  "
                f"head_err rms={score.heading_err_rms:.3f}rad  "
                f"v_lin_rms={score.linear_speed_rms:.2f}m/s  "
                f"smoothness(Σ|Δcmd|)={score.cmd_rate_integral:.2f}"
            )
            cohort_results[name] = asdict(score)
            executed_xy = [(t.pose.position.x, t.pose.position.y) for t in traj.ticks]
            (cohort_dir / f"{name}.svg").write_text(trajectory_to_svg(path, executed_xy))
            per_path_executed[name][cohort_name] = executed_xy

        all_results[cohort_name] = cohort_results

    # Composite: one SVG per path with all cohorts overlaid.
    composite_dir = out / "composite"
    composite_dir.mkdir(parents=True, exist_ok=True)
    for path_name, path in battery.items():
        [f"{c}: {all_results[c][path_name]['cte_rms'] * 100:.1f}cm rms" for c in COHORTS]
        svg = multi_trajectory_to_svg(
            path,
            per_path_executed[path_name],
            size_px=600,
            title=path_name,
        )
        (composite_dir / f"{path_name}.svg").write_text(svg)

    # Single index.html aggregating all composites in a grid.
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Sim baseline {sha}</title>",
        "<style>body{font-family:monospace;margin:20px;background:#fafafa}"
        "h1{margin:8px 0}h2{margin:24px 0 8px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px}"
        ".card{background:white;border:1px solid #ddd;padding:8px}"
        "table{border-collapse:collapse;font-size:12px}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "</style></head><body>",
        f"<h1>Sim baseline (commit {sha})</h1>",
        "<h2>Head-to-head: cte_rms (cm)</h2>",
        "<table><thead><tr><th>path</th>"
        + "".join(f"<th>{c}</th>" for c in COHORTS)
        + "</tr></thead><tbody>",
    ]
    for path_name in battery:
        html_parts.append(f"<tr><td>{path_name}</td>")
        for c in COHORTS:
            r = all_results[c][path_name]
            html_parts.append(f"<td>{r['cte_rms'] * 100:.1f} / {r['time_to_complete']:.1f}s</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    html_parts.append("<h2>Trajectories (per path, all cohorts)</h2>")
    html_parts.append("<div class='grid'>")
    for path_name in battery:
        html_parts.append(
            f"<div class='card'><img src='composite/{path_name}.svg' "
            f"width='600' alt='{path_name}'/></div>"
        )
    html_parts.append("</div></body></html>")

    (out / "index.html").write_text("\n".join(html_parts))

    summary_path = out / f"sim_baseline_{sha}.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {summary_path}")
    print(f"Wrote composite SVGs to {out}/composite/")
    print(f"Open in browser: file://{out.resolve()}/index.html")

    # Comparison table
    print(f"\n{'=' * 90}\nHead-to-head: cte_rms (cm)\n{'=' * 90}")
    cohort_names = list(COHORTS.keys())
    print(f"{'path':<22}  " + "  ".join(f"{c:>20}" for c in cohort_names))
    for path_name in battery:
        row = [path_name]
        for c in cohort_names:
            r = all_results[c][path_name]
            row.append(f"{r['cte_rms'] * 100:5.1f}cm/{r['time_to_complete']:5.1f}s")
        print(f"{row[0]:<22}  " + "  ".join(f"{r:>20}" for r in row[1:]))


if __name__ == "__main__":
    main()

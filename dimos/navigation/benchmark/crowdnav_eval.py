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

"""
CrowdNav-based planner evaluation for DimOS navigation modules.

Runs a planner module through dynamic pedestrian scenarios (powered by CrowdNav's
ORCA simulation) and collects practical metrics: collision rate, replan churn,
cmd_vel smoothness, time-to-goal, and more.

Usage:
    from dimos.navigation.benchmark import evaluate_planner
    from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner

    results = evaluate_planner(ReplanningAStarPlanner.blueprint())
    print(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

from dimos_lcm.std_msgs import Bool, String
import numpy as np

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Pedestrian:
    """A simulated pedestrian with ORCA-style linear motion."""

    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.3
    goal_x: float = 0.0
    goal_y: float = 0.0
    v_pref: float = 1.0

    def step(self, dt: float) -> None:
        """Move toward goal at preferred speed, stop when close."""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            self.vx = 0.0
            self.vy = 0.0
            return
        scale = min(self.v_pref, dist / dt) / dist
        self.vx = dx * scale
        self.vy = dy * scale
        self.x += self.vx * dt
        self.y += self.vy * dt


@dataclass
class Scenario:
    """A single benchmark scenario."""

    name: str
    # World bounds (meters, centered at origin)
    world_size: float = 20.0
    # Grid resolution (meters/cell)
    resolution: float = 0.1
    # Robot start/goal
    robot_start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw
    robot_goal: tuple[float, float] = (8.0, 0.0)
    # Pedestrians
    pedestrians: list[Pedestrian] = field(default_factory=list)
    # Static obstacles: list of (x, y, radius) circles
    static_obstacles: list[tuple[float, float, float]] = field(default_factory=list)
    # Time limit for this scenario
    time_limit: float = 60.0
    # Goal tolerance
    goal_tolerance: float = 0.5


def _default_scenarios() -> list[Scenario]:
    """Generate the default benchmark scenario suite."""
    scenarios: list[Scenario] = []

    # --- 1. Open field (sanity check) ---
    scenarios.append(
        Scenario(
            name="open_field",
            robot_start=(0.0, 0.0, 0.0),
            robot_goal=(8.0, 0.0),
            time_limit=30.0,
        )
    )

    # --- 2. Static obstacle wall ---
    wall_obs = [(4.0, y, 0.3) for y in np.arange(-3.0, 0.0, 0.7)]
    scenarios.append(
        Scenario(
            name="static_wall_gap",
            robot_start=(0.0, 0.0, 0.0),
            robot_goal=(8.0, 0.0),
            static_obstacles=wall_obs,
            time_limit=30.0,
        )
    )

    # --- 3. Circle crossing (CrowdNav classic) ---
    n_humans = 5
    circle_r = 4.0
    circle_peds = []
    for i in range(n_humans):
        angle = 2 * math.pi * i / n_humans
        px = circle_r * math.cos(angle)
        py = circle_r * math.sin(angle)
        circle_peds.append(
            Pedestrian(
                x=px,
                y=py,
                vx=0.0,
                vy=0.0,
                radius=0.3,
                goal_x=-px,
                goal_y=-py,
                v_pref=1.0,
            )
        )
    scenarios.append(
        Scenario(
            name="circle_crossing_5",
            robot_start=(-6.0, 0.0, 0.0),
            robot_goal=(6.0, 0.0),
            pedestrians=circle_peds,
            time_limit=40.0,
        )
    )

    # --- 4. Circle crossing (dense) ---
    n_humans_dense = 10
    dense_peds = []
    for i in range(n_humans_dense):
        angle = 2 * math.pi * i / n_humans_dense
        px = circle_r * math.cos(angle)
        py = circle_r * math.sin(angle)
        dense_peds.append(
            Pedestrian(
                x=px,
                y=py,
                vx=0.0,
                vy=0.0,
                radius=0.3,
                goal_x=-px,
                goal_y=-py,
                v_pref=1.0,
            )
        )
    scenarios.append(
        Scenario(
            name="circle_crossing_10",
            robot_start=(-6.0, 0.0, 0.0),
            robot_goal=(6.0, 0.0),
            pedestrians=dense_peds,
            time_limit=50.0,
        )
    )

    # --- 5. Narrow corridor with oncoming pedestrian ---
    corridor_obs: list[tuple[float, float, float]] = []
    for x in np.arange(1.0, 7.0, 0.5):
        corridor_obs.append((x, 1.2, 0.25))
        corridor_obs.append((x, -1.2, 0.25))
    scenarios.append(
        Scenario(
            name="narrow_corridor_oncoming",
            robot_start=(0.0, 0.0, 0.0),
            robot_goal=(8.0, 0.0),
            static_obstacles=corridor_obs,
            pedestrians=[
                Pedestrian(
                    x=7.0,
                    y=0.0,
                    vx=0.0,
                    vy=0.0,
                    radius=0.3,
                    goal_x=0.0,
                    goal_y=0.0,
                    v_pref=0.8,
                )
            ],
            time_limit=40.0,
        )
    )

    # --- 6. Perpendicular crossing stream ---
    perp_peds = []
    for i in range(6):
        perp_peds.append(
            Pedestrian(
                x=4.0,
                y=-5.0 + i * 2.0,
                vx=0.0,
                vy=0.0,
                radius=0.3,
                goal_x=4.0,
                goal_y=5.0 - i * 2.0,
                v_pref=0.8 + 0.1 * i,
            )
        )
    scenarios.append(
        Scenario(
            name="perpendicular_crossing",
            robot_start=(0.0, 0.0, 0.0),
            robot_goal=(8.0, 0.0),
            pedestrians=perp_peds,
            time_limit=40.0,
        )
    )

    # --- 7. Unreachable goal (enclosed) ---
    enclosure_obs = []
    for angle_deg in range(0, 360, 15):
        a = math.radians(angle_deg)
        enclosure_obs.append((8.0 + 1.5 * math.cos(a), 1.5 * math.sin(a), 0.3))
    scenarios.append(
        Scenario(
            name="unreachable_goal",
            robot_start=(0.0, 0.0, 0.0),
            robot_goal=(8.0, 0.0),
            static_obstacles=enclosure_obs,
            time_limit=30.0,
        )
    )

    return scenarios


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

ROBOT_RADIUS = 0.25


@dataclass
class StepRecord:
    """One simulation timestep worth of data."""

    t: float
    robot_x: float
    robot_y: float
    robot_yaw: float
    cmd_linear_x: float
    cmd_linear_y: float
    cmd_angular_z: float
    distance_to_goal: float
    min_pedestrian_dist: float
    min_obstacle_dist: float
    path_length: int  # number of waypoints in latest path
    nav_state: str


@dataclass
class ScenarioResult:
    """Metrics from a single scenario run."""

    scenario_name: str
    success: bool
    collision: bool
    timeout: bool
    time_to_goal: float
    total_time: float
    path_length_actual: float
    path_length_optimal: float
    path_length_ratio: float
    collision_count: int
    min_pedestrian_clearance: float
    min_obstacle_clearance: float
    replan_count: int
    cmd_vel_jerk_mean: float
    cmd_vel_jerk_max: float
    heading_oscillation: float
    forward_progress_rate: float
    avg_speed: float
    steps: list[StepRecord]


def _build_costmap(
    scenario: Scenario,
    pedestrians: list[Pedestrian],
    grid_cells: int,
    resolution: float,
) -> OccupancyGrid:
    """Build an OccupancyGrid from static obstacles + current pedestrian positions."""
    half_world = scenario.world_size / 2.0
    grid = np.zeros((grid_cells, grid_cells), dtype=np.int8)

    def world_to_cell(wx: float, wy: float) -> tuple[int, int]:
        cx = int((wx + half_world) / resolution)
        cy = int((wy + half_world) / resolution)
        return cx, cy

    def fill_circle(cx: int, cy: int, radius_cells: int, value: int) -> None:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    gx = cx + dx
                    gy = cy + dy
                    if 0 <= gx < grid_cells and 0 <= gy < grid_cells:
                        grid[gy, gx] = max(grid[gy, gx], value)

    # Static obstacles → lethal (100)
    for ox, oy, orad in scenario.static_obstacles:
        cx, cy = world_to_cell(ox, oy)
        rad_cells = math.ceil(orad / resolution)
        fill_circle(cx, cy, rad_cells, 100)

    # Pedestrians → high cost core (100) + inflation ring (50)
    inflation_m = 0.5
    for ped in pedestrians:
        cx, cy = world_to_cell(ped.x, ped.y)
        core_cells = math.ceil(ped.radius / resolution)
        inflation_cells = math.ceil((ped.radius + inflation_m) / resolution)
        fill_circle(cx, cy, inflation_cells, 50)
        fill_circle(cx, cy, core_cells, 100)

    origin = Pose(-half_world, -half_world, 0.0)
    return OccupancyGrid(
        grid=grid,
        resolution=resolution,
        origin=origin,
        frame_id="world",
    )


def _run_scenario(
    scenario: Scenario,
    publish_odom: Any,
    publish_costmap: Any,
    publish_goal: Any,
    get_latest_cmd_vel: Any,
    get_latest_path: Any,
    get_latest_nav_state: Any,
    dt: float = 0.1,
) -> ScenarioResult:
    """Run a single scenario through the planner and collect metrics."""
    grid_cells = int(scenario.world_size / scenario.resolution)

    # Robot state
    rx, ry, ryaw = scenario.robot_start
    gx, gy = scenario.robot_goal
    optimal_dist = math.hypot(gx - rx, gy - ry)

    # Deep copy pedestrians so scenarios are independent
    peds = [
        Pedestrian(
            x=p.x,
            y=p.y,
            vx=p.vx,
            vy=p.vy,
            radius=p.radius,
            goal_x=p.goal_x,
            goal_y=p.goal_y,
            v_pref=p.v_pref,
        )
        for p in scenario.pedestrians
    ]

    steps: list[StepRecord] = []
    collision_count = 0
    replan_count = 0
    prev_path_hash: int | None = None
    total_distance = 0.0
    prev_cmd_vels: list[tuple[float, float, float]] = []
    headings: list[float] = []
    success = False
    collision_flag = False

    # Publish initial state
    costmap = _build_costmap(scenario, peds, grid_cells, scenario.resolution)
    odom = PoseStamped(
        position=[rx, ry, 0.0], orientation=[0.0, 0.0, math.sin(ryaw / 2), math.cos(ryaw / 2)]
    )
    publish_odom(odom)
    publish_costmap(costmap)

    # Give planner a moment to initialize, then send goal
    time.sleep(0.3)
    goal_pose = PoseStamped(position=[gx, gy, 0.0])
    publish_goal(goal_pose)
    time.sleep(0.2)

    sim_time = 0.0
    while sim_time < scenario.time_limit:
        # Get latest cmd_vel from planner
        cmd = get_latest_cmd_vel()
        vx = cmd.linear.x if cmd else 0.0
        vy = cmd.linear.y if cmd else 0.0
        wz = cmd.angular.z if cmd else 0.0

        # Apply unicycle/holonomic kinematics
        ryaw += wz * dt
        rx += (vx * math.cos(ryaw) - vy * math.sin(ryaw)) * dt
        ry += (vx * math.sin(ryaw) + vy * math.cos(ryaw)) * dt
        total_distance += math.hypot(
            (vx * math.cos(ryaw) - vy * math.sin(ryaw)) * dt,
            (vx * math.sin(ryaw) + vy * math.cos(ryaw)) * dt,
        )

        # Step pedestrians
        for ped in peds:
            ped.step(dt)

        # Check collisions
        min_ped_dist = float("inf")
        for ped in peds:
            d = math.hypot(rx - ped.x, ry - ped.y) - ped.radius - ROBOT_RADIUS
            min_ped_dist = min(min_ped_dist, d)
        if min_ped_dist < 0:
            collision_count += 1
            collision_flag = True

        min_obs_dist = float("inf")
        for ox, oy, orad in scenario.static_obstacles:
            d = math.hypot(rx - ox, ry - oy) - orad - ROBOT_RADIUS
            min_obs_dist = min(min_obs_dist, d)
        if min_obs_dist < 0:
            collision_count += 1
            collision_flag = True

        # Check goal reached
        dist_to_goal = math.hypot(rx - gx, ry - gy)
        if dist_to_goal < scenario.goal_tolerance:
            success = True

        # Track replans
        cur_path = get_latest_path()
        if cur_path is not None:
            path_id = id(cur_path)
            if prev_path_hash is not None and path_id != prev_path_hash:
                replan_count += 1
            prev_path_hash = path_id

        # Track cmd_vel history for jerk/oscillation
        prev_cmd_vels.append((vx, vy, wz))
        headings.append(ryaw)

        nav_state_str = ""
        nav_state = get_latest_nav_state()
        if nav_state is not None:
            nav_state_str = str(nav_state)

        steps.append(
            StepRecord(
                t=sim_time,
                robot_x=rx,
                robot_y=ry,
                robot_yaw=ryaw,
                cmd_linear_x=vx,
                cmd_linear_y=vy,
                cmd_angular_z=wz,
                distance_to_goal=dist_to_goal,
                min_pedestrian_dist=min_ped_dist if peds else float("inf"),
                min_obstacle_dist=min_obs_dist if scenario.static_obstacles else float("inf"),
                path_length=len(cur_path.poses)
                if cur_path and hasattr(cur_path, "poses") and cur_path.poses
                else 0,
                nav_state=nav_state_str,
            )
        )

        # Publish updated state to planner
        costmap = _build_costmap(scenario, peds, grid_cells, scenario.resolution)
        odom = PoseStamped(
            position=[rx, ry, 0.0],
            orientation=[0.0, 0.0, math.sin(ryaw / 2), math.cos(ryaw / 2)],
        )
        publish_odom(odom)
        publish_costmap(costmap)

        sim_time += dt

        if success:
            break

        # Real-time pacing (sleep for dt minus compute time)
        time.sleep(max(0.0, dt * 0.5))

    # --- Compute aggregate metrics ---
    time_to_goal = sim_time if success else scenario.time_limit
    path_length_ratio = total_distance / optimal_dist if optimal_dist > 0 else float("inf")

    # Cmd_vel jerk (change in acceleration between steps)
    jerks: list[float] = []
    for i in range(2, len(prev_cmd_vels)):
        a1_x = prev_cmd_vels[i - 1][0] - prev_cmd_vels[i - 2][0]
        a1_y = prev_cmd_vels[i - 1][1] - prev_cmd_vels[i - 2][1]
        a2_x = prev_cmd_vels[i][0] - prev_cmd_vels[i - 1][0]
        a2_y = prev_cmd_vels[i][1] - prev_cmd_vels[i - 1][1]
        jerk = math.hypot(a2_x - a1_x, a2_y - a1_y) / dt
        jerks.append(jerk)
    jerk_mean = float(np.mean(jerks)) if jerks else 0.0
    jerk_max = float(np.max(jerks)) if jerks else 0.0

    # Heading oscillation (sum of absolute heading changes)
    heading_changes = [abs(headings[i] - headings[i - 1]) for i in range(1, len(headings))]
    heading_osc = sum(heading_changes)

    # Forward progress rate (distance closed to goal per second)
    if len(steps) >= 2:
        initial_dist = steps[0].distance_to_goal
        final_dist = steps[-1].distance_to_goal
        progress = initial_dist - final_dist
        forward_progress_rate = progress / sim_time if sim_time > 0 else 0.0
    else:
        forward_progress_rate = 0.0

    avg_speed = total_distance / sim_time if sim_time > 0 else 0.0

    min_ped_clearance = min((s.min_pedestrian_dist for s in steps), default=float("inf"))
    min_obs_clearance = min((s.min_obstacle_dist for s in steps), default=float("inf"))

    return ScenarioResult(
        scenario_name=scenario.name,
        success=success,
        collision=collision_flag,
        timeout=not success,
        time_to_goal=time_to_goal,
        total_time=sim_time,
        path_length_actual=total_distance,
        path_length_optimal=optimal_dist,
        path_length_ratio=path_length_ratio,
        collision_count=collision_count,
        min_pedestrian_clearance=min_ped_clearance,
        min_obstacle_clearance=min_obs_clearance,
        replan_count=replan_count,
        cmd_vel_jerk_mean=jerk_mean,
        cmd_vel_jerk_max=jerk_max,
        heading_oscillation=heading_osc,
        forward_progress_rate=forward_progress_rate,
        avg_speed=avg_speed,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Benchmark harness module
# ---------------------------------------------------------------------------


class _BenchmarkHarness(Module):
    """Simulated world that drives the planner under test.

    Publishes synthetic odom + costmap, consumes cmd_vel + path.
    The main evaluate_planner() function calls run_evaluation() via RPC.
    """

    # Outputs → connect to planner inputs
    odom: Out[PoseStamped]
    global_costmap: Out[OccupancyGrid]
    goal_request: Out[PoseStamped]

    # Inputs ← connect to planner outputs
    cmd_vel: In[Twist]
    path: In[Path]
    goal_reached: In[Bool]
    navigation_state: In[String]

    _latest_cmd_vel: Twist | None
    _latest_path: Path | None
    _latest_nav_state: String | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_cmd_vel = None
        self._latest_path = None
        self._latest_nav_state = None

    @rpc
    def start(self) -> None:
        super().start()
        self.cmd_vel.subscribe(self._on_cmd_vel)
        self.path.subscribe(self._on_path)
        self.navigation_state.subscribe(self._on_nav_state)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd_vel = msg

    def _on_path(self, msg: Path) -> None:
        self._latest_path = msg

    def _on_nav_state(self, msg: String) -> None:
        self._latest_nav_state = msg

    @rpc
    def run_evaluation(
        self,
        scenarios_data: list[dict[str, Any]],
        dt: float,
    ) -> list[dict[str, Any]]:
        """Run all scenarios and return serialized results."""
        scenarios = [_deserialize_scenario(s) for s in scenarios_data]
        results: list[dict[str, Any]] = []

        for scenario in scenarios:
            self._latest_cmd_vel = None
            self._latest_path = None
            self._latest_nav_state = None

            result = _run_scenario(
                scenario=scenario,
                publish_odom=self.odom.publish,
                publish_costmap=self.global_costmap.publish,
                publish_goal=self.goal_request.publish,
                get_latest_cmd_vel=lambda: self._latest_cmd_vel,
                get_latest_path=lambda: self._latest_path,
                get_latest_nav_state=lambda: self._latest_nav_state,
                dt=dt,
            )
            results.append(_serialize_result(result))

        return results


# ---------------------------------------------------------------------------
# Serialization helpers (for RPC transport)
# ---------------------------------------------------------------------------


def _serialize_scenario(s: Scenario) -> dict[str, Any]:
    return {
        "name": s.name,
        "world_size": s.world_size,
        "resolution": s.resolution,
        "robot_start": list(s.robot_start),
        "robot_goal": list(s.robot_goal),
        "pedestrians": [
            {
                "x": p.x,
                "y": p.y,
                "vx": p.vx,
                "vy": p.vy,
                "radius": p.radius,
                "goal_x": p.goal_x,
                "goal_y": p.goal_y,
                "v_pref": p.v_pref,
            }
            for p in s.pedestrians
        ],
        "static_obstacles": [list(o) for o in s.static_obstacles],
        "time_limit": s.time_limit,
        "goal_tolerance": s.goal_tolerance,
    }


def _deserialize_scenario(d: dict[str, Any]) -> Scenario:
    return Scenario(
        name=d["name"],
        world_size=d.get("world_size", 20.0),
        resolution=d.get("resolution", 0.1),
        robot_start=tuple(d["robot_start"]),  # type: ignore[arg-type]
        robot_goal=tuple(d["robot_goal"]),  # type: ignore[arg-type]
        pedestrians=[Pedestrian(**p) for p in d.get("pedestrians", [])],
        static_obstacles=[tuple(o) for o in d.get("static_obstacles", [])],  # type: ignore[misc]
        time_limit=d.get("time_limit", 60.0),
        goal_tolerance=d.get("goal_tolerance", 0.5),
    )


def _serialize_result(r: ScenarioResult) -> dict[str, Any]:
    return {
        "scenario_name": r.scenario_name,
        "success": r.success,
        "collision": r.collision,
        "timeout": r.timeout,
        "time_to_goal": round(r.time_to_goal, 3),
        "total_time": round(r.total_time, 3),
        "path_length_actual": round(r.path_length_actual, 3),
        "path_length_optimal": round(r.path_length_optimal, 3),
        "path_length_ratio": round(r.path_length_ratio, 3),
        "collision_count": r.collision_count,
        "min_pedestrian_clearance": round(r.min_pedestrian_clearance, 3),
        "min_obstacle_clearance": round(r.min_obstacle_clearance, 3),
        "replan_count": r.replan_count,
        "cmd_vel_jerk_mean": round(r.cmd_vel_jerk_mean, 4),
        "cmd_vel_jerk_max": round(r.cmd_vel_jerk_max, 4),
        "heading_oscillation": round(r.heading_oscillation, 3),
        "forward_progress_rate": round(r.forward_progress_rate, 3),
        "avg_speed": round(r.avg_speed, 3),
        # Omit per-step data from the serialized dict (too large for RPC).
        # The full steps are available when running _run_scenario directly.
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_planner(
    planner_blueprint: Blueprint,
    scenarios: list[Scenario] | None = None,
    dt: float = 0.1,
    n_workers: int = 2,
) -> dict[str, Any]:
    """Evaluate a DimOS planner module against CrowdNav-style scenarios.

    Args:
        planner_blueprint: Blueprint for a planner module. Must expose:
            - Inputs:  odom (PoseStamped), global_costmap (OccupancyGrid),
                       goal_request (PoseStamped)
            - Outputs: cmd_vel (Twist), path (Path)
        scenarios: List of Scenario objects. Defaults to a built-in suite
            covering open-field, static obstacles, circle crossings,
            narrow corridors, perpendicular crossings, and unreachable goals.
        dt: Simulation timestep in seconds.
        n_workers: Number of worker processes for the ModuleCoordinator.

    Returns:
        Dict with:
            - "scenarios": list of per-scenario result dicts
            - "aggregate": summary metrics across all scenarios
    """
    if scenarios is None:
        scenarios = _default_scenarios()

    harness_bp = _BenchmarkHarness.blueprint(
        rpc_timeouts={"run_evaluation": 600.0},
    )
    combined = autoconnect(harness_bp, planner_blueprint).global_config(n_workers=n_workers)

    coordinator = ModuleCoordinator.build(combined)

    try:
        # Get the harness proxy and run scenarios one at a time via RPC
        harness_proxy = coordinator.get_instance(_BenchmarkHarness)
        raw_results: list[dict[str, Any]] = []
        for scenario in scenarios:
            scenario_data = [_serialize_scenario(scenario)]
            batch: list[dict[str, Any]] = harness_proxy.run_evaluation(scenario_data, dt)
            raw_results.extend(batch)

        # Compute aggregate metrics
        successes = [r for r in raw_results if r["success"]]
        collisions = [r for r in raw_results if r["collision"]]

        aggregate: dict[str, Any] = {
            "total_scenarios": len(raw_results),
            "success_count": len(successes),
            "success_rate": len(successes) / len(raw_results) if raw_results else 0.0,
            "collision_count": len(collisions),
            "collision_rate": len(collisions) / len(raw_results) if raw_results else 0.0,
        }

        if successes:
            aggregate["avg_time_to_goal"] = round(
                sum(r["time_to_goal"] for r in successes) / len(successes), 3
            )
            aggregate["avg_path_length_ratio"] = round(
                sum(r["path_length_ratio"] for r in successes) / len(successes), 3
            )
            # SPL (Success weighted by Path Length) — standard metric
            spl_values = [
                r["path_length_optimal"] / max(r["path_length_actual"], r["path_length_optimal"])
                for r in successes
            ]
            aggregate["spl"] = round(sum(spl_values) / len(raw_results), 3)
        else:
            aggregate["avg_time_to_goal"] = None
            aggregate["avg_path_length_ratio"] = None
            aggregate["spl"] = 0.0

        aggregate["avg_cmd_vel_jerk"] = (
            round(sum(r["cmd_vel_jerk_mean"] for r in raw_results) / len(raw_results), 4)
            if raw_results
            else 0.0
        )

        aggregate["avg_replan_count"] = (
            round(sum(r["replan_count"] for r in raw_results) / len(raw_results), 1)
            if raw_results
            else 0.0
        )

        return {
            "scenarios": raw_results,
            "aggregate": aggregate,
        }
    finally:
        coordinator.stop()

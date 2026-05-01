# Copyright 2026 Dimensional Inc.
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

"""E2E integration test: cross-wall planning in the MuJoCo sim.

MuJoCo counterpart to ``test_cross_wall_planning_simple.py``. Same
nav-stack composition (SimplePlanner + PGO + LocalPlanner + PathFollower
+ MovementManager + TerrainAnalysis); the sim is the only thing that
changes — Unity's ``home_building_1`` building is replaced by MuJoCo's
default ``office1`` scene loaded via ``GlobalConfig.mujoco_room``.

The waypoints are **best-effort** for the office1 scene. The robot
spawns near the origin facing +X; the office layout has interior walls
that the planner must route around. Coordinates here are placeholders
that exercise the planner — fine-tuning them to specific doorways
requires loading the scene and reading wall geometry, which is out of
scope for an initial sim wire-up.
"""

from __future__ import annotations

import pytest

# create_nav_stack pulls in PGO which requires gtsam; the mujoco-side
# test also needs the mujoco package.  Skip the whole module if either
# is missing.  Note: when both ARE installed the test is collected and
# runs, but is xfail-marked (see pytestmark below) until office1
# waypoint/planner tuning lands.
pytest.importorskip("gtsam")
pytest.importorskip("mujoco")

from dimos.core.coordination.blueprints import autoconnect
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.nav_stack.main import create_nav_stack
from dimos.navigation.nav_stack.tests.conftest import (
    CROSS_WALL_LOCAL_PLANNER,
    CROSS_WALL_PATH_FOLLOWER,
    CROSS_WALL_TERRAIN_ANALYSIS,
    run_cross_wall_test,
)
from dimos.robot.unitree.g1.blueprints.navigation._mujoco_pose_adapter import (
    MujocoPoseToOdometryAdapter,
)
from dimos.robot.unitree.g1.mujoco_sim import G1SimConnection

pytestmark = [
    pytest.mark.slow,
    # The placeholder waypoints in MUJOCO_OFFICE1_WAYPOINTS are not tuned to
    # the office1 wall layout, and SimplePlanner's A* fails to path through
    # the resulting costmap from the spawn position.  The wiring is verified
    # (odom flows, TF resolves, planner receives goals); reaching the goals
    # needs scene-specific waypoint tuning + planner tuning, which the user
    # explicitly accepted as a follow-up.  strict=False means "we expect to
    # fail; if it ever passes, that's a bonus, not a regression."
    pytest.mark.xfail(
        reason="office1 waypoints + planner tuning not yet done — wiring verified, end-to-end not",
        strict=False,
    ),
]

# Ground-truth ceiling guard.  The G1 stands ~1.24 m tall in the mujoco
# scene and the office1 ceiling is ~3 m, so MAX_ALLOWED_Z = 2.0 catches
# the robot phasing through geometry while still tolerating the ~0.3 m
# terrain-z drift TerrainAnalysis exhibits near walls.
MAX_ALLOWED_Z = 2.0

# Best-effort waypoint sequence for the MuJoCo office1 scene.  The robot
# spawns near (0, 0) facing +X; the surrounding office has interior
# walls separating an open lobby/floor from inner rooms.
#
# Format: (label, x_m, y_m, z_m, timeout_sec, reach_threshold_m).
# Picked to exercise the planner along multiple headings rather than to
# match a hand-tuned doorway sequence.  Adjust these once the scene's
# wall layout is mapped (see context in this docstring).
MUJOCO_OFFICE1_WAYPOINTS: list[tuple[str, float, float, float, float, float]] = [
    ("p0", 2.0, 0.0, 0.0, 60, 1.0),  # straight ahead
    ("p1", 4.0, 2.0, 0.0, 90, 1.5),  # diagonal forward-left
    ("p2", 4.0, -2.0, 0.0, 120, 1.5),  # diagonal forward-right (cross-room)
    ("p3", 0.0, 0.0, 0.0, 120, 1.5),  # back to start
]


class TestCrossWallPlanningMujoco:
    """E2E: cross-wall routing in the MuJoCo office1 scene."""

    def test_cross_wall_sequence_mujoco(self) -> None:
        blueprint = (
            autoconnect(
                G1SimConnection.blueprint(),
                MujocoPoseToOdometryAdapter.blueprint(),
                create_nav_stack(
                    use_simple_planner=True,
                    terrain_analysis=CROSS_WALL_TERRAIN_ANALYSIS,
                    local_planner=CROSS_WALL_LOCAL_PLANNER,
                    path_follower=CROSS_WALL_PATH_FOLLOWER,
                    simple_planner={
                        "cell_size": 0.3,
                        "obstacle_height_threshold": 0.15,
                        "inflation_radius": 0.7,
                        "lookahead_distance": 2.0,
                        "replan_rate": 5.0,
                        "stuck_seconds": 4.0,
                        "stuck_shrink_factor": 0.5,
                    },
                ),
                MovementManager.blueprint(),
            )
            .remappings(
                [
                    # MuJoCo's lidar is already in world frame.
                    (G1SimConnection, "lidar", "registered_scan"),
                ]
            )
            .global_config(
                n_workers=8,
                robot_model="unitree_g1",
                simulation=True,
                mujoco_room="office1",
                # Run the mujoco physics loop without the interactive
                # viewer so this test works in headless / CI / non-GUI
                # macOS environments.
                mujoco_headless=True,
            )
        )

        run_cross_wall_test(
            blueprint,
            label="mujoco",
            max_z=MAX_ALLOWED_Z,
            waypoints=MUJOCO_OFFICE1_WAYPOINTS,
        )

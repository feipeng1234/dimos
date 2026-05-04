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

"""G1 nav sim (MuJoCo) -- SimplePlanner + PGO + local obstacle avoidance.

Uses ``G1MujocoPlanarSim`` as the sim source: a kinematic planar box with
a Mid-360-flavoured 3D lidar. The sim emits ``odom`` (nav_msgs/Odometry)
and ``lidar`` (PointCloud2 in world frame, i.e. already a registered
scan), so no pose adapter or coordinate transform is needed -- a couple
of stream-name remappings hand the data straight to the nav stack.

Data flow:
    Click -> MovementManager -> goal -> SimplePlanner
    -> way_point -> LocalPlanner -> path -> PathFollower
    -> nav_cmd_vel -> MovementManager -> cmd_vel -> G1MujocoPlanarSim

    G1MujocoPlanarSim.lidar (= registered_scan) -> TerrainAnalysis -> terrain_map
    G1MujocoPlanarSim.odom  (= odometry)        -> PGO
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.nav_stack.main import create_nav_stack
from dimos.robot.unitree.g1.config import G1_LOCAL_PLANNER_PRECOMPUTED_PATHS, G1_VEHICLE_HEIGHT
from dimos.robot.unitree.g1.mujoco_planar_sim import G1MujocoPlanarSim

unitree_g1_nav_mujoco_sim = (
    autoconnect(
        G1MujocoPlanarSim.blueprint(),
        create_nav_stack(
            use_simple_planner=True,
            vehicle_height=G1_VEHICLE_HEIGHT,
            simple_planner={
                "body_frame": "base_link",
                # The planar-sim box's odom z sits at ~0.66 m (box centre).
                # The default ground_offset_below_robot=1.3 was tuned for the
                # G1's full 1.24 m standing height -- with our shorter robot z
                # it would label the actual floor (z=0) as a 0.6 m obstacle and
                # block every cell. 0.7 matches the box centre + a little
                # slack so the floor stays below obstacle_height_threshold.
                "ground_offset_below_robot": 0.7,
            },
            terrain_analysis={
                "obstacle_height_threshold": 0.1,
                "ground_height_threshold": 0.05,
                "max_relative_z": 0.3,
                "min_relative_z": -1.5,
            },
            local_planner={
                "paths_dir": str(G1_LOCAL_PLANNER_PRECOMPUTED_PATHS),
                "max_speed": 2.0,
                "autonomy_speed": 2.0,
                "obstacle_height_threshold": 0.1,
                "max_relative_z": 0.3,
                "min_relative_z": -1.5,
                "freeze_ang": 180.0,
                "two_way_drive": False,
            },
            path_follower={
                "max_speed": 2.0,
                "autonomy_speed": 2.0,
                "max_acceleration": 4.0,
                "slow_down_distance_threshold": 0.5,
                "omni_dir_goal_threshold": 0.5,
                "two_way_drive": False,
            },
        ),
        MovementManager.blueprint(),
    )
    .remappings(
        [
            # Sim emits world-frame lidar -> the "registered_scan" the
            # nav stack consumes in TerrainAnalysis.
            (G1MujocoPlanarSim, "lidar", "registered_scan"),
            # Sim emits Odometry directly under "odom"; nav-stack subscribes
            # under "odometry".
            (G1MujocoPlanarSim, "odom", "odometry"),
            # Planner owns way_point -- disconnect MovementManager's click relay.
            (MovementManager, "way_point", "_mgr_way_point_unused"),
        ]
    )
    .global_config(
        n_workers=8,
        robot_model="unitree_g1",
        simulation=True,
        # Spawn inside the dining_room interior (matches the scene_209
        # layout used by the cross-wall test). Override at the CLI with
        # --mujoco-start-pos "x, y" if you want a different room.
        mujoco_start_pos="0.2, -3.2",
    )
)


__all__ = ["unitree_g1_nav_mujoco_sim"]

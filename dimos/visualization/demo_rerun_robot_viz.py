#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""Demo blueprint for comprehensive Rerun visualization.

This blueprint starts:
- Dashboard (Rerun server + web viewer)
- ROS→Rerun bridge (autonomy stack LiDAR/nav topics)
- ObjectSceneRegistration (perception + mesh visualization)

Everything that was previously visible in RViz is now also visible in Rerun,
while RViz continues to work unchanged (non-breaking).

Usage:
    python -m dimos.robot.cli.dimos run demo-rerun-robot-viz
"""

import rerun.blueprint as rrb

from dimos.core.blueprints import autoconnect
from dimos.dashboard.module import Dashboard
from dimos.perception.object_scene_registration import object_scene_registration_module
from dimos.visualization.ros_rerun_bridge import ros_rerun_bridge_module

# Custom blueprint for better default layout
custom_blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(
            name="World 3D",
            origin="/world",
        ),
        rrb.Vertical(
            rrb.Spatial2DView(
                name="Camera RGB",
                origin="/world/robot/base/camera/rgb",
            ),
            rrb.Spatial2DView(
                name="Detections Overlay",
                origin="/world/robot/base/camera/overlay",
            ),
            row_shares=[1, 1],
        ),
        column_shares=[2, 1],
    )
)


demo_rerun_robot_viz = autoconnect(
    Dashboard.blueprint(
        port=4001,  # Avoid conflicts with NoMachine on port 4000
        rerun_default_blueprint=custom_blueprint,
    ),
    ros_rerun_bridge_module(),
    object_scene_registration_module(
        mesh_pose_service_url="http://localhost:8080",
        auto_mesh_pose=True,
        mesh_pose_use_box_prompt=True,
        mesh_store_dir="/home/dimensional/dimos/meshes",
        mesh_marker_topic="/object_detections/mesh_markers",
    ),
)

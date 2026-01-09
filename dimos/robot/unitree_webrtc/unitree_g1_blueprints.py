#!/usr/bin/env python3
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

"""Blueprint configurations for Unitree G1 humanoid robot.

On `dev`, the canonical navigation stack is the same as GO2:
- TF drives motion (visualized via `tf_rerun()` snapshot polling)
- Planning uses `replanning_a_star_planner()` (no ROS nav dependency required)
- Connection selects WebRTC vs MuJoCo via `GlobalConfig.simulation`/`unitree_connection_type`
"""

from dimos_lcm.foxglove_msgs import SceneUpdate
from dimos_lcm.foxglove_msgs.ImageAnnotations import (
    ImageAnnotations,
)

from dimos.agents.agent import llm_agent
from dimos.agents.cli.human import human_input
from dimos.agents.skills.navigation import navigation_skill
from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE
from dimos.core.blueprints import autoconnect
from dimos.core.transport import LCMTransport, pSHMTransport
from dimos.dashboard.tf_rerun_module import tf_rerun
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.voxels import voxel_mapper
from dimos.msgs.geometry_msgs import (
    PoseStamped,
    Twist,
)
from dimos.msgs.sensor_msgs import CameraInfo, Image, PointCloud2
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.navigation.frontier_exploration import wavefront_frontier_explorer
from dimos.navigation.replanning_a_star.module import replanning_a_star_planner
from dimos.perception.detection.detectors.person.yolo import YoloPersonDetector
from dimos.perception.detection.module3D import Detection3DModule, detection3d_module
from dimos.perception.detection.moduleDB import ObjectDBModule, detectionDB_module
from dimos.perception.detection.person_tracker import PersonTracker, person_tracker_module
from dimos.perception.object_tracker import object_tracking
from dimos.perception.spatial_perception import spatial_memory
from dimos.robot.foxglove_bridge import foxglove_bridge
from dimos.robot.unitree.connection.g1 import G1Connection, g1_connection
from dimos.robot.unitree_webrtc.keyboard_teleop import keyboard_teleop
from dimos.robot.unitree_webrtc.unitree_g1_skill_container import g1_skills
from dimos.utils.monitoring import utilization
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis

basic = autoconnect(
    g1_connection(),
    foxglove_bridge(),
    websocket_vis(),
    tf_rerun(),  # TF snapshot polling -> Rerun under `world/tf/*`
).global_config(n_dask_workers=4, robot_model="unitree_g1")

nav = autoconnect(
    basic,
    voxel_mapper(voxel_size=0.1),
    cost_mapper(),
    replanning_a_star_planner(),
    wavefront_frontier_explorer(),
).global_config(n_dask_workers=6, robot_model="unitree_g1")

# Backwards-compat names expected by `all_blueprints.py`
basic_ros = basic
basic_sim = basic

_perception_and_memory = autoconnect(
    spatial_memory(),
    object_tracking(frame_id="camera_link"),
    utilization(),
)

standard = autoconnect(
    nav,
    _perception_and_memory,
).global_config(n_dask_workers=8)

standard_sim = standard

# Optimized configuration using shared memory for images
standard_with_shm = autoconnect(
    standard.transports(
        {
            ("color_image", Image): pSHMTransport(
                "/g1/color_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
            ),
        }
    ),
    foxglove_bridge(
        shm_channels=[
            "/g1/color_image#sensor_msgs.Image",
        ]
    ),
)

_agentic_skills = autoconnect(
    llm_agent(),
    human_input(),
    navigation_skill(),
    g1_skills(),
)

# Full agentic configuration with LLM and skills
agentic = autoconnect(
    standard,
    _agentic_skills,
)

agentic_sim = autoconnect(
    standard_sim,
    _agentic_skills,
)

# Configuration with joystick control for teleoperation
with_joystick = autoconnect(
    basic,
    keyboard_teleop(),  # Pygame-based joystick control
)

# Detection configuration with person tracking and 3D detection
detection = (
    autoconnect(
        nav,
        # Person detection modules with YOLO
        detection3d_module(
            camera_info=G1Connection.camera_info_static,
            detector=YoloPersonDetector,
        ),
        detectionDB_module(
            camera_info=G1Connection.camera_info_static,
            filter=lambda det: det.class_id == 0,  # Filter for person class only
        ),
        person_tracker_module(
            cameraInfo=G1Connection.camera_info_static,
        ),
    )
    .global_config(n_dask_workers=8)
    .remappings(
        [
            # Use the mapped global map (PointCloud2) for 3D projection/tracking
            (Detection3DModule, "pointcloud", "global_map"),
            (ObjectDBModule, "pointcloud", "global_map"),
            # Ensure the camera stream is the connection camera (not any other image stream)
            (Detection3DModule, "color_image", "color_image"),
            (ObjectDBModule, "color_image", "color_image"),
            (PersonTracker, "color_image", "color_image"),
        ]
    )
    .transports(
        {
            # Detection 3D module outputs
            ("detections", Detection3DModule): LCMTransport(
                "/detector3d/detections", Detection2DArray
            ),
            ("annotations", Detection3DModule): LCMTransport(
                "/detector3d/annotations", ImageAnnotations
            ),
            ("scene_update", Detection3DModule): LCMTransport(
                "/detector3d/scene_update", SceneUpdate
            ),
            ("detected_pointcloud_0", Detection3DModule): LCMTransport(
                "/detector3d/pointcloud/0", PointCloud2
            ),
            ("detected_pointcloud_1", Detection3DModule): LCMTransport(
                "/detector3d/pointcloud/1", PointCloud2
            ),
            ("detected_pointcloud_2", Detection3DModule): LCMTransport(
                "/detector3d/pointcloud/2", PointCloud2
            ),
            ("detected_image_0", Detection3DModule): LCMTransport("/detector3d/image/0", Image),
            ("detected_image_1", Detection3DModule): LCMTransport("/detector3d/image/1", Image),
            ("detected_image_2", Detection3DModule): LCMTransport("/detector3d/image/2", Image),
            # Detection DB module outputs
            ("detections", ObjectDBModule): LCMTransport(
                "/detectorDB/detections", Detection2DArray
            ),
            ("annotations", ObjectDBModule): LCMTransport(
                "/detectorDB/annotations", ImageAnnotations
            ),
            ("scene_update", ObjectDBModule): LCMTransport("/detectorDB/scene_update", SceneUpdate),
            ("detected_pointcloud_0", ObjectDBModule): LCMTransport(
                "/detectorDB/pointcloud/0", PointCloud2
            ),
            ("detected_pointcloud_1", ObjectDBModule): LCMTransport(
                "/detectorDB/pointcloud/1", PointCloud2
            ),
            ("detected_pointcloud_2", ObjectDBModule): LCMTransport(
                "/detectorDB/pointcloud/2", PointCloud2
            ),
            ("detected_image_0", ObjectDBModule): LCMTransport("/detectorDB/image/0", Image),
            ("detected_image_1", ObjectDBModule): LCMTransport("/detectorDB/image/1", Image),
            ("detected_image_2", ObjectDBModule): LCMTransport("/detectorDB/image/2", Image),
            # Person tracker outputs
            ("target", PersonTracker): LCMTransport("/person_tracker/target", PoseStamped),
        }
    )
)

# Full featured configuration with everything
full_featured = autoconnect(
    standard_with_shm,
    _agentic_skills,
    keyboard_teleop(),
)

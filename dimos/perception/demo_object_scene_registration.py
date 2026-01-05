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

"""Demo blueprint for ObjectSceneRegistration with hosted service.

This runs automatic object detection with optional mesh/pose enhancement.
By default, uses box-only prompting to avoid passing garbage YOLO-E labels.

Meshes are saved to `/home/dimensional/dimos/meshes` and published to RViz as
`visualization_msgs/MarkerArray` on `/object_detections/mesh_markers`.

For interactive control (list detections, trigger pipelines manually),
use demo_object_scene_registration_interactive instead.

Usage:
    # Auto mode (processes all detections with box-only prompting)
    python -m dimos.robot.cli.dimos run demo-object-scene-registration

    # Interactive mode (manual trigger, custom prompts)
    python -m dimos.perception.demo_object_scene_registration_interactive \\
        --service-url http://localhost:8080
"""

from dimos.core.blueprints import autoconnect
from dimos.perception.object_scene_registration import object_scene_registration_module

# Default configuration: auto mesh/pose enhancement with box-only prompting
# This avoids forwarding garbage YOLO-E labels to the hosted service
demo_object_scene_registration = autoconnect(
    object_scene_registration_module(
        mesh_pose_service_url="http://localhost:8080",
        auto_mesh_pose=True,  # Auto-enhance all detections
        mesh_pose_use_box_prompt=True,  # Use box-only (ignore YOLO-E labels)
        mesh_store_dir="/home/dimensional/dimos/meshes",
        mesh_marker_topic="/object_detections/mesh_markers",
    ),
)

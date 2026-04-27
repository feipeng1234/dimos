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

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.perception.optical_flow.optical_flow_module import OpticalFlowModule
from dimos.visualization.rerun.bridge import RerunBridgeModule


def _flow_only_blueprint() -> Any:
    """Single 2D view of the OpticalFlowModule's annotated flow_visualization."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial2DView(origin="world/flow_visualization", name="Optical Flow"),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


# Webcam → optical flow τ-based obstacle avoidance, visualized in Rerun.
# angular_velocity is left unconnected: with no IMU/odom on a bare webcam,
# the danger gating reduces to the raw backend signal.
webcam_optical_flow = autoconnect(
    CameraModule.blueprint(),
    OpticalFlowModule.blueprint(),
    RerunBridgeModule.blueprint(blueprint=_flow_only_blueprint),
).global_config(n_workers=2)

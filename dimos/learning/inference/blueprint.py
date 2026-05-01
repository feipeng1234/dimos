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

"""ACT inference blueprint. ActionReplayer is registered by the per-robot
coordinator blueprint (passed in below). v1 placeholder uses the existing
teleop coordinator; replace with a coordinator that registers ActionReplayer.
"""

from __future__ import annotations

from dimos.control.blueprints.teleop import coordinator_teleop_xarm7
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.learning.inference.chunk_policy_module import ChunkPolicyModule
from dimos.learning.policy.base import ActionChunk
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState

_T_COLOR_IMAGE = "/camera/color_image"
_T_JOINT_STATE = "/coordinator/joint_state"
_T_ACTION_CHUNK = "/learning/action_chunk"


learning_infer_xarm7 = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(
        policy_path="data/runs/act_pick_red",
        inference_rate_hz=30.0,
    ),
    coordinator_teleop_xarm7,  # TODO: replace with coordinator_action_replayer_xarm7
).transports(
    {
        ("color_image", Image): LCMTransport(_T_COLOR_IMAGE, Image),
        ("joint_state", JointState): LCMTransport(_T_JOINT_STATE, JointState),
        ("action_chunk", ActionChunk): LCMTransport(_T_ACTION_CHUNK, ActionChunk),
    }
)


__all__ = ["learning_infer_xarm7"]

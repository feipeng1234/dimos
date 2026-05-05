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

"""ACT inference blueprints.

`ChunkPolicyModule` publishes `joint_command` directly so a coordinator's
servo / position task can consume it without an `ActionReplayer` task in
the tick loop. Compose with the user's coordinator blueprint at the call
site, e.g.::

    autoconnect(learning_infer_chunkpolicy_only, my_servo_coordinator)
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.learning.inference.chunk_policy_module import ChunkPolicyModule
from dimos.learning.policy.base import ActionChunk
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState

# Stable topics so external tools (lcmspy, dimos topic echo) work without rebuild.
_T_COLOR_IMAGE   = "/camera/color_image"
_T_JOINT_STATE   = "/coordinator/joint_state"
_T_ACTION_CHUNK  = "/learning/action_chunk"
_T_JOINT_COMMAND = "/teleop/joint_command"  # matches coordinator_servo_* default

_INFER_TRANSPORTS = {
    ("color_image",   Image):       LCMTransport(_T_COLOR_IMAGE,   Image),
    ("joint_state",   JointState):  LCMTransport(_T_JOINT_STATE,   JointState),
    ("action_chunk",  ActionChunk): LCMTransport(_T_ACTION_CHUNK,  ActionChunk),
    ("joint_command", JointState):  LCMTransport(_T_JOINT_COMMAND, JointState),
}


learning_infer_chunkpolicy_only = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(
        policy_path="data/runs/act_pickplace_001",
        inference_rate_hz=30.0,
    ),
).transports(_INFER_TRANSPORTS)


__all__ = ["learning_infer_chunkpolicy_only"]

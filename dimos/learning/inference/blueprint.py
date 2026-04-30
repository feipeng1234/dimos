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

"""Inference blueprints for the DimOS Learning Framework.

Each blueprint composes:
    Camera (publishes color_image)
    ChunkPolicyModule (consumes obs, publishes ActionChunk at policy rate)
    ControlCoordinator with ActionReplayer task (replays chunks at 100 Hz)
    Hardware (consumes joint_command)

The same blueprint serves both ACT (vision) and pi0/pi0.5 (vision +
language) — `ChunkPolicyModule` auto-detects from the loaded checkpoint
whether the policy expects language. For VLA, an LLM agent skill or a
language-source Module publishes to the `language_text` topic.

Note: ActionReplayer is a ControlTask, not a Module. It runs inside the
ControlCoordinator and is registered via `task_type="action_replayer"` in
the coordinator's task config. The coordinator variants below currently
reference the existing teleop-IK coordinator blueprints; v1 implementation
adds learning-specific coordinator blueprints under
`dimos/control/blueprints/learning.py` that swap teleop_ik for action_replayer
(see plan §7 critical files).

Usage:
    dimos run learning-infer-xarm7 \\
        --ChunkPolicyModule.config.spec_path dataset.yaml \\
        --ChunkPolicyModule.config.policy_path runs/act_pick_red \\
        --ChunkPolicyModule.config.inference_rate_hz 30
"""

from __future__ import annotations

from dimos.control.blueprints.teleop import (
    coordinator_teleop_piper,
    coordinator_teleop_xarm6,
    coordinator_teleop_xarm7,
)
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.learning.inference.chunk_policy_module import ChunkPolicyModule
from dimos.learning.policy.base import ActionChunk
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState

# Topics shared across variants.
_T_COLOR_IMAGE = "/camera/color_image"
_T_JOINT_STATE = "/coordinator/joint_state"
_T_LANGUAGE = "/learning/language_text"
_T_ACTION_CHUNK = "/learning/action_chunk"


# ── XArm7 (ACT-rate, 30 Hz) ──────────────────────────────────────────────────

learning_infer_xarm7 = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(inference_rate_hz=30.0),
    coordinator_teleop_xarm7,  # TODO: replace with coordinator_action_replayer_xarm7
).transports(
    {
        ("color_image", Image): LCMTransport(_T_COLOR_IMAGE, Image),
        ("joint_state", JointState): LCMTransport(_T_JOINT_STATE, JointState),
        ("language_text", str): LCMTransport(_T_LANGUAGE, str),
        ("action_chunk", ActionChunk): LCMTransport(_T_ACTION_CHUNK, ActionChunk),
    }
)


# ── Piper (ACT-rate) ─────────────────────────────────────────────────────────

learning_infer_piper = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(inference_rate_hz=30.0),
    coordinator_teleop_piper,
).transports(
    {
        ("color_image", Image): LCMTransport(_T_COLOR_IMAGE, Image),
        ("joint_state", JointState): LCMTransport(_T_JOINT_STATE, JointState),
        ("language_text", str): LCMTransport(_T_LANGUAGE, str),
        ("action_chunk", ActionChunk): LCMTransport(_T_ACTION_CHUNK, ActionChunk),
    }
)


# ── XArm6 (ACT-rate) ─────────────────────────────────────────────────────────

learning_infer_xarm6 = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(inference_rate_hz=30.0),
    coordinator_teleop_xarm6,
).transports(
    {
        ("color_image", Image): LCMTransport(_T_COLOR_IMAGE, Image),
        ("joint_state", JointState): LCMTransport(_T_JOINT_STATE, JointState),
        ("language_text", str): LCMTransport(_T_LANGUAGE, str),
        ("action_chunk", ActionChunk): LCMTransport(_T_ACTION_CHUNK, ActionChunk),
    }
)


# ── XArm7 (VLA-rate, 5 Hz) ───────────────────────────────────────────────────
# Same wiring; only the policy thread rate differs. pi0/pi0.5 are slow
# enough that running them at 30 Hz wastes GPU.

learning_infer_vla_xarm7 = autoconnect(
    RealSenseCamera.blueprint(enable_pointcloud=False),
    ChunkPolicyModule.blueprint(inference_rate_hz=5.0),
    coordinator_teleop_xarm7,
).transports(
    {
        ("color_image", Image): LCMTransport(_T_COLOR_IMAGE, Image),
        ("joint_state", JointState): LCMTransport(_T_JOINT_STATE, JointState),
        ("language_text", str): LCMTransport(_T_LANGUAGE, str),
        ("action_chunk", ActionChunk): LCMTransport(_T_ACTION_CHUNK, ActionChunk),
    }
)


__all__ = [
    "learning_infer_piper",
    "learning_infer_vla_xarm7",
    "learning_infer_xarm6",
    "learning_infer_xarm7",
]

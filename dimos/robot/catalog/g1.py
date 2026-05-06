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

"""Unitree G1 catalog entries for the manipulation stack.

Treats one G1 arm as a stationary 7-DOF manipulator rooted at
``torso_link``.  The G1 ``WHOLE_BODY`` HardwareComponent already
publishes joint state under dimos canonical names
(``g1/left_shoulder_pitch``, …) but the G1 URDF uses the upstream
Unitree names (``left_shoulder_pitch_joint``, …) — we expose
``joint_name_mapping`` so the manipulation module can translate
between the two.

Caveats:
- Base motion: IK assumes the torso is static.  The robot must not
  be walking while a manipulation trajectory executes.
- Gripper: the G1 hand is articulated (14 finger joints), not a
  binary gripper.  No gripper config attached.
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.control.coordinator import TaskConfig
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.data import LfsPath

# URDF + meshes shipped under data/g1_urdf/.  Mirrors the structure
# of the cached newton-assets package; references like
# ``package://unitree_g1/meshes/...`` resolve via ``package_paths``.
_G1_URDF = LfsPath("g1_urdf/g1.urdf")
_G1_PACKAGE_DIR = LfsPath("g1_urdf")

# (URDF joint, dimos canonical joint) per arm.  The dimos names are
# what ``make_humanoid_joints("g1")`` emits — we hand-mirror them
# rather than slicing g1_arms to keep this file standalone.
_LEFT_ARM_JOINT_PAIRS = [
    ("left_shoulder_pitch_joint", "g1/left_shoulder_pitch"),
    ("left_shoulder_roll_joint", "g1/left_shoulder_roll"),
    ("left_shoulder_yaw_joint", "g1/left_shoulder_yaw"),
    ("left_elbow_joint", "g1/left_elbow"),
    ("left_wrist_roll_joint", "g1/left_wrist_roll"),
    ("left_wrist_pitch_joint", "g1/left_wrist_pitch"),
    ("left_wrist_yaw_joint", "g1/left_wrist_yaw"),
]
_RIGHT_ARM_JOINT_PAIRS = [
    ("right_shoulder_pitch_joint", "g1/right_shoulder_pitch"),
    ("right_shoulder_roll_joint", "g1/right_shoulder_roll"),
    ("right_shoulder_yaw_joint", "g1/right_shoulder_yaw"),
    ("right_elbow_joint", "g1/right_elbow"),
    ("right_wrist_roll_joint", "g1/right_wrist_roll"),
    ("right_wrist_pitch_joint", "g1/right_wrist_pitch"),
    ("right_wrist_yaw_joint", "g1/right_wrist_yaw"),
]


@dataclass(frozen=True)
class G1ArmCatalogEntry:
    """Pre-configured pair the blueprint composes into the sim.

    ``robot_model_config`` is the manipulation-module side (URDF joint
    names + the coord<->urdf mapping for state translation).
    ``task_config`` is the coordinator side (dimos canonical joint
    names so the trajectory task claims the right joints).
    """

    name: str
    robot_model_config: RobotModelConfig
    task_config: TaskConfig


def _g1_arm(
    name: str,
    pairs: list[tuple[str, str]],
    end_effector_link: str,
    *,
    grasp_offset_xyz: tuple[float, float, float],
    side: str,
    task_priority: int = 20,
) -> G1ArmCatalogEntry:
    urdf_joints = [u for u, _ in pairs]
    coord_joints = [c for _, c in pairs]
    coord_to_urdf = {c: u for u, c in pairs}

    rmc = RobotModelConfig(
        name=name,
        model_path=_G1_URDF,
        joint_names=urdf_joints,
        end_effector_link=end_effector_link,
        grasp_offset_xyz=grasp_offset_xyz,
        # Pelvis is the floating base.  weld_base=False leaves it as a
        # 6-DOF free body in Drake; G1ManipulationModule pushes the live
        # /odom pose into the plant before each plan, so Drake's world
        # frame stays aligned with MuJoCo's world frame.  No frame
        # transforms are needed at the IK / obstacle / EE-pose layers.
        base_link="pelvis",
        weld_base=False,
        package_paths={"unitree_g1": _G1_PACKAGE_DIR},
        joint_name_mapping=coord_to_urdf,
        coordinator_task_name=f"traj_{name}",
        # The G1 URDF references mesh files as .STL, which Drake's
        # collision pipeline rejects (MakeConvexHull only takes .obj /
        # .vtk / .gltf).  auto-convert at parse time.
        auto_convert_meshes=True,
        # Required by the schema even though weld_base=False ignores it.
        base_pose=PoseStamped(
            position=Vector3(0.0, 0.0, 0.0),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
        max_velocity=1.0,
        max_acceleration=2.5,
        # Home pose: zero everywhere (matches ARM_DEFAULT_POSE).
        home_joints=[0.0] * len(urdf_joints),
        # The G1 URDF's shoulder_yaw_link mesh extends back into the
        # torso area; at zero joint position the meshes overlap by a
        # few mm and Drake reports a constant penetration that blocks
        # every plan with COLLISION_AT_START. These pairs are NOT real
        # — they're URDF mesh artifacts of the structural connection
        # between torso and shoulder.
        collision_exclusion_pairs=[
            ("torso_link", f"{side}_shoulder_yaw_link"),
            ("torso_link", f"{side}_shoulder_roll_link"),
        ],
    )

    task = TaskConfig(
        name=f"traj_{name}",
        type="trajectory",
        joint_names=coord_joints,
        priority=task_priority,
    )

    return G1ArmCatalogEntry(name=name, robot_model_config=rmc, task_config=task)


# Calibrated grasp-center offsets from each wrist_yaw_link's origin.
# Cribbed from Matrix's hand_frames.py (palm_offset + grasp_center_offset),
# which were measured against the same Unitree G1 mesh.  The wrist_yaw link
# itself sits ~13 cm behind the palm grasp point, so without these offsets
# IK loses 13 cm of effective reach and aims at the wrist instead of where
# the fingers actually close.
_LEFT_GRASP_CENTER_FROM_WRIST_YAW = (0.12, -0.05, 0.0)
_RIGHT_GRASP_CENTER_FROM_WRIST_YAW = (0.12, 0.05, 0.0)


def g1_left_arm(name: str = "left_arm") -> G1ArmCatalogEntry:
    """Default name "left_arm" rather than "g1_left_arm" because LLMs reach
    for the natural English name first when the user says "the left arm"."""
    return _g1_arm(
        name,
        _LEFT_ARM_JOINT_PAIRS,
        "left_wrist_yaw_link",
        grasp_offset_xyz=_LEFT_GRASP_CENTER_FROM_WRIST_YAW,
        side="left",
    )


def g1_right_arm(name: str = "right_arm") -> G1ArmCatalogEntry:
    return _g1_arm(
        name,
        _RIGHT_ARM_JOINT_PAIRS,
        "right_wrist_yaw_link",
        grasp_offset_xyz=_RIGHT_GRASP_CENTER_FROM_WRIST_YAW,
        side="right",
    )


__all__ = ["G1ArmCatalogEntry", "g1_left_arm", "g1_right_arm"]

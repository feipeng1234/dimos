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

"""Drake optimization-based IK using SNOPT/IPOPT. Requires DrakeWorld."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dimos.manipulation.planning.spec.enums import IKStatus
from dimos.manipulation.planning.spec.models import IKResult, WorldRobotID
from dimos.manipulation.planning.spec.protocols import WorldSpec
from dimos.manipulation.planning.utils.kinematics_utils import compute_pose_error
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger
from dimos.utils.transform_utils import pose_to_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.multibody.inverse_kinematics import InverseKinematics
    from pydrake.solvers import Solve

    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False

logger = setup_logger()


class DrakeOptimizationIK:
    """Drake optimization-based IK solver using constrained nonlinear optimization.

    Requires DrakeWorld. For backend-agnostic IK, use JacobianIK.

    Two solve modes:
      - ``solve``: full 6-DOF pose constraint (position + orientation).
      - ``solve_pointing``: position + "EE forward axis points within
        ``angle_tolerance`` of a world direction". Leaves the rotation
        about the pointing axis free, dramatically widening the
        feasible set for "point at" gestures.

    For floating-base robots (e.g. G1 with ``weld_base=False``), this
    class locks the floating-base positions AND all non-arm joints to
    their current values via bounding-box constraints, so the optimizer
    can only vary the arm joints. Without these locks the optimizer
    would happily move the pelvis or bend the legs to satisfy the EE
    constraint.
    """

    def __init__(self) -> None:
        if not DRAKE_AVAILABLE:
            raise ImportError("Drake is not installed. Install with: pip install drake")

    def _validate_world(self, world: WorldSpec) -> IKResult | None:
        from dimos.manipulation.planning.world.drake_world import DrakeWorld

        if not isinstance(world, DrakeWorld):
            return _create_failure_result(
                IKStatus.NO_SOLUTION, "DrakeOptimizationIK requires DrakeWorld"
            )
        if not world.is_finalized:
            return _create_failure_result(IKStatus.NO_SOLUTION, "World must be finalized before IK")
        return None

    def _ee_offset(self, world: WorldSpec, robot_id: WorldRobotID) -> NDArray[np.float64]:
        """Get the grasp_offset (in EE body frame) from the world's robot data."""
        rd = world._robots[robot_id]  # type: ignore[attr-defined]
        offset = getattr(rd, "grasp_offset_in_body", None)
        if offset is None:
            return np.zeros(3)
        return np.asarray(offset, dtype=np.float64)

    def _full_seed_and_locks(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        arm_seed: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], list[int], list[int]]:
        """Build a full plant-position vector that uses ``arm_seed`` for
        this robot's arm joints and the live state for everything else.

        Returns:
            (full_positions, arm_position_indices, locked_position_indices)
            ``locked_position_indices`` is the complement of arm indices —
            the floating base + every other joint, all to be pinned at
            their live values during the IK solve.
        """
        rd = world._robots[robot_id]  # type: ignore[attr-defined]
        plant = world.plant  # type: ignore[attr-defined]
        live_ctx = world._plant_context  # type: ignore[attr-defined]
        full = plant.GetPositions(live_ctx).copy()
        arm_indices = list(rd.joint_indices)
        for i, joint_idx in enumerate(arm_indices):
            full[joint_idx] = arm_seed[i]
        all_indices = set(range(plant.num_positions()))
        locked = sorted(all_indices - set(arm_indices))
        return full, arm_indices, locked

    def solve(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        target_pose: PoseStamped,
        seed: JointState | None = None,
        position_tolerance: float = 0.001,
        orientation_tolerance: float = 0.01,
        check_collision: bool = True,
        max_attempts: int = 10,
    ) -> IKResult:
        """Full pose IK (position + orientation). Multi-restart with random seeds on failure."""
        err = self._validate_world(world)
        if err is not None:
            return err

        target_matrix = Transform(
            translation=target_pose.position,
            rotation=target_pose.orientation,
        ).to_matrix()
        target_transform = RigidTransform(target_matrix)

        lower_limits, upper_limits = world.get_joint_limits(robot_id)

        if seed is None:
            with world.scratch_context() as ctx:
                seed = world.get_joint_state(ctx, robot_id)
        joint_names = seed.name
        seed_positions = np.array(seed.position, dtype=np.float64)

        best_result: IKResult | None = None
        best_error = float("inf")

        for attempt in range(max_attempts):
            current_seed = (
                seed_positions if attempt == 0 else np.random.uniform(lower_limits, upper_limits)
            )
            result = self._solve_pose(
                world=world,
                robot_id=robot_id,
                target_transform=target_transform,
                seed=current_seed,
                joint_names=joint_names,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
            if result.is_success() and result.joint_state is not None:
                if check_collision and not world.check_config_collision_free(
                    robot_id, result.joint_state
                ):
                    continue
                total = result.position_error + result.orientation_error
                if total < best_error:
                    best_error = total
                    best_result = result
                if (
                    result.position_error <= position_tolerance
                    and result.orientation_error <= orientation_tolerance
                ):
                    return result

        if best_result is not None:
            return best_result
        return _create_failure_result(
            IKStatus.NO_SOLUTION, f"IK failed after {max_attempts} attempts"
        )

    def solve_pointing(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        ee_position: NDArray[np.float64],
        direction_world: NDArray[np.float64],
        seed: JointState | None = None,
        position_tolerance: float = 0.02,
        angle_tolerance: float = 0.2,
        check_collision: bool = True,
        max_attempts: int = 10,
    ) -> IKResult:
        """IK for "point the EE forward axis at a world direction".

        Constrains:
          1. Grasp center (EE body origin + grasp_offset) within
             ``position_tolerance`` of ``ee_position``.
          2. Angle between EE forward axis (= grasp_offset normalized
             in EE body frame) and ``direction_world`` is at most
             ``angle_tolerance`` rad.

        Critically, the rotation about the pointing axis is unconstrained
        — IK can pick whatever wrist roll makes the position reachable.
        This is the key difference from ``solve`` and is what unlocks
        directions where strict look-at orientation is infeasible.

        Args:
            ee_position: target world-frame position for grasp center
            direction_world: target world-frame direction (need not be unit)
            position_tolerance: meters (looser than full-pose IK; "point
                at" doesn't need millimeter accuracy)
            angle_tolerance: radians (default 0.2 ≈ 11.5°)
        """
        err = self._validate_world(world)
        if err is not None:
            return err

        ee_position = np.asarray(ee_position, dtype=np.float64).reshape(3)
        direction_world = np.asarray(direction_world, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(direction_world))
        if n < 1e-9:
            return _create_failure_result(IKStatus.NO_SOLUTION, "direction_world is zero")
        dir_unit = direction_world / n

        # EE forward axis in body frame = direction of grasp_offset.
        # Falls back to body +x if no offset is configured.
        offset = self._ee_offset(world, robot_id)
        if float(np.linalg.norm(offset)) < 1e-9:
            forward_body = np.array([1.0, 0.0, 0.0])
        else:
            forward_body = offset / np.linalg.norm(offset)

        lower_limits, upper_limits = world.get_joint_limits(robot_id)

        if seed is None:
            with world.scratch_context() as ctx:
                seed = world.get_joint_state(ctx, robot_id)
        joint_names = seed.name
        seed_positions = np.array(seed.position, dtype=np.float64)

        best_result: IKResult | None = None
        best_error = float("inf")

        for attempt in range(max_attempts):
            current_seed = (
                seed_positions if attempt == 0 else np.random.uniform(lower_limits, upper_limits)
            )
            result = self._solve_pointing_single(
                world=world,
                robot_id=robot_id,
                ee_position=ee_position,
                direction_world=dir_unit,
                forward_body=forward_body,
                seed=current_seed,
                joint_names=joint_names,
                position_tolerance=position_tolerance,
                angle_tolerance=angle_tolerance,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
            if result.is_success() and result.joint_state is not None:
                if check_collision and not world.check_config_collision_free(
                    robot_id, result.joint_state
                ):
                    continue
                # Score by position error + orientation error against
                # the implicit "forward = direction" target.
                total = result.position_error + result.orientation_error
                if total < best_error:
                    best_error = total
                    best_result = result
                if (
                    result.position_error <= position_tolerance
                    and result.orientation_error <= angle_tolerance
                ):
                    return result

        if best_result is not None:
            return best_result
        return _create_failure_result(
            IKStatus.NO_SOLUTION, f"solve_pointing failed after {max_attempts} attempts"
        )

    def _solve_pose(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        target_transform: RigidTransform,
        seed: NDArray[np.float64],
        joint_names: list[str],
        position_tolerance: float,
        orientation_tolerance: float,
        lower_limits: NDArray[np.float64],
        upper_limits: NDArray[np.float64],
    ) -> IKResult:
        rd = world._robots[robot_id]  # type: ignore[attr-defined]
        plant = world.plant  # type: ignore[attr-defined]
        ee_frame = rd.ee_frame
        offset = self._ee_offset(world, robot_id)

        ik = InverseKinematics(plant)

        # Position constraint at the grasp center, not the body origin
        ik.AddPositionConstraint(
            frameB=ee_frame,
            p_BQ=offset,  # type: ignore[arg-type]
            frameA=plant.world_frame(),
            p_AQ_lower=target_transform.translation() - np.array([position_tolerance] * 3),
            p_AQ_upper=target_transform.translation() + np.array([position_tolerance] * 3),
        )

        # Strict orientation
        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=target_transform.rotation(),
            frameBbar=ee_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=orientation_tolerance,
        )

        prog = ik.get_mutable_prog()
        q = ik.q()

        full_seed, arm_indices, locked_indices = self._full_seed_and_locks(world, robot_id, seed)
        prog.SetInitialGuess(q, full_seed)
        # Lock everything that isn't an arm joint to its live value
        if locked_indices:
            locked_vars = q[locked_indices]
            locked_values = full_seed[locked_indices]
            prog.AddBoundingBoxConstraint(locked_values, locked_values, locked_vars)

        result = Solve(prog)
        if not result.is_success():
            return _create_failure_result(
                IKStatus.NO_SOLUTION,
                f"Optimization failed: {result.get_solution_result()}",
            )

        full_solution = result.GetSolution(q)
        joint_solution = np.array([full_solution[idx] for idx in arm_indices])
        joint_solution = np.clip(joint_solution, lower_limits, upper_limits)

        # Verify with FK
        solution_state = JointState(name=joint_names, position=joint_solution.tolist())
        with world.scratch_context() as ctx:
            world.set_joint_state(ctx, robot_id, solution_state)
            actual_pose = world.get_ee_pose(ctx, robot_id)

        pos_err, ori_err = compute_pose_error(
            pose_to_matrix(actual_pose),
            target_transform.GetAsMatrix4(),  # type: ignore[arg-type]
        )
        return _create_success_result(
            joint_names=joint_names,
            joint_positions=joint_solution,
            position_error=pos_err,
            orientation_error=ori_err,
            iterations=1,
        )

    def _solve_pointing_single(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        ee_position: NDArray[np.float64],
        direction_world: NDArray[np.float64],
        forward_body: NDArray[np.float64],
        seed: NDArray[np.float64],
        joint_names: list[str],
        position_tolerance: float,
        angle_tolerance: float,
        lower_limits: NDArray[np.float64],
        upper_limits: NDArray[np.float64],
    ) -> IKResult:
        rd = world._robots[robot_id]  # type: ignore[attr-defined]
        plant = world.plant  # type: ignore[attr-defined]
        ee_frame = rd.ee_frame
        offset = self._ee_offset(world, robot_id)

        ik = InverseKinematics(plant)

        # Position: grasp center near ee_position
        ik.AddPositionConstraint(
            frameB=ee_frame,
            p_BQ=offset,  # type: ignore[arg-type]
            frameA=plant.world_frame(),
            p_AQ_lower=ee_position - np.array([position_tolerance] * 3),
            p_AQ_upper=ee_position + np.array([position_tolerance] * 3),
        )

        # Pointing: angle between EE forward axis and world direction <= tol
        ik.AddAngleBetweenVectorsConstraint(
            frameA=plant.world_frame(),
            na_A=direction_world,
            frameB=ee_frame,
            nb_B=forward_body,
            angle_lower=0.0,
            angle_upper=angle_tolerance,
        )

        prog = ik.get_mutable_prog()
        q = ik.q()
        full_seed, arm_indices, locked_indices = self._full_seed_and_locks(world, robot_id, seed)
        prog.SetInitialGuess(q, full_seed)
        if locked_indices:
            locked_vars = q[locked_indices]
            locked_values = full_seed[locked_indices]
            prog.AddBoundingBoxConstraint(locked_values, locked_values, locked_vars)

        result = Solve(prog)
        if not result.is_success():
            return _create_failure_result(
                IKStatus.NO_SOLUTION,
                f"solve_pointing optimization failed: {result.get_solution_result()}",
            )

        full_solution = result.GetSolution(q)
        joint_solution = np.array([full_solution[idx] for idx in arm_indices])
        joint_solution = np.clip(joint_solution, lower_limits, upper_limits)

        # Verify: compute actual position + actual angle
        solution_state = JointState(name=joint_names, position=joint_solution.tolist())
        with world.scratch_context() as ctx:
            world.set_joint_state(ctx, robot_id, solution_state)
            ee_pose = world.get_ee_pose(ctx, robot_id)

        actual_xyz = np.array(
            [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z], dtype=np.float64
        )
        pos_err = float(np.linalg.norm(actual_xyz - ee_position))

        # actual EE forward axis in world = R_W_E @ forward_body
        # Compose: R_W_E built from quaternion in returned PoseStamped
        q_xyzw = (
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        )
        R_we = _quat_to_rot_matrix(q_xyzw)
        forward_world_actual = R_we @ forward_body
        cos_ang = float(np.clip(np.dot(forward_world_actual, direction_world), -1.0, 1.0))
        ang_err = float(np.arccos(cos_ang))

        return _create_success_result(
            joint_names=joint_names,
            joint_positions=joint_solution,
            position_error=pos_err,
            orientation_error=ang_err,
            iterations=1,
        )


def _quat_to_rot_matrix(q_xyzw: tuple[float, float, float, float]) -> NDArray[np.float64]:
    x, y, z, w = q_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _create_success_result(
    joint_names: list[str],
    joint_positions: NDArray[np.float64],
    position_error: float,
    orientation_error: float,
    iterations: int,
) -> IKResult:
    return IKResult(
        status=IKStatus.SUCCESS,
        joint_state=JointState(name=joint_names, position=joint_positions.tolist()),
        position_error=position_error,
        orientation_error=orientation_error,
        iterations=iterations,
        message="IK solution found",
    )


def _create_failure_result(status: IKStatus, message: str, iterations: int = 0) -> IKResult:
    return IKResult(
        status=status,
        joint_state=None,
        iterations=iterations,
        message=message,
    )

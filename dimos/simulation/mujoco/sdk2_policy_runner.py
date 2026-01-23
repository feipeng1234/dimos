#!/usr/bin/env python3
"""SDK2 Policy Runner - runs ONNX policies over SDK2 interface.

This allows mjlab-trained policies to run on both simulation and real robots
using the same SDK2 interface (zero code change deployment).

Usage:
    # Simulation (domain_id=1, interface=lo0 on macOS, lo on Linux)
    python -m dimos.simulation.mujoco.sdk2_policy_runner policy.onnx

    # Real robot (domain_id=0, interface=eth0)
    python -m dimos.simulation.mujoco.sdk2_policy_runner policy.onnx --real eth0
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HG_LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HG_LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as GO_LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as GO_LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as HG_LowCmd_default,
    unitree_go_msg_dds__LowCmd_ as GO_LowCmd_default,
)

if TYPE_CHECKING:
    pass


# G1 motor index to joint name mapping (matches SDK2 LowCmd/LowState motor order)
# This is the order motors appear in msg.motor_state[i] and cmd.motor_cmd[i]
G1_MOTOR_JOINT_NAMES: list[str] = [
    # Left leg (0-5)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@dataclass
class PolicyMetadata:
    """Metadata extracted from ONNX policy."""

    joint_names: list[str]
    joint_stiffness: NDArray[np.floating]
    joint_damping: NDArray[np.floating]
    default_joint_pos: NDArray[np.floating]
    action_scale: NDArray[np.floating]
    observation_names: list[str]
    command_names: list[str]


def parse_csv_floats(s: str) -> NDArray[np.floating]:
    """Parse comma-separated floats from string."""
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def parse_csv_strings(s: str) -> list[str]:
    """Parse comma-separated strings."""
    return [x.strip() for x in s.split(",")]


def load_policy_metadata(session: ort.InferenceSession) -> PolicyMetadata:
    """Load metadata from ONNX policy."""
    meta = session.get_modelmeta().custom_metadata_map

    return PolicyMetadata(
        joint_names=parse_csv_strings(meta["joint_names"]),
        joint_stiffness=parse_csv_floats(meta["joint_stiffness"]),
        joint_damping=parse_csv_floats(meta["joint_damping"]),
        default_joint_pos=parse_csv_floats(meta["default_joint_pos"]),
        action_scale=parse_csv_floats(meta["action_scale"]),
        observation_names=parse_csv_strings(meta["observation_names"]),
        command_names=parse_csv_strings(meta["command_names"]),
    )


class SDK2PolicyRunner:
    """Runs ONNX policies over SDK2 interface."""

    def __init__(
        self,
        policy_path: str,
        robot_type: str = "g1",
        domain_id: int = 1,
        interface: str = "lo0",
        control_dt: float = 0.02,  # 50 Hz
    ) -> None:
        self.robot_type = robot_type
        self.control_dt = control_dt
        self.num_joints = 29 if robot_type == "g1" else 12

        # Load ONNX policy
        self.session = ort.InferenceSession(policy_path)
        self.metadata = load_policy_metadata(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.obs_dim = self.session.get_inputs()[0].shape[1]

        # State buffers
        self.joint_pos = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_vel = np.zeros(self.num_joints, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z
        self.prev_actions = np.zeros(self.num_joints, dtype=np.float32)

        # Command (velocity target from external source)
        self.command = np.zeros(3, dtype=np.float32)  # vx, vy, wz

        # Safety/arming state
        self._enabled = False
        self._estop = False
        # When disabled we publish a "hold current pose" command; keep gains at full strength
        # so the robot doesn't go limp/collapse before the UI enables the policy.
        self._hold_kp_scale = 1.0

        # Thread lock for safe access to sensor data (callbacks run in background threads)
        self._data_lock = threading.Lock()

        # For humanoid (unitree_hg) LowCmd fields: mode_machine/mode_pr must be set consistently.
        # We mirror mode_machine from LowState and default to PR mode (0) unless overridden.
        self._mode_machine: int = 0
        self._mode_pr: int = 0  # Mode.PR from Unitree examples

        # Initialize SDK2
        ChannelFactoryInitialize(domain_id, interface)

        # Select message types based on robot
        if robot_type == "g1":
            self.LowCmd = HG_LowCmd_
            self.LowState = HG_LowState_
            self._create_lowcmd = HG_LowCmd_default
        else:
            self.LowCmd = GO_LowCmd_
            self.LowState = GO_LowState_
            self._create_lowcmd = GO_LowCmd_default

        # Initialize state BEFORE starting subscribers (callbacks can fire immediately)
        self._state_received = False
        self._debug_counter = 0

        # Build joint name mappings for proper ordering
        self._build_joint_mappings()

        # Publishers and subscribers
        self.cmd_pub = ChannelPublisher("rt/lowcmd", self.LowCmd)
        self.cmd_pub.Init()

        self.state_sub = ChannelSubscriber("rt/lowstate", self.LowState)
        self.state_sub.Init(self._lowstate_callback, 10)

        self.sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sport_sub.Init(self._sportstate_callback, 10)

        print(f"SDK2 Policy Runner initialized")
        print(f"  Robot: {robot_type}")
        print(f"  Domain ID: {domain_id}")
        print(f"  Interface: {interface}")
        print(f"  Policy: {policy_path}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Joints: {self.num_joints}")
        print(f"  Default pos (legs 0-5): {self.metadata.default_joint_pos[:6]}")

    def _build_joint_mappings(self) -> None:
        """Build mappings between motor indices and policy joint indices.

        The SDK2 motor_cmd/motor_state arrays have a fixed ordering (G1_MOTOR_JOINT_NAMES).
        The ONNX policy may have a different joint ordering (metadata.joint_names).

        We build two mappings:
        - motor_to_policy: motor_idx -> policy_idx (for applying actions)
        - policy_to_motor: policy_idx -> motor_idx (for reading observations)
        """
        if self.robot_type == "g1":
            motor_joint_names = G1_MOTOR_JOINT_NAMES
        else:
            # GO2 has 12 joints - would need similar mapping
            raise NotImplementedError(f"Joint mapping not implemented for {self.robot_type}")

        # Build policy joint name to index mapping
        policy_name_to_idx = {name: i for i, name in enumerate(self.metadata.joint_names)}

        # motor_to_policy[motor_idx] = policy_idx
        # Used when applying actions: policy output at policy_idx goes to motor motor_idx
        self._motor_to_policy: list[int] = []
        for motor_idx, joint_name in enumerate(motor_joint_names):
            if joint_name not in policy_name_to_idx:
                raise ValueError(
                    f"Motor joint '{joint_name}' (motor idx {motor_idx}) "
                    f"not found in policy joint_names: {self.metadata.joint_names}"
                )
            self._motor_to_policy.append(policy_name_to_idx[joint_name])

        # policy_to_motor[policy_idx] = motor_idx
        # Used when building observations: motor state at motor_idx goes to obs position policy_idx
        motor_name_to_idx = {name: i for i, name in enumerate(motor_joint_names)}
        self._policy_to_motor: list[int] = []
        for policy_idx, joint_name in enumerate(self.metadata.joint_names):
            if joint_name not in motor_name_to_idx:
                raise ValueError(
                    f"Policy joint '{joint_name}' (policy idx {policy_idx}) "
                    f"not found in motor joint names"
                )
            self._policy_to_motor.append(motor_name_to_idx[joint_name])

        # Log any ordering differences
        if self._motor_to_policy != list(range(self.num_joints)):
            print(f"  Joint order remapping active (policy order differs from motor order)")

    def _lowstate_callback(self, msg: HG_LowState_ | GO_LowState_) -> None:
        """Handle LowState message."""
        with self._data_lock:
            for i in range(self.num_joints):
                self.joint_pos[i] = msg.motor_state[i].q
                self.joint_vel[i] = msg.motor_state[i].dq

            # Humanoid (hg) state carries the current machine mode; copy it so LowCmd is accepted.
            if hasattr(msg, "mode_machine"):
                try:
                    self._mode_machine = int(msg.mode_machine)
                except Exception:
                    pass

            # IMU data
            self.imu_quat[0] = msg.imu_state.quaternion[0]  # w
            self.imu_quat[1] = msg.imu_state.quaternion[1]  # x
            self.imu_quat[2] = msg.imu_state.quaternion[2]  # y
            self.imu_quat[3] = msg.imu_state.quaternion[3]  # z

            self.base_ang_vel[0] = msg.imu_state.gyroscope[0]
            self.base_ang_vel[1] = msg.imu_state.gyroscope[1]
            self.base_ang_vel[2] = msg.imu_state.gyroscope[2]

            if not self._state_received:
                # First state received - verify robot at keyframe pose
                expected = [-0.312, 0, 0, 0.669, -0.363, 0]
                actual = self.joint_pos[:6]
                match = all(abs(a - e) < 0.01 for a, e in zip(actual, expected))
                if match:
                    print(f"  Initial pose: OK (at keyframe)")
                else:
                    print(f"  WARNING: Initial pose mismatch!")
                    print(f"    Actual: {actual}")
                    print(f"    Expected: {expected}")
            self._state_received = True

    def _sportstate_callback(self, msg: SportModeState_) -> None:
        """Handle SportModeState message for velocity estimation."""
        with self._data_lock:
            self.base_lin_vel[0] = msg.velocity[0]
            self.base_lin_vel[1] = msg.velocity[1]
            self.base_lin_vel[2] = msg.velocity[2]

    def _quat_to_projected_gravity(self, quat: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert quaternion to projected gravity vector in body frame.

        World gravity is [0, 0, -1] (pointing down). When the robot is upright,
        the body-frame gravity should be [0, 0, -1].
        """
        w, x, y, z = quat
        # Rotate world gravity [0, 0, -1] to body frame using quaternion
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -1.0 + 2.0 * (x * x + y * y)
        return np.array([gx, gy, gz], dtype=np.float32)

    def _world_to_body_velocity(
        self, world_vel: NDArray[np.floating], quat: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Transform velocity from world frame to body frame using quaternion.

        Args:
            world_vel: Velocity in world frame [vx, vy, vz]
            quat: Quaternion [w, x, y, z] representing body orientation

        Returns:
            Velocity in body frame [vx, vy, vz]
        """
        w, x, y, z = quat

        # Rotation matrix R (body to world) from quaternion
        # To go world→body, we use R^T (transpose)
        # R^T @ world_vel = body_vel

        # R^T row 0: [1-2(y²+z²), 2(xy+wz), 2(xz-wy)]
        # R^T row 1: [2(xy-wz), 1-2(x²+z²), 2(yz+wx)]
        # R^T row 2: [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]

        vx = (1 - 2*(y*y + z*z)) * world_vel[0] + 2*(x*y + w*z) * world_vel[1] + 2*(x*z - w*y) * world_vel[2]
        vy = 2*(x*y - w*z) * world_vel[0] + (1 - 2*(x*x + z*z)) * world_vel[1] + 2*(y*z + w*x) * world_vel[2]
        vz = 2*(x*z + w*y) * world_vel[0] + 2*(y*z - w*x) * world_vel[1] + (1 - 2*(x*x + y*y)) * world_vel[2]

        return np.array([vx, vy, vz], dtype=np.float32)

    def _build_observation(self) -> NDArray[np.floating]:
        """Build observation vector for policy.

        Thread-safe: acquires lock before reading sensor data.
        Applies clipping to prevent extreme values from destabilizing policy.

        IMPORTANT: Joint data from SDK2 is in MOTOR order, but the policy expects
        joints in POLICY order. We use _policy_to_motor to reorder.
        """
        with self._data_lock:
            # Copy sensor data under lock (joint data is in MOTOR order)
            base_lin_vel_world = self.base_lin_vel.copy()  # This is in world frame from framelinvel
            base_ang_vel = self.base_ang_vel.copy()
            imu_quat = self.imu_quat.copy()
            joint_pos_motor_order = self.joint_pos.copy()
            joint_vel_motor_order = self.joint_vel.copy()

        # Reorder joint data from motor order to policy order
        # policy_to_motor[policy_idx] = motor_idx
        # So joint_pos_policy[policy_idx] = joint_pos_motor[policy_to_motor[policy_idx]]
        joint_pos = np.array([joint_pos_motor_order[self._policy_to_motor[i]]
                              for i in range(self.num_joints)], dtype=np.float32)
        joint_vel = np.array([joint_vel_motor_order[self._policy_to_motor[i]]
                              for i in range(self.num_joints)], dtype=np.float32)

        # Transform world-frame velocity to body-frame (policy expects body frame)
        base_lin_vel = self._world_to_body_velocity(base_lin_vel_world, imu_quat)

        # Order: base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command
        proj_gravity = self._quat_to_projected_gravity(imu_quat)

        # Normalize joint positions relative to default (both now in policy order)
        joint_pos_normalized = joint_pos - self.metadata.default_joint_pos

        # Command (no scaling - mjlab policy expects raw values)
        obs = np.concatenate([
            base_lin_vel,                # 3
            base_ang_vel,                # 3
            proj_gravity,                # 3
            joint_pos_normalized,        # num_joints (policy order)
            joint_vel,                   # num_joints (policy order)
            self.prev_actions,           # num_joints (policy order)
            self.command,                # 3 (vx, vy, wz) - no scaling
        ]).astype(np.float32)

        # Clip observation to prevent extreme values from destabilizing policy
        obs = np.clip(obs, -100.0, 100.0)

        return obs.reshape(1, -1)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Set velocity command."""
        self.command[0] = vx
        self.command[1] = vy
        self.command[2] = wz

    def set_enabled(self, enabled: bool) -> None:
        with self._data_lock:
            self._enabled = bool(enabled)
        print(f"[SDK2PolicyRunner] enabled={self._enabled}")

    def set_estop(self, estop: bool) -> None:
        with self._data_lock:
            self._estop = bool(estop)
            if self._estop:
                # When E-stop is asserted, force policy disabled.
                self._enabled = False
        print(f"[SDK2PolicyRunner] estop={self._estop} enabled={self._enabled}")

    def step(self) -> None:
        """Run one policy step."""
        if not self._state_received:
            return

        with self._data_lock:
            enabled = bool(self._enabled)
            estop = bool(self._estop)
            mode_machine = int(self._mode_machine)
            mode_pr = int(self._mode_pr)
            # Snapshot current motor-order joint state for safe/hold commands.
            joint_pos_motor_order = self.joint_pos.copy()
            joint_vel_motor_order = self.joint_vel.copy()

        # Build observation
        obs = self._build_observation()

        # Debug: print first few observations to verify everything is working
        self._debug_counter += 1
        if self._debug_counter <= 3:
            obs_flat = obs.flatten()
            joint_pos_norm = obs_flat[9:15]  # First 6 normalized joint positions
            # Normalized positions should be near 0 if robot is at default pose
            max_deviation = max(abs(p) for p in joint_pos_norm)
            print(f"  Step {self._debug_counter}: gravity={obs_flat[6:9]}, joint_norm_max={max_deviation:.3f}")

        # If E-stop: publish "limp" command (kp=kd=tau=0).
        if estop:
            cmd = self._create_lowcmd()
            if hasattr(cmd, "mode_machine"):
                cmd.mode_machine = mode_machine
            if hasattr(cmd, "mode_pr"):
                cmd.mode_pr = mode_pr

            for motor_idx in range(self.num_joints):
                # HG: mode is enable(1)/disable(0). GO: mode is motor control mode.
                cmd.motor_cmd[motor_idx].mode = 0
                cmd.motor_cmd[motor_idx].q = float(joint_pos_motor_order[motor_idx])
                cmd.motor_cmd[motor_idx].dq = 0.0
                cmd.motor_cmd[motor_idx].kp = 0.0
                cmd.motor_cmd[motor_idx].kd = 0.0
                cmd.motor_cmd[motor_idx].tau = 0.0

            self.cmd_pub.Write(cmd)
            return

        # If not enabled: publish "hold current pose" command with modest gains.
        if not enabled:
            cmd = self._create_lowcmd()
            if hasattr(cmd, "mode_machine"):
                cmd.mode_machine = mode_machine
            if hasattr(cmd, "mode_pr"):
                cmd.mode_pr = mode_pr

            for motor_idx in range(self.num_joints):
                policy_idx = self._motor_to_policy[motor_idx]
                kp = float(self.metadata.joint_stiffness[policy_idx]) * float(self._hold_kp_scale)
                kd = float(self.metadata.joint_damping[policy_idx]) * float(self._hold_kp_scale)
                cmd.motor_cmd[motor_idx].mode = 1
                cmd.motor_cmd[motor_idx].q = float(joint_pos_motor_order[motor_idx])
                cmd.motor_cmd[motor_idx].dq = 0.0
                cmd.motor_cmd[motor_idx].kp = kp
                cmd.motor_cmd[motor_idx].kd = kd
                cmd.motor_cmd[motor_idx].tau = 0.0

            self.cmd_pub.Write(cmd)
            return

        # Run inference - actions are in POLICY joint order
        actions = self.session.run(None, {self.input_name: obs})[0][0]

        # Debug: print action range
        if self._debug_counter <= 3:
            print(f"           actions range: [{actions.min():.3f}, {actions.max():.3f}]")

        # Clip actions to prevent extreme motor commands
        actions = np.clip(actions, -10.0, 10.0)
        self.prev_actions = actions.copy()

        # Build and send command (use factory function for proper initialization)
        cmd = self._create_lowcmd()

        # Set header (required for real robot)
        if hasattr(cmd, "mode_machine"):
            cmd.mode_machine = mode_machine
        if hasattr(cmd, "mode_pr"):
            cmd.mode_pr = mode_pr

        # Apply actions using joint name mapping
        # motor_cmd[motor_idx] gets the action from policy index _motor_to_policy[motor_idx]
        for motor_idx in range(self.num_joints):
            policy_idx = self._motor_to_policy[motor_idx]

            # Get action, default position, and gains for this joint (in policy order)
            action = actions[policy_idx]
            default_pos = self.metadata.default_joint_pos[policy_idx]
            action_scale = self.metadata.action_scale[policy_idx]
            kp = self.metadata.joint_stiffness[policy_idx]
            kd = self.metadata.joint_damping[policy_idx]

            # Compute target position
            target_pos = default_pos + action * action_scale

            cmd.motor_cmd[motor_idx].mode = 1  # HG: enable, GO: typically PMSM
            cmd.motor_cmd[motor_idx].q = float(target_pos)
            cmd.motor_cmd[motor_idx].kp = float(kp)
            cmd.motor_cmd[motor_idx].dq = 0.0
            cmd.motor_cmd[motor_idx].kd = float(kd)
            cmd.motor_cmd[motor_idx].tau = 0.0

        # CRC for real robot
        if hasattr(cmd, "crc"):
            try:
                from unitree_sdk2py.utils.crc import CRC
                crc = CRC()
                cmd.crc = crc.Crc(cmd)
            except Exception:
                pass

        self.cmd_pub.Write(cmd)

    def run(self) -> None:
        """Run policy loop."""
        print("Waiting for robot state...")
        while not self._state_received:
            time.sleep(0.01)

        print("Running policy (Ctrl+C to stop)...")
        try:
            while True:
                step_start = time.perf_counter()
                self.step()
                elapsed = time.perf_counter() - step_start
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopping policy...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX policy over SDK2")
    parser.add_argument("policy", help="Path to ONNX policy file")
    parser.add_argument("--robot", default="g1", choices=["g1", "go2"], help="Robot type")
    parser.add_argument("--real", metavar="INTERFACE", help="Run on real robot with given interface")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command")
    parser.add_argument("--wz", type=float, default=0.0, help="Angular velocity command")
    parser.add_argument("--hz", type=float, default=50.0, help="Control frequency")

    args = parser.parse_args()

    if args.real:
        domain_id = 0
        interface = args.real
    else:
        domain_id = 1
        interface = "lo0"

    runner = SDK2PolicyRunner(
        policy_path=args.policy,
        robot_type=args.robot,
        domain_id=domain_id,
        interface=interface,
        control_dt=1.0 / args.hz,
    )
    runner.set_command(args.vx, args.vy, args.wz)
    runner.run()


if __name__ == "__main__":
    main()

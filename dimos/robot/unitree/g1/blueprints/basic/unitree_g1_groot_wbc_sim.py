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

"""Unitree G1 GR00T WBC blueprint — MuJoCo sim.

Sim counterpart to ``unitree_g1_groot_wbc``.  Same coordinator layout
(WHOLE_BODY g1 component + groot_wbc on legs+waist + servo_arms on
arms), only the WholeBodyAdapter swaps: sim_mujoco_g1 (SHM) instead
of unitree_g1 (DDS).  Sim safety profile is laxer (auto_arm=True,
auto_dry_run=False) so the robot walks immediately.

Usage:
    dimos run unitree-g1-groot-wbc-sim
"""

from __future__ import annotations

import os
import sys

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.whole_body.spec import WholeBodyConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool as DimosBool
from dimos.robot.unitree.g1.blueprints.basic._groot_wbc_common import (
    ARM_DEFAULT_POSE,
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_arms,
    g1_joints,
    g1_legs_waist,
)
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
from dimos.utils.data import get_data
from dimos.utils.logging_config import setup_logger
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

logger = setup_logger()

# Resolved to an absolute path so MujocoSimModule (parent) and the
# DIMOS_MUJOCO_VIEW=1 subprocess can both open the file regardless of
# the shell's CWD.  get_data also auto-extracts the LFS tarball on
# first run.
_MJCF_PATH = str(get_data("mujoco_sim") / "g1_gear_wbc.xml")

_g1_engine = MujocoSimModule.blueprint(
    address=_MJCF_PATH,
    headless=True,
    dof=29,
    enable_color=False,
    enable_depth=False,
    enable_pointcloud=False,
    inject_legacy_assets=True,
).transports(
    {
        ("odom", PoseStamped): LCMTransport("/sim/odom", PoseStamped),
    }
)

_g1_coordinator = ControlCoordinator.blueprint(
    # 50 Hz coordinator tick — matches the rate the GR00T policy was
    # trained at (decoupled_wbc/control/envs/g1/utils/joint_safety.py:38).
    # Combined with the 200 Hz physics in g1_gear_wbc.xml gives a 4:1
    # sim/control ratio.  Running faster (e.g. tick_rate=500) only burns
    # CPU on duplicate inference.
    tick_rate=50.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[
        HardwareComponent(
            hardware_id="g1",
            hardware_type=HardwareType.WHOLE_BODY,
            joints=g1_joints,
            adapter_type="sim_mujoco_g1",
            address=_MJCF_PATH,
            auto_enable=True,
            wb_config=WholeBodyConfig(kp=tuple(G1_GROOT_KP), kd=tuple(G1_GROOT_KD)),
        ),
    ],
    tasks=[
        TaskConfig(
            name="groot_wbc",
            type="groot_wbc",
            joint_names=g1_legs_waist,
            priority=50,
            model_path=str(get_data("groot")),
            hardware_id="g1",
            auto_start=True,
            auto_arm=True,
            auto_dry_run=False,
            # No ramp — policy commands torques the moment it arms, so
            # the robot doesn't free-fall between MJCF spawn and policy
            # takeover.  Sim only; real hardware uses a non-zero ramp.
            default_ramp_seconds=0.0,
            # decimation=1 with tick_rate=50 → policy at 50 Hz (training
            # rate).  GrootWBCTaskConfig defaults to 10 (paired with the
            # legacy 500 Hz tick); leaving the default with our 50 Hz
            # tick yields 5 Hz policy and the robot tips over.
            decimation=1,
        ),
        TaskConfig(
            name="servo_arms",
            type="servo",
            joint_names=g1_arms,
            priority=10,
            default_positions=ARM_DEFAULT_POSE,
            auto_start=True,
        ),
    ],
).transports(
    {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
        ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
        ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
        ("twist_command", Twist): LCMTransport("/g1/cmd_vel", Twist),
        ("activate", DimosBool): LCMTransport("/g1/activate", DimosBool),
        ("dry_run", DimosBool): LCMTransport("/g1/dry_run", DimosBool),
    }
)

_g1_ws_vis = WebsocketVisModule.blueprint().transports(
    {
        ("cmd_vel", Twist): LCMTransport("/g1/cmd_vel", Twist),
        ("activate", DimosBool): LCMTransport("/g1/activate", DimosBool),
        ("dry_run", DimosBool): LCMTransport("/g1/dry_run", DimosBool),
    },
)

unitree_g1_groot_wbc_sim = autoconnect(_g1_engine, _g1_coordinator, _g1_ws_vis)


# Optional native MuJoCo viewer in a separate process — read-only, mirrors
# the engine's joint state via LCM (no physics, no perf hit on the engine).
# Spawn from MainProcess only — worker imports of this module must be
# no-ops (workers are daemonic and can't spawn children).
import multiprocessing as _mp

if (
    os.environ.get("DIMOS_MUJOCO_VIEW", "0") not in ("", "0")
    and _mp.current_process().name == "MainProcess"
):
    import shutil
    import subprocess

    # mujoco.viewer.launch_passive needs ``mjpython`` on macOS; Linux
    # runs fine under regular python.
    if sys.platform == "darwin":
        _viewer_python = shutil.which("mjpython") or shutil.which("python")
    else:
        _viewer_python = sys.executable
    if _viewer_python is None:
        logger.warning(
            "DIMOS_MUJOCO_VIEW=1: couldn't locate mjpython/python on PATH; viewer not launched"
        )
    else:
        _viewer_proc = subprocess.Popen(
            [
                _viewer_python,
                "-m",
                "dimos.simulation.engines.mujoco_view_subprocess",
                _MJCF_PATH,
            ],
        )
        logger.info(
            f"DIMOS_MUJOCO_VIEW=1: MuJoCo viewer subprocess started "
            f"(pid={_viewer_proc.pid}, executable={_viewer_python}, mjcf={_MJCF_PATH})"
        )

__all__ = ["unitree_g1_groot_wbc_sim"]

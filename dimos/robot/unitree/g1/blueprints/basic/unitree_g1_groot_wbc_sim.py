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

"""Unitree G1 GR00T whole-body-control blueprint — MuJoCo sim.

Sim counterpart to ``unitree_g1_groot_wbc``.  Same coordinator + tasks,
swap the real DDS adapter for the MuJoCo sim adapter, drop the
real-hw safety ritual (no operator → auto-arm, no ramp, no dry-run),
add the viser viewer + splat-rendered head camera so perception /
memory / agents can consume the same wire format real cameras produce.

Architecture:
    dashboard WASD ──▶ WebsocketVisModule ──▶ LCM /g1/cmd_vel
                                                       │
                              coordinator twist_command ──▶ GrootWBCTask
                                                       │
    ControlCoordinator ──joint_state, odom──▶ LCM
                              │
                              ▼
                  ViserRenderModule (browser at :8082)
                              │
                              ▼
                  SplatCameraModule ──▶ /splat/color_image
                                        /splat/camera_info

Splat + alignment YAML are pulled via the standard Git-LFS data flow:
``get_data("dimos_office")`` triggers a one-time pull of
``data/.lfs/dimos_office.tar.gz`` and decompresses to
``data/dimos_office/``.  YAML schema: ``SplatAlignment`` in
``dimos/visualization/viser/splat.py``.  If the pull fails (no
Git-LFS, offline, etc.) the viser modules are skipped and the rest
of the blueprint runs unchanged.

Usage:
    dimos run unitree-g1-groot-wbc-sim
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.navigation import NavigationSkillContainer
from dimos.agents.skills.sim_g1_locomotion import G1SimLocomotion
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.stream import In
from dimos.core.transport import LCMTransport
from dimos.hardware.whole_body.spec import WholeBodyConfig
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.mesh_camera import MeshCameraModule
from dimos.mapping.static_costmap import StaticCostmapModule
from dimos.mapping.voxels import VoxelGridMapper
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Bool import Bool as DimosBool
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.patrolling.module import PatrollingModule
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.perception.experimental.temporal_memory.temporal_memory import TemporalMemory
from dimos.perception.perceive_loop_skill import PerceiveLoopSkill
from dimos.perception.spatial_perception import SpatialMemory
from dimos.robot.catalog.g1 import g1_left_arm, g1_right_arm
from dimos.robot.unitree.g1.blueprints.basic._groot_wbc_common import (
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_joints,
    g1_legs_waist,
)
from dimos.robot.unitree.g1.g1_manipulation import G1ManipulationModule
from dimos.robot.unitree.g1.system_prompt import G1_SYSTEM_PROMPT
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
from dimos.utils.data import get_data
from dimos.utils.logging_config import setup_logger
from dimos.visualization.viser import SplatCameraModule, ViserRenderModule
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

logger = setup_logger()


class G1MemoryConfig(RecorderConfig):
    db_path: str | Path = "recording_g1.db"


class G1Memory(Recorder):
    """G1 ``Recorder`` subclass — records the visual + spatial streams.

    Mirrors ``Go2Memory`` shape so memory2's existing playback / search
    tooling works on G1 recordings without special-casing.
    """

    color_image: In[Image]
    lidar: In[PointCloud2]
    config: G1MemoryConfig


# MJCF the GR00T policies were trained against — torque actuators, the
# subprocess (and now the in-process MujocoEngine) computes PD itself.
_MJCF_PATH = "data/mujoco_sim/g1_gear_wbc.xml"

# Scene mesh — used by:
#   * viser viewer (renders the colored mesh in the browser)
#   * MeshCameraModule (ray-casts the head-camera RGB feed)
#   * MuJoCo physics (when DIMOS_SCENE_MESH_COLLISION=1, the default —
#     the mesh is baked into a wrapped MJCF and added as a static
#     collidable so the robot can't phase through walls).
#
# Default: an artist-built mesh of the dimos_office, shipped via Git-LFS
# (``data/.lfs/dimos_office_mesh.tar.gz``).  Despite using the splat
# point cloud as a Reference object in Blender, the artist worked in
# Blender's native Z-up at metric meters — the mesh's aggregate world
# bbox is 11.3m x 7.9m x 3.0m with floor at z=0, ceiling at z=3.0m.
# The .blend's Reference cloud sits at a totally different scale (vertex
# spans of 2300+ units), so the splat-alignment YAML does NOT apply to
# the mesh; the GLB is exported with ``export_yup=False`` to preserve
# Z-up, and dimos loads it with identity alignment.  Run with no env
# vars and the mesh lands directly in dimos world.
#
# Override via env vars to swap in a different scene (e.g. a Sketchfab
# USDZ) — when DIMOS_SCENE_MESH_PATH is set explicitly, the alignment
# defaults below revert to scale=0.05 / zero translation+rotation /
# y_up=true (typical Sketchfab USDZ in centimeters):
#   DIMOS_SCENE_MESH_PATH   = path to .usdz/.glb/.obj/etc.
#   DIMOS_SCENE_MESH_SCALE  = float
#   DIMOS_SCENE_MESH_TRANSLATION = "x,y,z" world-frame offset
#   DIMOS_SCENE_MESH_ROTATION_ZYX_DEG = "z,y,x" extra euler in degrees
#   DIMOS_SCENE_MESH_Y_UP   = "0" to disable the y-up→z-up swap
#   DIMOS_SCENE_MESH_COLLISION = "0" to skip baking + use the bare
#       robot MJCF (visualization-only).  Default: bake.
#   DIMOS_SCENE_MESH_AUTO_GROUND = "1" to auto-translate the scene so
#       the *first* surface a ray-down at origin hits lands at world
#       z=0.  Off by default — for multi-story scenes the first hit is
#       usually a ceiling / upper floor, not the ground.  Use only when
#       you know origin is over the surface you want to stand on.
_scene_mesh_path_override = os.environ.get("DIMOS_SCENE_MESH_PATH") or None
if _scene_mesh_path_override:
    # User-supplied scene; keep historical Sketchfab-cm defaults so old
    # invocations continue to work.
    _scene_mesh_path = _scene_mesh_path_override
    _scene_mesh_scale = float(os.environ.get("DIMOS_SCENE_MESH_SCALE", "0.05"))
    _scene_mesh_translation = tuple(
        float(x) for x in os.environ.get("DIMOS_SCENE_MESH_TRANSLATION", "0,0,0").split(",")
    )
    _scene_mesh_rotation = tuple(
        float(x) for x in os.environ.get("DIMOS_SCENE_MESH_ROTATION_ZYX_DEG", "0,0,0").split(",")
    )
    _scene_mesh_y_up = os.environ.get("DIMOS_SCENE_MESH_Y_UP", "1") != "0"
else:
    # Default: Git-LFS-shipped artist mesh, identity alignment (mesh is
    # already in dimos-world meters with floor at z=0, exported Z-up).
    _scene_mesh_path = str(get_data("dimos_office_mesh") / "dimos_office_mesh.glb")
    _scene_mesh_scale = float(os.environ.get("DIMOS_SCENE_MESH_SCALE", "1.0"))
    _scene_mesh_translation = tuple(
        float(x) for x in os.environ.get("DIMOS_SCENE_MESH_TRANSLATION", "0,0,0").split(",")
    )
    _scene_mesh_rotation = tuple(
        float(x) for x in os.environ.get("DIMOS_SCENE_MESH_ROTATION_ZYX_DEG", "0,0,0").split(",")
    )
    _scene_mesh_y_up = os.environ.get("DIMOS_SCENE_MESH_Y_UP", "0") != "0"
_scene_mesh_collision = os.environ.get("DIMOS_SCENE_MESH_COLLISION", "1") not in ("", "0")
_scene_mesh_auto_ground = os.environ.get("DIMOS_SCENE_MESH_AUTO_GROUND", "0") not in ("", "0")
# Perf knob: kill the entire lidar/voxel/costmap pipeline.  When set, the
# 3-camera depth render in MujocoSimModule, VoxelGridMapper, and CostMapper
# all skip — StaticCostmapModule fills in with an all-free 50x50m grid so
# click-to-nav still works against open space.  Useful for isolating
# locomotion / control-loop perf from the perception firehose.
_disable_lidar = os.environ.get("DIMOS_DISABLE_LIDAR", "0") not in ("", "0")

if _scene_mesh_path:
    from dimos.mapping.mesh_scene import SceneMeshAlignment, floor_z_under_origin

    # If auto-ground is on, ray-cast the scene under (0, 0, ·) once with
    # the user's alignment, then subtract that floor z from translation
    # so the floor at origin lands exactly on world z=0.  All three
    # downstream views (viser, mesh camera, MuJoCo physics) get the
    # *same* corrected SceneMeshAlignment, so geometry is identical
    # everywhere — robot's feet rest on the same surface they're drawn
    # on.  Disable with DIMOS_SCENE_MESH_AUTO_GROUND=0.
    if _scene_mesh_auto_ground:
        try:
            _probe_align = SceneMeshAlignment(
                scale=_scene_mesh_scale,
                rotation_zyx_deg=_scene_mesh_rotation,
                translation=_scene_mesh_translation,
                y_up=_scene_mesh_y_up,
            )
            _floor_z = floor_z_under_origin(_scene_mesh_path, alignment=_probe_align)
            if abs(_floor_z) > 1e-6:
                _scene_mesh_translation = (
                    _scene_mesh_translation[0],
                    _scene_mesh_translation[1],
                    _scene_mesh_translation[2] - _floor_z,
                )
                logger.info(
                    f"Scene-mesh auto-ground: floor under origin was at z={_floor_z:+.3f} m; "
                    f"translating scene by dz={-_floor_z:+.3f} m so it lands at z=0"
                )
        except Exception as e:
            logger.warning(f"Scene-mesh auto-ground probe failed: {e}; using user alignment as-is")

    if _scene_mesh_collision:
        from dimos.mapping.usdz_to_mjcf import bake_scene_mjcf

        try:
            _MJCF_PATH = str(
                bake_scene_mjcf(
                    scene_mesh_path=_scene_mesh_path,
                    robot_mjcf_path=_MJCF_PATH,
                    alignment=SceneMeshAlignment(
                        scale=_scene_mesh_scale,
                        rotation_zyx_deg=_scene_mesh_rotation,
                        translation=_scene_mesh_translation,
                        y_up=_scene_mesh_y_up,
                    ),
                )
            )
            logger.info(f"Scene-mesh collision: using wrapped MJCF {_MJCF_PATH}")
        except Exception as e:
            logger.warning(
                f"Failed to bake scene mesh into MJCF: {e}; falling back to bare robot MJCF "
                f"(visualization will still show the mesh, but physics won't collide with it)"
            )

# Optional MuJoCo native viewer.  ``MujocoSimModule`` runs MuJoCo on a
# *worker* thread and on macOS that can't host ``viewer.launch_passive``
# (glfw needs the main thread).  Spawn a separate process *from the
# dimos CLI main process*, that subscribes to ``/coordinator/joint_state``
# + ``/odom`` over LCM and mirrors the live state into its own
# ``MjData``.  No physics in the viewer — just rendering — so what you
# see is exactly what dimos's engine is producing.
#
# IMPORTANT: this blueprint module is also imported inside dimos worker
# processes when they look up module classes.  Workers are daemonic
# (multiprocessing forbids them from spawning children).  Only spawn
# from MainProcess so worker imports are no-ops here.
import multiprocessing as _mp

if (
    os.environ.get("DIMOS_MUJOCO_VIEW", "0") not in ("", "0")
    and _mp.current_process().name == "MainProcess"
):
    import shutil
    import subprocess

    # ``mujoco.viewer.launch_passive`` checks ``sys.executable`` and
    # raises on macOS unless it's ``mjpython`` (a glfw-bootstrap launcher
    # shipped with the mujoco package).  Linux has no such restriction
    # and runs fine under regular ``python``.
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

# Manipulation: G1 left + right arms as 7-DOF stationary manipulators
# rooted at the floating-base pelvis.  Drake-driven IK + RRT plans
# trajectories; the coordinator's "trajectory" task on the same joint
# subset executes them.
#
# Both arms share a single URDF parse — Drake's "load g1.urdf twice ->
# two complete G1s welded at world origin -> COLLISION_AT_START" trap
# is sidestepped by ManipulationModule._initialize_planning, which
# dedupes by model_path and registers the second arm via
# DrakeWorld.add_robot(share_model_with=...) — one parse, two views
# (left_wrist_yaw vs right_wrist_yaw).
_g1_left_arm_cfg = g1_left_arm()
_g1_right_arm_cfg = g1_right_arm()

_g1_coordinator = (
    ControlCoordinator.blueprint(
        # 50 Hz control loop — matches upstream GR00T-WBC's
        # `control_frequency=50` (decoupled_wbc/control/envs/g1/utils/
        # joint_safety.py:38).  The policy was trained at this rate;
        # running it faster doesn't make the robot smarter, just burns
        # CPU on redundant inference.  Combined with the 200 Hz physics
        # in g1_gear_wbc.xml, gives the upstream 4:1 sim/control ratio.
        tick_rate=50.0,
        publish_joint_state=True,
        joint_state_frame_id="coordinator",
        hardware=[
            HardwareComponent(
                hardware_id="g1",
                hardware_type=HardwareType.WHOLE_BODY,
                joints=g1_joints,
                # In-process engine via MujocoSimModule — adapter and
                # engine share state through SHM keyed on the MJCF path.
                adapter_type="sim_mujoco_g1",
                address=_MJCF_PATH,
                domain_id=0,
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
                model_path=os.getenv("GROOT_MODEL_DIR", str(get_data("groot"))),
                hardware_id="g1",
                auto_start=True,
                # Sim convenience: the MuJoCo subprocess holds the MJCF
                # init pose until the first command arrives, so no
                # operator-arm ritual or ramp is needed.  Dry-run off.
                auto_arm=True,
                auto_dry_run=False,
                default_ramp_seconds=0.0,
                # Coordinator runs at 50 Hz (TickLoop), so decimation=1
                # gives the policy 50 Hz inference — matches the rate
                # the model was trained at.  GrootWBCTaskConfig's own
                # default is 10 (paired with the legacy 500 Hz tick);
                # leaving it at 10 with our 50 Hz tick produces 5 Hz
                # policy and the robot tips over.
                decimation=1,
            ),
            # Per-arm trajectory followers driven by ManipulationModule.
            # When idle the arms dangle under the WBC's kp/kd damping; when
            # a trajectory is loaded for one of them the task wins
            # arbitration on those 7 joints.
            _g1_left_arm_cfg.task_config,
            _g1_right_arm_cfg.task_config,
        ],
    )
    .transports(
        {
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
            ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
            ("twist_command", Twist): LCMTransport("/cmd_vel", Twist),
            ("activate", DimosBool): LCMTransport("/g1/activate", DimosBool),
            ("dry_run", DimosBool): LCMTransport("/g1/dry_run", DimosBool),
        }
    )
    .global_config(
        # global_config.simulation gates real-vs-sim adapter selection
        # in some upstream blueprints; harmless to set even though the
        # in-process engine + adapter pair don't read robot_model anymore.
        robot_model="unitree_g1",
    )
)

# In-process MuJoCo engine.  Owns the MujocoEngine (single thread, no
# subprocess) and publishes joint state + IMU into SHM for the WB
# adapter, plus camera/lidar/pointcloud streams for downstream consumers.
# The G1 GR00T MJCF has no head_camera so we point camera_name at the
# torso lidar instead — a separate splat camera handles RGB perception.
_g1_engine = MujocoSimModule.blueprint(
    address=_MJCF_PATH,
    # Always headless.  The integrated `viewer.launch_passive` calls
    # `m_viewer.sync()` from inside `_step_once` every physics tick and
    # measures 8.4ms/call on Linux — that's a hard ceiling of ~119 Hz
    # regardless of the MJCF timestep target.  The viser viewer at
    # :8082 is the canonical 3D view; the native MuJoCo window is
    # available via DIMOS_MUJOCO_VIEW=1 which spawns a *separate*
    # process that mirrors live state without blocking the engine.
    headless=True,
    dof=29,
    # SplatCameraModule is the canonical RGB source for this sim; suppress
    # MujocoSimModule's own RGB to keep /splat/color_image single-publisher
    # (autoconnect merges any module's `color_image` Out into one shared
    # channel, so per-module transports can't separate them — must gate
    # the publish itself).
    enable_color=False,
    # head_color depth is unused now that ObjectSceneRegistration is out
    # of the stack (MeshCameraModule does its own ray-cast for color and
    # the lidar pipeline below uses the 3 torso lidar cameras instead).
    # Leaving it on registers head_color with the engine, which then
    # blocks the 500 Hz sim thread inside _step_once doing GPU renders
    # nobody consumes.  Re-enable when wiring back ObjectSceneRegistration.
    enable_depth=False,
    enable_pointcloud=not _disable_lidar,
    pointcloud_fps=2.0,
    # head_color camera in the MJCF mirrors g1_d435_default's pose
    # (torso-mounted, 47.6° downward pitch) so MuJoCo's depth would align
    # pixel-for-pixel with the splat-rendered RGB if anything were
    # consuming it.
    camera_name="head_color",
    width=320,
    height=180,
    # Multi-camera 360° lidar — three 160°-FOV depth cameras on the torso
    # (defined in g1_gear_wbc.xml) rendered + back-projected + stitched
    # into a single world-frame pointcloud per scan.  This is what the
    # legacy mujoco_process.py has done for the Go2 sim since #862; without
    # it, VoxelGridMapper's column-carving only sees the forward cone, leaves
    # phantom obstacles behind a moving robot, and A* fails after a while.
    lidar_camera_names=(
        []
        if _disable_lidar
        else [
            "lidar_front_camera",
            "lidar_left_camera",
            "lidar_right_camera",
        ]
    ),
    lidar_camera_width=640,
    lidar_camera_height=360,
    lidar_voxel_size=0.05,
    # G1 GR00T MJCF references meshes by bare filename (menagerie convention);
    # without the legacy asset injection MjModel.from_xml_path can't find them.
    inject_legacy_assets=True,
).transports(
    {
        # ShmMujocoG1WholeBodyAdapter.read_odom returns None (no SHM
        # base-pose channel); MujocoSimModule publishes the floating
        # base pose directly so the viser viewer + nav stack see it.
        ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
        # Bridge pointcloud → /lidar topic so downstream consumers
        # (VoxelGridMapper, G1Memory) with ``lidar`` In ports can
        # subscribe by topic regardless of port-name mismatch.
        ("pointcloud", PointCloud2): LCMTransport("/lidar", PointCloud2),
        # Depth + intrinsics flow to ObjectSceneRegistration so it can
        # back-project 2D detections into 3D world poses for grasping.
        ("depth_image", Image): LCMTransport("/head/depth_image", Image),
        ("camera_info", CameraInfo): LCMTransport("/head/camera_info", CameraInfo),
        ("depth_camera_info", CameraInfo): LCMTransport("/head/depth_camera_info", CameraInfo),
    }
)

# Embedded shell at http://localhost:7779/ — WASD teleop, viser iframe,
# camera MJPEG.  TODO(perf): move /splat/color_image to JpegShmTransport
# across the splat → ws-vis → memory chain to drop LCM JSON overhead.
_g1_ws_vis = WebsocketVisModule.blueprint().transports(
    {
        ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        ("activate", DimosBool): LCMTransport("/g1/activate", DimosBool),
        ("dry_run", DimosBool): LCMTransport("/g1/dry_run", DimosBool),
        ("color_image", Image): LCMTransport("/splat/color_image", Image),
    },
)

# Splat viewer + splat-rendered head camera, gated on the splat asset
# being pullable.  When the LFS pull fails the modules are silently
# skipped and the rest of the sim runs as a flat-floor MuJoCo viewer.
_viser_modules: tuple = ()
try:
    _splat_dir = get_data("dimos_office")
    _legacy_splat_path = _splat_dir / "dimos_office.ply"
    _alignment_yaml = _splat_dir / "dimos_office.yaml"
except Exception as e:
    logger.warning(f"Splat asset unavailable: {e}; viser viewer + splat camera disabled")
    _legacy_splat_path = None
# When on the default office path, use the joint-exported Z-up splat from
# the artist .blend — same Blender world frame as the mesh GLB, so identity
# alignment puts both in the same dimos world (no YAML, no env vars).  When
# a user-supplied DIMOS_SCENE_MESH_PATH is set, fall back to the legacy
# Y-up dimos_office.ply with its (unrelated) YAML alignment.
_splat_path = _legacy_splat_path
if _scene_mesh_path_override is None:
    try:
        _office_pointcloud = get_data("dimos_office_mesh") / "dimos_office_pointcloud.ply"
        if _office_pointcloud.exists():
            _splat_path = _office_pointcloud
    except Exception:
        pass
if _splat_path is not None and _splat_path.exists():
    # Show the splat alongside the mesh when we're using the default office
    # mesh (so the user can visually compare both in the same world frame).
    # When the user provides their *own* scene mesh, hide the splat — it's
    # unrelated geometry that just confuses the picture.  The
    # SplatCameraModule below still uses the splat for the head-camera feed
    # in either case.
    _viser_splat_path = None if _scene_mesh_path_override else str(_splat_path)
    # Splat alignment policy:
    #   * Default office path: identity.  The joint-exported PLY is in the
    #     SAME Blender Z-up world frame as the artist mesh GLB, so no
    #     transform is needed — they overlay directly, the way the artist
    #     authored them in Blender.  ``SplatAlignment()`` defaults to
    #     y_up=True, so we still need a runtime YAML to set y_up=false.
    #   * Custom DIMOS_SCENE_MESH_PATH: legacy ``dimos_office.yaml``.
    import tempfile

    _office_splat_yaml = Path(tempfile.gettempdir()) / "dimos_office_identity_alignment.yaml"
    _office_splat_yaml.write_text(
        "# Identity alignment for the joint-exported office splat.\n"
        "# Both mesh GLB and splat PLY are in the same Blender Z-up frame.\n"
        "scale: 1.0\n"
        "translation: [0.0, 0.0, 0.0]\n"
        "rotation_zyx: [0.0, 0.0, 0.0]\n"
        "y_up: false\n"
    )
    _splat_alignment_yaml = (
        str(_office_splat_yaml)
        if _scene_mesh_path_override is None
        else (str(_alignment_yaml) if _alignment_yaml.exists() else None)
    )
    _g1_viser = ViserRenderModule.blueprint(
        splat_path=_viser_splat_path,
        mjcf_path=_MJCF_PATH,
        alignment_yaml=_splat_alignment_yaml,
        port=8082,
        scene_mesh_path=_scene_mesh_path,
        scene_mesh_scale=_scene_mesh_scale,
        scene_mesh_translation=_scene_mesh_translation,
        scene_mesh_rotation_zyx_deg=_scene_mesh_rotation,
        scene_mesh_y_up=_scene_mesh_y_up,
    ).transports(
        {
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
            # Pointcloud overlay — toggle via "Show lidar pointcloud" in
            # the viser UI panel.  Subscribed to /global_map (the
            # accumulated voxel cloud after column-carving) instead of
            # /lidar (per-scan, transient) so the overlay shows the
            # robot's full obstacle memory rather than just the most
            # recent sweep.  Port is named pointcloud_overlay (not
            # lidar) to dodge the (port_name, type) transport-map
            # collision with VoxelGridMapper.lidar.
            ("pointcloud_overlay", PointCloud2): LCMTransport("/global_map", PointCloud2),
        },
    )
    # Camera publisher.  When the user provided their own scene mesh we
    # render the head camera by ray-casting that mesh (so /splat/color_image
    # actually shows the loaded scene); otherwise we fall back to the splat
    # rasterizer driving dimos_office.  Set ``DIMOS_DISABLE_MESH_CAMERA=1``
    # to force the splat path even when a mesh is provided (escape hatch
    # while debugging — accepts any non-empty value other than "0").
    _disable_mesh_cam = os.environ.get("DIMOS_DISABLE_MESH_CAMERA", "0") not in ("", "0")
    if _scene_mesh_path and not _disable_mesh_cam:
        _g1_camera = MeshCameraModule.blueprint(
            scene_path=_scene_mesh_path,
            mjcf_path=_MJCF_PATH,
            scene_scale=_scene_mesh_scale,
            scene_translation=_scene_mesh_translation,
            scene_rotation_zyx_deg=_scene_mesh_rotation,
            scene_y_up=_scene_mesh_y_up,
            render_hz=10.0,
            # ObjectSceneRegistration's tf.get(target_frame, color.frame_id, ts)
            # needs color.frame_id == the MJCF camera's TF frame so the
            # head/depth_image frame_id matches.
            frame_id="head_color_color_optical_frame",
        ).transports(
            {
                ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
                ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
                ("color_image", Image): LCMTransport("/splat/color_image", Image),
                ("camera_info", CameraInfo): LCMTransport("/splat/camera_info", CameraInfo),
            },
        )
    else:
        _g1_camera = SplatCameraModule.blueprint(
            splat_path=str(_splat_path),
            mjcf_path=_MJCF_PATH,
            alignment_yaml=_splat_alignment_yaml,
            render_hz=10.0,
            # Match MujocoSimModule's depth camera frame so RGB and depth
            # share frame_id; ObjectSceneRegistration's tf lookup needs that.
            frame_id="head_color_color_optical_frame",
        ).transports(
            {
                ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
                ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
                ("color_image", Image): LCMTransport("/splat/color_image", Image),
                ("camera_info", CameraInfo): LCMTransport("/splat/camera_info", CameraInfo),
            },
        )
    _viser_modules = (_g1_viser, _g1_camera)

# Mapping + planning + memory + telemetry layered on top of the base
# sim.  The base sim publishes pointcloud → /lidar (see the engine
# transports above) and color_image → /splat/color_image; downstream
# subscribers bind to those topics by name.
_g1_perception_stack = (
    # Lidar-driven occupancy pipeline.  Skipped entirely when
    # DIMOS_DISABLE_LIDAR=1 — useful for measuring locomotion + agent
    # CPU without VoxelGrid + CostMapper running.  StaticCostmapModule
    # below picks up the slack (planner still gets an all-free grid).
    *(
        ()
        if _disable_lidar
        else (
            VoxelGridMapper.blueprint().transports(
                {("lidar", PointCloud2): LCMTransport("/lidar", PointCloud2)}
            ),
            CostMapper.blueprint(),
        )
    ),
    # Publish a constant all-free OccupancyGrid alongside CostMapper when:
    #   * macOS — the depth-render-based ``/lidar`` pipeline is silent
    #     (mujoco.Renderer can't build Metal pipeline state in a forkserver
    #     child — see splat_camera.py's MlxBackend for the same XPC issue),
    #     so CostMapper would sit idle.
    #   * No scene mesh loaded — the bare g1_gear_wbc.xml has no walls, so
    #     CostMapper's lidar-built grid stays empty and the planner can't
    #     find paths beyond the robot's local LOS.  Static all-free map is
    #     correct for the actual physics (flat floor, nothing to hit).
    #   * DIMOS_DISABLE_LIDAR=1 — CostMapper isn't running, planner
    #     would have no costmap source at all.
    # When a scene mesh IS loaded AND lidar is enabled, skip — CostMapper
    # builds the real obstacle costmap from lidar hitting walls and we
    # don't want a second publisher overwriting it.
    *(
        (StaticCostmapModule.blueprint(),)
        if sys.platform == "darwin" or not _scene_mesh_path or _disable_lidar
        else ()
    ),
    ReplanningAStarPlanner.blueprint(),
    # Visual perception (semantic spatial memory).  Matches Go2 canonical
    # — ObjectTracking and the G1Memory recorder dropped to keep the
    # module set close to unitree-go2-temporal-memory.
    SpatialMemory.blueprint(),
)

# Agentic stack — Go2 parity minus xArm and minus PersonFollow.
# UnitreeG1SkillContainer is still skipped (its move()/arm-gesture/mode
# skills need G1ConnectionSpec which our in-process engine doesn't
# provide); G1SimLocomotion gives the agent move() via /cmd_vel instead.
# Vision is via PerceiveLoopSkill (Qwen API), memory introspection via
# TemporalMemory.query().  Requires OPENAI_API_KEY (LLM + TTS) and
# ALIBABA_API_KEY (Qwen-VL for navigate_with_text + look_out_for).
#
# Note: on macOS, McpServer.on_system_modules can stall past its 120 s
# budget because the agentic + heavy perception modules cost ~30+ s of
# cold-start. If running on Mac, gate this stack off and rely only on
# G1SimLocomotion + PatrollingModule + WavefrontFrontierExplorer.
_g1_agentic_stack = (
    McpServer.blueprint(),
    McpClient.blueprint(system_prompt=G1_SYSTEM_PROMPT),
    G1SimLocomotion.blueprint().transports(
        {
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        }
    ),
    NavigationSkillContainer.blueprint(),
    SpeakSkill.blueprint(),
    # Manipulation — Drake IK + RRT planner driving the G1 left arm via
    # the coordinator's trajectory task.  Subscribes to coordinator
    # joint_state for live state sync.  Meshcat viz off in this
    # composed sim (we already have viser as the live 3D view).
    G1ManipulationModule.blueprint(
        robots=[
            _g1_left_arm_cfg.robot_model_config,
            _g1_right_arm_cfg.robot_model_config,
        ],
        planning_timeout=10.0,
        # Drake's nonlinear-program-based IK (SNOPT under the hood) —
        # robust to seed quality and supports `solve_pointing` (an
        # angle-between-vectors constraint) which `point_at` uses to
        # leave the wrist roll about the pointing axis free.  Eval
        # showed JacobianIK with strict look-at could only reach
        # ~25% of random pointing directions; this expands that.
        kinematics_name="drake_optimization",
        # Meshcat viewer for what Drake actually sees: URDF model in
        # the planner's frame, planned trajectories, world-monitored
        # obstacles. Logged at startup as
        # "Visualization started: http://localhost:7000/".
        enable_viz=True,
        # Easy-mode handle: the reach_for_sim_object skill loads this
        # MJCF separately and reads body world poses straight from it,
        # bypassing perception.  Useful for isolating manipulation
        # bugs from perception bugs (YOLO-E labels, RGBD back-proj
        # accuracy, frame transforms).
        sim_mjcf_path=_MJCF_PATH,
    ).transports(
        {
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
        }
    ),
    PerceiveLoopSkill.blueprint().transports(
        {
            ("color_image", Image): LCMTransport("/splat/color_image", Image),
        }
    ),
    TemporalMemory.blueprint(new_memory=global_config.new_memory),
    PatrollingModule.blueprint(),
    WavefrontFrontierExplorer.blueprint(),
)

unitree_g1_groot_wbc_sim = autoconnect(
    _g1_coordinator,
    _g1_engine,
    _g1_ws_vis,
    *_viser_modules,
    *_g1_perception_stack,
    *_g1_agentic_stack,
).global_config(n_workers=18, detection_model="qwen")

__all__ = ["unitree_g1_groot_wbc_sim"]

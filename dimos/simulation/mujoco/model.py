#!/usr/bin/env python3

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


from pathlib import Path
import xml.etree.ElementTree as ET

from etils import epath
import mujoco
from mujoco_playground._src import mjx_env
import numpy as np

from dimos.core.global_config import GlobalConfig
from dimos.mapping.occupancy.extrude_occupancy import generate_mujoco_scene
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.simulation.mujoco.input_controller import InputController
from dimos.simulation.mujoco.policy import G1OnnxController, Go1OnnxController, OnnxController
from dimos.utils.data import get_data


def _get_data_dir() -> epath.Path:
    return epath.Path(str(get_data("mujoco_sim")))


def _get_hssd_house_dir() -> epath.Path:
    """Path to the unpacked HSSD house archive (data/hssd_house/)."""
    return epath.Path(str(get_data("hssd_house")))


def get_assets(mujoco_room: str | None = None) -> dict[str, bytes]:
    """Load mesh + texture assets for the active scene.

    Pass ``mujoco_room`` to also load scene-specific assets (e.g.
    ``"hssd_house"`` pulls in the meshes from the HSSD archive).
    """
    data_dir = _get_data_dir()
    assets: dict[str, bytes] = {}

    # Assets used from https://sketchfab.com/3d-models/mersus-office-8714be387bcd406898b2615f7dae3a47
    # Created by Ryan Cassidy and Coleman Costello
    mjx_env.update_assets(assets, data_dir, "*.xml")
    mjx_env.update_assets(assets, data_dir / "scene_office1/textures", "*.png")
    mjx_env.update_assets(assets, data_dir / "scene_office1/office_split", "*.obj")
    mjx_env.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_go1" / "assets")
    mjx_env.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_g1" / "assets")

    # From: https://sketchfab.com/3d-models/jeong-seun-34-42956ca979404a038b8e0d3e496160fd
    person_dir = epath.Path(str(get_data("person")))
    mjx_env.update_assets(assets, person_dir, "*.obj")
    mjx_env.update_assets(assets, person_dir, "*.png")

    # HSSD-derived multi-room house scene (assets via SceneSmith
    # nepfaff/scenesmith-example-scenes, House subset, scene_186).
    # Only loaded when the scene is selected — 200+ MB of OBJ meshes
    # would be wasted memory for office1 / scene_empty runs.
    if mujoco_room == "hssd_house":
        hssd_meshes = _get_hssd_house_dir() / "meshes"
        mjx_env.update_assets(assets, hssd_meshes, "*.obj")
        mjx_env.update_assets(assets, hssd_meshes, "*.png")

    return assets


def load_model(
    input_device: InputController,
    robot: str,
    scene_xml: str,
    mujoco_room: str | None = None,
    *,
    kinematic_robot: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    mujoco.set_mjcb_control(None)

    xml_string = get_model_xml(robot, scene_xml)
    model = mujoco.MjModel.from_xml_string(xml_string, assets=get_assets(mujoco_room))
    data = mujoco.MjData(model)

    # Initialise data from mjModel.qpos0 first.  qpos0 is built from each
    # body's authored <body pos="..." quat="..."> attributes, so HSSD-style
    # scenes whose furniture each have a <joint type="free"/> get every
    # piece placed where it was authored (bed in the bedroom, etc.).
    #
    # Calling mj_resetDataKeyframe directly would clobber that — the
    # robot's "home" keyframe only specifies the G1's 36 qpos slots, so
    # MuJoCo zero-pads the rest, dumping every furniture body at world
    # origin and producing the "all furniture overlaps + glitches"
    # behaviour we hit on scene_186.
    mujoco.mj_resetData(model, data)
    # Now overwrite the robot's own qpos slots with the home keyframe so
    # the G1 starts in its standing pose.  Robot joints occupy the first
    # (7 + model.nu) qpos slots: 7 for the floating-base free joint and
    # one per actuator.  Furniture qpos lives after that and stays at
    # qpos0 from the line above.
    robot_qpos_len = 7 + int(model.nu)
    if model.nkey > 0:
        data.qpos[:robot_qpos_len] = model.key_qpos[0][:robot_qpos_len]

    match robot:
        case "unitree_g1":
            sim_dt = 0.002
        case _:
            sim_dt = 0.005

    ctrl_dt = 0.02
    n_substeps = round(ctrl_dt / sim_dt)
    model.opt.timestep = sim_dt

    # Robots have one actuator per controllable joint, so model.nu is the
    # robot's joint count regardless of how many extra articulated bodies
    # the surrounding scene contributes.  Without this slice, keyframe
    # "home"'s qpos[7:] is the FULL model qpos minus the free joint, which
    # picks up every HSSD-style scene joint and produces a 9× too-large
    # default_angles vector.  That balloons the ONNX policy's obs and
    # crashes the first control step with INVALID_ARGUMENT.
    home_qpos = np.array(model.keyframe("home").qpos[7 : 7 + model.nu])

    # Kinematic mode skips the ONNX walking policy entirely: the floating
    # base will be driven by ``data.qpos[0:7]`` updates from cmd_vel each
    # tick (see ``mujoco_process._step_once``) and the joints stay frozen
    # at the home pose.  Useful for nav tests where we don't care about
    # gait, just camera/lidar-driven planning.
    if kinematic_robot:
        return model, data

    params = {
        "policy_path": (_get_data_dir() / f"{robot}_policy.onnx").as_posix(),
        "default_angles": home_qpos,
        "n_substeps": n_substeps,
        "action_scale": 0.5,
        "input_controller": input_device,
        "ctrl_dt": ctrl_dt,
    }

    match robot:
        case "unitree_go1":
            policy: OnnxController = Go1OnnxController(**params)
        case "unitree_g1":
            policy = G1OnnxController(**params, drift_compensation=[-0.18, 0.0, -0.09])
        case _:
            raise ValueError(f"Unknown robot policy: {robot}")

    mujoco.set_mjcb_control(policy.get_control)

    return model, data


def get_model_xml(robot: str, scene_xml: str) -> str:
    root = ET.fromstring(scene_xml)
    root.set("model", f"{robot}_scene")
    root.insert(0, ET.Element("include", file=f"{robot}.xml"))

    # Ensure visual/map element exists with znear and zfar
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    map_elem = visual.find("map")
    if map_elem is None:
        map_elem = ET.SubElement(visual, "map")
    map_elem.set("znear", "0.01")
    map_elem.set("zfar", "10000")

    _add_person_object(root)

    return ET.tostring(root, encoding="unicode")


def _add_person_object(root: ET.Element) -> None:
    asset = root.find("asset")

    if asset is None:
        asset = ET.SubElement(root, "asset")

    ET.SubElement(asset, "mesh", name="person_mesh", file="jeong_seun_34.obj")
    ET.SubElement(asset, "texture", name="person_texture", file="material_0.png", type="2d")
    ET.SubElement(asset, "material", name="person_material", texture="person_texture")

    worldbody = root.find("worldbody")

    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    person_body = ET.SubElement(worldbody, "body", name="person", pos="0 0 0", mocap="true")

    ET.SubElement(
        person_body,
        "geom",
        type="mesh",
        mesh="person_mesh",
        material="person_material",
        euler="1.5708 0 0",
    )


def load_scene_xml(config: GlobalConfig) -> str:
    if config.mujoco_room_from_occupancy:
        path = Path(config.mujoco_room_from_occupancy)
        return generate_mujoco_scene(OccupancyGrid.from_path(path))

    mujoco_room = config.mujoco_room or "office1"

    # The HSSD house scene lives in its own LFS archive because its
    # meshes are 200+ MB (vs the 60 MB mujoco_sim archive).  Pull from
    # that archive instead of the default mujoco_sim data dir.
    if mujoco_room == "hssd_house":
        scene_dir = _get_hssd_house_dir()
    else:
        scene_dir = _get_data_dir()

    xml_file = (scene_dir / f"scene_{mujoco_room}.xml").as_posix()
    with open(xml_file) as f:
        return f.read()

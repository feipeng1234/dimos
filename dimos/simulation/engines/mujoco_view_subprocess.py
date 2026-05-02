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

"""MuJoCo passive viewer subprocess for the in-process dimos sim.

dimos's ``MujocoSimModule`` runs MuJoCo on a *worker* thread; on macOS
``mujoco.viewer.launch_passive`` requires the *main* thread (a glfw
constraint), which is why ``MujocoSimConfig.headless`` defaults to True
on Darwin.  This module is a tiny standalone entry point that:

  1. Loads the same MJCF dimos compiled.
  2. Subscribes to ``/coordinator/joint_state`` + ``/odom`` over LCM —
     the topics the in-process engine already publishes.
  3. Mirrors that state into its own ``mujoco.MjData`` and renders via
     ``mujoco.viewer.launch_passive`` from *its* main thread.

It runs no physics — it's a viewer, not a second simulator — so what
you see is exactly the state the dimos engine is producing.

Spawned by the GR00T sim blueprint when ``DIMOS_MUJOCO_VIEW=1``; the
blueprint uses ``multiprocessing.spawn`` so the subprocess gets a fresh
main thread regardless of how the dimos CLI was launched.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def main(mjcf_path: str) -> None:
    """Subprocess entry point.  Blocks until the viewer window is closed."""
    import mujoco  # type: ignore[import-untyped]
    import mujoco.viewer as viewer  # type: ignore[import-untyped]

    from dimos.core.transport import LCMTransport
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.sensor_msgs.JointState import JointState
    from dimos.visualization.viser.robot_meshes import dimos_joint_to_mjcf

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Map joint name → qpos index (only hinge / slide joints have a
    # 1-to-1 mapping with msg.position; the free joint at index 0
    # we update separately via /odom).
    name_to_qposadr: dict[str, int] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name_to_qposadr[name] = int(model.jnt_qposadr[jid])

    # Latest state — written by LCM callbacks, read by the viewer loop.
    latest: dict[str, Any] = {"joints": {}, "base_pos": None, "base_wxyz": None}

    def _on_joint_state(msg: JointState) -> None:
        # Coordinator publishes dimos canonical names ("g1_LeftHipPitch")
        # but the MJCF uses MuJoCo names ("left_hip_pitch_joint"); translate
        # so the lookup against ``name_to_qposadr`` actually hits.
        try:
            for n, q in zip(list(msg.name), list(msg.position), strict=False):
                latest["joints"][dimos_joint_to_mjcf(str(n))] = float(q)
        except Exception:
            pass

    def _on_odom(msg: PoseStamped) -> None:
        try:
            latest["base_pos"] = np.array(
                [msg.position.x, msg.position.y, msg.position.z], dtype=np.float64
            )
            latest["base_wxyz"] = np.array(
                [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z],
                dtype=np.float64,
            )
        except Exception:
            pass

    js_t: LCMTransport[JointState] = LCMTransport("/coordinator/joint_state", JointState)
    js_t.start()
    js_t.subscribe(_on_joint_state)
    od_t: LCMTransport[PoseStamped] = LCMTransport("/odom", PoseStamped)
    od_t.start()
    od_t.subscribe(_on_odom)

    # The first qpos slots come from the free joint (px, py, pz, qw, qx, qy, qz).
    # If the robot doesn't have one, fall back to leaving them alone.
    has_freejoint = bool(model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
    free_qposadr = int(model.jnt_qposadr[0]) if has_freejoint else -1

    # Use ``mj_kinematics`` (forward kinematics only) — we want to render
    # whatever pose dimos is producing, not advance physics.
    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as v:
        while v.is_running():
            joints = dict(latest["joints"])
            base_pos = latest["base_pos"]
            base_wxyz = latest["base_wxyz"]
            for name, q in joints.items():
                adr = name_to_qposadr.get(name)
                if adr is not None:
                    data.qpos[adr] = q
            if has_freejoint and base_pos is not None and base_wxyz is not None:
                data.qpos[free_qposadr : free_qposadr + 3] = base_pos
                data.qpos[free_qposadr + 3 : free_qposadr + 7] = base_wxyz
            mujoco.mj_kinematics(model, data)
            v.sync()
            time.sleep(1.0 / 60.0)

    js_t.stop()
    od_t.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m dimos.simulation.engines.mujoco_view_subprocess <mjcf>")
        sys.exit(2)
    main(sys.argv[1])

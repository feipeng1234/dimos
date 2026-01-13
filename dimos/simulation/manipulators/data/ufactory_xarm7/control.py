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

import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("./scene.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # Apply control: zero all actuators and set only one actuator (desired_idx).
        desired_idx = 3  # change this to the actuator index you want to move
        desired_value = 1  # control value for that single actuator
        d.ctrl[:] = 0.0
        d.ctrl[desired_idx] = desired_value

        # mj_step will now use the control applied above.
        mujoco.mj_step(m, d)
        # print velocities
        # print("Joint velocities:", d.qvel[:m.nv])

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

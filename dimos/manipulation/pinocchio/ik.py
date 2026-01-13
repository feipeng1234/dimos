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

import numpy as np
from numpy.linalg import norm, solve
import pinocchio

mjcf_path = "/home/ruthwik/Documents/dimos/dimos/simulation/manipulators/data/ufactory_xarm7/xarm7_nohand.xml"
model = pinocchio.buildModelFromMJCF(mjcf_path)
data = model.createData()

JOINT_ID = 7  # joint7 is the end-effector for xarm7
oMdes = pinocchio.SE3(np.eye(3), np.array([0.4, 0.2, 0.2]))

q = pinocchio.neutral(model)
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0

print(f"initial: {q.flatten().tolist()}")
while True:
    pinocchio.forwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)

    err = pinocchio.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print(f"{i}: error = {err.T}")
    i += 1

if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

print(f"\nresult (rad): {q.flatten().tolist()}")
print(f"result (deg): {np.degrees(q).flatten().tolist()}")
print(f"\nfinal error: {err.T}")

# Verify with FK
pinocchio.forwardKinematics(model, data, q)
print(f"\nFK verification - EE position: {data.oMi[JOINT_ID].translation.T}")
print(f"FK verification - EE rotation:\n{data.oMi[JOINT_ID].rotation}")

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

from pathlib import Path
from sys import argv

import pinocchio

# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = "/home/ruthwik/Documents/dimos/dimos/simulation/manipulators/data/ufactory_xarm7/xarm7_nohand.xml"

# You should change here to set up your own URDF file or just pass it as an argument of
# this example.
urdf_filename = pinocchio_model_dir if len(argv) < 2 else argv[1]

# Load the urdf model
model = pinocchio.buildModelFromMJCF(urdf_filename)
print("model name: " + model.name)

# Create data required by the algorithms
data = model.createData()

# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print(f"q: {q.T}")

# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi, strict=False):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

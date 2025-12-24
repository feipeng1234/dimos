# Copyright 2025 Dimensional Inc.
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

import pytest
from lcm_msgs.foxglove_msgs import SceneUpdate

from dimos.core import LCMTransport, start
from dimos.msgs.foxglove_msgs import ImageAnnotations
from dimos.msgs.geometry_msgs import PoseStamped
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.perception.detection.module3D import Detection3DModule
from dimos.perception.detection.moduleDB import ObjectDBModule
from dimos.protocol.service import lcmservice as lcm
from dimos.robot.unitree_webrtc.modular import deploy_connection, deploy_navigation
from dimos.robot.unitree_webrtc.modular.connection_module import ConnectionModule


@pytest.fixture(scope="module")
def dimos_cluster():
    dimos = start(5)
    yield dimos
    dimos.stop()


@pytest.mark.module
def test_module3d(dimos_cluster):
    connection = deploy_connection(dimos_cluster)

    module = dimos_cluster.deploy(
        Detection3DModule,
        camera_info=ConnectionModule._camera_info(),
        # goto=lambda obj_id: print(f"Going to {obj_id}"),
    )
    module.image.connect(connection.video)
    module.pointcloud.connect(connection.lidar)

    module.annotations.transport = LCMTransport("/annotations", ImageAnnotations)
    module.detections.transport = LCMTransport("/detections", Detection2DArray)

    module.detected_pointcloud_0.transport = LCMTransport("/detected/pointcloud/0", PointCloud2)
    module.detected_pointcloud_1.transport = LCMTransport("/detected/pointcloud/1", PointCloud2)
    module.detected_pointcloud_2.transport = LCMTransport("/detected/pointcloud/2", PointCloud2)

    module.detected_image_0.transport = LCMTransport("/detected/image/0", Image)
    module.detected_image_1.transport = LCMTransport("/detected/image/1", Image)
    module.detected_image_2.transport = LCMTransport("/detected/image/2", Image)

    module.scene_update.transport = LCMTransport("/scene_update", SceneUpdate)
    # module.target.transport = LCMTransport("/target", PoseStamped)

    connection.start()
    module.start()

    time.sleep(3)
    print("VLM QUERY START")
    res = module.query_vlm("a chair")
    print("VLM QUERY RESULT:", res)

    time.sleep(30)

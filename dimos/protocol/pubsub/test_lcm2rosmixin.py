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

from dimos_lcm.geometry_msgs import PoseStamped, Vector3
import pytest

from dimos.protocol.pubsub.rospubsub import ROS_AVAILABLE, LCM2ROSMixin

pytestmark = pytest.mark.skipif(not ROS_AVAILABLE, reason="ROS not installed")


def test_simple_encode():
    """Test simple Vector3 roundtrip conversion."""
    mixin = LCM2ROSMixin()

    # Create dimos_lcm Vector3
    vec = Vector3()
    vec.x, vec.y, vec.z = 1.0, 2.0, 3.0

    # Encode to ROS
    ros_msg = mixin.encode(vec)
    print(f"ROS message: {ros_msg}")

    # Decode back to dimos_lcm
    decoded = mixin.decode(ros_msg)
    print(f"Decoded: {decoded}")

    # Verify roundtrip
    assert decoded.x == 1.0
    assert decoded.y == 2.0
    assert decoded.z == 3.0


def test_complex_encode():
    """Test nested PoseStamped roundtrip conversion."""
    mixin = LCM2ROSMixin()

    # Create dimos_lcm PoseStamped with nested structure
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.header.stamp.sec = 123
    pose.header.stamp.nsec = 456
    pose.pose.position.x = 1.0
    pose.pose.position.y = 2.0
    pose.pose.position.z = 3.0
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0

    # Encode to ROS
    ros_msg = mixin.encode(pose)
    print(f"ROS message: {ros_msg}")

    # Decode back to dimos_lcm
    decoded = mixin.decode(ros_msg)
    print(f"Decoded: {decoded}")

    # Verify roundtrip preserves structure
    assert decoded.header.frame_id == "base_link"
    assert decoded.pose.position.x == 1.0
    assert decoded.pose.orientation.w == 1.0

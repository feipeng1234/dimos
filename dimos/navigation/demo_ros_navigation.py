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

import time

from geometry_msgs.msg import PoseStamped, TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Int8


class NavDemo(Node):
    def __init__(self):
        super().__init__("nav_demo")

        best_effort_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.cancel_goal_pub = self.create_publisher(Bool, "/cancel_goal", 10)
        self.soft_stop_pub = self.create_publisher(Int8, "/stop", 10)
        self.joy_pub = self.create_publisher(Joy, "/joy", 10)

        self.goal_reached_sub = self.create_subscription(
            Bool, "/goal_reached", self._on_goal_reached, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped, "/cmd_vel", self._on_cmd_vel, best_effort_qos
        )

        self.goal_reached = None

    def _on_goal_reached(self, msg: Bool):
        self.goal_reached = msg.data

    def _on_cmd_vel(self, msg: TwistStamped):
        pass  # received; could forward via LCM here if needed

    def set_autonomy_mode(self):
        joy = Joy()
        joy.axes = [0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        joy.buttons = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.joy_pub.publish(joy)
        self.get_logger().info("Autonomy mode enabled via Joy")

    def send_goal(self, x: float, y: float, frame_id: str = "map"):
        self.goal_reached = None
        self.set_autonomy_mode()

        soft_stop = Int8()
        soft_stop.data = 0
        self.soft_stop_pub.publish(soft_stop)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self.goal_pose_pub.publish(pose)
        self.get_logger().info(f"Goal sent: ({x}, {y}) in {frame_id}")

    def cancel_goal(self):
        cancel = Bool()
        cancel.data = True
        self.cancel_goal_pub.publish(cancel)

        soft_stop = Int8()
        soft_stop.data = 2
        self.soft_stop_pub.publish(soft_stop)
        self.get_logger().info("Goal cancelled")


def main():
    rclpy.init()
    node = NavDemo()

    node.get_logger().info("Waiting 2s for nav stack to be ready...")
    end = time.time() + 2.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.send_goal(10.0, 10.0)

    node.get_logger().info("Waiting up to 5s then cancelling...")
    end = time.time() + 5.0
    while time.time() < end and node.goal_reached is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.cancel_goal()
    node.get_logger().info("NavDemo running. Ctrl+C to stop.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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

"""ROS2 to Rerun bridge for autonomy stack visualization.

This module subscribes to ROS2 topics from the autonomy stack and logs them
to Rerun using Jeff's RerunConnection pattern. It mirrors all RViz-visible
streams without modifying any ROS publishers.
"""

import threading
import time
from collections import deque
from typing import Any

from geometry_msgs.msg import PointStamped, PolygonStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import rclpy
import rclpy.duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformException

try:
    from numpy.lib.recfunctions import structured_to_unstructured
    from sensor_msgs_py import point_cloud2
except ImportError:
    point_cloud2 = None
    structured_to_unstructured = None

import rerun as rr

from dimos.core import Module, rpc
from dimos.dashboard.module import RerunConnection
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class PointCloudBuffer:
    """Buffer that accumulates pointclouds with timestamps for 5-second decay effect."""
    
    def __init__(self, decay_seconds: float = 5.0):
        self.decay_seconds = decay_seconds
        self.scans = deque()  # List of (timestamp, points_array)
    
    def add_scan(self, points: np.ndarray) -> None:
        """Add a new scan with current timestamp."""
        now = time.time()
        self.scans.append((now, points))
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove scans older than decay_seconds."""
        now = time.time()
        cutoff = now - self.decay_seconds
        while self.scans and self.scans[0][0] < cutoff:
            self.scans.popleft()
    
    def get_all_points(self) -> np.ndarray:
        """Get all points from buffered scans."""
        self._cleanup()
        if not self.scans:
            return np.zeros((0, 3), dtype=np.float32)
        
        all_points = [p for t, p in self.scans]
        return np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=np.float32)


class RosRerunBridgeNode(Node):
    """ROS2 node that bridges autonomy stack topics to Rerun."""

    def __init__(self, rc: RerunConnection) -> None:
        super().__init__("ros_rerun_bridge")

        self.rc = rc
        self._log_lock = threading.Lock()

        # ReentrantCallbackGroup for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # Point cloud buffers for 5-second decay
        self._registered_scan_buffer = PointCloudBuffer(decay_seconds=5.0)
        self._terrain_map_buffer = PointCloudBuffer(decay_seconds=5.0)
        self._overall_map_buffer = PointCloudBuffer(decay_seconds=10.0)  # Keep map longer
        
        # Frame counter for sequence timeline
        self._frame_count = 0
        
        # TF buffer for camera transforms
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Camera intrinsics (logged once as static)
        self._camera_intrinsics_logged = False
        
        # Debug: Check if RerunConnection has a stream
        if self.rc.stream is None:
            logger.warning(
                "[RosRerunBridge] RerunConnection.stream is None - Dashboard lock file may not exist"
            )
        else:
            logger.info(f"[RosRerunBridge] RerunConnection.stream initialized: {self.rc.stream}")

        # Set coordinate system (ROS uses right-handed Z-up for map frame)
        self.rc.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        logger.info("[RosRerunBridge] Logged world coordinate system")

        # Subscribe to autonomy stack topics
        self._setup_subscriptions()

        logger.info("[RosRerunBridge] ROS2 node initialized")

    def _log_with_time(self, entity_path: str, archetype: Any, stamp: Any) -> None:
        """Thread-safe helper to set timeline and log atomically."""
        try:
            time_ns = Time.from_msg(stamp).nanoseconds
            with self._log_lock:
                # Correct RecordingStream API: set_time with timestamp parameter
                self.rc.stream.set_time("ros_time", timestamp=np.datetime64(time_ns, "ns"))
                self.rc.log(entity_path, archetype)
        except Exception as e:
            logger.error(f"[RosRerunBridge] Failed to log {entity_path}: {e}", exc_info=True)

    def _setup_subscriptions(self) -> None:
        """Create all ROS2 subscriptions."""
        # PointCloud2 topics
        self.create_subscription(
            PointCloud2,
            "/registered_scan",
            self._on_registered_scan,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointCloud2, "/terrain_map", self._on_terrain_map, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PointCloud2,
            "/terrain_map_ext",
            self._on_terrain_map_ext,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointCloud2, "/overall_map", self._on_overall_map, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PointCloud2, "/free_paths", self._on_free_paths, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PointCloud2,
            "/explored_areas",
            self._on_explored_areas,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointCloud2, "/trajectory", self._on_trajectory, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PointCloud2,
            "/added_obstacles",
            self._on_added_obstacles,
            5,
            callback_group=self.callback_group,
        )

        # Planning topics
        self.create_subscription(
            Path, "/path", self._on_path, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped, "/goal_pose", self._on_goal_pose, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PointStamped, "/way_point", self._on_waypoint, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            PolygonStamped,
            "/navigation_boundary",
            self._on_boundary,
            5,
            callback_group=self.callback_group,
        )

        # Odometry for robot pose
        self.create_subscription(
            Odometry, "/state_estimation", self._on_odometry, 10, callback_group=self.callback_group
        )

        # Planner visualization markers
        self.create_subscription(
            Marker, "/viz_path_topic", self._on_viz_path, 5, callback_group=self.callback_group
        )
        self.create_subscription(
            MarkerArray,
            "/viz_node_topic",
            self._on_viz_nodes,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            MarkerArray,
            "/viz_graph_topic",
            self._on_viz_graph,
            5,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            MarkerArray,
            "/viz_viewpoint_extend_topic",
            self._on_viz_viewpoint,
            5,
            callback_group=self.callback_group,
        )

        logger.info("[RosRerunBridge] All subscriptions created")

    def _parse_pointcloud(self, msg: PointCloud2, max_points: int = 200_000) -> np.ndarray | None:
        """Parse PointCloud2 message to numpy array."""
        if point_cloud2 is None or structured_to_unstructured is None:
            logger.warning("[RosRerunBridge] sensor_msgs_py not available, skipping pointcloud")
            return None

        try:
            pts = point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
            pts = structured_to_unstructured(pts)

            # Downsample if too many points
            if len(pts) > max_points:
                idx = np.random.choice(len(pts), size=max_points, replace=False)
                pts = pts[idx]

            return pts
        except Exception as e:
            logger.error(f"[RosRerunBridge] Failed to parse pointcloud: {e}", exc_info=True)
            return None
    
    # PointCloud2 callbacks with 5-second accumulation
    def _on_registered_scan(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=250_000)
        if pts is not None:
            self._registered_scan_buffer.add_scan(pts)
            all_pts = self._registered_scan_buffer.get_all_points()
            
            # Use sequence timeline for auto-cleanup + white color
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/registered_scan",
                        rr.Points3D(all_pts, colors=[[255, 255, 255]], radii=0.005)
                    )
                self._frame_count += 1
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log registered_scan: {e}", exc_info=True)
    
    def _on_terrain_map(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=200_000)
        if pts is not None:
            self._terrain_map_buffer.add_scan(pts)
            all_pts = self._terrain_map_buffer.get_all_points()
            
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/terrain_map",
                        rr.Points3D(all_pts, colors=[[255, 255, 255]], radii=0.005)
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log terrain_map: {e}", exc_info=True)
    
    def _on_terrain_map_ext(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=200_000)
        if pts is not None:
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/terrain_map_ext",
                        rr.Points3D(pts, colors=[[255, 255, 255]], radii=0.005)
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log terrain_map_ext: {e}", exc_info=True)
    
    def _on_overall_map(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=300_000)
        if pts is not None:
            self._overall_map_buffer.add_scan(pts)
            all_pts = self._overall_map_buffer.get_all_points()
            
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/overall_map",
                        rr.Points3D(all_pts, colors=[[255, 255, 255]], radii=0.005)
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log overall_map: {e}", exc_info=True)
    
    def _on_free_paths(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=150_000)
        if pts is not None:
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/free_paths",
                        rr.Points3D(pts, colors=[[0, 170, 255]], radii=0.005)  # Light blue like RViz
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log free_paths: {e}", exc_info=True)
    
    def _on_explored_areas(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=200_000)
        if pts is not None:
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/explored_areas",
                        rr.Points3D(pts, colors=[[255, 255, 255]], radii=0.005),
                        static=True  # Explored areas persist
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log explored_areas: {e}", exc_info=True)
    
    def _on_trajectory(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=100_000)
        if pts is not None:
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/robot/trajectory",
                        rr.Points3D(pts, colors=[[255, 255, 0]], radii=0.01),  # Yellow trail
                        static=True
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log trajectory: {e}", exc_info=True)
    
    def _on_added_obstacles(self, msg: PointCloud2) -> None:
        pts = self._parse_pointcloud(msg, max_points=100_000)
        if pts is not None:
            try:
                with self._log_lock:
                    self.rc.stream.set_time("frame", sequence=self._frame_count)
                    self.rc.log(
                        "world/lidar/added_obstacles",
                        rr.Points3D(pts, colors=[[255, 25, 0]], radii=0.01)  # Red obstacles
                    )
            except Exception as e:
                logger.error(f"[RosRerunBridge] Failed to log added_obstacles: {e}", exc_info=True)
    
    # Planning callbacks
    def _on_path(self, msg: Path) -> None:
        if not msg.poses:
            return

        try:
            pts = np.array(
                [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                dtype=np.float32,
            )

            if len(pts) >= 2:
                self._log_with_time("world/planner/path", rr.LineStrips3D([pts]), msg.header.stamp)
        except Exception as e:
            logger.debug(f"[RosRerunBridge] Failed to log path: {e}")

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        try:
            p = msg.pose.position
            q = msg.pose.orientation

            # Log as an arrow pointing in the direction of the goal
            from scipy.spatial.transform import Rotation as R

            rot = R.from_quat([q.x, q.y, q.z, q.w])
            forward = rot.apply([1.0, 0.0, 0.0])  # Forward direction

            origin = np.array([[p.x, p.y, p.z]], dtype=np.float32)
            vector = np.array([forward], dtype=np.float32)

            self._log_with_time(
                "world/planner/goal",
                rr.Arrows3D(origins=origin, vectors=vector, colors=[0, 255, 0]),
                msg.header.stamp,
            )
        except Exception as e:
            logger.debug(f"[RosRerunBridge] Failed to log goal pose: {e}")

    def _on_waypoint(self, msg: PointStamped) -> None:
        try:
            p = msg.point
            self._log_with_time(
                "world/planner/waypoint",
                rr.Points3D([[p.x, p.y, p.z]], radii=0.3, colors=[204, 41, 204]),
                msg.header.stamp,
            )
        except Exception as e:
            logger.debug(f"[RosRerunBridge] Failed to log waypoint: {e}")

    def _on_boundary(self, msg: PolygonStamped) -> None:
        if not msg.polygon.points:
            return

        try:
            pts = np.array([[p.x, p.y, p.z] for p in msg.polygon.points], dtype=np.float32)

            # Close the loop
            if len(pts) >= 2:
                pts = np.vstack([pts, pts[0]])
                self._log_with_time(
                    "world/planner/boundary",
                    rr.LineStrips3D([pts], colors=[0, 255, 0]),
                    msg.header.stamp,
                )
        except Exception as e:
            logger.debug(f"[RosRerunBridge] Failed to log boundary: {e}")

    def _on_odometry(self, msg: Odometry) -> None:
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation

            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation as R

            rot_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            
            # Log robot base pose
            with self._log_lock:
                self.rc.stream.set_time("frame", sequence=self._frame_count)
                self.rc.log(
                    "world/robot/base",
                    rr.Transform3D(
                        translation=[p.x, p.y, p.z],
                        mat3x3=rot_matrix,
                        axis_length=0.5
                    )
                )
                
                # Log forward direction arrow (yellow)
                self.rc.log(
                    "world/robot/base/heading",
                    rr.Arrows3D(
                        origins=[[0, 0, 0]],
                        vectors=[[0.3, 0, 0]],  # X-forward
                        colors=[[255, 255, 0]],
                        radii=0.02
                    )
                )
            
            # Look up camera transform relative to base_link
            try:
                ros_time = Time.from_msg(msg.header.stamp)
                tf = self.tf_buffer.lookup_transform(
                    "base_link",
                    "camera_color_optical_frame",
                    ros_time,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                t = tf.transform.translation
                q_cam = tf.transform.rotation
                
                with self._log_lock:
                    self.rc.log(
                        "world/robot/base/camera",
                        rr.Transform3D(
                            translation=[t.x, t.y, t.z],
                            rotation=rr.Quaternion(xyzw=[q_cam.x, q_cam.y, q_cam.z, q_cam.w])
                        )
                    )
            except (TransformException, Exception) as tf_err:
                # TF not available yet or frame names don't match - not critical
                pass
                
        except Exception as e:
            logger.error(f"[RosRerunBridge] Failed to log odometry: {e}", exc_info=True)
    
    # Marker callbacks
    def _convert_marker_to_rerun(self, marker: Marker, entity_path: str) -> None:
        """Convert a single RViz marker to Rerun archetype."""
        try:
            # LINE_STRIP
            if marker.type == Marker.LINE_STRIP:
                if marker.points:
                    pts = np.array([[p.x, p.y, p.z] for p in marker.points], dtype=np.float32)
                    color = [
                        int(marker.color.r * 255),
                        int(marker.color.g * 255),
                        int(marker.color.b * 255),
                        int(marker.color.a * 255),
                    ]
                    self._log_with_time(
                        entity_path, rr.LineStrips3D([pts], colors=[color]), marker.header.stamp
                    )

            # LINE_LIST (pairs of points forming line segments)
            elif marker.type == Marker.LINE_LIST:
                if marker.points and len(marker.points) >= 2:
                    pts = np.array([[p.x, p.y, p.z] for p in marker.points], dtype=np.float32)
                    # Group into line segments (every 2 points)
                    segments = []
                    for i in range(0, len(pts) - 1, 2):
                        segments.append(pts[i : i + 2])
                    if segments:
                        color = [
                            int(marker.color.r * 255),
                            int(marker.color.g * 255),
                            int(marker.color.b * 255),
                            int(marker.color.a * 255),
                        ]
                        self._log_with_time(
                            entity_path,
                            rr.LineStrips3D(segments, colors=[color]),
                            marker.header.stamp,
                        )

            # SPHERE_LIST
            elif marker.type == Marker.SPHERE_LIST:
                if marker.points:
                    pts = np.array([[p.x, p.y, p.z] for p in marker.points], dtype=np.float32)
                    radius = marker.scale.x / 2.0 if marker.scale.x > 0 else 0.1
                    color = [
                        int(marker.color.r * 255),
                        int(marker.color.g * 255),
                        int(marker.color.b * 255),
                        int(marker.color.a * 255),
                    ]
                    self._log_with_time(
                        entity_path,
                        rr.Points3D(pts, radii=radius, colors=[color]),
                        marker.header.stamp,
                    )

            # CUBE_LIST
            elif marker.type == Marker.CUBE_LIST:
                if marker.points:
                    pts = np.array([[p.x, p.y, p.z] for p in marker.points], dtype=np.float32)
                    half_size = [marker.scale.x / 2.0, marker.scale.y / 2.0, marker.scale.z / 2.0]
                    color = [
                        int(marker.color.r * 255),
                        int(marker.color.g * 255),
                        int(marker.color.b * 255),
                        int(marker.color.a * 255),
                    ]
                    # Log as boxes at each point
                    for pt in pts:
                        self._log_with_time(
                            entity_path,
                            rr.Boxes3D(half_sizes=[half_size], centers=[pt], colors=[color]),
                            marker.header.stamp,
                        )

            # SPHERE (single)
            elif marker.type == Marker.SPHERE:
                p = marker.pose.position
                radius = marker.scale.x / 2.0 if marker.scale.x > 0 else 0.1
                color = [
                    int(marker.color.r * 255),
                    int(marker.color.g * 255),
                    int(marker.color.b * 255),
                    int(marker.color.a * 255),
                ]
                self._log_with_time(
                    entity_path,
                    rr.Points3D([[p.x, p.y, p.z]], radii=radius, colors=[color]),
                    marker.header.stamp,
                )

        except Exception as e:
            logger.debug(f"[RosRerunBridge] Failed to convert marker: {e}")

    def _on_viz_path(self, msg: Marker) -> None:
        self._convert_marker_to_rerun(msg, "world/planner/viz/path")

    def _on_viz_nodes(self, msg: MarkerArray) -> None:
        for i, marker in enumerate(msg.markers):
            self._convert_marker_to_rerun(marker, f"world/planner/viz/nodes/{marker.ns or i}")

    def _on_viz_graph(self, msg: MarkerArray) -> None:
        for i, marker in enumerate(msg.markers):
            self._convert_marker_to_rerun(marker, f"world/planner/viz/graph/{marker.ns or i}")

    def _on_viz_viewpoint(self, msg: MarkerArray) -> None:
        for i, marker in enumerate(msg.markers):
            self._convert_marker_to_rerun(marker, f"world/planner/viz/viewpoint/{marker.ns or i}")


class RosRerunBridgeModule(Module):
    """Dimos module that runs the ROS→Rerun bridge."""

    _node: RosRerunBridgeNode | None = None
    _executor: MultiThreadedExecutor | None = None
    _spin_thread: threading.Thread | None = None
    _rc: RerunConnection | None = None

    @rpc
    def start(self) -> None:
        super().start()

        # Initialize Rerun connection
        self._rc = RerunConnection()

        # Debug: Check lock file
        from pathlib import Path

        from dimos.dashboard.module import config

        lock_path = Path(config["dashboard_started_lock"])
        if lock_path.exists():
            logger.info(f"[RosRerunBridge] Dashboard lock file exists: {lock_path}")
        else:
            logger.error(f"[RosRerunBridge] Dashboard lock file MISSING: {lock_path}")

        logger.info("[RosRerunBridge] RerunConnection initialized")

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()

        # Create node
        self._node = RosRerunBridgeNode(self._rc)

        # Create multi-threaded executor
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)

        # Spin in background thread
        def spin_executor() -> None:
            try:
                self._executor.spin()
            except Exception as e:
                logger.error(f"[RosRerunBridge] Executor error: {e}")

        self._spin_thread = threading.Thread(target=spin_executor, daemon=True)
        self._spin_thread.start()

        logger.info("[RosRerunBridge] Module started successfully")

    @rpc
    def stop(self) -> None:
        if self._executor:
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()
        super().stop()
        logger.info("[RosRerunBridge] Module stopped")


# Blueprint factory
ros_rerun_bridge_module = RosRerunBridgeModule.blueprint

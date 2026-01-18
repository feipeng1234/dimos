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

from dataclasses import dataclass
import os
from threading import Event, Thread
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dimos.core import In, Module, ModuleConfig, rpc
from dimos.core.rpc_client import RpcCall
from dimos.msgs.geometry_msgs import Quaternion, Transform, Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()
# Directory where this file is located
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ArucoTrackerConfig(ModuleConfig):
    """Configuration for ArUco tracker."""

    marker_size: float = 0.1  # Marker size in meters
    aruco_dict: int = cv2.aruco.DICT_4X4_50  # ArUco dictionary to use
    camera_frame_id: str = "camera_color_optical_frame"  # Frame ID for the camera
    target_marker_id: int | None = None  # Specific marker ID to track (None = track all)
    save_images: bool = True  # Whether to save annotated images
    output_dir: str = os.path.join(_THIS_DIR, "aruco_output")  # Directory to save images
    processing_rate: float = 1  # Processing rate in Hz (how often to process latest image)
    max_loops: int = 5  # Maximum number of loops to process
    move_robot_to_aruco: bool = True  # Whether to move the robot to the ArUco marker
    safety_max_delta: float = 0.10  # Max allowed 3D distance (meters) between commands
    hardware_id: str = "arm"  # Hardware ID for ControlOrchestrator EE pose lookup


class ArucoTracker(Module[ArucoTrackerConfig]):
    """
    ArUco marker tracker that detects markers in camera images and computes their transforms.

    Subscribes to camera images and camera info, detects ArUco markers,
    and publishes their transforms relative to the camera frame.
    """

    # Input ports
    color_image: In[Image]
    camera_info: In[CameraInfo]

    # Output ports
    # marker_transforms: Out[list[Transform]]

    config: ArucoTrackerConfig
    default_config = ArucoTrackerConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ArUco detector
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(self.config.aruco_dict)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        # Camera intrinsics (will be updated from camera_info)
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None

        # Latest image storage (thread-safe access)
        self._latest_image: Image | None = None
        self._latest_image_lock = Event()

        # Image counter for saving
        self._image_counter = 0

        # Threading for processing loop
        self._stop_event = Event()
        self._processing_thread: Thread | None = None
        self._loop_count = 0  # Track number of processing loops

        # RPC calls to ControlOrchestrator
        self._get_ee_positions_rpc: RpcCall | None = None
        self._move_to_cartesian_pose_rpc: RpcCall | None = None

        # Safety: track last commanded position for delta check
        self._last_commanded_pos: tuple[float, float, float] | None = None

        # Create output directory if saving images
        if self.config.save_images:
            os.makedirs(self.config.output_dir, exist_ok=True)

    @rpc
    def start(self) -> None:
        """Start the ArUco tracker by subscribing to camera streams."""
        super().start()

        # Subscribe to camera info to get intrinsics
        self._disposables.add(self.camera_info.observable().subscribe(self._update_camera_info))

        # Subscribe to color images to store latest image (don't process immediately)
        self._disposables.add(self.color_image.observable().subscribe(self._store_latest_image))

        # Reset loop counter and start processing loop thread
        self._loop_count = 0
        self._stop_event.clear()
        self._processing_thread = Thread(
            target=self._processing_loop, daemon=True, name="ArucoTracker-Processing"
        )
        self._processing_thread.start()

        # Publish static transform from base_link to world_frame
        self._publish_robot_base_to_world_transform()

    def _store_latest_image(self, image: Image) -> None:
        """Store the latest image for processing."""
        self._latest_image = image
        self._latest_image_lock.set()  # Signal that we have an image

    def _update_camera_info(self, camera_info: CameraInfo) -> None:
        """Update camera intrinsics from CameraInfo message."""
        if len(camera_info.K) == 9:
            fx, _, cx, _, fy, cy, _, _, _ = camera_info.K
            self._camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            self._dist_coeffs = (
                np.array(camera_info.D, dtype=np.float32) if camera_info.D else np.zeros(5)
            )

    @rpc
    def set_ControlOrchestrator_get_ee_positions(self, rpc_call: RpcCall) -> None:
        """Wire get_ee_positions RPC from ControlOrchestrator."""
        self._get_ee_positions_rpc = rpc_call
        self._get_ee_positions_rpc.set_rpc(self.rpc)

    @rpc
    def set_ControlOrchestrator_move_to_cartesian_pose(self, rpc_call: RpcCall) -> None:
        """Wire move_to_cartesian_pose RPC from ControlOrchestrator."""
        self._move_to_cartesian_pose_rpc = rpc_call
        self._move_to_cartesian_pose_rpc.set_rpc(self.rpc)

    def _check_safety_delta(self, x: float, y: float, z: float) -> bool:
        """Check if the new position is within safety limits of the last commanded position.

        Returns True if safe to move, False if delta exceeds limit.
        """
        if self._last_commanded_pos is None:
            # First command - initialize from current EE position if available
            if self._get_ee_positions_rpc is not None:
                try:
                    ee_positions = self._get_ee_positions_rpc()
                    if ee_positions and self.config.hardware_id in ee_positions:
                        pose = ee_positions[self.config.hardware_id]
                        if pose:
                            self._last_commanded_pos = (pose["x"], pose["y"], pose["z"])
                            logger.debug(
                                f"Initialized last_commanded_pos from EE: {self._last_commanded_pos}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to get initial EE position: {e}")

            # If still None, allow first command
            if self._last_commanded_pos is None:
                return True

        # Calculate 3D distance
        dx = x - self._last_commanded_pos[0]
        dy = y - self._last_commanded_pos[1]
        dz = z - self._last_commanded_pos[2]
        delta = (dx**2 + dy**2 + dz**2) ** 0.5

        if delta > self.config.safety_max_delta:
            logger.warning(
                f"Safety check failed: delta {delta:.3f}m exceeds limit {self.config.safety_max_delta:.3f}m"
            )
            return False

        return True

    def _publish_robot_base_to_world_transform(self) -> None:
        """Publish static transform from base_link to world."""
        robot_base_to_world_transform = Transform(
            translation=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            frame_id="world_frame",
            child_frame_id="base_link",
            ts=time.time(),
        )
        self.tf.publish(robot_base_to_world_transform)

    def _processing_loop(self) -> None:
        """Processing loop that runs at the configured rate."""
        period = 1.0 / self.config.processing_rate
        next_time = time.perf_counter() + period
        logger.info(f"ArUco processing loop started at {self.config.processing_rate}Hz")

        while not self._stop_event.is_set() and self._loop_count < self.config.max_loops:
            try:
                # Check if an image is available (non-blocking check)
                if self._latest_image_lock.is_set() and self._latest_image is not None:
                    # Process the latest image
                    self._process_image(self._latest_image)
                    self._latest_image_lock.clear()  # Reset for next image

                # Increment loop counter
                self._loop_count += 1
                logger.debug(f"Completed processing loop {self._loop_count}")

                # Exit after max loops
                if self._loop_count >= self.config.max_loops:
                    logger.debug(f"Processed {self.config.max_loops} loops, stopping...")
                    break

                # Rate control - maintain precise timing
                next_time += period
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Fell behind - reset timing
                    next_time = time.perf_counter() + period
                    if sleep_time < -period:
                        logger.warning(f"Processing loop fell behind by {-sleep_time:.3f}s")

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

        logger.info(f"ArUco processing loop completed after {self._loop_count} iterations")
        # Thread will exit naturally here - do NOT call self.stop() as it would try to join itself

    def _process_image(self, image: Image) -> None:
        """Process image to detect ArUco markers."""
        if self._camera_matrix is None or self._dist_coeffs is None:
            return  # Skip if camera info not ready yet

        # Convert image for visualization (keep original format)
        if image.format.name == "RGB":
            display_image = image.data.copy()
            gray = cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY)
        elif image.format.name == "BGR":
            display_image = image.data.copy()
            gray = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
        else:
            display_image = image.data.copy()
            gray = image.data

        # Detect ArUco markers
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return  # No markers detected, continue processing

        # Estimate pose for each detected marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.config.marker_size, self._camera_matrix, self._dist_coeffs
        )

        # Convert poses to Transform messages and collect filtered data for drawing
        transforms: list[Transform] = []
        filtered_indices: list[int] = []
        filtered_ids: list[int] = []
        filtered_corners: list[np.ndarray] = []
        filtered_rvecs: list[np.ndarray] = []
        filtered_tvecs: list[np.ndarray] = []

        for i, marker_id in enumerate(ids.flatten()):
            # Skip if tracking specific marker and this isn't it
            if (
                self.config.target_marker_id is not None
                and marker_id != self.config.target_marker_id
            ):
                continue

            # Collect filtered data for drawing
            filtered_indices.append(i)
            filtered_ids.append(int(marker_id))
            filtered_corners.append(corners[i])
            filtered_rvecs.append(rvecs[i])
            filtered_tvecs.append(tvecs[i])

            # Convert rotation vector to quaternion
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            rot_matrix, _ = cv2.Rodrigues(rvec)
            quat = Rotation.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

            # Create and publish transforms (ArUco marker and EE)
            transform = self._set_transforms(marker_id, tvec, quat, image.ts)
            if transform:
                transforms.append(transform)

            # logger.debug("--------------------------------")
            # logger.debug(f"frame_id: {self.config.camera_frame_id}")
            # logger.debug("--------------------------------")
            # logger.debug(f"parent-camera_color_optical_frame to child-aruco: {self.tf.get('camera_color_optical_frame', 'aruco_0')}")
            # logger.debug(f"parent-camera_color_frame to child-camera_color_optical_frame: {self.tf.get('camera_color_frame', 'camera_color_optical_frame')}")
            # logger.debug(f"parent-camera_link to child-camera_color_frame: {self.tf.get('camera_link', 'camera_color_frame')}")
            # logger.debug(f"parent-ee_link to child-camera_color_optical_frame: {self.tf.get('ee_link', 'camera_color_optical_frame')}")

            aruco_wrt_robot_base = self.tf.get("base_link", "aruco_0")
            logger.debug(f"aruco wrt robot base: {aruco_wrt_robot_base}")
            if aruco_wrt_robot_base is not None:
                # Calculate reach pose: adding 150mm to the z-axis and setting orientation
                # Position: aruco position + 150mm offset in z
                reach_x = aruco_wrt_robot_base.translation.x - 0.05  # 5cm offset in x
                reach_y = aruco_wrt_robot_base.translation.y
                reach_z = aruco_wrt_robot_base.translation.z + 0.20  # 20cm offset

                # Orientation: roll=-3.13, pitch=0, yaw=0 (in radians)
                reach_roll = -3.13
                reach_pitch = 0.0
                reach_yaw = 0.0

                logger.debug(
                    f"Computed reach pose: x={reach_x:.3f}, y={reach_y:.3f}, z={reach_z:.3f}, roll={reach_roll:.3f}, pitch={reach_pitch:.3f}, yaw={reach_yaw:.3f}"
                )

                # RPC call to ControlOrchestrator to move to pose
                if self.config.move_robot_to_aruco:
                    # Safety check: ensure delta is within limits
                    if not self._check_safety_delta(reach_x, reach_y, reach_z):
                        logger.warning("Skipping move command due to safety check failure")
                    elif self._move_to_cartesian_pose_rpc is not None:
                        try:
                            success = self._move_to_cartesian_pose_rpc(
                                hardware_id=self.config.hardware_id,
                                x=reach_x,
                                y=reach_y,
                                z=reach_z,
                                roll=reach_roll,
                                pitch=reach_pitch,
                                yaw=reach_yaw,
                            )
                            if not success:
                                logger.warning("Failed to move to pose, stopping processing loop")
                                self._loop_count = self.config.max_loops
                                break
                            else:
                                self._last_commanded_pos = (reach_x, reach_y, reach_z)
                                logger.debug("Successfully commanded move to pose")
                        except Exception as e:
                            logger.error(f"Error calling move_to_cartesian_pose: {e}")
                            self._loop_count = self.config.max_loops
                            break
                    else:
                        logger.warning("move_to_cartesian_pose RPC not available")

        # Draw markers and save image (using filtered data)
        if transforms:
            filtered_ids_array = np.array(filtered_ids).reshape(-1, 1)
            self._draw_markers(
                display_image,
                filtered_corners,
                filtered_ids_array,
                np.array(filtered_rvecs),
                np.array(filtered_tvecs),
                transforms,
                image.format.name,
            )

    def _set_transforms(
        self, marker_id: int, tvec: np.ndarray, quat: np.ndarray, timestamp: float
    ) -> Transform | None:
        """Set transforms: aruco marker to camera, and base_link to ee_link (from RPC call).

        Args:
            marker_id: ArUco marker ID
            tvec: Translation vector from marker detection
            quat: Quaternion rotation from marker detection [x, y, z, w]
            timestamp: Timestamp for the transforms

        Returns:
            The ArUco marker transform, or None if creation failed
        """
        # Create ArUco marker transform (camera_optical -> aruco_{marker_id})
        aruco_transform = Transform(
            translation=Vector3(float(tvec[0]), float(tvec[1]), float(tvec[2])),
            rotation=Quaternion(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            frame_id=self.config.camera_frame_id,
            child_frame_id=f"aruco_{marker_id}",
            ts=timestamp,
        )

        # Publish ArUco marker transform
        self.tf.publish(aruco_transform)

        # Publish base_link -> ee_link transform from ControlOrchestrator
        if self._get_ee_positions_rpc is not None:
            try:
                # Get end-effector positions from ControlOrchestrator via RPC
                # Returns dict[hardware_id, dict[x,y,z,roll,pitch,yaw]]
                ee_positions = self._get_ee_positions_rpc()
                if ee_positions is not None:
                    hw_pose = ee_positions.get(self.config.hardware_id)
                    if hw_pose is not None:
                        x, y, z = hw_pose["x"], hw_pose["y"], hw_pose["z"]
                        roll, pitch, yaw = hw_pose["roll"], hw_pose["pitch"], hw_pose["yaw"]
                        # Convert Euler angles to quaternion
                        from dimos.utils.transform_utils import euler_to_quaternion

                        orientation = euler_to_quaternion(Vector3(roll, pitch, yaw))
                        # Create and publish transform from base_link to ee_link
                        ee_transform = Transform(
                            translation=Vector3(float(x), float(y), float(z)),
                            rotation=orientation,
                            frame_id="base_link",
                            child_frame_id="ee_link",
                            ts=timestamp,
                        )
                        self.tf.publish(ee_transform)
                    else:
                        logger.warning(f"No EE pose for hardware_id '{self.config.hardware_id}'")
            except Exception as e:
                logger.error(f"Error getting EE pose from ControlOrchestrator: {e}")
        else:
            logger.warning(
                "ControlOrchestrator RPC not available, cannot publish base_link -> ee_link transform"
            )
        return aruco_transform

    def _draw_markers(
        self,
        display_image: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray,
        rvecs: np.ndarray,
        tvecs: np.ndarray,
        transforms: list[Transform],
        image_format: str,
    ) -> None:
        """
        Draw ArUco markers, axes, and text overlays on the image.

        Args:
            display_image: Image to draw on (will be modified in place)
            corners: Detected marker corners
            ids: Detected marker IDs
            rvecs: Rotation vectors for each marker
            tvecs: Translation vectors for each marker
            transforms: List of transforms for each marker
            image_format: Format of the image (for saving)
        """
        # Draw all detected markers
        cv2.aruco.drawDetectedMarkers(display_image, corners, ids)

        # Draw axes and text for each marker
        text_y_offset = 30
        for i, (marker_id, _transform) in enumerate(zip(ids.flatten(), transforms, strict=False)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            # Draw coordinate axes (length = marker_size)
            # Uses OpenCV ArUco convention: Z out of marker, X/Y in marker plane
            # Axes rotate with the marker when paper is rotated
            axis_length = self.config.marker_size * 0.5
            cv2.drawFrameAxes(
                display_image,
                self._camera_matrix,
                self._dist_coeffs,
                rvec,
                tvec,
                axis_length,
            )

            # Draw text with transform info for this marker
            text_y = text_y_offset + (i * 50)
            cv2.putText(
                display_image,
                f"Marker ID: {marker_id}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            text_y += 25
            cv2.putText(
                display_image,
                f"Pos: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Save annotated image (convert RGB to BGR if needed for OpenCV)
        if self.config.save_images:
            save_image = display_image.copy()
            if image_format == "RGB":
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            filename = os.path.join(
                self.config.output_dir, f"aruco_detection_{self._image_counter:05d}.png"
            )
            cv2.imwrite(filename, save_image)
            logger.debug(f"Saved annotated image to: {filename}")
            self._image_counter += 1

    @rpc
    def stop(self) -> None:
        """Stop the ArUco tracker."""
        # Signal processing thread to stop
        self._stop_event.set()

        # Wait for processing thread to finish
        if self._processing_thread is not None and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        super().stop()


aruco_tracker = ArucoTracker.blueprint

__all__ = ["ArucoTracker", "ArucoTrackerConfig", "aruco_tracker"]

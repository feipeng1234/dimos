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

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any
import uuid

import cv2
from dimos_lcm.geometry_msgs import Pose
from dimos_lcm.vision_msgs import ObjectHypothesis, ObjectHypothesisWithPose
import numpy as np
import open3d as o3d

from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Transform, Vector3
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.msgs.std_msgs import Header
from dimos.msgs.vision_msgs import Detection3D as ROSDetection3D, Detection3DArray
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg
from dimos.perception.detection.type.detection3d.base import Detection3D

if TYPE_CHECKING:
    from dimos_lcm.sensor_msgs import CameraInfo

    from dimos.perception.detection.type.detection2d import ImageDetections2D


@dataclass
class Object(Detection3D):
    """3D object detection combining bounding box and pointcloud representations.

    Represents a detected object in 3D space with support for accumulating
    multiple detections over time. Optionally includes mesh data and accurate
    6D pose from hosted mesh/pose service.
    """

    object_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    pointcloud: PointCloud2 | None = None
    camera_transform: Transform | None = None
    center: Vector3 | None = None
    size: Vector3 | None = None
    detections_count: int = 1

    # Mesh/pose enhancement (optional, from hosted service)
    mesh_obj: bytes | None = None
    mesh_path: str | None = None
    mesh_dimensions: tuple[float, float, float] | None = None
    fp_position: tuple[float, float, float] | None = None
    fp_orientation: tuple[float, float, float, float] | None = None

    @property
    def has_mesh(self) -> bool:
        """Check if mesh data is available from hosted service."""
        return self.mesh_obj is not None and len(self.mesh_obj) > 0

    @property
    def has_accurate_pose(self) -> bool:
        """Check if FoundationPose data is available from hosted service."""
        return self.fp_position is not None and self.fp_orientation is not None

    @property
    def pose(self) -> PoseStamped | None:
        """Get 6D pose, preferring FoundationPose if available."""
        if self.has_accurate_pose:
            return PoseStamped(
                ts=self.ts,
                frame_id=self.frame_id,
                position=Vector3(*self.fp_position),
                orientation=self.fp_orientation,
            )
        # Fallback to pointcloud center with identity rotation
        if self.center is not None:
            return PoseStamped(
                ts=self.ts,
                frame_id=self.frame_id,
                position=self.center,
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
        return None

    def save_mesh(self, path: str) -> bool:
        """Save mesh to an .obj file if available.

        Args:
            path: File path to save the mesh to.

        Returns:
            True if saved successfully, False if no mesh data available.
        """
        if not self.has_mesh:
            return False
        try:
            with open(path, "wb") as f:
                f.write(self.mesh_obj)
            return True
        except Exception:
            return False

    def update_object(self, other: Object) -> None:
        """Update this object with data from another detection.

        Accumulates pointclouds by transforming the new pointcloud to world frame
        and adding it to the existing pointcloud. Updates center and camera_transform,
        and increments the detections_count.

        Args:
            other: Another Object instance with newer detection data.
        """
        # Accumulate pointclouds if both exist and transforms are available
        if (
            self.pointcloud is not None
            and other.pointcloud is not None
            and other.camera_transform is not None
        ):
            # Transform new pointcloud to world frame and add to existing
            self.pointcloud = self.pointcloud + other.pointcloud
            self.pointcloud = self.pointcloud.voxel_downsample(voxel_size=0.025)

            # Recompute center from accumulated pointcloud
            pc_center = self.pointcloud.center
            self.center = Vector3(pc_center.x, pc_center.y, pc_center.z)
        else:
            # No transform available, just replace
            self.pointcloud = other.pointcloud
            self.center = other.center

        self.camera_transform = other.camera_transform
        self.detections_count += 1

    def get_oriented_bounding_box(self):
        """Get oriented bounding box of the pointcloud."""
        if self.pointcloud is None:
            raise ValueError("No pointcloud available")
        return self.pointcloud.get_oriented_bounding_box()

    def scene_entity_label(self) -> str:
        """Get label for scene visualization."""
        if self.detections_count > 1:
            return f"{self.name} ({self.detections_count})"
        return f"{self.track_id}/{self.name} ({self.confidence:.0%})"

    def to_ros_detection3d(self) -> ROSDetection3D:
        """Convert to ROS Detection3D message."""
        msg = ROSDetection3D()
        msg.header = Header(self.ts, self.frame_id)
        msg.results = [
            ObjectHypothesisWithPose(
                hypothesis=ObjectHypothesis(
                    class_id=str(self.class_id),
                    score=self.confidence,
                )
            )
        ]

        obb = self.get_oriented_bounding_box()
        obb_center = obb.center
        obb_extent = obb.extent
        orientation = Quaternion.from_rotation_matrix(obb.R)

        msg.bbox.center = Pose(
            position=Vector3(obb_center[0], obb_center[1], obb_center[2]),
            orientation=orientation,
        )
        msg.bbox.size = Vector3(obb_extent[0], obb_extent[1], obb_extent[2])

        return msg

    def agent_encode(self) -> dict[str, Any]:
        """Encode for agent consumption."""
        return {
            "id": self.track_id,
            "name": self.name,
            "detections": self.detections_count,
            "last_seen": f"{round(time.time() - self.ts)}s ago",
        }

    @classmethod
    def from_2d(
        cls,
        detections_2d: ImageDetections2D,
        color_image: Image,
        depth_image: Image,
        camera_info: CameraInfo,
        camera_transform: Transform | None = None,
        depth_scale: float = 1.0,
        depth_trunc: float = 10.0,
        statistical_nb_neighbors: int = 10,
        statistical_std_ratio: float = 0.5,
        mask_erode_pixels: int = 3,
    ) -> list[Object]:
        """Create 3D Objects from 2D detections and RGBD images.

        Uses Open3D's optimized RGBD projection for efficient processing.

        Args:
            detections_2d: 2D detections with segmentation masks
            color_image: RGB color image
            depth_image: Depth image (in meters if depth_scale=1.0)
            camera_info: Camera intrinsics
            camera_transform: Optional transform from camera frame to world frame.
                If provided, pointclouds will be transformed to world frame.
            depth_scale: Scale factor for depth (1.0 for meters, 1000.0 for mm)
            depth_trunc: Maximum depth value in meters
            statistical_nb_neighbors: Neighbors for statistical outlier removal
            statistical_std_ratio: Std ratio for statistical outlier removal
            mask_erode_pixels: Number of pixels to erode the mask by to remove
                              noisy depth edge points. Set to 0 to disable.

        Returns:
            List of Object instances with pointclouds
        """
        color_cv = color_image.to_opencv()
        if color_cv.ndim == 3 and color_cv.shape[2] == 3:
            color_cv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)

        depth_cv = depth_image.to_opencv()
        h, w = depth_cv.shape[:2]

        # Build Open3D camera intrinsics
        fx, fy = camera_info.K[0], camera_info.K[4]
        cx, cy = camera_info.K[2], camera_info.K[5]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        objects: list[Object] = []

        for det in detections_2d.detections:
            if isinstance(det, Detection2DSeg):
                mask = det.mask
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, det.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 255

            if mask_erode_pixels > 0:
                mask_uint8 = mask.astype(np.uint8)
                if mask_uint8.max() == 1:
                    mask_uint8 = mask_uint8 * 255
                kernel_size = 2 * mask_erode_pixels + 1
                erode_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                mask = cv2.erode(mask_uint8, erode_kernel)

            depth_masked = depth_cv.copy()
            depth_masked[mask == 0] = 0

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_cv.astype(np.uint8)),
                o3d.geometry.Image(depth_masked.astype(np.float32)),
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)

            pcd_filtered, _ = pcd.remove_statistical_outlier(
                nb_neighbors=statistical_nb_neighbors,
                std_ratio=statistical_std_ratio,
            )

            if len(pcd_filtered.points) < 10:
                continue

            pc = PointCloud2(
                pcd_filtered,
                frame_id=depth_image.frame_id,
                ts=depth_image.ts,
            )

            # Transform pointcloud to world frame if camera_transform is provided
            if camera_transform is not None:
                pc = pc.transform(camera_transform)
                frame_id = camera_transform.frame_id
            else:
                frame_id = depth_image.frame_id

            # Compute center from pointcloud
            pc_center = pc.center
            center = Vector3(pc_center.x, pc_center.y, pc_center.z)

            objects.append(
                cls(
                    bbox=det.bbox,
                    track_id=det.track_id,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    name=det.name,
                    ts=det.ts,
                    image=det.image,
                    frame_id=frame_id,
                    pointcloud=pc,
                    center=center,
                    camera_transform=camera_transform,
                )
            )

        return objects


def aggregate_pointclouds(objects: list[Object]) -> PointCloud2 | None:
    """Aggregate all object pointclouds into a single colored pointcloud.

    Each object's points are colored based on its track_id.

    Args:
        objects: List of Object instances with pointclouds

    Returns:
        Combined PointCloud2 with all points colored by object, or None if empty.
    """
    if not objects:
        return None

    all_points = []
    all_colors = []

    for i, obj in enumerate(objects):
        if obj.pointcloud is None:
            continue

        points, colors = obj.pointcloud.as_numpy(include_colors=True)
        if len(points) == 0:
            continue

        track_id = obj.track_id if obj.track_id >= 0 else i
        np.random.seed(abs(track_id))
        track_color = np.random.randint(50, 255, 3) / 255.0

        if colors is not None:
            blended = np.clip(0.6 * colors + 0.4 * track_color, 0.0, 1.0)
        else:
            blended = np.tile(track_color, (len(points), 1))

        all_points.append(points)
        all_colors.append(blended)

    if not all_points:
        return None

    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    pc = PointCloud2.from_numpy(
        combined_points,
        frame_id=objects[0].frame_id,
        timestamp=objects[0].ts,
    )
    pc.pointcloud.colors = o3d.utility.Vector3dVector(combined_colors)

    return pc


def to_detection3d_array(objects: list[Object]) -> Detection3DArray:
    """Convert a list of Objects to a ROS Detection3DArray message.

    Args:
        objects: List of Object instances

    Returns:
        Detection3DArray ROS message
    """
    array = Detection3DArray()

    if objects:
        array.header = Header(objects[0].ts, objects[0].frame_id)

    for obj in objects:
        array.detections.append(obj.to_ros_detection3d())

    return array

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

"""2D detection → world-frame 3D pointcloud via lidar back-projection.

Single function ``back_project_bbox`` projects a world-frame
pointcloud into the camera image plane (using the camera's intrinsics
+ world→optical transform), keeps only points that fall inside the
2D bbox and pass simple geometric sanity filters (radius outlier,
statistical outlier), and returns the resulting world-frame
pointcloud.  Logic ported and simplified from
``perception.detection.type.detection3d.pointcloud.Detection3DPC.from_2d``
— same algorithm, no Detection2DBBox / Object inheritance baggage.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def back_project_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    world_pointcloud: PointCloud2,
    camera_info: CameraInfo,
    world_to_optical: Transform,
    *,
    margin_px: int = 5,
    radius_outlier_nb: int = 3,
    radius_outlier_radius: float = 0.20,
    statistical_outlier_nb: int = 6,
    statistical_outlier_std_ratio: float = 3.0,
    min_points: int = 3,
) -> np.ndarray | None:
    """Slice a world-frame pointcloud by a 2D image bbox; return ``(N, 3)``.

    Returns ``None`` if no valid points remain after filtering.

    Parameters
    ----------
    bbox_xyxy:
        2D bbox in image pixels: ``(x_min, y_min, x_max, y_max)``.
    world_pointcloud:
        Lidar pointcloud already in world frame (e.g. the fused
        ``/lidar`` topic from ``MujocoSimModule``).
    camera_info:
        Pinhole intrinsics; ``K`` matters, ``D`` ignored (we don't
        undistort here).
    world_to_optical:
        Transform whose ``to_matrix()`` maps homogeneous world points
        to the camera optical frame.  Sourced from
        ``module.tf.get("camera_optical", world_pointcloud.frame_id, ts)``.
    margin_px:
        Extra pixels to expand the bbox by before slicing — accounts
        for slight bbox imprecision and lidar discretization.
    radius_outlier_nb / radius_outlier_radius:
        Open3D radius outlier removal — drops points without
        ``radius_outlier_nb`` neighbors in ``radius_outlier_radius`` m.
        Defaults are tuned for the MujocoSimModule lidar fused at
        5 cm voxel size — radius 0.20 m at nb=3 keeps small clusters
        while still rejecting isolated points; tighter values
        (the dense-RGBD defaults of nb=8 / r=0.05 m) drop everything
        because points are ~5 cm apart by construction.
    statistical_outlier_nb / statistical_outlier_std_ratio:
        Open3D statistical outlier removal — drops points whose mean
        neighbor distance exceeds the per-cluster mean by more than
        ``std_ratio`` standard deviations.
    min_points:
        Below this count post-filter we treat the slice as noise and
        return None instead of a tiny pointcloud.  Default 3 because
        tight VLM bboxes on a sparse fused lidar typically yield
        single-digit point counts; demanding more loses real detections.
    """
    fx, fy = camera_info.K[0], camera_info.K[4]
    cx, cy = camera_info.K[2], camera_info.K[5]
    width = camera_info.width
    height = camera_info.height
    K_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    world_xyz, _ = world_pointcloud.as_numpy()
    if world_xyz.size == 0:
        logger.info("back_project_bbox: empty world pointcloud")
        return None
    n_total = world_xyz.shape[0]

    # World → camera optical
    world_h = np.hstack([world_xyz, np.ones((world_xyz.shape[0], 1))])
    extrinsics = world_to_optical.to_matrix()
    cam_xyz = (extrinsics @ world_h.T).T[:, :3]

    in_front = cam_xyz[:, 2] > 0
    n_in_front = int(in_front.sum())
    if not np.any(in_front):
        logger.info(
            "back_project_bbox bbox=%s: 0/%d points in front of camera",
            bbox_xyxy,
            n_total,
        )
        return None
    cam_xyz = cam_xyz[in_front]
    world_xyz = world_xyz[in_front]

    # Optical → image plane
    pix_h = (K_mat @ cam_xyz.T).T
    pix = pix_h[:, :2] / pix_h[:, 2:3]

    in_image = (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    n_in_image = int(in_image.sum())
    if not np.any(in_image):
        logger.info(
            "back_project_bbox bbox=%s: 0/%d in image (front=%d, img=%dx%d, K=fx=%.1f cx=%.1f)",
            bbox_xyxy,
            n_total,
            n_in_front,
            width,
            height,
            fx,
            cx,
        )
        return None
    pix = pix[in_image]
    world_xyz = world_xyz[in_image]

    x_min, y_min, x_max, y_max = bbox_xyxy
    in_bbox = (
        (pix[:, 0] >= x_min - margin_px)
        & (pix[:, 0] <= x_max + margin_px)
        & (pix[:, 1] >= y_min - margin_px)
        & (pix[:, 1] <= y_max + margin_px)
    )
    n_in_bbox = int(in_bbox.sum())
    detection_pts = world_xyz[in_bbox]
    if detection_pts.shape[0] < min_points:
        logger.info(
            "back_project_bbox bbox=%s: only %d in bbox (front=%d, image=%d, total=%d)",
            bbox_xyxy,
            n_in_bbox,
            n_in_front,
            n_in_image,
            n_total,
        )
        return None

    # Outlier filtering via Open3D — same defaults the old
    # Detection3DPC pipeline uses, just exposed as kwargs.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(detection_pts)
    pcd, _ = pcd.remove_radius_outlier(nb_points=radius_outlier_nb, radius=radius_outlier_radius)
    n_after_radius = len(pcd.points)
    if n_after_radius < min_points:
        logger.info(
            "back_project_bbox bbox=%s: radius filter dropped %d -> %d (radius=%.3fm, nb=%d)",
            bbox_xyxy,
            n_in_bbox,
            n_after_radius,
            radius_outlier_radius,
            radius_outlier_nb,
        )
        return None
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=statistical_outlier_nb,
        std_ratio=statistical_outlier_std_ratio,
    )
    n_after_stat = len(pcd.points)
    if n_after_stat < min_points:
        logger.info(
            "back_project_bbox bbox=%s: statistical filter dropped %d -> %d",
            bbox_xyxy,
            n_after_radius,
            n_after_stat,
        )
        return None

    logger.info(
        "back_project_bbox bbox=%s: kept %d points (in_bbox=%d, after_radius=%d)",
        bbox_xyxy,
        n_after_stat,
        n_in_bbox,
        n_after_radius,
    )
    return np.asarray(pcd.points, dtype=np.float32)


__all__ = ["back_project_bbox"]

#!/usr/bin/env python3

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

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import open3d as o3d  # type: ignore[import-untyped]

from dimos.simulation.mujoco.constants import MAX_HEIGHT, MAX_RANGE, MIN_RANGE


def depth_image_to_point_cloud(
    depth_image: NDArray[Any],
    camera_pos: NDArray[Any],
    camera_mat: NDArray[Any],
    fov_degrees: float = 120,
    max_range: float = MAX_RANGE,
    max_height: float = MAX_HEIGHT,
    min_range: float = MIN_RANGE,
) -> NDArray[Any]:
    """Convert a depth image from a camera to a 3D point cloud."""
    height, width = depth_image.shape

    # Calculate camera intrinsics similar to StackOverflow approach
    fovy = math.radians(fov_degrees)
    f = height / (2 * math.tan(fovy / 2))  # focal length in pixels
    cx = width / 2  # principal point x
    cy = height / 2  # principal point y

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, cx, cy)
    o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32))
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, cam_intrinsics)
    camera_points: NDArray[Any] = np.asarray(o3d_cloud.points)

    if camera_points.size == 0:
        return np.array([]).reshape(0, 3)

    # Flip y and z axes
    camera_points[:, 1] = -camera_points[:, 1]
    camera_points[:, 2] = -camera_points[:, 2]

    # y (index 1) is up here
    valid_mask = (
        (np.abs(camera_points[:, 0]) <= max_range)
        & (np.abs(camera_points[:, 1]) <= max_height)
        & (np.abs(camera_points[:, 2]) >= min_range)
        & (np.abs(camera_points[:, 2]) <= max_range)
    )
    camera_points = camera_points[valid_mask]

    if camera_points.size == 0:
        return np.array([]).reshape(0, 3)

    # Transform to world coordinates
    world_points: NDArray[Any] = (camera_mat @ camera_points.T).T + camera_pos

    return world_points

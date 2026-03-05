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

"""Visualization helpers for Memory2 search results.

Produces LCM-publishable messages (OccupancyGrid, PoseStamped) and
Rerun time-series plots from embedding search observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dimos.memory.types import Observation
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid


def similarity_heatmap(
    observations: list[Observation] | Any,
    *,
    resolution: float = 0.1,
    padding: float = 1.0,
    frame_id: str = "world",
) -> OccupancyGrid:
    """Build an OccupancyGrid heatmap from observations with similarity scores.

    Each observation's pose maps to a grid cell; the cell value is
    ``int(similarity * 100)`` (0-100 scale).  Unknown cells stay at -1.

    Args:
        observations: Iterable of Observation (must have .pose and .similarity).
        resolution: Grid resolution in metres/cell.
        padding: Extra metres around the bounding box.
        frame_id: Coordinate frame for the grid.

    Returns:
        OccupancyGrid publishable via LCMTransport.
    """
    from dimos.memory.types import EmbeddingObservation
    from dimos.msgs.geometry_msgs.Pose import Pose
    from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid as OG

    posed: list[tuple[float, float, float]] = []
    for obs in observations:
        if obs.pose is None:
            continue
        sim = (
            obs.similarity
            if isinstance(obs, EmbeddingObservation) and obs.similarity is not None
            else 0.0
        )
        p = obs.pose.position
        posed.append((p.x, p.y, sim))

    if not posed:
        return OG(width=1, height=1, resolution=resolution, frame_id=frame_id)

    xs = [p[0] for p in posed]
    ys = [p[1] for p in posed]

    min_x = min(xs) - padding
    min_y = min(ys) - padding
    max_x = max(xs) + padding
    max_y = max(ys) + padding

    width = max(1, int((max_x - min_x) / resolution) + 1)
    height = max(1, int((max_y - min_y) / resolution) + 1)

    grid = np.full((height, width), -1, dtype=np.int8)

    for px, py, sim in posed:
        gx = int((px - min_x) / resolution)
        gy = int((py - min_y) / resolution)
        gx = min(gx, width - 1)
        gy = min(gy, height - 1)
        val = int(sim * 100)
        # Keep max similarity per cell
        if grid[gy, gx] < val:
            grid[gy, gx] = np.int8(val)

    origin = Pose(
        position=[min_x, min_y, 0.0],
        orientation=[0.0, 0.0, 0.0, 1.0],
    )

    return OG(grid=grid, resolution=resolution, origin=origin, frame_id=frame_id)


def similarity_poses(observations: list[Observation] | Any) -> list[PoseStamped]:
    """Extract PoseStamped from observations for spatial arrow rendering.

    Args:
        observations: Iterable of Observation with .pose.

    Returns:
        List of PoseStamped suitable for LCMTransport publishing.
    """
    result: list[PoseStamped] = []
    for obs in observations:
        if obs.pose is not None:
            result.append(obs.pose)
    return result


def log_similarity_timeline(
    observations: list[Observation] | Any,
    entity_path: str = "memory/similarity",
) -> None:
    """Log similarity scores as a Rerun time-series plot.

    Each observation is logged at its timestamp with its similarity score.
    Rerun auto-generates an interactive time-series graph in the timeline panel.

    Args:
        observations: Iterable of EmbeddingObservation with .similarity and .ts.
        entity_path: Rerun entity path for the scalar series.
    """
    import rerun as rr

    from dimos.memory.types import EmbeddingObservation

    for obs in observations:
        if not isinstance(obs, EmbeddingObservation):
            continue
        if obs.similarity is None or obs.ts is None:
            continue
        rr.set_time("memory_time", timestamp=obs.ts)
        rr.log(entity_path, rr.Scalars(obs.similarity))

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

"""Canonical reference path battery for the controller benchmark.

Every path starts at the origin facing +x in the robot frame. Each
:class:`PoseStamped` waypoint carries the path-tangent yaw at that point.
"""

from __future__ import annotations

import math

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def _path_from_xy(xs: list[float], ys: list[float]) -> Path:
    """Build a Path with tangent yaw at each waypoint."""
    n = len(xs)
    poses: list[PoseStamped] = []
    for i in range(n):
        if i < n - 1:
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
        else:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
        yaw = math.atan2(dy, dx)
        poses.append(_pose(xs[i], ys[i], yaw))
    return Path(poses=poses)


# ---------------------------------------------------------------------------
# Path generators
# ---------------------------------------------------------------------------


def straight_line(length: float = 5.0, step: float = 0.05) -> Path:
    n = round(length / step)
    xs = [i * step for i in range(n + 1)]
    ys = [0.0] * (n + 1)
    return _path_from_xy(xs, ys)


def single_corner(leg_length: float = 2.0, angle_deg: float = 90.0, step: float = 0.05) -> Path:
    """Two straight legs meeting at one corner.

    Robot starts at origin going +x, drives ``leg_length``, turns by
    ``angle_deg`` (left positive), drives another ``leg_length``.
    """
    angle = math.radians(angle_deg)
    n_leg = round(leg_length / step)

    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_leg + 1):
        xs.append(i * step)
        ys.append(0.0)
    corner_x, corner_y = xs[-1], ys[-1]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for i in range(1, n_leg + 1):
        d = i * step
        xs.append(corner_x + d * cos_a)
        ys.append(corner_y + d * sin_a)
    return _path_from_xy(xs, ys)


def circle(radius: float = 1.0, n_points: int = 100) -> Path:
    """Closed circle, robot starts at origin going +x, curves left.

    Center at (0, ``radius``). Last waypoint coincides with the first.
    """
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_points + 1):
        theta = 2.0 * math.pi * i / n_points
        xs.append(radius * math.sin(theta))
        ys.append(radius * (1.0 - math.cos(theta)))
    return _path_from_xy(xs, ys)


def figure_eight(loop_radius: float = 1.0, n_points: int = 200) -> Path:
    """Lemniscate of Gerono.

    x(t) = R sin(2t), y(t) = R sin(t), t in [0, 2pi].
    Starts at origin going +x.
    """
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_points + 1):
        t = 2.0 * math.pi * i / n_points
        xs.append(loop_radius * math.sin(2.0 * t))
        ys.append(loop_radius * math.sin(t))
    return _path_from_xy(xs, ys)


def slalom(
    cone_spacing: float = 1.0,
    lateral_offset: float = 0.5,
    n_cones: int = 5,
    points_per_cone: int = 20,
) -> Path:
    """Smooth slalom past ``n_cones`` cones, alternating sides.

    Cones sit at (i * cone_spacing, +/-lateral_offset). The path is a
    sinusoid that crosses the centerline between cones.
    """
    total_length = (n_cones + 1) * cone_spacing
    n = n_cones * points_per_cone + points_per_cone
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n + 1):
        x = total_length * i / n
        y = lateral_offset * math.sin(math.pi * x / cone_spacing)
        xs.append(x)
        ys.append(y)
    return _path_from_xy(xs, ys)


def square(side: float = 2.0, step: float = 0.05) -> Path:
    """Closed square. Origin → +x → +y → -x → -y back to origin."""
    n_side = round(side / step)

    xs: list[float] = []
    ys: list[float] = []
    # leg 1: +x
    for i in range(n_side + 1):
        xs.append(i * step)
        ys.append(0.0)
    # leg 2: +y
    for i in range(1, n_side + 1):
        xs.append(side)
        ys.append(i * step)
    # leg 3: -x
    for i in range(1, n_side + 1):
        xs.append(side - i * step)
        ys.append(side)
    # leg 4: -y
    for i in range(1, n_side + 1):
        xs.append(0.0)
        ys.append(side - i * step)
    return _path_from_xy(xs, ys)


# ---------------------------------------------------------------------------
# Battery registry
# ---------------------------------------------------------------------------


def default_battery() -> dict[str, Path]:
    """All canonical paths used for the standard benchmark report."""
    return {
        "straight_5m": straight_line(length=5.0),
        "corner_90": single_corner(leg_length=2.0, angle_deg=90.0),
        "circle_R0.5": circle(radius=0.5),
        "circle_R1.0": circle(radius=1.0),
        "circle_R2.0": circle(radius=2.0),
        "figure_eight_R1.0": figure_eight(loop_radius=1.0),
        "slalom_5cones": slalom(),
        "square_2m": square(side=2.0),
    }


# ---------------------------------------------------------------------------
# SVG rendering (for visual fixtures)
# ---------------------------------------------------------------------------


def path_to_svg(path: Path, size_px: int = 400, margin_px: int = 20) -> str:
    """Render a Path as an SVG polyline (for visual inspection)."""
    if not path.poses:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}"/>'

    xs = [p.position.x for p in path.poses]
    ys = [p.position.y for p in path.poses]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    scale = (size_px - 2 * margin_px) / max(span_x, span_y)

    def _xy(p: PoseStamped) -> tuple[float, float]:
        # Flip y so +y points up in SVG coords.
        sx = margin_px + (p.position.x - x_min) * scale
        sy = size_px - (margin_px + (p.position.y - y_min) * scale)
        return sx, sy

    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (_xy(p) for p in path.poses))
    start_x, start_y = _xy(path.poses[0])
    end_x, end_y = _xy(path.poses[-1])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<polyline points="{pts}" stroke="black" fill="none" stroke-width="1.5"/>'
        f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="4" fill="green"/>'
        f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="4" fill="red"/>'
        f"</svg>"
    )


def trajectory_to_svg(
    reference: Path,
    executed_xy: list[tuple[float, float]],
    size_px: int = 500,
    margin_px: int = 20,
) -> str:
    """Render a reference path (gray) overlaid with executed trajectory (blue)."""
    if not reference.poses or not executed_xy:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}"/>'

    ref_xs = [p.position.x for p in reference.poses]
    ref_ys = [p.position.y for p in reference.poses]
    exe_xs = [x for x, _ in executed_xy]
    exe_ys = [y for _, y in executed_xy]

    x_min = min(min(ref_xs), min(exe_xs))
    x_max = max(max(ref_xs), max(exe_xs))
    y_min = min(min(ref_ys), min(exe_ys))
    y_max = max(max(ref_ys), max(exe_ys))

    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    scale = (size_px - 2 * margin_px) / max(span_x, span_y)

    def _xy(x: float, y: float) -> tuple[float, float]:
        sx = margin_px + (x - x_min) * scale
        sy = size_px - (margin_px + (y - y_min) * scale)
        return sx, sy

    ref_pts = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (_xy(p.position.x, p.position.y) for p in reference.poses)
    )
    exe_pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (_xy(x, y) for x, y in executed_xy))
    sx, sy = _xy(executed_xy[0][0], executed_xy[0][1])
    ex, ey = _xy(executed_xy[-1][0], executed_xy[-1][1])

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<polyline points="{ref_pts}" stroke="lightgray" fill="none" stroke-width="3"/>'
        f'<polyline points="{exe_pts}" stroke="#1f77b4" fill="none" stroke-width="1.5"/>'
        f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="4" fill="green"/>'
        f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="4" fill="red"/>'
        f"</svg>"
    )


_COHORT_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
]


def multi_trajectory_to_svg(
    reference: Path,
    cohorts: dict[str, list[tuple[float, float]]],
    size_px: int = 600,
    margin_px: int = 30,
    title: str | None = None,
) -> str:
    """Render reference + multiple executed trajectories on one SVG with a legend."""
    if not reference.poses:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}"/>'

    ref_xs = [p.position.x for p in reference.poses]
    ref_ys = [p.position.y for p in reference.poses]
    all_xs = list(ref_xs)
    all_ys = list(ref_ys)
    for xy in cohorts.values():
        all_xs.extend(x for x, _ in xy)
        all_ys.extend(y for _, y in xy)

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    plot_h = size_px - 2 * margin_px - (20 + 18 * len(cohorts))  # leave room for legend
    plot_w = size_px - 2 * margin_px
    scale = min(plot_w / span_x, plot_h / span_y)

    def _xy(x: float, y: float) -> tuple[float, float]:
        sx = margin_px + (x - x_min) * scale
        sy = margin_px + plot_h - (y - y_min) * scale
        return sx, sy

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if title:
        parts.append(
            f'<text x="{size_px // 2}" y="20" font-family="monospace" font-size="13" '
            f'text-anchor="middle" font-weight="bold">{title}</text>'
        )

    ref_pts = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (_xy(p.position.x, p.position.y) for p in reference.poses)
    )
    parts.append(f'<polyline points="{ref_pts}" stroke="lightgray" fill="none" stroke-width="4"/>')

    legend_y = margin_px + plot_h + 16
    for i, (name, xy) in enumerate(cohorts.items()):
        color = _COHORT_COLORS[i % len(_COHORT_COLORS)]
        if xy:
            exe_pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (_xy(x, y) for x, y in xy))
            parts.append(
                f'<polyline points="{exe_pts}" stroke="{color}" fill="none" stroke-width="1.5"/>'
            )
            sx, sy = _xy(xy[0][0], xy[0][1])
            parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3" fill="{color}"/>')
        # Legend row
        ly = legend_y + i * 18
        parts.append(
            f'<line x1="{margin_px}" y1="{ly}" x2="{margin_px + 20}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{margin_px + 26}" y="{ly + 4}" font-family="monospace" '
            f'font-size="11">{name}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


__all__ = [
    "circle",
    "default_battery",
    "figure_eight",
    "multi_trajectory_to_svg",
    "path_to_svg",
    "single_corner",
    "slalom",
    "square",
    "straight_line",
    "trajectory_to_svg",
]

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

"""Smoke tests + visual fixture rendering for the path battery."""

from __future__ import annotations

import math
from pathlib import Path as FsPath

import pytest

from dimos.utils.benchmarking.paths import (
    circle,
    default_battery,
    figure_eight,
    path_to_svg,
    single_corner,
    slalom,
    square,
    straight_line,
)


def test_straight_line_starts_at_origin_facing_x() -> None:
    p = straight_line(length=5.0, step=0.05)
    assert p.poses[0].position.x == pytest.approx(0.0)
    assert p.poses[0].position.y == pytest.approx(0.0)
    assert p.poses[0].orientation.euler[2] == pytest.approx(0.0, abs=1e-9)
    assert p.poses[-1].position.x == pytest.approx(5.0, abs=1e-6)
    assert len(p.poses) == 101


def test_single_corner_endpoint_geometry() -> None:
    p = single_corner(leg_length=2.0, angle_deg=90.0, step=0.05)
    # Last point: corner at (2,0), then 2.0 along +y → (2, 2)
    assert p.poses[-1].position.x == pytest.approx(2.0, abs=1e-6)
    assert p.poses[-1].position.y == pytest.approx(2.0, abs=1e-6)
    # First yaw 0, last yaw pi/2
    assert p.poses[0].orientation.euler[2] == pytest.approx(0.0, abs=1e-9)
    assert p.poses[-1].orientation.euler[2] == pytest.approx(math.pi / 2, abs=1e-6)


def test_circle_closes_and_starts_facing_x() -> None:
    p = circle(radius=1.0, n_points=100)
    # Start at origin facing +x
    assert p.poses[0].position.x == pytest.approx(0.0, abs=1e-9)
    assert p.poses[0].position.y == pytest.approx(0.0, abs=1e-9)
    assert p.poses[0].orientation.euler[2] == pytest.approx(0.0, abs=0.1)
    # Closes: last waypoint coincides with first
    assert p.poses[-1].position.x == pytest.approx(0.0, abs=1e-6)
    assert p.poses[-1].position.y == pytest.approx(0.0, abs=1e-6)
    # All points on circle of radius 1 centered at (0, 1)
    for pose in p.poses:
        d = math.hypot(pose.position.x, pose.position.y - 1.0)
        assert d == pytest.approx(1.0, abs=1e-6)


def test_figure_eight_starts_at_origin() -> None:
    p = figure_eight(loop_radius=1.0, n_points=200)
    assert p.poses[0].position.x == pytest.approx(0.0, abs=1e-9)
    assert p.poses[0].position.y == pytest.approx(0.0, abs=1e-9)
    # Figure-8 is closed
    assert p.poses[-1].position.x == pytest.approx(0.0, abs=1e-6)
    assert p.poses[-1].position.y == pytest.approx(0.0, abs=1e-6)


def test_slalom_passes_centerline_at_each_cone_spacing() -> None:
    p = slalom(cone_spacing=1.0, lateral_offset=0.5, n_cones=5)
    assert p.poses[0].position.x == pytest.approx(0.0, abs=1e-9)
    assert p.poses[0].position.y == pytest.approx(0.0, abs=1e-9)
    # At every multiple of cone_spacing, sin(pix) = 0 → y = 0
    for pose in p.poses:
        if abs(pose.position.x - round(pose.position.x)) < 1e-6:
            assert pose.position.y == pytest.approx(0.0, abs=1e-6)


def test_square_closes() -> None:
    p = square(side=2.0, step=0.05)
    assert p.poses[0].position.x == pytest.approx(0.0, abs=1e-9)
    assert p.poses[0].position.y == pytest.approx(0.0, abs=1e-9)
    assert p.poses[-1].position.x == pytest.approx(0.0, abs=1e-6)
    assert p.poses[-1].position.y == pytest.approx(0.0, abs=1e-6)


def test_default_battery_keys() -> None:
    battery = default_battery()
    expected = {
        "straight_5m",
        "corner_90",
        "circle_R0.5",
        "circle_R1.0",
        "circle_R2.0",
        "figure_eight_R1.0",
        "slalom_5cones",
        "square_2m",
    }
    assert set(battery.keys()) == expected
    for name, path in battery.items():
        assert len(path.poses) > 1, f"path {name} has too few poses"


def test_render_battery_to_svg_artifacts(tmp_path: FsPath) -> None:
    """Render every path to SVG. Inspect by eye if you doubt the geometry."""
    out_dir = tmp_path / "svg"
    out_dir.mkdir()
    for name, path in default_battery().items():
        svg = path_to_svg(path)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        (out_dir / f"{name}.svg").write_text(svg)
    # Sanity: 8 SVGs written
    assert len(list(out_dir.glob("*.svg"))) == 8

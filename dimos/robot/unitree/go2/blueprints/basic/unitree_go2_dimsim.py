#!/usr/bin/env python3
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

"""Go2 + DimSim blueprint — browser-based sim with nav stack.

Drop-in replacement for unitree_go2 that uses DimSim instead of
hardware or MuJoCo. Works on macOS and ARM Linux.
"""

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.robot.sim.bridge import DimSimBridge
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


class _SimLCM(LCM):  # type: ignore[misc]
    """LCM that JPEG-decodes image topics (with fallback to standard decode)."""

    _JPEG_TOPICS = frozenset({"/color_image"})

    def decode(self, msg: bytes, topic: Any) -> Any:  # type: ignore[override]
        topic_str = getattr(topic, "topic", "") or ""
        bare_topic = topic_str.split("#")[0]
        if bare_topic in self._JPEG_TOPICS:
            try:
                return Image.lcm_jpeg_decode(msg)
            except ValueError:
                return super().decode(msg, topic)
        return super().decode(msg, topic)


def _go2_sim_rerun_blueprint() -> Any:
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="camera/color_image", name="Camera"),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.2),
                ),
            ),
            column_shares=[1, 2],
        ),
    )


def _convert_color_image(image: Any) -> Any:
    rerun_data = image.to_rerun()
    return [
        ("camera/color_image", rerun_data),
        ("world/tf/camera_optical/image", rerun_data),
    ]


def _convert_navigation_costmap(grid: Any) -> Any:
    return grid.to_rerun(
        colormap="Accent",
        z_offset=0.015,
        opacity=0.2,
        background="#484981",
    )


def _static_base_link(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(
            half_sizes=[0.35, 0.155, 0.2],
            colors=[(0, 255, 127)],
            fill_mode="wireframe",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


rerun_config = {
    "blueprint": _go2_sim_rerun_blueprint,
    "pubsubs": [_SimLCM()],
    "visual_override": {
        "world/color_image": _convert_color_image,
        "world/navigation_costmap": _convert_navigation_costmap,
    },
    "static": {
        "world/tf/base_link": _static_base_link,
        "world/tf/camera_optical": DimSimBridge.rerun_static_pinhole,
    },
}

if global_config.viewer.startswith("rerun"):
    from dimos.visualization.rerun.bridge import RerunBridgeModule, _resolve_viewer_mode

    with_vis = autoconnect(
        RerunBridgeModule.blueprint(viewer_mode=_resolve_viewer_mode(), **rerun_config),
    )
else:
    with_vis = autoconnect()

unitree_go2_dimsim = (
    autoconnect(
        with_vis,
        DimSimBridge.blueprint(
            scene="apt",
            local=True,
            vehicle_height=0.3,
        ),
        VoxelGridMapper.blueprint(voxel_size=0.1),
        CostMapper.blueprint(),
        ReplanningAStarPlanner.blueprint(),
        WavefrontFrontierExplorer.blueprint(),
        WebsocketVisModule.blueprint(),
    )
    .global_config(n_workers=6, robot_model="unitree_go2", simulation=True)
)

__all__ = ["unitree_go2_dimsim"]

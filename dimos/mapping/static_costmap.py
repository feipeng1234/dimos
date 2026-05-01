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

"""Constant empty global costmap publisher.

Useful on platforms where the depth-render-based lidar pipeline isn't
available (e.g. macOS, where ``mujoco.Renderer`` fails to build Metal
pipeline state inside the dimos worker — same XPC-context-in-forkserver
-child problem the splat camera works around with a subprocess).  The
``CostMapper`` module relies on ``/lidar → /global_map → /global_costmap``;
without a working lidar source, the planner has no costmap and click-
to-nav fails with ``No current global costmap available`` on every
goal.

This module publishes a constant ``width_m`` x ``height_m`` all-free
``OccupancyGrid`` to ``/global_costmap`` on a slow timer (default 1 Hz).
The grid is centered on world origin and configurable in size.

The G1 GR00T sim ships a flat-floor MJCF with no collidable obstacles,
so the all-free grid is correct for the sim's actual physics — clicking
anywhere in the grid triggers a straight-line plan to that point.

Run alongside ``CostMapper`` only when needed: include this module only
on macOS / when you know the depth-render pipeline is silent, otherwise
both will publish to ``/global_costmap`` and the planner will see
duplicate messages (still functionally fine, but noisy).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class Config(ModuleConfig):
    """Configuration for ``StaticCostmapModule``."""

    width_m: float = 50.0
    """Grid width in meters; centered on world origin."""

    height_m: float = 50.0
    """Grid height in meters; centered on world origin."""

    resolution: float = 0.1
    """Grid cell size in meters/cell.  10 cm by default."""

    publish_hz: float = 1.0
    """Publish rate.  Slow is fine — the grid never changes."""

    frame_id: str = "world"


class StaticCostmapModule(Module):
    """Publishes a constant all-free ``OccupancyGrid`` on a timer."""

    config: Config
    global_costmap: Out[OccupancyGrid]

    @rpc
    def start(self) -> None:
        super().start()

        cfg = self.config
        n_cols = round(cfg.width_m / cfg.resolution)
        n_rows = round(cfg.height_m / cfg.resolution)
        self._grid_data = np.zeros((n_rows, n_cols), dtype=np.int8)

        # Centre the grid on world origin.  ``OccupancyGrid.origin`` is the
        # world-frame pose of the grid's (0, 0) cell, so to centre at
        # origin we shift it by minus half the extent.
        self._origin = Pose(-cfg.width_m / 2.0, -cfg.height_m / 2.0, 0.0)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._publish_loop,
            name="static-costmap-publisher",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            f"StaticCostmapModule publishing {n_cols}x{n_rows} all-free "
            f"costmap (res={cfg.resolution}m, extent={cfg.width_m}x"
            f"{cfg.height_m}m centred at origin) at {cfg.publish_hz} Hz"
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        super().stop()

    def _publish_loop(self) -> None:
        cfg = self.config
        period = 1.0 / cfg.publish_hz if cfg.publish_hz > 0 else 1.0
        while not self._stop_event.is_set():
            grid = OccupancyGrid(
                grid=self._grid_data,
                resolution=cfg.resolution,
                origin=self._origin,
                frame_id=cfg.frame_id,
                ts=time.time(),
            )
            try:
                self.global_costmap.publish(grid)
            except Exception as e:
                logger.debug(f"StaticCostmapModule publish failed: {e}")
            self._stop_event.wait(period)


static_costmap = StaticCostmapModule.blueprint

__all__ = ["Config", "StaticCostmapModule", "static_costmap"]

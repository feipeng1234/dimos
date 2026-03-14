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

"""
Agentic integration test: LLM agent navigates the robot via skills.

Starts the full agentic stack (ROSNav + Agent + navigation skill proxy)
in Unity simulation mode, sends a natural-language instruction to the
agent, and verifies that:
  1. The agent calls the correct navigation skill (``goto_global``)
  2. The robot's odom shows it moved toward the target
  3. The navigation completes

The agent discovers navigation skills through a thin worker-side skill
module (``NavSkillProxy``) that uses ``rpc_calls`` to call the Docker-
hosted ROSNav module's ``goto_global`` method via LCM RPC — the same
architecture used by the production ``unitree_g1_agentic_sim`` blueprint.

Uses MockModel in playback mode for deterministic, offline-capable runs.
Set RECORD=1 to re-record fixtures with a real LLM.

Run:
    pytest dimos/navigation/rosnav/test_rosnav_agentic.py -m slow -s

Record new fixture:
    RECORD=1 pytest dimos/navigation/rosnav/test_rosnav_agentic.py -m slow -s
"""

import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
import pytest
from reactivex.disposable import Disposable

from dimos.agents.agent import Agent
from dimos.agents.agent_test_runner import AgentTestRunner
from dimos.agents.annotation import skill
from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.docker_runner import DockerModule
from dimos.core.module import Module
from dimos.core.rpc_client import RPCClient
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.rosnav.rosnav_module import ROSNav

# Where we ask the agent to go.
GOAL_X = 2.0
GOAL_Y = 0.0
POSITION_TOLERANCE = 1.5  # metres

# Timeouts
ODOM_WAIT_SEC = 30
NAV_TIMEOUT_SEC = 180  # total test timeout

FIXTURE_DIR = Path(__file__).parent / "fixtures"

SYSTEM_PROMPT = (
    "You are a robot navigation assistant. You can move the robot using "
    "the goto_global skill which takes (x, y) coordinates in the map frame. "
    "The robot starts at (0, 0). When the user asks you to go somewhere, "
    "call goto_global with the requested coordinates. Do not ask for clarification."
)


class TestAgent(Agent):
    """Agent subclass that filters out DockerModules from on_system_modules.

    DockerModule proxies cannot be unpickled in worker processes (their
    LCM RPC connections are bound to the host process), so we filter
    them out before the base Agent tries to call get_skills() on them.
    Worker-side skill proxies (like NavSkillProxy) provide the bridge.
    """

    @rpc
    def on_system_modules(self, modules: list[RPCClient]) -> None:
        worker_modules = [m for m in modules if not isinstance(m, DockerModule)]
        super().on_system_modules(worker_modules)


class NavSkillProxy(Module):
    """Thin worker-side module that exposes ROSNav's goto_global as an agent skill.

    Uses ``rpc_calls`` to reference the Docker-hosted ROSNav module's
    ``goto_global`` method, which gets wired at build time via LCM RPC.
    This is the same pattern used in the production agentic blueprints.
    """

    rpc_calls: list[str] = ["ROSNav.goto_global"]

    @skill
    def goto_global(self, x: float, y: float) -> str:
        """Go to map coordinates (x, y). The robot starts at (0, 0).

        Args:
            x: X coordinate in the map frame (metres).
            y: Y coordinate in the map frame (metres).

        Returns:
            Status message from the navigation module.
        """
        goto_rpc = self.get_rpc_calls("ROSNav.goto_global")
        return goto_rpc(x, y)


class OdomTracker(Module):
    """Records odom for test assertions."""

    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_odom: PoseStamped | None = None
        self._odom_count = 0

    @rpc
    def start(self) -> None:
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = msg
            self._odom_count += 1

    @rpc
    def get_odom(self) -> PoseStamped | None:
        with self._lock:
            return self._latest_odom

    @rpc
    def get_odom_count(self) -> int:
        with self._lock:
            return self._odom_count

    @rpc
    def stop(self) -> None:
        pass


@pytest.mark.slow
def test_rosnav_agentic_goto():
    """LLM agent uses goto_global skill to navigate the robot to a target."""

    messages = [
        HumanMessage(f"Go to map coordinates ({GOAL_X}, {GOAL_Y})."),
    ]

    agent_kwargs: dict[str, Any] = {"system_prompt": SYSTEM_PROMPT}
    fixture = FIXTURE_DIR / "test_rosnav_agentic_goto.json"
    if bool(os.getenv("RECORD")) or fixture.exists():
        agent_kwargs["model_fixture"] = str(fixture)

    coordinator = (
        autoconnect(
            ROSNav.blueprint(mode="simulation"),
            OdomTracker.blueprint(),
            NavSkillProxy.blueprint(),
            TestAgent.blueprint(**agent_kwargs),
            AgentTestRunner.blueprint(messages=messages),
        )
        .global_config(viewer="none", n_workers=4)
        .build()
    )

    try:
        odom_tracker = coordinator.get_instance(OdomTracker)

        # --- Wait for odom (sim is live) ---
        t0 = time.time()
        while odom_tracker.get_odom_count() == 0:
            if time.time() - t0 > ODOM_WAIT_SEC:
                pytest.fail(f"No odom within {ODOM_WAIT_SEC}s")
            time.sleep(1)

        start_odom = odom_tracker.get_odom()
        print(f"  initial odom: ({start_odom.position.x:.2f}, {start_odom.position.y:.2f})")

        # --- Wait for the robot to reach the target ---
        # The Agent receives the message, calls goto_global via NavSkillProxy,
        # which calls ROSNav.goto_global via RPC (blocking until nav completes).
        # We poll odom to verify position convergence.
        t0 = time.time()
        closest_dist = float("inf")

        while time.time() - t0 < NAV_TIMEOUT_SEC:
            odom = odom_tracker.get_odom()
            if odom is not None:
                dx = odom.position.x - GOAL_X
                dy = odom.position.y - GOAL_Y
                dist = math.sqrt(dx * dx + dy * dy)
                closest_dist = min(closest_dist, dist)

                if dist < POSITION_TOLERANCE:
                    print(
                        f"  robot reached target after {time.time() - t0:.1f}s  "
                        f"pos=({odom.position.x:.2f}, {odom.position.y:.2f})  "
                        f"error={dist:.2f}m"
                    )
                    return  # SUCCESS

            time.sleep(2)

        # -- Timeout --
        final_odom = odom_tracker.get_odom()
        if final_odom:
            dx = final_odom.position.x - GOAL_X
            dy = final_odom.position.y - GOAL_Y
            final_dist = math.sqrt(dx * dx + dy * dy)
            start_dx = final_odom.position.x - (start_odom.position.x if start_odom else 0)
            start_dy = final_odom.position.y - (start_odom.position.y if start_odom else 0)
            moved = math.sqrt(start_dx * start_dx + start_dy * start_dy)

            pytest.fail(
                f"Navigation did not converge within {NAV_TIMEOUT_SEC}s.\n"
                f"  start:     ({start_odom.position.x:.2f}, {start_odom.position.y:.2f})\n"
                f"  final:     ({final_odom.position.x:.2f}, {final_odom.position.y:.2f})\n"
                f"  moved:     {moved:.2f}m\n"
                f"  dist→goal: {final_dist:.2f}m (closest: {closest_dist:.2f}m)\n"
                f"  tolerance: {POSITION_TOLERANCE}m"
            )
        else:
            pytest.fail("No odom received — sim may have crashed")

    finally:
        coordinator.stop()

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

"""Fast tests for Go2 MuJoCo velocity dance stand-in (no MuJoCo subprocess)."""

from __future__ import annotations

from dimos.robot.unitree.mujoco_connection import _go2_mujoco_dance_plan


def test_go2_mujoco_dance_plan_nonempty_and_timed() -> None:
    for variant in (1, 2):
        plan = _go2_mujoco_dance_plan(variant)
        assert len(plan) >= 4
        total_dur = sum(dur for _, dur in plan)
        assert total_dur > 2.0


def test_go2_mujoco_dance_plan_distinct_variants() -> None:
    p1 = _go2_mujoco_dance_plan(1)
    p2 = _go2_mujoco_dance_plan(2)
    assert p1 != p2

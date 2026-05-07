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

"""Hosted teleop subclasses (WebRTC-via-Cloudflare-Realtime transport).

Mirrors the role of ``dimos/teleop/quest/quest_extensions.py`` but for the
hosted module — small overrides on top of ``HostedTeleopModule`` for arm
teleop (per-hand task names + analog trigger packing into button bits).
"""

from typing import Any

from pydantic import Field

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.teleop.quest.quest_teleop_module import Hand
from dimos.teleop.quest.quest_types import Buttons, QuestControllerState
from dimos.teleop.quest_hosted.hosted_teleop_module import (
    HostedTeleopConfig,
    HostedTeleopModule,
)


class HostedArmTeleopConfig(HostedTeleopConfig):
    """Adds ``task_names`` for routing per-hand commands to coordinator tasks.

    ``task_names`` maps lower-case hand names (``"left"``, ``"right"``) to
    the coordinator task name (e.g. ``"teleop_xarm"``). Used to set
    ``frame_id`` on the published ``PoseStamped`` so the coordinator routes
    to the correct ``TeleopIKTask``.
    """

    task_names: dict[str, str] = Field(default_factory=dict)


class HostedArmTeleopModule(HostedTeleopModule):
    """Hosted teleop with per-hand task_name routing + analog trigger packing.

    Same overrides as ``ArmTeleopModule`` but on top of the WebRTC-via-broker
    ``HostedTeleopModule`` instead of the local-WebSocket ``QuestTeleopModule``.
    """

    config: HostedArmTeleopConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task_names: dict[Hand, str] = {
            Hand[k.upper()]: v for k, v in self.config.task_names.items()
        }

    def _publish_msg(self, hand: Hand, output_msg: PoseStamped) -> None:
        """Stamp ``frame_id`` with the configured task name, then publish."""
        task_name = self._task_names.get(hand)
        if task_name:
            output_msg = PoseStamped(
                position=output_msg.position,
                orientation=output_msg.orientation,
                ts=output_msg.ts,
                frame_id=task_name,
            )
        super()._publish_msg(hand, output_msg)

    def _publish_button_state(
        self,
        left: QuestControllerState | None,
        right: QuestControllerState | None,
    ) -> None:
        """Publish ``Buttons`` with analog triggers packed into bits 16-29."""
        buttons = Buttons.from_controllers(left, right)
        buttons.pack_analog_triggers(
            left=left.trigger if left is not None else 0.0,
            right=right.trigger if right is not None else 0.0,
        )
        self.buttons.publish(buttons)

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

"""unitree-go2-twitch — Twitch Plays Go2.

Usage::

    export DIMOS_TWITCH_TOKEN=oauth:your_token
    export DIMOS_CHANNEL_NAME=your_channel
    dimos run unitree-go2-twitch --robot-ip 192.168.123.161
"""

from __future__ import annotations

import time

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.stream.twitch.votes import TwitchChoice, TwitchVotes
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class _ChoiceToCmdVel(Module):
    config: ModuleConfig
    command_duration: float = 1.0

    chat_vote_choice: In[TwitchChoice]
    cmd_vel: Out[Twist]

    _connection: GO2ConnectionSpec

    @rpc
    def start(self) -> None:
        super().start()
        self.chat_vote_choice.subscribe(self._on_choice)

    def _on_choice(self, choice: TwitchChoice) -> None:
        logger.info("[TwitchPlays] Executing: %s", choice.winner)

        if choice.winner == "sit":
            self._do_sport_command("Sit")
            return
        elif choice.winner == "stand":
            self._do_sport_command("StandUp")
            return

        t = Twist()
        if choice.winner == "forward":
            t.linear.x = 0.3
        elif choice.winner == "back":
            t.linear.x = -0.3
        elif choice.winner == "left":
            t.angular.z = 0.5
        elif choice.winner == "right":
            t.angular.z = -0.5

        end = time.time() + self.command_duration
        while time.time() < end:
            self.cmd_vel.publish(t)
            time.sleep(0.1)

        self.cmd_vel.publish(Twist())

    def _do_sport_command(self, command_name: str) -> None:
        api_id = SPORT_CMD[command_name]
        logger.info("[TwitchPlays] Sport command: %s (api_id=%d)", command_name, api_id)
        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": api_id})
        except Exception:
            logger.exception("[TwitchPlays] Failed to execute %s", command_name)


unitree_go2_twitch = autoconnect(
    unitree_go2_basic,
    TwitchVotes.blueprint(
        vote_window_seconds=5.0,
        vote_mode="plurality",
    ),
    _ChoiceToCmdVel.blueprint(),
).global_config(n_workers=4, robot_model="unitree_go2")

__all__ = ["unitree_go2_twitch"]

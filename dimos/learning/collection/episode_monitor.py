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

"""Single point of teleop-input → EpisodeStatus translation.

Watches buttons / keyboard, runs the start/save/discard state machine,
publishes EpisodeStatus on every transition. RecordReplay captures that
stream into session.db; DataPrep reads only the recorded EpisodeStatus
events offline — never raw buttons or keypresses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.teleop.quest.quest_types import Buttons

# Friendly names → Quest Buttons attribute names. Override by supplying an
# attribute name directly in `button_map`.
BUTTON_ALIASES: dict[str, str] = {
    "A": "right_primary",
    "B": "right_secondary",
    "X": "left_primary",
    "Y": "left_secondary",
    "LT": "left_trigger",
    "RT": "right_trigger",
    "LG": "left_grip",
    "RG": "right_grip",
    "MENU_L": "left_menu",
    "MENU_R": "right_menu",
}


class EpisodeStatus(BaseModel):
    state: Literal["idle", "recording"]
    episodes_saved: int
    episodes_discarded: int
    current_episode_start_ts: float | None
    last_event: Literal["start", "save", "discard", "init"] = "init"
    task_label: str | None = None


class KeyPress(BaseModel):
    """Single keypress event from a keyboard input source."""

    key: str
    ts: float


class EpisodeMonitorModuleConfig(ModuleConfig):
    button_map: dict[Literal["start", "save", "discard"], str] = {
        "start": "A",
        "save": "B",
        "discard": "X",
    }
    keyboard_map: dict[Literal["start", "save", "discard"], str] = {}
    default_task_label: str | None = None


class EpisodeMonitorModule(Module):
    config: EpisodeMonitorModuleConfig

    buttons: In[Buttons]
    keyboard: In[KeyPress]
    status: Out[EpisodeStatus]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state: Literal["idle", "recording"] = "idle"
        self._saved: int = 0
        self._discarded: int = 0
        self._current_start_ts: float | None = None
        self._prev_bits: dict[str, bool] = {}  # rising-edge detection for buttons

    @rpc
    def start(self) -> None:
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        raise NotImplementedError

    @rpc
    def reset_counters(self) -> EpisodeStatus:
        raise NotImplementedError

    @rpc
    def get_status(self) -> EpisodeStatus:
        raise NotImplementedError

    def _on_buttons(self, msg: Buttons) -> None:
        """Rising-edge detect against `config.button_map`; advance state machine."""
        raise NotImplementedError

    def _on_keyboard(self, msg: KeyPress) -> None:
        """Match `msg.key` against `config.keyboard_map`; advance state machine."""
        raise NotImplementedError

    def _transition(self, event: Literal["start", "save", "discard"], ts: float) -> None:
        """Apply the state-machine transition and publish EpisodeStatus.

        IDLE      --start-->     RECORDING
        RECORDING --save-->      IDLE        (commit, saved += 1)
        RECORDING --discard-->   IDLE        (drop, discarded += 1)
        RECORDING --start-->     RECORDING   (auto-commit prev, begin new)
        """
        raise NotImplementedError

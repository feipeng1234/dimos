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

"""Live episode-status feedback during teleop recording.

Watches the buttons stream and runs the same start/save/discard state
machine that `DataPrep.extract_episodes` runs offline — but here it runs
live so the operator can see counters update in real time. Pure observability:
this module does NOT write anything. The recording itself is RecordReplay's
job; episode boundary extraction still happens post-hoc inside DataPrep.

Why a separate live state-machine instead of just consuming DataPrep's offline
output? Because the operator wants feedback *during* the session ("episodes
saved: 12") to know when to stop, retry a bad demo, etc.

Agent surface: `get_status()` returns the latest counters; `reset_counters()`
zeroes them between recording sessions without restarting the blueprint.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.teleop.quest.quest_types import Buttons


class EpisodeStatus(BaseModel):
    """Live counters published every state transition."""

    state: Literal["idle", "recording"]
    episodes_saved: int
    episodes_discarded: int
    current_episode_start_ts: float | None  # None when state == "idle"
    last_event: Literal["start", "save", "discard", "init"] = "init"


class EpisodeMonitorModuleConfig(ModuleConfig):
    """Match the same fields used by `EpisodeConfig` in the dataset spec
    so the live monitor and the offline extractor agree on what each button
    means. Friendly names ("A", "B", "X") resolve via BUTTON_ALIASES.
    """

    button_stream: str = "buttons"
    start: str = "A"
    save: str = "B"
    discard: str = "X"


class EpisodeMonitorModule(Module):
    """Live operator feedback for teleop recording sessions."""

    config: EpisodeMonitorModuleConfig

    buttons: In[Buttons]

    status: Out[EpisodeStatus]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state: Literal["idle", "recording"] = "idle"
        self._saved: int = 0
        self._discarded: int = 0
        self._current_start_ts: float | None = None
        # Previous bit-state of each watched button, for rising-edge detection.
        self._prev_bits: dict[str, bool] = {}

    @rpc
    def start(self) -> None:
        """Subscribe to `buttons` and emit an initial idle status."""
        raise NotImplementedError

    @rpc
    def stop(self) -> None:
        """Unsubscribe and call super().stop()."""
        raise NotImplementedError

    @rpc
    def reset_counters(self) -> EpisodeStatus:
        """Zero the saved/discarded counters and force state back to idle.
        Returns the new status."""
        raise NotImplementedError

    @rpc
    def get_status(self) -> EpisodeStatus:
        """Return the current EpisodeStatus snapshot."""
        raise NotImplementedError

    # ── internals ────────────────────────────────────────────────────────────

    def _on_buttons(self, msg: Buttons) -> None:
        """Detect rising edges on start/save/discard buttons; advance state
        machine; publish EpisodeStatus on every transition.

        State machine — must mirror DataPrep.extract_episodes in BUTTONS mode:
            IDLE       --start press--> RECORDING (begin)
            RECORDING  --save press---> IDLE      (saved += 1)
            RECORDING  --discard ----->  IDLE      (discarded += 1)
            RECORDING  --start press--> RECORDING (auto-commit prev, begin new)
        """
        raise NotImplementedError

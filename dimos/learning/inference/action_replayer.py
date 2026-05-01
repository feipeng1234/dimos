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

"""Replay policy-emitted ActionChunks at the coordinator's tick rate.

`ChunkPolicyModule` runs slow (1–30 Hz) and emits sequences of future actions
(ActionChunks). The coordinator runs at 100 Hz. This task bridges them:
subscribe to the chunk topic, maintain a small buffer of pending (target_ts,
positions) entries, and on each tick interpolate to the current time.

Lives in the tick loop because hardware writes happen there. Designed so a
slow / stalled policy doesn't crash the controller — see "fault behavior"
below.
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.control.task import (
    BaseControlTask,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
)
from dimos.learning.policy.base import ActionChunk


@dataclass
class ActionReplayerConfig:
    """Configuration for ActionReplayer.

    Attributes:
        joint_names: joints this task commands. Must match the policy's
            `joint_names` (caller is responsible — typically wired from the
            checkpoint's `dimos_meta.json`).
        chunk_topic: name of the topic ChunkPolicyModule publishes on.
            ActionReplayer subscribes via the coordinator's transport.
        priority: tick-loop arbitration priority.
        max_chunk_age_s: drop any incoming chunk whose `ts` is more than this
            many seconds old at receive time. Guards against stalls.
        hold_on_stall: if the buffer empties (policy fell behind / died),
            hold the last commanded position instead of returning None
            (which would let lower-priority tasks take over).
        temporal_ensemble: when overlapping chunks arrive, exponentially
            weight predictions for the same target time (ACT trick).
            Off by default; v1 nice-to-have.
    """

    joint_names: list[str]
    chunk_topic: str = "action_chunk"
    priority: int = 10
    max_chunk_age_s: float = 0.5
    hold_on_stall: bool = True
    temporal_ensemble: bool = False


class ActionReplayer(BaseControlTask):
    """ControlTask that replays policy chunks into joint commands at tick rate.

    Behavior:
      - On each new chunk, drop any buffered targets at or after the new
        chunk's first target_ts (latest chunk wins).
      - On each tick, interpolate (or look up nearest) target for `state.now`.
      - If `state.now` is past the buffer end:
          - hold last position if `hold_on_stall=True`
          - else go inactive (return None)
      - Stale chunks (`now - chunk.ts > max_chunk_age_s`) are dropped.

    Fault behavior:
      - Policy dies / module crashes: buffer drains, behavior degrades to
        "hold last position" (or inactive). Hardware never sees zero or NaN.
    """

    def __init__(self, name: str, config: ActionReplayerConfig) -> None:
        """Initialize. Subscription to `chunk_topic` is set up by the coordinator
        when the task is registered (we expose `on_action_chunk` for it to call).
        """
        raise NotImplementedError

    # ── ControlTask interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        raise NotImplementedError

    def claim(self) -> ResourceClaim:
        """Claim `config.joint_names` at `config.priority`."""
        raise NotImplementedError

    def is_active(self) -> bool:
        """Active iff the buffer has a non-stale target for `now` (or
        `hold_on_stall` is true and we've ever received a chunk)."""
        raise NotImplementedError

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        """Return interpolated joint targets for `state.now`.

        Pure lookup over the buffered chunk; no model inference happens here.
        Must complete in well under 10 ms to not jeopardize the 100 Hz loop.
        """
        raise NotImplementedError

    # ── chunk handling ───────────────────────────────────────────────────────

    def on_action_chunk(self, msg: ActionChunk) -> None:
        """Push a new chunk's actions into the buffer.

        Steps:
          1. If `time_now - msg.ts > max_chunk_age_s`: drop and log.
          2. Compute target_ts for each action: `msg.ts + i * msg.dt`.
          3. Drop any buffered entries with target_ts >= msg.ts + msg.dt.
          4. Append the new (target_ts, positions) pairs in order.
        """
        raise NotImplementedError

    # ── internals ────────────────────────────────────────────────────────────

    def _interpolate(self, t: float) -> JointCommandOutput | None:
        """Look up or linearly interpolate the buffer at time `t`. Returns
        None if `t` is outside the buffered range and `hold_on_stall=False`."""
        raise NotImplementedError

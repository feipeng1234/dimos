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

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

from dimos.protocol.pubsub.encoders import (
    JpegEncoderMixin,
    LCMEncoderMixin,
    PickleEncoderMixin,
)
from dimos.protocol.pubsub.patterns import Glob
from dimos.protocol.pubsub.spec import AllPubSub
from dimos.protocol.service.lcmservice import LCMConfig, LCMService, NotStartedError, autoconf
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    import threading

    from dimos.msgs import DimosMsg

logger = setup_logger()


@dataclass
class Topic:
    topic: str | re.Pattern[str] | Glob
    lcm_type: type[DimosMsg] | None = None

    @property
    def is_pattern(self) -> bool:
        return isinstance(self.topic, re.Pattern | Glob)

    @property
    def pattern(self) -> str:
        if isinstance(self.topic, re.Pattern):
            return self.topic.pattern
        if isinstance(self.topic, Glob):
            return self.topic.pattern
        return self.topic

    def __str__(self) -> str:
        if self.lcm_type is None:
            return self.pattern
        return f"{self.pattern}#{self.lcm_type.msg_name}"

    @staticmethod
    def from_channel_str(channel: str, default_lcm_type: type[DimosMsg] | None = None) -> Topic:
        """Create Topic from channel string.

        Channel format: /topic#module.ClassName
        Falls back to default_lcm_type if type cannot be parsed.
        """
        from dimos.msgs import resolve_msg_type

        if "#" not in channel:
            return Topic(topic=channel, lcm_type=default_lcm_type)

        topic_str, type_name = channel.rsplit("#", 1)
        lcm_type = resolve_msg_type(type_name)
        return Topic(topic=topic_str, lcm_type=lcm_type or default_lcm_type)


class LCMPubSubBase(LCMService, AllPubSub[Topic, Any]):
    """LCM-based PubSub with native regex subscription support.

    LCM natively supports regex patterns in subscribe(), so we implement
    RegexSubscribable directly without needing discovery-based fallback.
    """

    default_config = LCMConfig
    _stop_event: threading.Event
    _thread: threading.Thread | None
    _que_capacity: int = 10000

    def publish(self, topic: Topic | str, message: bytes) -> None:
        """Publish a message to the specified channel."""
        lcm_obj = self._l
        if not lcm_obj:
            logger.error("Tried to publish after LCM was stopped (or never started)")
            return
        topic_str = str(topic) if isinstance(topic, Topic) else topic
        lcm_obj.publish(topic_str, message)

    def subscribe_all(self, callback: Callable[[bytes, Topic], Any]) -> Callable[[], None]:
        return self.subscribe(Topic(re.compile(".*")), callback)  # type: ignore[arg-type]

    def subscribe(
        self, topic: Topic, callback: Callable[[bytes, Topic], None]
    ) -> Callable[[], None]:
        lcm_obj = self._l
        if not lcm_obj:
            logger.error("Tried to subscribe after LCM was closed")
            return lambda: None

        if topic.is_pattern:

            def handler(channel: str, msg: bytes) -> None:
                if channel == "LCM_SELF_TEST":
                    return
                callback(msg, Topic.from_channel_str(channel, topic.lcm_type))

            pattern_str = str(topic)
            if not pattern_str.endswith("*"):
                pattern_str = f"{pattern_str}(#.*)?"

            lcm_subscription = lcm_obj.subscribe(pattern_str, handler)
        else:
            topic_str = str(topic)
            lcm_subscription = lcm_obj.subscribe(topic_str, lambda _, msg: callback(msg, topic))

        # Set queue capacity to 10000 to handle high-volume bursts
        lcm_subscription.set_queue_capacity(self._que_capacity)

        def unsubscribe() -> None:
            try:
                self.l.unsubscribe(lcm_subscription)
            except NotStartedError:
                pass

        return unsubscribe


# these ignoress might be unsolvable
# and should use composition not inheritance for encoding/decoding


class LCM(  # type: ignore[misc]
    LCMEncoderMixin,  # type: ignore[type-arg]
    LCMPubSubBase,
): ...


class PickleLCM(  # type: ignore[misc]
    PickleEncoderMixin,  # type: ignore[type-arg]
    LCMPubSubBase,
): ...


class JpegLCM(  # type: ignore[misc]
    JpegEncoderMixin,  # type: ignore[type-arg]
    LCMPubSubBase,
): ...


__all__ = [
    "LCM",
    "Glob",
    "JpegLCM",
    "LCMEncoderMixin",
    "LCMPubSubBase",
    "PickleLCM",
    "Topic",
    "autoconf",
]

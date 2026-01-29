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

"""Rerun bridge for logging pubsub messages with to_rerun() methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import rerun as rr

from dimos.core import Module, rpc
from dimos.core.module import ModuleConfig
from dimos.protocol.pubsub.impl.lcmpubsub import LCM

if TYPE_CHECKING:
    from collections.abc import Callable

    from dimos.protocol.pubsub.spec import SubscribeAllCapable

ViewerMode = Literal["native", "web", "none"]


@dataclass
class Config(ModuleConfig):
    """Configuration for RerunBridgeModule."""

    pubsubs: list[SubscribeAllCapable[Any, Any]] = field(
        default_factory=lambda: [LCM(autoconf=True)]
    )
    entity_prefix: str = "world"
    topic_to_entity: Callable[[Any], str] | None = None
    viewer_mode: ViewerMode = "native"


class RerunBridgeModule(Module):
    """Bridge that logs messages from pubsubs to Rerun.

    Spawns its own Rerun viewer and subscribes to all topics on each provided
    pubsub. Any message that has a to_rerun() method is automatically logged.

    Example:
        from dimos.protocol.pubsub.impl.lcmpubsub import LCM

        lcm = LCM(autoconf=True)
        bridge = RerunBridgeModule(pubsubs=[lcm])
        bridge.start()
        # All messages with to_rerun() are now logged to Rerun
        bridge.stop()
    """

    default_config = Config
    config: Config

    def _get_entity_path(self, topic: Any) -> str:
        """Convert a topic to a Rerun entity path."""
        if self.config.topic_to_entity:
            return self.config.topic_to_entity(topic)

        # Default: use topic.name if available (LCM Topic), else str
        topic_str = getattr(topic, "name", None) or str(topic)
        return f"{self.config.entity_prefix}/{topic_str}"

    def _on_message(self, msg: Any, topic: Any) -> None:
        """Handle incoming message - log to rerun if it has to_rerun."""
        if not hasattr(msg, "to_rerun"):
            return

        rerun_data = msg.to_rerun()

        # TFMessage returns list of (entity_path, transform) tuples
        if isinstance(rerun_data, list):
            for entity_path, archetype in rerun_data:
                rr.log(entity_path, archetype)
        else:
            entity_path = self._get_entity_path(topic)
            rr.log(entity_path, rerun_data)

    @rpc
    def start(self) -> None:
        from reactivex.disposable import Disposable

        super().start()

        # Initialize and spawn Rerun viewer
        rr.init("dimos-bridge")
        if self.config.viewer_mode == "native":
            rr.spawn(connect=True)
        elif self.config.viewer_mode == "web":
            rr.serve_web_viewer(open_browser=True)
        # "none" - just init, no viewer (connect externally)

        # Start pubsubs and subscribe to all messages
        for pubsub in self.config.pubsubs:
            if hasattr(pubsub, "start"):
                pubsub.start()  # type: ignore[union-attr]
            unsub = pubsub.subscribe_all(self._on_message)
            self._disposables.add(Disposable(unsub))

        # Add pubsub stop as disposable
        for pubsub in self.config.pubsubs:
            if hasattr(pubsub, "stop"):
                self._disposables.add(Disposable(pubsub.stop))  # type: ignore[union-attr]


rerun_bridge = RerunBridgeModule.blueprint


def main() -> None:
    """CLI entry point for rerun-bridge."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Rerun bridge for LCM messages")
    parser.add_argument(
        "--viewer-mode",
        choices=["native", "web", "none"],
        default="native",
        help="Viewer mode: native (desktop), web (browser), none (headless)",
    )
    args = parser.parse_args()

    bridge = RerunBridgeModule(viewer_mode=args.viewer_mode)
    bridge.start()

    signal.signal(signal.SIGINT, lambda *_: bridge.stop())
    signal.pause()


if __name__ == "__main__":
    main()

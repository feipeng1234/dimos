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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stream import Stream, TextStream
    from .transformer import Transformer
    from .types import PoseProvider, StreamInfo


class Session(ABC):
    """A session against a memory store. Creates and manages streams."""

    @abstractmethod
    def stream(
        self,
        name: str,
        payload_type: type | None = None,
        *,
        pose_provider: PoseProvider | None = None,
    ) -> Stream[Any]:
        """Get or create a stored stream backed by the database."""

    @abstractmethod
    def text_stream(
        self,
        name: str,
        payload_type: type | None = None,
        *,
        tokenizer: str = "unicode61",
        pose_provider: PoseProvider | None = None,
    ) -> TextStream[Any]:
        """Get or create a text stream with FTS index."""

    @abstractmethod
    def list_streams(self) -> list[StreamInfo]: ...

    @abstractmethod
    def materialize_transform(
        self,
        name: str,
        source: Stream[Any],
        transformer: Transformer[Any, Any],
        *,
        live: bool = False,
        backfill_only: bool = False,
    ) -> Stream[Any]:
        """Create a stored stream from a transform pipeline."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class Store(ABC):
    """Top-level entry point — wraps a database file."""

    @abstractmethod
    def session(self) -> Session: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

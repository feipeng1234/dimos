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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from dimos.models.embedding.base import Embedding
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

PoseProvider: TypeAlias = Callable[[], Any]  # () -> PoseLike | None

_UNSET: Any = object()


@dataclass
class Observation:
    id: int
    ts: float | None = None
    pose: PoseStamped | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    parent_id: int | None = field(default=None, repr=False)
    _data: Any = field(default=_UNSET, repr=False)
    _data_loader: Callable[[], Any] | None = field(default=None, repr=False, compare=False)

    @property
    def data(self) -> Any:
        if self._data is not _UNSET:
            return self._data
        if self._data_loader is not None:
            self._data = self._data_loader()
            return self._data
        raise LookupError("No data available; observation was not fetched with payload")


@dataclass
class EmbeddingObservation(Observation):
    """Returned by EmbeddingStream terminals.

    .data auto-projects to the source stream's payload type.
    .embedding gives the Embedding vector.
    """

    _embedding: Embedding | None = field(default=None, repr=False)
    _embedding_loader: Callable[[], Embedding] | None = field(
        default=None, repr=False, compare=False
    )
    _source_data_loader: Callable[[], Any] | None = field(default=None, repr=False, compare=False)

    @property
    def data(self) -> Any:
        if self._data is not _UNSET:
            return self._data
        if self._source_data_loader is not None:
            self._data = self._source_data_loader()
            return self._data
        return super().data

    @property
    def embedding(self) -> Embedding:
        if self._embedding is not None:
            return self._embedding
        if self._embedding_loader is not None:
            self._embedding = self._embedding_loader()
            return self._embedding
        raise LookupError("No embedding available")


@dataclass
class StreamInfo:
    name: str
    payload_type: str | None = None
    count: int = 0


# ── Filter types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AfterFilter:
    t: float


@dataclass(frozen=True)
class BeforeFilter:
    t: float


@dataclass(frozen=True)
class TimeRangeFilter:
    t1: float
    t2: float


@dataclass(frozen=True)
class AtFilter:
    t: float
    tolerance: float


@dataclass(frozen=True)
class NearFilter:
    pose: Any  # PoseLike
    radius: float


@dataclass(frozen=True)
class TagsFilter:
    tags: dict[str, Any]


@dataclass(frozen=True)
class EmbeddingSearchFilter:
    query: list[float]
    k: int


@dataclass(frozen=True)
class TextSearchFilter:
    text: str
    k: int | None


Filter: TypeAlias = (
    AfterFilter
    | BeforeFilter
    | TimeRangeFilter
    | AtFilter
    | NearFilter
    | TagsFilter
    | EmbeddingSearchFilter
    | TextSearchFilter
)


@dataclass(frozen=True)
class StreamQuery:
    """Immutable bundle of query parameters passed to backends."""

    filters: tuple[Filter, ...] = ()
    order_field: str | None = None
    order_desc: bool = False
    limit_val: int | None = None
    offset_val: int | None = None
